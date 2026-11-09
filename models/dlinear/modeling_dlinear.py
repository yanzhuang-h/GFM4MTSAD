import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class DLinear(nn.Module):
    def __init__(self, cfg):
        super(DLinear, self).__init__()
        self.seq_len = cfg.DATA.WIN_SIZE
        self.feature_dim = cfg.DLINEAR.FEATURE_DIM

        self.individual = cfg.DLINEAR.INDIVIDUAL
        self.channels = cfg.DATA.N_VAR
        self.moving_avg_kernel = int(
            getattr(cfg.DLINEAR, "MOVING_AVG_KERNEL", 25)
        )
        if self.moving_avg_kernel <= 0 or self.moving_avg_kernel % 2 == 0:
            raise ValueError(
                "DLINEAR.MOVING_AVG_KERNEL must be a positive odd integer"
            )
        self.decomposition = SeriesDecomp(
            kernel_size=self.moving_avg_kernel
        )

        if self.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            for _ in range(self.channels):
                seasonal = nn.Linear(self.seq_len, self.feature_dim)
                trend = nn.Linear(self.seq_len, self.feature_dim)
                seasonal.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.feature_dim, self.seq_len])
                )
                trend.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.feature_dim, self.seq_len])
                )
                self.linear_seasonal.append(seasonal)
                self.linear_trend.append(trend)
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.feature_dim)
            self.linear_trend = nn.Linear(self.seq_len, self.feature_dim)
            self.linear_seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.feature_dim, self.seq_len])
            )
            self.linear_trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.feature_dim, self.seq_len])
            )

        self._init_biases()
        if cfg.DLINEAR.FREEZE:
            self.freeze()

    def _init_biases(self):
        if self.individual:
            modules = list(self.linear_seasonal) + list(self.linear_trend)
        else:
            modules = [self.linear_seasonal, self.linear_trend]
        for module in modules:
            nn.init.zeros_(module.bias)

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        seasonal_output, trend_output = self.forward_components(x)
        return seasonal_output + trend_output

    def forward_channel(self, x, channel_index):
        """Encode one channel without evaluating unrelated sensor branches.

        DLinear's decomposition is channel independent.  The GFM4MTSAD v2.2
        negative selector exploits that contract to batch candidate edits for
        one directly intervened sensor while producing exactly the same token
        as a full ``forward`` call.
        """
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3 or x.size(-1) != 1:
            raise ValueError(
                "DLinear.forward_channel expects [B, T] or [B, T, 1], "
                f"got {tuple(x.shape)}"
            )
        channel_index = int(channel_index)
        if not 0 <= channel_index < self.channels:
            raise IndexError(
                f"DLinear channel index {channel_index} outside [0, {self.channels})"
            )
        seasonal, trend = self.decomposition(x)
        seasonal = seasonal[:, :, 0]
        trend = trend[:, :, 0]
        if self.individual:
            seasonal = self.linear_seasonal[channel_index](seasonal)
            trend = self.linear_trend[channel_index](trend)
        else:
            seasonal = self.linear_seasonal(seasonal)
            trend = self.linear_trend(trend)
        return seasonal + trend

    def forward_components(self, x):
        """返回显式 seasonal/trend token，forward 保持原有求和语义。"""
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = seasonal_init.new_zeros(
                seasonal_init.size(0), seasonal_init.size(1), self.feature_dim
            )
            trend_output = trend_init.new_zeros(
                trend_init.size(0), trend_init.size(1), self.feature_dim
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        return seasonal_output, trend_output
