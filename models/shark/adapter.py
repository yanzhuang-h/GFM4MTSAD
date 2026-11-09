import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAdapter(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(FeatureAdapter, self).__init__()
        self.method = cfg.SHARK.ADAPTER.METHOD
        self.input_dim = input_dim
        self.output_dim = output_dim
        if self.output_dim > self.input_dim:
            raise ValueError("Adapter output_dim must be <= input_dim")
        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=True)
        self._init_slice_projection()
        self.fitted = self.method != "pca"
        if cfg.SHARK.ADAPTER.FREEZE:
            self.freeze()

    def _init_slice_projection(self):
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            self.proj.weight[:, :self.output_dim] = torch.eye(self.output_dim)

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, node_features):
        return self.proj(node_features)

    @torch.no_grad()
    def fit(self, node_features):
        if self.method != "pca":
            return
        rows = node_features.reshape(-1, node_features.size(-1)).float()
        rows = rows[torch.isfinite(rows).all(dim=1)]
        if rows.size(0) < 2:
            return
        mean = rows.mean(dim=0)
        centered = rows - mean
        _, _, vh = torch.linalg.svd(centered.cpu(), full_matrices=False)
        components = vh[:self.output_dim].to(device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        bias = -torch.matmul(components, mean.to(components.device, components.dtype))
        with torch.no_grad():
            self.proj.weight.copy_(components)
            self.proj.bias.copy_(bias)
        self.fitted = True


def build_adapter(cfg, input_dim, output_dim):
    name = cfg.SHARK.ADAPTER.NAME
    if name == "FeatureAdapter":
        return FeatureAdapter(cfg, input_dim, output_dim)
    raise ValueError(f"Unsupported SHARK.ADAPTER.NAME: {name}")


class SensorGraphPrompt(nn.Module):
    """A small PEFT layer that adapts sensor-graph features before AnomalyGFM."""

    def __init__(self, cfg, feature_dim):
        super(SensorGraphPrompt, self).__init__()
        tuning_enabled = bool(getattr(cfg.SHARK.GFM_TUNING, "ENABLE", False))
        strategy = getattr(cfg.SHARK.GFM_TUNING, "STRATEGY", "frozen")
        self.enabled = tuning_enabled and self._uses_sensor_graph_prompt(strategy)
        self.scale = float(cfg.SHARK.GFM_TUNING.SENSOR_GRAPH_PROMPT_SCALE)
        self.dropout = nn.Dropout(float(cfg.SHARK.GFM_TUNING.SENSOR_GRAPH_PROMPT_DROPOUT))

        self.node_prompt = nn.Parameter(torch.zeros(feature_dim))
        self.graph_prompt = nn.Parameter(torch.zeros(feature_dim))
        self.context_proj = nn.Sequential(
            nn.Linear(feature_dim + 1, feature_dim),
            nn.Tanh(),
        )
        self._init_parameters(float(cfg.SHARK.GFM_TUNING.SENSOR_GRAPH_PROMPT_INIT_STD))

        if not self.enabled:
            self.freeze()

    @staticmethod
    def _uses_sensor_graph_prompt(strategy):
        return strategy in {
            "sensor_graph_prompt",
            "sensor_graph_prompt_anomaly",
            "sensor_graph_prompt_prompt",
            "sensor_graph_prompt_adapter",
        }

    def _init_parameters(self, std):
        nn.init.normal_(self.node_prompt, mean=0.0, std=std)
        nn.init.normal_(self.graph_prompt, mean=0.0, std=std)
        for module in self.context_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, node_features, adjacency):
        if not self.enabled:
            return node_features
        if adjacency.dim() != 3:
            raise ValueError(f"SensorGraphPrompt expects adjacency [B, N, N], got {tuple(adjacency.shape)}")
        if node_features.dim() != 3:
            raise ValueError(f"SensorGraphPrompt expects node_features [B, N, F], got {tuple(node_features.shape)}")

        degree = adjacency.float().clamp_min(0.0).sum(dim=-1, keepdim=True)
        degree = degree / degree.amax(dim=1, keepdim=True).clamp_min(1e-6)
        graph_context = node_features.mean(dim=1, keepdim=True).expand_as(node_features)
        context_prompt = self.context_proj(torch.cat([graph_context, degree], dim=-1))
        prompt = self.node_prompt.view(1, 1, -1) + degree * self.graph_prompt.view(1, 1, -1)
        prompt = prompt + context_prompt
        return node_features + self.scale * self.dropout(F.normalize(prompt, p=2, dim=-1))


def build_sensor_graph_prompt(cfg, feature_dim):
    return SensorGraphPrompt(cfg, feature_dim)
