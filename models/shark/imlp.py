import torch.nn as nn

from models.dlinear.modeling_dlinear import DLinear


class DLinear_MLP(nn.Module):
    def __init__(self, cfg):
        super(DLinear_MLP, self).__init__()
        self.dlinear = DLinear(cfg)

    def forward(self, x):
        return self.dlinear(x)

    def forward_channel(self, x, channel_index):
        return self.dlinear.forward_channel(x, channel_index)

    def forward_components(self, x):
        return self.dlinear.forward_components(x)
