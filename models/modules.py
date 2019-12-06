import torch.nn as nn
from torch.nn.utils import spectral_norm

class Conv1d(nn.Module):

    "Conv1d for spectral normalisation and orthogonal initialisation"

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 dilation=1,
                 groups=1):
        super(Conv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        pad = dilation * (kernel_size - 1) // 2

        layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=pad, dilation=dilation, groups=groups)
        nn.init.orthogonal_(layer.weight)
        self.layer = spectral_norm(layer)

    def forward(self, inputs):
        return self.layer(inputs)

class Linear(nn.Module):

    "Linear for spectral normalisation and orthogonal initialisation"

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        layer = nn.Linear(in_features, out_features, bias)
        nn.init.orthogonal_(layer.weight)
        self.layer = spectral_norm(layer)

    def forward(self, inputs):
        return self.layer(inputs)
