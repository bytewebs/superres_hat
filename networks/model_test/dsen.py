import math
import torch
import functools
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F


class DSen2(nn.Module):
    def __init__(self, scale=4, num_layers=32, feature_size=256):
        super(DSen2, self).__init__()
        
        self.scale = scale
        self.conv1 = nn.Conv2d(8, feature_size, (3, 3), (1, 1), (1, 1))
        
        trunk = []
        for _ in range(num_layers):
            trunk.append(ResBlock(feature_size))
        self.trunk = nn.Sequential(*trunk)
        
        self.conv_out = nn.Conv2d(feature_size, 4, (3, 3), (1, 1), (1, 1))
        self.relu = nn.ReLU(True)
        
    def forward(self, x, ref_x):
        x_bar = F.interpolate(x, scale_factor=self.scale, mode="bicubic")
        out = self.relu(self.conv1(torch.cat([ref_x, x_bar], 1)))
        out = self.trunk(out)
        out = self.conv_out(out)
        
        # Adding initial upsampling back
        x = torch.add(x_bar, out)
        x = torch.clamp_(x, 0.0, 1.0)
        
        return x
        

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))
        self.relu = nn.ReLU(True)
        self.identity = nn.Identity()
        
    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.identity(self.conv2(out))
        x = torch.mul(out, 0.2)
        x = torch.add(x, identity)
        
        return x
