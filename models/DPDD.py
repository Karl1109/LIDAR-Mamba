'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui1109@stud.tjut.edu.cn, liuhui@ieee.org
'''

import torch.nn as nn
from models.LDMK_Conv import LDMK_Conv

class DPDD(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = LDMK_Conv(channels, channels, channels//8)
        self.norm = nn.GroupNorm(num_groups=16, num_channels=channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        combined = avg_out + max_out
        out = self.conv(combined)
        out = self.norm(out)
        out = self.activation(out)
        return out + x