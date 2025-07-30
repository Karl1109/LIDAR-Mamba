'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui1109@stud.tjut.edu.cn, liuhui@ieee.org
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class WidthPredictor(nn.Module):
    def __init__(self, in_channels, max_mid_channels, ema_decay=0.9):
        super().__init__()
        self.max_mid = max_mid_channels
        self.ema_decay = ema_decay
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max_mid_channels * 2),
            nn.ReLU(),
            nn.Linear(max_mid_channels * 2, max_mid_channels),
            nn.Sigmoid()
        )
        self.register_buffer('ema_scale', torch.tensor(0.0))

    def forward(self, x):
        attended_x = x
        gap = F.adaptive_avg_pool2d(attended_x, 1).flatten(1)
        scores = self.fc(gap)
        if self.training:
            current_scale = scores.mean(dim=1).mean()
            self.ema_scale = (self.ema_decay * self.ema_scale +
                              (1 - self.ema_decay) * current_scale.detach())

        scale = torch.clamp(self.ema_scale if self.training else scores.mean(),
                            min=0.25, max=1.0)
        k = torch.ceil(self.max_mid * scale).int().clamp(min=max(4, self.max_mid // 4))

        _, indices = torch.topk(scores, k.item(), dim=1)
        mask = torch.zeros_like(scores).scatter(1, indices, 1.0)
        return mask.unsqueeze(-1).unsqueeze(-1), k.item()


class LDMK_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, max_mid_channels,
                 kernel_sizes=[3, 5, 7], stride=1, groups=1):
        super().__init__()
        self.stride = stride
        self.groups = groups
        self.kernel_sizes = kernel_sizes
        self.num_branches = len(kernel_sizes)
        self.max_mid = max_mid_channels

        self.pointwise_1 = nn.Conv2d(in_channels, max_mid_channels, 1)
        self.predictor = WidthPredictor(max_mid_channels, max_mid_channels)

        self.dw_bases = nn.ParameterList([
            nn.Parameter(torch.Tensor(max_mid_channels, 1, ks, ks))
            for ks in self.kernel_sizes
        ])

        self.offset_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(max_mid_channels, self.num_branches * 2, 1),
            nn.Tanh()
        )

        self.pw_weight = nn.Parameter(
            torch.Tensor(out_channels, max_mid_channels * self.num_branches // self.groups, 1, 1)
        )
        self.pw_bias = nn.Parameter(torch.Tensor(out_channels))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        for w in self.dw_bases:
            nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pw_weight, mode='fan_out')
        nn.init.zeros_(self.pw_bias)

    def dynamic_conv(self, x, mask, k):
        B, C, H, W = x.shape

        offsets = self.offset_generator(x).view(B, self.num_branches, 2)

        branch_outputs = []
        for i, (ks, base_weight) in enumerate(zip(self.kernel_sizes, self.dw_bases)):
            branch_offset = offsets[:, i, :]

            scale = branch_offset[:, 0].sigmoid().view(B, 1, 1, 1, 1)
            shift = branch_offset[:, 1].tanh().view(B, 1, 1, 1, 1)

            selected_x = x * mask
            selected_x = selected_x[:, :k]  # [B, k, H, W]

            selected_weight = base_weight[:k].unsqueeze(0) * (1 + scale) + shift
            selected_weight = selected_weight.view(B * k, 1, ks, ks)

            conv_input = selected_x.reshape(1, B * k, H, W)
            conv = F.conv2d(
                conv_input,
                selected_weight,
                stride=self.stride,
                padding=(ks - 1) // 2,
                groups=B * k
            )
            branch_outputs.append(conv.view(B, k, H // self.stride, W // self.stride))

        fused = torch.cat(branch_outputs, dim=1)  # [B, k*num_branches, H, W]

        in_channels = k * self.num_branches
        selected_pw = self.pw_weight[:, :in_channels // self.groups]

        return F.conv2d(
            fused,
            selected_pw,
            self.pw_bias,
            groups=self.groups
        )

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.pointwise_1(x)
        mask, k = self.predictor(x)

        x = F.relu(x)
        dynamic_out = self.dynamic_conv(x, mask, k)
        return dynamic_out + residual
