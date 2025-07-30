'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui1109@stud.tjut.edu.cn, liuhui@ieee.org
'''


import torch
import torch.nn as nn
from models.LDMK_Conv import LDMK_Conv

class AFDP(nn.Module):
    def __init__(self, in_channels, init_r=5, temperature=10.0):
        super(AFDP, self).__init__()
        self.r = nn.Parameter(torch.tensor(float(init_r)))
        self.temperature = temperature
        self.weight_conv_high_h = nn.Conv2d(in_channels, 1, kernel_size=(1, 3), padding=(0, 1))
        self.weight_conv_high_v = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        self.weight_conv_low_h = nn.Conv2d(in_channels, 1, kernel_size=(1, 3), padding=(0, 1))
        self.weight_conv_low_v = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        self.fuse_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv_h = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=(1, 3),
                                padding=(0, 1), groups=in_channels, bias=False)
        self.conv_v = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=(3, 1),
                                padding=(1, 0), groups=in_channels, bias=False)

        self.bn_h = nn.BatchNorm2d(in_channels * 2)
        self.bn_v = nn.BatchNorm2d(in_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.align_conv = LDMK_Conv(in_channels, in_channels, in_channels//4)

    def forward(self, x):
        B, C, H_orig, W_orig = x.shape
        ffted = torch.fft.rfft2(x, norm='ortho')
        B, C, H_fft, W_fft = ffted.shape
        ffted = torch.view_as_real(ffted).permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(B, -1, H_fft, W_fft)

        ffted_h = self.relu(self.bn_h(self.conv_h(ffted)))
        ffted_v = self.relu(self.bn_v(self.conv_v(ffted)))
        ffted = ffted_h + ffted_v

        ffted = ffted.view(B, C, 2, H_fft, W_fft).permute(0, 1, 3, 4, 2).contiguous()
        ffted_complex = torch.view_as_complex(ffted)
        restored = torch.fft.irfft2(ffted_complex, s=(H_orig, W_orig), norm='ortho')
        restored_shift = torch.fft.fftshift(restored, dim=(-2, -1))

        Y, X = torch.meshgrid(torch.arange(H_orig, device=x.device),
                              torch.arange(W_orig, device=x.device),
                              indexing='ij')
        center_h, center_w = H_orig // 2, W_orig // 2
        current_r = self.r.abs()
        distance_h = (X - center_w).abs()
        distance_v = (Y - center_h).abs()
        high_mask_h = torch.sigmoid((distance_h - current_r) * self.temperature).float().unsqueeze(0)
        high_mask_v = torch.sigmoid((distance_v - current_r) * self.temperature).float().unsqueeze(0)
        low_mask = 1 - torch.max(high_mask_h, high_mask_v)
        high_freq_h = torch.fft.ifftshift(restored_shift * high_mask_h, dim=(-2, -1))
        high_freq_v = torch.fft.ifftshift(restored_shift * high_mask_v, dim=(-2, -1))
        low_freq = torch.fft.ifftshift(restored_shift * low_mask, dim=(-2, -1))
        high_freq_h = self.align_conv(high_freq_h)
        high_freq_v = self.align_conv(high_freq_v)
        low_freq = self.align_conv(low_freq)

        high_weight_h = self.sigmoid(self.weight_conv_high_h(high_freq_h))
        high_weight_v = self.sigmoid(self.weight_conv_high_v(high_freq_v))
        low_weight_h = self.sigmoid(self.weight_conv_low_h(low_freq))
        low_weight_v = self.sigmoid(self.weight_conv_low_v(low_freq))
        weights = torch.cat([high_weight_h + high_weight_v, low_weight_h + low_weight_v], dim=1)
        fused_weights = self.sigmoid(self.fuse_conv(weights))
        enhanced_high = (high_freq_h * high_weight_h + high_freq_v * high_weight_v) / 2
        suppressed_low = low_freq * (1 - (low_weight_h + low_weight_v) / 2)

        return x * fused_weights[:, 0] + enhanced_high + suppressed_low
