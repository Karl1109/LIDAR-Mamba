'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui1109@stud.tjut.edu.cn, liuhui@ieee.org
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.LDMK_Conv import LDMK_Conv
from models.AFDP import AFDP

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

class MLP(nn.Module):
    def __init__(self, input_dim=256, embed_dim=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x

class LD3CF(nn.Module):
    def __init__(self, num_modalities, embedding_dim):
        super().__init__()
        self.num_modes = num_modalities

        self.main_enhance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 64, 1),
            nn.Sigmoid()
        )

        self.weights_gen = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 8, 1),
                nn.ReLU(),
                nn.Conv2d(8, 2, 1)
            ) for _ in range(num_modalities - 1)
        ])

        self.cross_gate = nn.ModuleDict({
            f'gate_{i}': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(64, 16, 1),
                nn.ReLU(),
                nn.Conv2d(16, 64, 1),
                nn.Sigmoid()
            ) for i in range(3)
        })

        self.AFDP = AFDP(64)
        self.LDMK_Conv_1 = LDMK_Conv(64, 64, 64//8, kernel_sizes=[3, 5, 7])
        self.LDMK_Conv_2 = LDMK_Conv(64, 64, 64//8, kernel_sizes=[3, 5, 7])
        self.LDMK_Conv_3 = LDMK_Conv(64, 64, 64//8, kernel_sizes=[3, 5, 7])
        self.GN_1 = nn.GroupNorm(num_channels=64, num_groups=64//16)
        self.GN_2 = nn.GroupNorm(num_channels=64, num_groups=64//16)
        self.GN_3 = nn.GroupNorm(num_channels=64, num_groups=16)
        self.ReLU = nn.ReLU()

        self.embedding_dim = embedding_dim
        self.linear_64 = MLP(input_dim=64, embed_dim=embedding_dim)
        self.LDMK_Conv_8 = LDMK_Conv(8, 8, 4)
        self.conv_pred_8 = nn.Conv2d(8, 1, kernel_size=1)
        self.conv_pred_1 = nn.Conv2d(1, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)
        self.DySample_C_8 = DySample(embedding_dim, scale=8)

    def _dual_pool_fusion(self, main, aux_feats):
        enhanced_main = main * self.main_enhance(main)
        fused = enhanced_main
        for i, feat in enumerate(aux_feats):
            concat = torch.cat([enhanced_main, feat], dim=1)
            weights = self.weights_gen[i](concat)
            avg_weight, max_weight = torch.chunk(weights, 2, dim=1)
            avg_feat = torch.mean(feat, dim=[2, 3], keepdim=True).expand_as(feat)
            max_feat = torch.max(feat.view(feat.size(0), feat.size(1), -1), dim=2)[0].unsqueeze(-1).unsqueeze(-1)

            am_feat = avg_weight * avg_feat + max_weight * max_feat
            am_feat_residual = am_feat
            am_feat_1 = self.ReLU(self.GN_1(self.LDMK_Conv_1(am_feat)))
            am_feat_2 = self.ReLU(self.GN_2(self.LDMK_Conv_2(am_feat)))
            am_feat = am_feat_1 * am_feat_2
            am_feat = self.ReLU(self.GN_3(self.LDMK_Conv_3(am_feat))) + am_feat_residual

            fused += am_feat

        return fused

    def forward(self, inputs):

        for i in range(len(inputs)):
            for j in range(len(inputs[i])):
                inputs[i][j] = self.AFDP(inputs[i][j])

        main_feats = inputs[0]
        aux_feats_list = inputs[1:]
        fused_features = []
        prev_feat = None

        for level in range(4):
            main = main_feats[level]
            aux_feats = [aux[level] for aux in aux_feats_list]
            fused = self._dual_pool_fusion(main, aux_feats)

            if prev_feat is not None and level < 3:
                gate = self.cross_gate[f'gate_{level}'](prev_feat)
                fused = fused * gate + prev_feat * (1 - gate)

            fused_features.append(fused)
            prev_feat = fused

        target_size = fused_features[0].shape[2:]
        aligned_features = []
        for feat in fused_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)

        final_feat = sum(aligned_features)

        b, c, h, w = final_feat.shape
        out = self.linear_64(final_feat.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out = self.DySample_C_8(out)
        out_c = self.LDMK_Conv_8(out)
        out_c = self.dropout(out_c)
        x = self.conv_pred_1(self.conv_pred_8(out_c))

        return x