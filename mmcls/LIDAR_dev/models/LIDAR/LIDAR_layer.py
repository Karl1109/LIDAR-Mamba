'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui1109@stud.tjut.edu.cn, liuhui@ieee.org
'''

import math
from einops import repeat
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_dropout
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layernorm import RMSNorm
from models.LDMK_Conv import LDMK_Conv

class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x

class LinearPositionEncoding(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.linear = nn.Linear(2, embed_dim)

    def forward(self, x):
        B, num_patches, embed_dim = x.shape
        grid_size = int(num_patches ** 0.5)
        coords = torch.stack(torch.meshgrid(
            torch.arange(grid_size),
            torch.arange(grid_size), indexing='ij'
        ), dim=-1).float().to(x.device)
        coords = (coords / (grid_size - 1)) * 2 - 1
        pos_emb = self.linear(coords)
        pos_emb = pos_emb.view(1, num_patches, embed_dim)
        return pos_emb

class LIDAR2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_size=7,
        bias=False,
        init_layer_scale=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert conv_size % 2 == 1
        self.conv2d = BottConv(in_channels=self.d_inner, out_channels=self.d_inner, mid_channels=self.d_inner//16, kernel_size=3, padding=1, stride=1)

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )

        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.pos_emb = LinearPositionEncoding(d_model)
        self.dir_emb = nn.Parameter(torch.zeros(4, self.d_inner))
        nn.init.normal_(self.dir_emb, mean=0, std=0.01)

    def get_permute_order_parallel_original(self, hw_shape):
        H, W = hw_shape
        L = H * W

        o1, o2, o3, o4 = [], [], [], []
        o1_inverse = [-1 for _ in range(L)]
        o2_inverse = [-1 for _ in range(L)]
        o3_inverse = [-1 for _ in range(L)]
        o4_inverse = [-1 for _ in range(L)]

        d1, d2, d3, d4 = [], [], [], []

        for i in range(H):
            for j in range(W):
                idx = i * W + j
                o1.append(idx)
                o1_inverse[idx] = len(o1) - 1
                d1.append(1 if j < W - 1 else 4)

        for i in range(H - 1, -1, -1):
            for j in range(W - 1, -1, -1):
                idx = i * W + j
                o2.append(idx)
                o2_inverse[idx] = len(o2) - 1
                d2.append(2 if j > 0 else 3)

        for j in range(W):
            for i in range(H):
                idx = i * W + j
                o3.append(idx)
                o3_inverse[idx] = len(o3) - 1
                d3.append(4 if i < H - 1 else 1)

        for j in range(W - 1, -1, -1):
            for i in range(H - 1, -1, -1):
                idx = i * W + j  # 计算索引
                o4.append(idx)
                o4_inverse[idx] = len(o4) - 1
                d4.append(3 if i > 0 else 2)

        return (tuple(o1), tuple(o2), tuple(o3), tuple(o4)), \
            (tuple(o1_inverse), tuple(o2_inverse), tuple(o3_inverse), tuple(o4_inverse)), \


    def forward(self, x, scan_orders, hw_shape):
        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner
        conv_state, ssm_state = None, None
        pos_emb = self.pos_emb(x)
        xz = self.in_proj(x + pos_emb)
        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=-1)
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        dt = dt.permute(0, 2, 1).contiguous()
        B = B.permute(0, 2, 1).contiguous()
        C = C.permute(0, 2, 1).contiguous()

        assert self.activation in ["silu", "swish"]

        if scan_orders is None:
            orders, inverse_orders = self.get_permute_order_parallel_original(hw_shape)
        else:
            orders = (tuple(scan_orders[0]), tuple(scan_orders[2]), tuple(scan_orders[4]), tuple(scan_orders[6]))
            inverse_orders = (tuple(scan_orders[1]), tuple(scan_orders[3]), tuple(scan_orders[5]), tuple(scan_orders[7]))

        ys = []
        for idx, (o, inv_o) in enumerate(zip(orders, inverse_orders)):
            dir_emb = self.dir_emb[idx].expand(batch_size, -1).unsqueeze(1)  # [B, 1, E]
            x_dir = x_conv[:, o, :] + dir_emb
            y = selective_scan_fn(
                x_dir.permute(0, 2, 1).contiguous(),
                dt,
                A,
                B.contiguous(),
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ).permute(0, 2, 1)[:, inv_o, :]
            ys.append(y)

        y = sum(ys) * self.act(z)
        out = self.out_proj(y)

        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out

class LIDARLayer(nn.Module):
    def __init__(
        self,
        embed_dims,
        use_rms_norm,
        with_dwconv,
        drop_path_rate,
        mamba_cfg,
    ):

        super(LIDARLayer, self).__init__()
        mamba_cfg.update({'d_model': embed_dims})
        if use_rms_norm:
            self.norm = RMSNorm(embed_dims)
        else:
            self.norm = nn.LayerNorm(embed_dims)

        self.with_dwconv = with_dwconv
        if self.with_dwconv:
            self.dw = nn.Sequential(
                nn.Conv2d(
                    embed_dims,
                    embed_dims,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    bias=False,
                    groups=embed_dims
                ),
                nn.BatchNorm2d(embed_dims),
                nn.GELU(),
            )
        self.mamba = LIDAR2D(**mamba_cfg)
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        self.linear_256 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.GN_256 = nn.GroupNorm(num_channels=256, num_groups=16)

        self.LDMK_Conv_1 = LDMK_Conv(embed_dims, embed_dims, embed_dims//8, kernel_sizes=[3, 5, 7])
        self.LDMK_Conv_2 = LDMK_Conv(embed_dims, embed_dims, embed_dims//8, kernel_sizes=[3, 5, 7])
        self.LDMK_Conv_3 = LDMK_Conv(embed_dims, embed_dims, embed_dims//8, kernel_sizes=[3, 5, 7])
        self.GN_1 = nn.GroupNorm(num_channels=embed_dims, num_groups=embed_dims//16)
        self.GN_2 = nn.GroupNorm(num_channels=embed_dims, num_groups=embed_dims//16)
        self.GN_3 = nn.GroupNorm(num_channels=embed_dims, num_groups=16)
        self.ReLU = nn.ReLU()

    def forward(self, x, scan_orders, hw_shape):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        for i in range(2):
            x_residual = x
            x_1 = self.ReLU(self.GN_1(self.LDMK_Conv_1(x)))
            x_2 = self.ReLU(self.GN_2(self.LDMK_Conv_2(x)))
            x = x_1 * x_2
            x = self.ReLU(self.GN_3(self.LDMK_Conv_3(x))) + x_residual

        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        mixed_x = self.drop_path(self.mamba(self.norm(x), scan_orders, hw_shape))
        mixed_x_res = self.linear_256(self.GN_256(mixed_x.permute(0, 2, 1)).permute(0, 2, 1))

        return mixed_x + mixed_x_res
