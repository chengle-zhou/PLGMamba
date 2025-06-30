import torch
import math
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm import Mamba
import torch.nn.functional as F


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, dilation=1):
        super(BaseConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.dilation = dilation
        if dilation == None:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        elif dilation == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=2, bias=bias, dilation=dilation)
        else:
            padding = int((kernel_size - 1) / 2) * dilation
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding,
                                  bias=bias, dilation=dilation)
    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(BaseConv(n_feats, 4 * n_feats, 3, 1, bias, 1))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(BaseConv(n_feats, 9 * n_feats, 3, 1, bias, 1))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1.):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(BaseConv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResAttentionBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, reduction=16, bn=False, act=nn.ReLU(True), res_scale=1.):
        super(ResAttentionBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(BaseConv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        m.append(CALayer(n_feats, reduction))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResMambaBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, mlp_ratio=4, bn=False, act=nn.ReLU(True), res_scale=1.):
        super(ResMambaBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(BaseConv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        m.append(MambaBlock(dim=n_feats, mlp_ratio=mlp_ratio))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResLearnBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, res_scale=1.0):
        super(ResLearnBlock, self).__init__()
        self.rb = ResBlock(dim, 3, res_scale=res_scale)
        self.rmb = ResMambaBlock(dim, 3, mlp_ratio=mlp_ratio, res_scale=res_scale)
        self.rab = ResAttentionBlock(dim, 3, res_scale=res_scale)

    def forward(self, x):
        y = self.rb(x)
        y = self.rmb(y)
        y = self.rab(y)
        return y


class Res3DBlock(nn.Module):
    def __init__(self, n_feats, bias=True, act=nn.ReLU(True), res_scale=1):
        super(Res3DBlock, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(1, n_feats, (3, 1, 1), 1, (1, 0, 0), bias=bias),
                                  act,
                                  nn.Conv3d(n_feats, 1, (1, 3, 3), 1, (0, 1, 1), bias=bias)
                                  )
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x.unsqueeze(1)) + x.unsqueeze(1)
        return x.squeeze(1)


class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MambaBlock(nn.Module):
    def __init__(self,
                 dim, d_state=64, d_conv=4, expand=2, mlp_ratio=4,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(MambaBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Mamba(  # MambaBlock(d_model=dim)
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.permute(0, 2, 1).reshape(N, C, H, W)
        return x


class PLGMamba(nn.Module):
    """SuperMamba: Hyperspectral Image Super-resolution With Polymorphic State Space Model"""
    def __init__(self, in_channels=102, g_num=8, scale=4, fea_dim=64, res_scale=1.,
                 mlp_ratio=4, block_num_g=6, block_num_a=3):
        super(PLGMamba, self).__init__()
        self.g_dim = in_channels // g_num
        g_res_dim = in_channels % g_num
        self.g_pad_dim = self.g_dim - g_res_dim if g_res_dim != 0 else 0
        self.n_feats = fea_dim
        self.g_num = g_num
        self.g_red_num = self.g_num + 1 if g_res_dim != 0 else self.g_num
        self.scale = scale
        self.layer1 = BaseConv(self.g_dim+self.n_feats+self.g_dim*self.scale ** 2, self.n_feats, 3)
        self.out_layer1 = BaseConv(self.n_feats, self.g_dim, 3)
        self.out_layer2 = BaseConv(self.n_feats, self.n_feats, 3)
        body1 = [ResLearnBlock(self.n_feats, mlp_ratio=mlp_ratio, res_scale=res_scale) for _ in range(block_num_g)]
        self.RB1 = nn.Sequential(*body1)
        self.upsample = Upsampler(scale, fea_dim)
        self.downsample = nn.PixelUnshuffle(downscale_factor=self.scale)
        self.act = nn.ReLU(True)
        # body2 = [Res3DBlock(seq_len) for _ in range(block_num_a)]
        body2 = [ResMambaBlock(in_channels, 3, mlp_ratio=mlp_ratio, res_scale=res_scale) for _ in range(block_num_a)]
        self.body2 = nn.Sequential(*body2)

    def forward(self, x):
        residual = x
        out = []
        B, C, h, w = residual.shape
        x_pad = torch.flip(residual[:, -self.g_pad_dim:, :, :], dims=[1])
        x = torch.cat((x, x_pad), dim=1)
        h1 = torch.zeros(B, self.n_feats, h, w, device=x.device)
        sr = torch.zeros(B, self.g_dim*self.scale ** 2, h, w, device=x.device)
        y = torch.split(x, self.g_dim, dim=1)
        for i in range(self.g_red_num):
            x_ilr = y[i]
            h1 = self.act(self.layer1(torch.cat([h1, sr, x_ilr], dim=1)))
            h1 = self.RB1(h1)
            sr = self.out_layer1(self.upsample(h1)) + F.interpolate(x_ilr, (h * self.scale, w * self.scale))
            h1 = self.out_layer2(h1)
            out.append(sr)
            sr = self.downsample(sr)
        out = torch.cat(out[:], 1)[:, 0:C, :, :]
        out = self.body2(out)
        return out
