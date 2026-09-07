"""Inference-only Big-LaMa (large) architecture.

Ported from manga-image-translator's FFC generator so we can load
`lama_large_512px.ckpt` from dreMaz/AnimeMangaInpainting.
Training-only pieces (discriminator, MPE) are omitted.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _as_local_global(x):
    if isinstance(x, tuple):
        return x
    return x, 0


class FourierUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groups=1,
        spatial_scale_factor=None,
        spatial_scale_mode="bilinear",
        spectral_pos_encoding=False,
        ffc3d=False,
        fft_norm="ortho",
    ):
        super().__init__()
        self.groups = groups
        extra_channels = 2 if spectral_pos_encoding else 0
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2 + extra_channels,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        orig_size = None
        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(
                x,
                scale_factor=self.spatial_scale_factor,
                mode=self.spatial_scale_mode,
                align_corners=False,
            )

        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = (
                torch.linspace(0, 1, height, device=ffted.device)
                .view(1, 1, height, 1)
                .expand(batch, 1, height, width)
            )
            coords_hor = (
                torch.linspace(0, 1, width, device=ffted.device)
                .view(1, 1, 1, width)
                .expand(batch, 1, height, width)
            )
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        ffted = self.relu(self.bn(self.conv_layer(ffted)))
        ffted = ffted.view((batch, -1, 2) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2
        ).contiguous()
        if ffted.dtype in (torch.float16, torch.bfloat16):
            ffted = ffted.float()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(
            ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm
        )
        if self.spatial_scale_factor is not None and orig_size is not None:
            output = F.interpolate(
                output,
                size=orig_size,
                mode=self.spatial_scale_mode,
                align_corners=False,
            )
        return output


class SpectralTransform(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        groups=1,
        enable_lfu=True,
        **fu_kwargs,
    ):
        super().__init__()
        self.enable_lfu = enable_lfu
        self.downsample = (
            nn.AvgPool2d(kernel_size=(2, 2), stride=2)
            if stride == 2
            else nn.Identity()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        if self.enable_lfu:
            c = x.shape[1]
            split_s = x.shape[2] // 2
            xs = torch.cat(torch.split(x[:, : c // 4], split_s, dim=-2), dim=1)
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1)
            xs = self.lfu(xs).repeat(1, 1, 2, 2)
        else:
            xs = 0
        return self.conv2(x + output + xs)


class FFC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin,
        ratio_gout,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        enable_lfu=True,
        padding_type="reflect",
        gated=False,
        **spectral_kwargs,
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("Stride should be 1 or 2.")

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg
        self.gated = gated

        local_local = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        local_global = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        global_local = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        global_global = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform

        self.convl2l = local_local(
            in_cl,
            out_cl,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        self.convl2g = local_global(
            in_cl,
            out_cg,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        self.convg2l = global_local(
            in_cg,
            out_cl,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        self.convg2g = global_global(
            in_cg,
            out_cg,
            stride,
            1 if groups == 1 else groups // 2,
            enable_lfu,
            **spectral_kwargs,
        )
        gate = nn.Identity if in_cg == 0 or out_cl == 0 or not gated else nn.Conv2d
        self.gate = gate(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = _as_local_global(x)
        if self.gated:
            parts = [x_l]
            if torch.is_tensor(x_g):
                parts.append(x_g)
            g2l_gate, l2g_gate = torch.sigmoid(self.gate(torch.cat(parts, dim=1))).chunk(
                2, dim=1
            )
        else:
            g2l_gate, l2g_gate = 1, 1

        out_xl = 0
        out_xg = 0
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin,
        ratio_gout,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.Identity,
        padding_type="reflect",
        enable_lfu=True,
        **kwargs,
    ):
        super().__init__()
        self.ffc = FFC(
            in_channels,
            out_channels,
            kernel_size,
            ratio_gin,
            ratio_gout,
            stride,
            padding,
            dilation,
            groups,
            bias,
            enable_lfu,
            padding_type=padding_type,
            **kwargs,
        )
        global_channels = int(out_channels * ratio_gout)
        local_norm = nn.Identity if ratio_gout == 1 else norm_layer
        global_norm = nn.Identity if ratio_gout == 0 else norm_layer
        local_act = nn.Identity if ratio_gout == 1 else activation_layer
        global_act = nn.Identity if ratio_gout == 0 else activation_layer
        self.bn_l = local_norm(out_channels - global_channels)
        self.bn_g = global_norm(global_channels)
        self.act_l = local_act(inplace=True)
        self.act_g = global_act(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        return self.act_l(self.bn_l(x_l)), self.act_g(self.bn_g(x_g))


class FFCResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        padding_type,
        norm_layer,
        activation_layer=nn.ReLU,
        dilation=1,
        inline=False,
        **conv_kwargs,
    ):
        super().__init__()
        self.inline = inline
        self.conv1 = FFC_BN_ACT(
            dim,
            dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            padding_type=padding_type,
            **conv_kwargs,
        )
        self.conv2 = FFC_BN_ACT(
            dim,
            dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            padding_type=padding_type,
            **conv_kwargs,
        )

    def forward(self, x):
        if self.inline:
            x_l = x[:, : -self.conv1.ffc.global_in_num]
            x_g = x[:, -self.conv1.ffc.global_in_num :]
        else:
            x_l, x_g = _as_local_global(x)

        y_l, y_g = self.conv2(self.conv1((x_l, x_g)))
        out = (x_l + y_l, x_g + y_g)
        if self.inline:
            return torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        x_l, x_g = x
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
    def __init__(
        self,
        input_nc=4,
        output_nc=3,
        ngf=64,
        n_downsampling=3,
        n_blocks=9,
        norm_layer=nn.BatchNorm2d,
        padding_type="reflect",
        activation_layer=nn.ReLU,
        up_norm_layer=nn.BatchNorm2d,
        up_activation=None,
        init_conv_kwargs=None,
        downsample_conv_kwargs=None,
        resnet_conv_kwargs=None,
        add_out_act="sigmoid",
        max_features=1024,
    ):
        super().__init__()
        if n_blocks < 0:
            raise ValueError("n_blocks must be >= 0")
        if up_activation is None:
            up_activation = nn.ReLU(True)
        init_conv_kwargs = init_conv_kwargs or {}
        downsample_conv_kwargs = downsample_conv_kwargs or {}
        resnet_conv_kwargs = resnet_conv_kwargs or {}

        model = [
            nn.ReflectionPad2d(3),
            FFC_BN_ACT(
                input_nc,
                ngf,
                kernel_size=7,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                **init_conv_kwargs,
            ),
        ]

        for i in range(n_downsampling):
            mult = 2**i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs["ratio_gout"] = resnet_conv_kwargs.get("ratio_gin", 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model.append(
                FFC_BN_ACT(
                    min(max_features, ngf * mult),
                    min(max_features, ngf * mult * 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    **cur_conv_kwargs,
                )
            )

        feats_num_bottleneck = min(max_features, ngf * 2**n_downsampling)
        for _ in range(n_blocks):
            model.append(
                FFCResnetBlock(
                    feats_num_bottleneck,
                    padding_type=padding_type,
                    activation_layer=activation_layer,
                    norm_layer=norm_layer,
                    **resnet_conv_kwargs,
                )
            )
        model.append(ConcatTupleLayer())

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model.extend(
                [
                    nn.ConvTranspose2d(
                        min(max_features, ngf * mult),
                        min(max_features, int(ngf * mult / 2)),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    up_norm_layer(min(max_features, int(ngf * mult / 2))),
                    up_activation,
                ]
            )

        model.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                nn.Sigmoid() if add_out_act == "sigmoid" else nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*model)

    def forward(self, img, mask):
        masked_img = torch.cat([img * (1 - mask), mask], dim=1)
        return self.model(masked_img)


class LamaLargeGenerator(nn.Module):
    """Big-LaMa large generator (18 FFC residual blocks, no MPE)."""

    def __init__(self):
        super().__init__()
        ffc_kwargs = {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False}
        self.generator = FFCResNetGenerator(
            4,
            3,
            add_out_act="sigmoid",
            n_blocks=18,
            init_conv_kwargs=ffc_kwargs,
            downsample_conv_kwargs=ffc_kwargs,
            resnet_conv_kwargs={
                "ratio_gin": 0.75,
                "ratio_gout": 0.75,
                "enable_lfu": False,
            },
        )

    def forward(self, img, mask):
        predicted = self.generator(img, mask)
        return predicted * mask + (1 - mask) * img


def load_lama_large_generator(ckpt_path: Path, device: torch.device) -> LamaLargeGenerator:
    """Build the large generator and load `gen_state_dict` from a Lightning ckpt."""
    model = LamaLargeGenerator()
    # Official dreMaz ckpt is a Lightning pickle, not a pure tensor dict.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "gen_state_dict" not in ckpt:
        raise ValueError(
            f"Unexpected LaMa checkpoint format in {ckpt_path}: missing gen_state_dict"
        )
    model.generator.load_state_dict(ckpt["gen_state_dict"])
    model.eval()
    return model.to(device)
