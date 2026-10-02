"""Inference-only DBNet text detector from manga-image-translator.

Loads `detect-20241225.ckpt` (zyddnys/manga-image-translator). Training-only
pieces are omitted. Architecture must stay key-compatible with that checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet34


def _resnet34_uninitialized() -> nn.Module:
    try:
        return resnet34(weights=None)
    except TypeError:
        return resnet34(pretrained=False)


class DBHead(nn.Module):
    def __init__(self, in_channels, out_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, 2, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 4, 2, 1),
        )
        self.binarize.apply(self.weights_init)
        self.thresh = self._init_thresh(in_channels)
        self.thresh.apply(self.weights_init)

    def forward(self, x):
        shrink_maps = self.binarize(x)
        threshold_maps = self.thresh(x)
        if self.training:
            binary_maps = self.step_function(shrink_maps.sigmoid(), threshold_maps)
            return torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        return torch.cat((shrink_maps, threshold_maps), dim=1)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        return nn.Sequential(
            nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels // 4, smooth, bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth, bias),
            nn.Sigmoid(),
        )

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias),
            ]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=1, stride=1, padding=1
                    )
                )
            return nn.Sequential(*module_list)
        return nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1, planes=256):
        super().__init__()
        self.planes = planes
        self.down = nn.AvgPool2d(2, stride=2) if stride > 1 else None
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch + mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        return self.conv(x)


class DoubleConvUp(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, planes=256):
        super().__init__()
        self.planes = planes
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch + mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                mid_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TextDetection(nn.Module):
    """DBNet-ResNet34 used by manga-image-translator's default detector."""

    def __init__(self):
        super().__init__()
        self.backbone = _resnet34_uninitialized()
        self.conv_db = DBHead(64, 0)
        self.conv_mask = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.down_conv1 = DoubleConv(0, 512, 512, 2)
        self.down_conv2 = DoubleConv(0, 512, 512, 2)
        self.down_conv3 = DoubleConv(0, 512, 512, 2)
        self.upconv1 = DoubleConvUp(0, 512, 256)
        self.upconv2 = DoubleConvUp(256, 512, 256)
        self.upconv3 = DoubleConvUp(256, 512, 256)
        self.upconv4 = DoubleConvUp(256, 512, 256, planes=128)
        self.upconv5 = DoubleConvUp(256, 256, 128, planes=64)
        self.upconv6 = DoubleConvUp(128, 128, 64, planes=32)
        self.upconv7 = DoubleConvUp(64, 64, 64, planes=16)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        h4 = self.backbone.layer1(x)
        h8 = self.backbone.layer2(h4)
        h16 = self.backbone.layer3(h8)
        h32 = self.backbone.layer4(h16)
        h64 = self.down_conv1(h32)
        h128 = self.down_conv2(h64)
        h256 = self.down_conv3(h128)
        up256 = self.upconv1(h256)
        up128 = self.upconv2(torch.cat([up256, h128], dim=1))
        up64 = self.upconv3(torch.cat([up128, h64], dim=1))
        up32 = self.upconv4(torch.cat([up64, h32], dim=1))
        up16 = self.upconv5(torch.cat([up32, h16], dim=1))
        up8 = self.upconv6(torch.cat([up16, h8], dim=1))
        up4 = self.upconv7(torch.cat([up8, h4], dim=1))
        return self.conv_db(up8), self.conv_mask(up4)


def load_dbnet_detector(ckpt_path: Path, device: torch.device) -> TextDetection:
    model = TextDetection()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model.to(device)
