""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, pooling_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(tuple(pooling_size)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, pooling_size, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=tuple(pooling_size), mode='trilinear', align_corners=True)
            self.conv_r = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
            self.conv = DoubleConv(out_channels * 2, out_channels, int(out_channels * 1.5))
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv_r(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DeepSuperVision(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(DeepSuperVision, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        return x



class UNet_3D(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=False):
        super(UNet_3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.pool_size = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        self.channel_list = [16, 32, 32, 64, 128]

        self.inc = (DoubleConv(n_channels, 16))

        p_len = len(self.pool_size)
        self.down = []
        for i in range(p_len):
            self.down.append(
                (Down(self.channel_list[i], self.channel_list[i+1], self.pool_size[i]))
            )

        self.up = []
        for i in range(p_len):
            self.up.append(
                (Up(self.channel_list[p_len-i], self.channel_list[p_len-i-1], self.pool_size[p_len-i-1], trilinear))
            )


        self.outc = (OutConv(16, n_classes))

        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        x = self.inc(x)
        skips = [x]

        for i in range(len(self.down)-1):
            x = self.down[i](x)
            skips.append(x)

        x = self.down[-1](x)

        for i in range(len(self.up)):
            x = self.up[i](x, skips[-(i+1)])

        logits = self.outc(x)

        return logits
