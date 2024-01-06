""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(nn.Conv3d(in_channel, in_channel//ratio, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv3d(in_channel//ratio, in_channel, kernel_size=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class GlobalAttn(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(GlobalAttn, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channel, in_channel//ratio, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv3d(in_channel//ratio, in_channel, kernel_size=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        return self.sigmoid(x)


class MAF(nn.Module):
    def __init__(self, in_channels, strides=1):
        super(MAF, self).__init__()
        self.convBlock = nn.Sequential(torch.nn.Conv3d(in_channels, in_channels, strides),
                                        torch.nn.InstanceNorm3d(in_channels),
                                        nn.LeakyReLU(),
                                        torch.nn.Conv3d(in_channels, in_channels, strides),
                                        torch.nn.InstanceNorm3d(in_channels))
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        self.ga = GlobalAttn(in_channels)


    def forward(self, x):
        res = x
        x = self.convBlock(x)
        x1 = self.ga(x) * x
        x2 = self.ca(x) * x
        x3 = self.sa(x) * x
        return x1 + x2 + x3 + res


class ScaleAttn(nn.Module):
    def __init__(self, in_channels, strides=1):
        super(ScaleAttn, self).__init__()
        self.convBlock = nn.Sequential(torch.nn.Conv3d(in_channels, in_channels, strides),
                                        torch.nn.InstanceNorm3d(in_channels),
                                        nn.LeakyReLU(),
                                        torch.nn.Conv3d(in_channels, in_channels, strides),
                                        torch.nn.InstanceNorm3d(in_channels))
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()


    def forward(self, x):
        res = x
        x = self.convBlock(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = x + res
        return x


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
            DoubleConv(in_channels, out_channels),
            MAF(out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, pooling_size, bilinear=True):
        super().__init__()

        if bilinear:
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


class DecFeatureUp(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(DecFeatureUp, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        return x


class DecAttnFusion(nn.Module):
    def __init__(self, in_c, out_c):
        super(DecAttnFusion, self).__init__()
        self.attn = ScaleAttn(in_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm3d(out_c)

    def forward(self, x):
        res = x
        x = self.attn(x)
        x = x + res
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ALIEN(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=False):
        super(ALIEN, self).__init__()
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

        self.sk_op = []
        for sk in range(len(self.down)):
            ops = []

            if sk == 0:
                scale_f = self.pool_size[sk]
                ops.append(
                    nn.Sequential(
                        nn.Conv3d(self.channel_list[sk], self.channel_list[sk], kernel_size=1, padding=0),
                        nn.BatchNorm3d(self.channel_list[sk]),
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                )
                ops.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=tuple(scale_f), mode='nearest', align_corners=None),
                        nn.Conv3d(self.channel_list[sk+1], self.channel_list[sk], kernel_size=1, padding=0),
                        nn.BatchNorm3d(self.channel_list[sk]),
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                )

            elif sk == len(self.down)-1:
                stride = self.pool_size[sk-1]
                k_s = [i+1 if i != 1 else 1 for i in stride]
                pd = [int((i-1)/2) for i in k_s]
                ops.append(
                    nn.Sequential(
                        nn.Conv3d(self.channel_list[sk-1], self.channel_list[sk], kernel_size=tuple(k_s),
                                  stride=tuple(stride), padding=tuple(pd)),
                        nn.BatchNorm3d(self.channel_list[sk]),
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                )
                ops.append(
                    nn.Sequential(
                        nn.Conv3d(self.channel_list[sk], self.channel_list[sk], kernel_size=1, padding=0),
                        nn.BatchNorm3d(self.channel_list[sk]),
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                )

            else:
                stride = self.pool_size[sk-1]
                k_s = [i+1 if i != 1 else 1 for i in stride]
                pd = [int((i-1)/2) for i in k_s]
                ops.append(
                    nn.Sequential(
                        nn.Conv3d(self.channel_list[sk-1], self.channel_list[sk], kernel_size=tuple(k_s),
                                  stride=tuple(stride), padding=tuple(pd)),
                        nn.BatchNorm3d(self.channel_list[sk]),
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                )
                ops.append(
                    nn.Sequential(
                        nn.Conv3d(self.channel_list[sk], self.channel_list[sk], kernel_size=1, padding=0),
                        nn.BatchNorm3d(self.channel_list[sk]),
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                )
                scale_f = self.pool_size[sk]
                ops.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=tuple(scale_f), mode='nearest', align_corners=None),
                        nn.Conv3d(self.channel_list[sk+1], self.channel_list[sk], kernel_size=1, padding=0),
                        nn.BatchNorm3d(self.channel_list[sk]),
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                )
            self.sk_op.append(nn.ModuleList(ops))

        self.sk_fusion = []  # czh
        for i in range(len(self.down)):  # czh
            if i != 0 and i != len(self.down)-1:
                self.sk_fusion.append(
                    nn.Sequential(
                        nn.Conv3d(self.channel_list[i] * 3, self.channel_list[i], kernel_size=1, padding=0),
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                )
            else:
                self.sk_fusion.append(
                    nn.Sequential(
                        nn.Conv3d(self.channel_list[i] * 2, self.channel_list[i], kernel_size=1, padding=0),
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                )

        self.ds = []
        for i in range(1, p_len):
            scale_f = [1, 1, 1]
            for sf in self.pool_size[0: -i]:
                scale_f = [scale_f[i] * sf[i] for i in range(3)]
            self.ds.append(
                (DecFeatureUp(self.channel_list[p_len-i], 4, scale_factor=scale_f))
            )
        self.ds0 = nn.Conv3d(self.channel_list[0], 4, kernel_size=1, padding=0)

        self.DecAttnFusion = DecAttnFusion(4*4, 4)

        self.outc = (OutConv(4, n_classes))

        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)
        self.sk_op = nn.ModuleList(self.sk_op)
        self.sk_fusion = nn.ModuleList(self.sk_fusion)
        self.ds = nn.ModuleList(self.ds)


    def forward(self, x):
        logits = []

        x = self.inc(x)
        skips = [x]
        for i in range(len(self.down)-1):
            x = self.down[i](x)
            skips.append(x)

        x = self.down[-1](x)

        skips_res = []
        for l_sk in range(len(self.sk_op)):
            if l_sk == 0:
                master_r = self.sk_op[l_sk][0](skips[l_sk])
                high_r = self.sk_op[l_sk][1](skips[l_sk+1])
                att_map_h = torch.sigmoid(master_r * high_r)
                fea_map = skips[l_sk] * att_map_h
                fea_map_res = self.sk_fusion[l_sk](torch.cat([skips[l_sk], fea_map], dim=1))
            elif l_sk == len(self.sk_op)-1:
                low_r = self.sk_op[l_sk][0](skips[l_sk-1])
                master_r = self.sk_op[l_sk][1](skips[l_sk])
                att_map_l = torch.sigmoid(master_r + low_r)
                fea_map = skips[l_sk] * att_map_l
                fea_map_res = self.sk_fusion[l_sk](torch.cat([fea_map, skips[l_sk]], dim=1))
            else:
                low_r = self.sk_op[l_sk][0](skips[l_sk-1])
                master_r = self.sk_op[l_sk][1](skips[l_sk])
                high_r = self.sk_op[l_sk][2](skips[l_sk+1])
                att_map_l = torch.sigmoid(master_r + low_r)
                att_map_h = torch.sigmoid(master_r * high_r)
                fea_map_l = skips[l_sk] * att_map_l
                fea_map_h = skips[l_sk] * att_map_h
                fea_map_res = self.sk_fusion[l_sk](torch.cat([fea_map_l, skips[l_sk], fea_map_h], dim=1))
            skips_res.append(fea_map_res)

        for i in range(len(self.up)):
            x = self.up[i](x, skips_res[-(i+1)])
            if i < len(self.ds):
                logits.append(self.ds[i](x))
        logits.append(self.ds0(x))

        x = self.DecAttnFusion(torch.cat(logits, dim=1))
        logits = self.outc(x)

        return logits