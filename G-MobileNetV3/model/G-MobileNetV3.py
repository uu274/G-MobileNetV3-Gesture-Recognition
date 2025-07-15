import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupMixAttention(nn.Module):
    """
    Group-Mix Attention as described in G-MobileNetV3 paper.
    Splits Q,K,V into groups, aggregates some segments, and performs attention.
    """
    def __init__(self, dim, num_heads=1, group_size=4, aggregator_kernel_sizes=[3,5,7,9]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.group_size = group_size
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_out = nn.Linear(dim, dim)
        # Define aggregator convs for group proxies
        self.aggregators = nn.ModuleList([
            nn.Conv1d(group_size, group_size, k, padding=k//2, groups=group_size, bias=False)
            for k in aggregator_kernel_sizes
        ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3C)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each (B, N, C)

        # Split into groups along the sequence dimension
        # pad N to multiple of group_size
        if N % self.group_size != 0:
            pad = self.group_size - (N % self.group_size)
            q = F.pad(q, (0,0,0,pad), value=0)
            k = F.pad(k, (0,0,0,pad), value=0)
            v = F.pad(v, (0,0,0,pad), value=0)
        G = q.size(1) // self.group_size
        # reshape to (B, G, group_size, C)
        qg = q.view(B, G, self.group_size, C)
        kg = k.view(B, G, self.group_size, C)
        vg = v.view(B, G, self.group_size, C)

        # Generate group proxies by aggregating each group segment
        proxies = []
        for conv in self.aggregators:
            # transpose to (B, C, N) then split
            xg = q.transpose(1,2).view(B, C, G, self.group_size)
            # merge G into batch to apply conv
            xg = xg.reshape(B*C*G, self.group_size, 1)
            agg = conv(xg).view(B, C, G, self.group_size)
            agg = agg.view(B, C, N + pad if N % self.group_size != 0 else N)
            proxies.append(agg.transpose(1,2))  # (B, N, C)
        # concatenate original and proxies
        q_mix = torch.cat([q] + proxies, dim=-1)
        k_mix = torch.cat([k] + proxies, dim=-1)
        v_mix = torch.cat([v] + proxies, dim=-1)

        # Standard attention
        attn = self.softmax(torch.matmul(q_mix, k_mix.transpose(-2, -1)) / (C**0.5))
        out = torch.matmul(attn, v_mix)  # (B, N, C_total)
        # project back to dim
        out = self.attn_out(out)
        # truncate to original length
        out = out[:, :N]
        return out


class CrossStageConnect(nn.Module):
    """
    Cross-stage residual connection block: splits channels and fuses after convolution.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid = out_channels // 2
        # convolution path
        self.conv = nn.Conv2d(in_channels//2, mid, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, H, W)
        c = x.size(1)
        x1, x2 = torch.split(x, c//2, dim=1)
        out_conv = self.act(self.bn(self.conv(x1)))
        # fuse
        out = torch.cat([out_conv, x2], dim=1)
        return out


class BottleneckGMA(nn.Module):
    """
    Linear bottleneck with GroupMixAttention and optional PReLU.
    """
    def __init__(self, in_channels, out_channels, stride, expansion, use_se, act_layer, gma=False):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        layers = []
        # expand
        if expansion != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(act_layer())
        # depthwise
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                                 padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(act_layer())
        if use_se:
            # SE
            se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, hidden_dim//4, kernel_size=1),
                act_layer(),
                nn.Conv2d(hidden_dim//4, hidden_dim, kernel_size=1),
                nn.Sigmoid())
            layers.append(se)
        # project
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(*layers)
        # optional GMA on feature map
        self.gma = GroupMixAttention(dim=out_channels) if gma else None

    def forward(self, x):
        out = self.conv(x)
        if self.gma is not None:
            B, C, H, W = out.shape
            # flatten spatial
            feat = out.view(B, C, H*W).transpose(1,2)  # (B, N, C)
            feat = self.gma(feat)
            out = feat.transpose(1,2).view(B, C, H, W)
        if self.use_res_connect:
            return x + out
        return out


class GMobileNetV3(nn.Module):
    """
    G-MobileNetV3: MobileNetV3 backbone with GMA, CSC, and PReLU modifications.
    """
    def __init__(self, num_classes=6, width_mult=1.0):
        super().__init__()
        # setting of layers: t, c, n, s, se, nl, gma
        # from Table1 in paper
        self.cfg = [
            # t, c, n, s, se, nl, gma
            [1,  16, 1, 2, False, nn.Hardswish, False],
            [4,  24, 2, 2, False, nn.ReLU,      False],
            [3,  24, 2, 1, False, nn.ReLU,      False],
            [3,  40, 3, 2, True,  nn.Hardswish, False],
            [3,  40, 3, 1, True,  nn.Hardswish, False],
            [6,  80, 4, 2, False, nn.Hardswish, True ],  # add GMA
            [6,  112,2, 1, True,  nn.Hardswish, True ],
            [6,  160,2, 2, True,  nn.Hardswish, True ],
        ]
        input_channel = int(16 * width_mult)
        layers = []
        # initial conv
        layers.append(nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(input_channel))
        layers.append(nn.Hardswish())
        # bottlenecks
        for t, c, n, s, se, nl, gma in self.cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(
                    BottleneckGMA(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        stride=stride,
                        expansion=t,
                        use_se=se,
                        act_layer=nl,
                        gma=gma
                    )
                )
                input_channel = output_channel
        # final layers
        last_channel = int(960 * width_mult)
        layers.append(nn.Conv2d(input_channel, last_channel, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(last_channel))
        layers.append(nn.Hardswish())
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Conv2d(last_channel, 1024, kernel_size=1))
        layers.append(nn.Hardswish())
        self.features = nn.Sequential(*layers)
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # test
    model = GMobileNetV3(num_classes=6)
    inp = torch.randn(1,3,224,224)
    out = model(inp)
    print(out.shape)
