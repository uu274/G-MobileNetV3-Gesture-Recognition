import sys
import os
from typing import Callable, List, Optional
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from group_mix_attention import GroupMixAttention

def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,
                 norm_layer=None, activation_layer=None):
        padding = (kernel_size - 1) // 2
        norm_layer = norm_layer or nn.BatchNorm2d
        activation_layer = activation_layer or nn.PReLU
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer()
        )

class SqueezeExcitation(nn.Module):
    def __init__(self, input_c, squeeze_factor=4):
        super().__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = F.hardsigmoid(self.fc2(scale), inplace=True)
        return x * scale

class CrossStageBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU()
        )

    def forward(self, x, shortcut):
        x = torch.cat([x, shortcut], dim=1)
        return self.fuse(x)

class InvertedResidualConfig:
    def __init__(self, input_c, kernel, expanded_c, out_c, use_se, activation, stride, width_multi):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride

    @staticmethod
    def adjust_channels(channels, width_multi):
        return _make_divisible(channels * width_multi, 8)

class InvertedResidual(nn.Module):
    def __init__(self, cnf, norm_layer):
        super().__init__()
        self.use_res_connect = cnf.stride == 1 and cnf.input_c == cnf.out_c
        activation_layer = nn.PReLU

        layers = []
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c, cnf.expanded_c, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer))
        layers.append(ConvBNActivation(cnf.expanded_c, cnf.expanded_c, kernel_size=cnf.kernel,
                                       stride=cnf.stride, groups=cnf.expanded_c,
                                       norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        layers.append(ConvBNActivation(cnf.expanded_c, cnf.out_c, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c

    def forward(self, x):
        return x + self.block(x) if self.use_res_connect else self.block(x)

class MobileNetV3(nn.Module):
    def __init__(self, inverted_residual_setting, last_channel, num_classes=1000,
                 block=None, norm_layer=None):
        super().__init__()
        if block is None:
            block = InvertedResidual
        norm_layer = norm_layer or partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        self.features = nn.ModuleList()
        self.cross_stage = nn.ModuleList()

        firstconv_output_c = inverted_residual_setting[0].input_c
        self.stem = ConvBNActivation(3, firstconv_output_c, kernel_size=3, stride=2,
                                     norm_layer=norm_layer, activation_layer=nn.PReLU)

        prev_out = firstconv_output_c
        for i, cnf in enumerate(inverted_residual_setting):
            self.features.append(block(cnf, norm_layer))
            if i >= 4:
                self.cross_stage.append(CrossStageBlock(prev_out, cnf.out_c))
            else:
                self.cross_stage.append(None)
            prev_out = cnf.out_c

        self.attention = GroupMixAttention(dim=prev_out, num_groups=4)

        lastconv_output_c = 6 * prev_out
        self.conv_last = ConvBNActivation(prev_out, lastconv_output_c, kernel_size=1,
                                          norm_layer=norm_layer, activation_layer=nn.PReLU)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_c, last_channel),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        for i, layer in enumerate(self.features):
            shortcut = x
            x = layer(x)
            if self.cross_stage[i]:
                x = self.cross_stage[i](x, shortcut)
        x = self.attention(x)
        x = self.conv_last(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def mobilenet_v3_large(num_classes=6):
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, True, "RE", 2),
        bneck_conf(16, 3, 72, 24, False, "RE", 2),
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96, True, "HS", 2),
        bneck_conf(96, 5, 576, 96, True, "HS", 1),
        bneck_conf(96, 5, 576, 96, True, "HS", 1)
    ]
    last_channel = _make_divisible(1024)
    return MobileNetV3(inverted_residual_setting, last_channel, num_classes=num_classes)

