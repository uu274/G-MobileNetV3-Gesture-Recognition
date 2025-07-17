# group_mix_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupMixAttention(nn.Module):
    def __init__(self, dim: int, num_groups: int = 4):
        super(GroupMixAttention, self).__init__()
        assert dim % num_groups == 0, "Channels must be divisible by num_groups"
        self.num_groups = num_groups
        self.group_dim = dim // num_groups

        # Q, K, V projections with group-wise Conv
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, groups=num_groups, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, groups=num_groups, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, groups=num_groups, bias=False)

        # Final output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape

        Q = self.q_proj(x)  # (B, C, H, W)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # reshape to (B, num_groups, group_dim, H*W)
        Q = Q.view(B, self.num_groups, self.group_dim, -1)  # (B, G, Cg, N)
        K = K.view(B, self.num_groups, self.group_dim, -1)
        V = V.view(B, self.num_groups, self.group_dim, -1)

        attention = torch.matmul(Q.transpose(-2, -1), K) / (self.group_dim ** 0.5)  # (B, G, N, N)
        attention = self.softmax(attention)

        out = torch.matmul(attention, V.transpose(-2, -1)).transpose(-2, -1)  # (B, G, Cg, N)
        out = out.contiguous().view(B, C, H, W)

        return self.out_proj(out)

