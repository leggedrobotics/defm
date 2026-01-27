# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class WeightedFusion(nn.Module):
    """Learnable weighted sum for BiFPN feature fusion."""
    def __init__(self, n_inputs):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))

    def forward(self, features):
        w = F.relu(self.weights)
        w = w / (w.sum() + 1e-6)
        return sum(w[i] * f for i, f in enumerate(features))


class BiFPNLayer(nn.Module):
    def __init__(self, out_channels, n_levels):
        super().__init__()
        self.out_channels = out_channels
        self.n_levels = n_levels

        # fusion modules
        self.fuse_topdown = nn.ModuleList([WeightedFusion(2) for _ in range(n_levels - 1)])
        self.fuse_bottomup = nn.ModuleList([WeightedFusion(2) for _ in range(n_levels - 1)])

        # convs for smoothing after fusion
        self.convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(n_levels)])

    def forward(self, feats):
        # feats: list of feature maps [P3 (/8), P4 (/16), ..., Pn]

        # --- Top-down pass ---
        td_feats = [None] * self.n_levels
        td_feats[-1] = feats[-1]  # top stays
        for i in range(self.n_levels - 2, -1, -1):
            up = F.interpolate(td_feats[i + 1], size=feats[i].shape[-2:], mode="nearest")
            td_feats[i] = self.fuse_topdown[i]([feats[i], up])

        # --- Bottom-up pass ---
        out_feats = [None] * self.n_levels
        out_feats[0] = td_feats[0]
        for i in range(1, self.n_levels):
            down = F.max_pool2d(out_feats[i - 1], kernel_size=2, stride=2)
            out_feats[i] = self.fuse_bottomup[i - 1]([td_feats[i], down])

        # --- Apply convs ---
        out_feats = [conv(f) for conv, f in zip(self.convs, out_feats)]
        return out_feats


class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, n_blocks=1):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.blocks = nn.ModuleList([BiFPNLayer(out_channels, len(in_channels_list)) for _ in range(n_blocks)])

    def forward(self, inputs: OrderedDict):
        # Project backbone features to same channels
        feats = [proj(inputs[k]) for proj, k in zip(self.proj, inputs.keys())]

        for block in self.blocks:
            feats = block(feats)

        return OrderedDict([(k, f) for k, f in zip(inputs.keys(), feats)])
