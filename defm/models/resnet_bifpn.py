# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from torchvision.models.feature_extraction import create_feature_extractor

from .bifpn import BiFPN
from functools import partial


def replace_bn_with_gn(model: nn.Module, num_groups: int = 32):
    """Recursively replace all nn.BatchNorm2d with nn.GroupNorm."""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=module.num_features)
            setattr(model, name, gn)
        else:
            replace_bn_with_gn(module, num_groups=num_groups)
    return model


class ResNetBiFPN(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        out_channels=128,
        n_blocks=1,
        pretrained=True,
        gn_groups=32,
        **kwargs,
    ):
        super().__init__()

        # pick backbone
        backbone_fn = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
            "resnet101": resnet101,
        }[backbone_name]

        backbone = backbone_fn(weights="IMAGENET1K_V1" if pretrained else None)
        backbone = replace_bn_with_gn(backbone, num_groups=gn_groups)

        # extract stages: C3=/8, C4=/16, C5=/32
        return_nodes = {
            "layer2": "c3",  # /8
            "layer3": "c4",  # /16
            "layer4": "c5",  # /32
        }
        self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)

        # pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # infer channels
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            feats = self.backbone(dummy)
            cls = self.global_pool(feats["c5"]).flatten(1)
            embed_dim = cls.shape[1]
        in_channels_list = [f.shape[1] for f in feats.values()]
        print(f"Backbone {backbone_name} feature channels: {in_channels_list}")

        # BiFPN
        self.fpn = BiFPN(in_channels_list, out_channels, n_blocks=n_blocks)

        # final norm
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.norm_cls = norm_layer(embed_dim)
        self.norm_patch = norm_layer(out_channels)

    def forward(self, x, norm=True):
        feats = self.backbone(x)  # dict with c3, c4, c5
        feats = OrderedDict([(k, v) for k, v in feats.items()])

        # Global CLS proxy = pooled backbone C5
        global_feat_backbone = self.global_pool(feats["c5"]).flatten(1)

        # BiFPN
        feats_bifpn = self.fpn(feats)

        if norm:
            global_feat_backbone = self.norm_cls(global_feat_backbone)
            for k in feats_bifpn.keys():
                # B, C, H, W -> B, H, W, C
                feats_bifpn[k] = feats_bifpn[k].permute(0, 2, 3, 1)
                feats_bifpn[k] = self.norm_patch(feats_bifpn[k])
                # B, H, W, C -> B, C, H, W
                feats_bifpn[k] = feats_bifpn[k].permute(0, 3, 1, 2)

        # Dense maps
        dense_feats_bifpn = {
            "P3": feats_bifpn["c3"],
            "P4": feats_bifpn["c4"],
            "P5": feats_bifpn["c5"],
        }

        return {
            "global_backbone": global_feat_backbone,
            "dense_bifpn": dense_feats_bifpn,
        }

    def forward_no_bifpn(self, x, norm=True):
        feats = self.backbone(x)  # dict with c3, c4, c5
        feats = OrderedDict([(k, v) for k, v in feats.items()])

        # Global CLS proxy = pooled backbone C5
        global_feat_backbone = self.global_pool(feats["c5"]).flatten(1)

        if norm:
            global_feat_backbone = self.norm_cls(global_feat_backbone)

        # Dense maps
        dense_feats = {
            "P3": feats["c3"],
            "P4": feats["c4"],
            "P5": feats["c5"],
        }

        return {
            "global_backbone": global_feat_backbone,
            "dense_feats": dense_feats,
        }


if __name__ == "__main__":
    backbones = ["resnet18", "resnet34", "resnet50", "resnet101"]
    out_channels = [128, 128, 256, 256]  # recommended BiFPN widths

    for i, backbone in enumerate(backbones):
        print(f"\nTesting backbone: {backbone}")
        model = ResNetBiFPN(
            backbone_name=backbone, out_channels=out_channels[i], n_blocks=1
        )
        x = torch.randn(1, 3, 256, 256)
        out = model(x)

        print("Global backbone feature:", out["global_backbone"].shape)
        for k, v in out["dense_bifpn"].items():
            print(k, v.shape)

        print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(
            f"Backbone params: {sum(p.numel() for p in model.backbone.parameters()) / 1e6:.2f}M"
        )
        print(
            f"BiFPN params: {sum(p.numel() for p in model.fpn.parameters()) / 1e6:.2f}M"
        )
