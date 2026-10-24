# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Depth -> encoder input, per encoder family.

- "defm":     official DeFM metric-aware 3-channel encoding + DeFM mean/std
- "imagenet": per-image min-max depth, stacked to 3 channels + ImageNet mean/std (DINOv2, DINOv3, DAv2)
- "unit":     per-image min-max depth, stacked to 3 channels, left in [0, 1] (RADIO and Theia normalize internally)

Encodings are computed at the native resolution; resizing happens afterwards on the float tensor.
"""

import numpy as np
import torch

from defm.utils.utils import DEFM_MEAN, DEFM_STD, convert_to_3channel_metric

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NORM_STATS = {
    "defm": (DEFM_MEAN, DEFM_STD),
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "unit": None,
}


def encode_metric_depth(depth_m: np.ndarray, input_norm: str) -> torch.Tensor:
    """Metric depth in meters [H, W] (0 = invalid) -> float tensor [3, H, W] before normalization."""
    depth_m = np.nan_to_num(depth_m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if input_norm == "defm":
        x = convert_to_3channel_metric(depth_m)
    else:
        lo, hi = depth_m.min(), depth_m.max()
        x = np.repeat(((depth_m - lo) / (hi - lo + 1e-6))[..., None], 3, axis=-1)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).permute(2, 0, 1)


def normalize(x: torch.Tensor, input_norm: str) -> torch.Tensor:
    """Apply the encoder's mean/std to a [..., 3, H, W] tensor."""
    stats = NORM_STATS[input_norm]
    if stats is None:
        return x
    mean = torch.tensor(stats[0], dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor(stats[1], dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x - mean) / std
