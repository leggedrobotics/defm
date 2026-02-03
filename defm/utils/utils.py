# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
import torchvision.transforms.v2 as tt
import math

DEFM_MEAN = [0.248880, 0.495620, 0.492858]
DEFM_STD = [0.139357, 0.271314, 0.297177]


def make_norm_transform(height: int, width: int, to_tensor: bool = True) -> tt.Compose:
    """
    Create a DeFM normalization transform for a given (H, W).

    Parameters:
      - height, width: target size for Resize
      - to_tensor: if True include tt.ToTensor() as the first step.
                   If False assume input is already a Tensor and only Resize+Normalize are applied.

    Usage:
        norm_transform = make_norm_transform(final_h, final_w)
        norm_transform_no_to_tensor = make_norm_transform(final_h, final_w, to_tensor=False)
    """
    transforms = []
    if to_tensor:
        transforms.append(tt.ToTensor())
    transforms.extend(
        [
            tt.Resize((height, width)),
            tt.Normalize(mean=DEFM_MEAN, std=DEFM_STD),
        ]
    )
    return tt.Compose(transforms)


def convert_to_3channel_metric(metric_depth, max_depth_c1=100.0, max_depth_c2=9.0):
    """
    Core DeFM Metric-Aware Normalization logic.
    Note: max_depth_c2 is set to 9.0 instead of 10.0 (different from the paper) since we inadvertently used
    log(10) instead of log1p(10) during pretraining. To offset this bug we set the max depth for C2 to 9.0 as
    log1p(9) ~= log(10).
    """
    # Convert NaNs and Infs to zeros
    metric_depth = np.nan_to_num(metric_depth, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip at max depth
    metric_depth = np.clip(metric_depth, 0, max_depth_c1)

    log_depth = np.log1p(metric_depth)

    # C1: Global (100m) | C2: Mid-range (10m)
    c1 = log_depth / np.log1p(max_depth_c1)
    c2 = np.clip(log_depth / np.log1p(max_depth_c2), 0, 1)

    # C3: Local Relative (per-image min-max)
    min_log, max_log = np.log1p(metric_depth.min()), np.log1p(metric_depth.max())
    if max_log > min_log:
        c3 = (log_depth - min_log) / (max_log - min_log)
    else:
        c3 = np.zeros_like(log_depth)

    return np.stack([c1, c2, c3], axis=-1)


def get_target_size(
    current_size: Tuple[int, int],
    target_size: Optional[Union[int, Tuple[int, int]]] = None,
    patch_size: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Determines the final (H, W) based on target_size and patch_size constraints.
    """
    curr_h, curr_w = current_size

    # 1. Handle target_size
    if target_size is None:
        new_h, new_w = curr_h, curr_w
    elif isinstance(target_size, int):
        new_h, new_w = target_size, target_size
    else:
        new_h, new_w = target_size

    # 2. Adjust for patch_size divisibility
    if patch_size is not None:
        new_h = (new_h // patch_size) * patch_size
        new_w = (new_w // patch_size) * patch_size

    return new_h, new_w


def preprocess_depth_image(
    input_data: Union[np.ndarray, torch.Tensor, Image.Image],
    target_size: Optional[Union[int, Tuple[int, int]]] = None,
    patch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    DeFM Preprocessing Pipeline.
    input_data: Depth image as Numpy array, Torch Tensor, or PIL Image.
    Supports:
    - Flexible target sizes: int, (H, W), or None.
    - Automatic patch alignment (divisibility by patch_size).
    """
    # Standardize to Numpy [H, W]
    if isinstance(input_data, Image.Image):
        img_np = np.array(input_data).astype(np.float32)
    elif isinstance(input_data, torch.Tensor):
        img_np = input_data.detach().cpu().numpy().astype(np.float32)
    else:
        img_np = input_data.astype(np.float32)

    if img_np.ndim == 3:
        img_np = img_np[:, :, 0]

    # Calculate and apply resizing
    final_h, final_w = get_target_size(img_np.shape[-2:], target_size, patch_size)

    # 3-channel normalization
    metric_stack = convert_to_3channel_metric(img_np)

    norm_transform = make_norm_transform(final_h, final_w)

    tensor = norm_transform(metric_stack).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def preprocess_depth_dav2(
    input_data: Union[np.ndarray, torch.Tensor, Image.Image],
    target_size: Optional[Union[int, Tuple[int, int]]] = None,
    patch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Specifically for non-metric depth (Depth Anything V2).
    1. Converts uint8 inverse depth to pseudo-metric via exponential decay.
    2. Passes through standard DeFM 3-channel normalization.
    """
    # Standardize to Numpy
    if isinstance(input_data, Image.Image):
        img_np = np.array(input_data).astype(np.float32)
    elif isinstance(input_data, torch.Tensor):
        img_np = input_data.detach().cpu().numpy().astype(np.float32)
    else:
        img_np = input_data.astype(np.float32)

    if img_np.ndim == 3:
        img_np = img_np[:, :, 0]

    # --- DAV2 Specific: Inverse Depth to Pseudo-Metric ---
    # Normalize to [0, 1]
    img_min, img_max = img_np.min(), img_np.max()
    img_norm = (img_np - img_min) / (img_max - img_min + 1e-8)

    # Exponential decay (This fixed transform was used in pretraining for the MDE datasets)
    # Higher pixel values in DAV2 usually mean 'closer', so we treat as inverse depth
    pseudo_metric_depth = 10.0 * np.exp(-5.0 * img_norm)

    # --- Standard DeFM Processing ---
    # Calculate target size with patch alignment
    final_h, final_w = get_target_size(
        pseudo_metric_depth.shape, target_size, patch_size
    )

    # 3-channel normalization (C1, C2, C3)
    metric_stack = convert_to_3channel_metric(pseudo_metric_depth)

    norm_transform = make_norm_transform(final_h, final_w)

    tensor = norm_transform(metric_stack).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def preprocess_depth_batch(
    input_batch: Union[np.ndarray, torch.Tensor],
    target_size: Optional[Union[int, Tuple[int, int]]] = None,
    patch_size: Optional[int] = None,
    max_depth_c1: float = 100.0,
    max_depth_c2: float = 9.0,
    device: Optional[Union[str, torch.device]] = "cpu",
) -> torch.Tensor:
    """
    Vectorized batch preprocessing for DeFM depth images (no Python loop).
    Accepts input shapes:
      - (N, H, W)
      - (N, 1, H, W)
    Returns:
      - torch.Tensor of shape (N, 3, H_out, W_out), dtype=torch.float32 on `device`.
    Behavior:
      - Builds the 3-channel metric representation (C1, C2, C3) in a vectorized way.
      - Resizes using torch.nn.functional.interpolate (fast, batched).
      - Applies DeFM per-channel mean/std normalization.
    """

    # Convert numpy -> torch
    if isinstance(input_batch, np.ndarray):
        t = torch.from_numpy(input_batch)
    elif isinstance(input_batch, torch.Tensor):
        t = input_batch
    else:
        raise TypeError("input_batch must be a numpy array or torch.Tensor")

    # Ensure float32
    t = t.to(dtype=torch.float32)

    # Normalize shapes to (N,1,H,W)
    if t.ndim == 3:
        # (N,H,W) -> (N,1,H,W)
        t = t.unsqueeze(1)
    elif t.ndim == 4:
        if t.shape[1] != 1:
            # take first channel if multi-channel provided
            t = t[:, :1, :, :]
    else:
        raise ValueError("input_batch must have shape (N,H,W) or (N,1,H,W)")

    # Move to device
    device = torch.device(device)
    t = t.to(device)

    N, _, H, W = t.shape

    # Determine final target size
    final_h, final_w = get_target_size((H, W), target_size, patch_size)

    # Replace NaN/inf with 0
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip to max depth for C1 computation
    t_clamped = torch.clamp(t, min=0.0, max=max_depth_c1)

    # log1p
    log_depth = torch.log1p(t_clamped)

    # C1: normalized by log1p(max_depth_c1)
    denom_c1 = math.log1p(max_depth_c1)
    c1 = log_depth / denom_c1

    # C2: clipped ratio using max_depth_c2
    denom_c2 = math.log1p(max_depth_c2)
    c2 = torch.clamp(log_depth / denom_c2, min=0.0, max=1.0)

    # C3: per-image min-max normalization over spatial dims
    log_flat = log_depth.view(N, -1)
    min_log = log_flat.min(dim=1)[0].view(N, 1, 1, 1)
    max_log = log_flat.max(dim=1)[0].view(N, 1, 1, 1)
    denom = max_log - min_log
    # avoid divide-by-zero: where denom <= 0 set denom=1 and result will be zero after subtract
    denom_safe = torch.where(denom > 0.0, denom, torch.ones_like(denom))
    c3 = (log_depth - min_log) / denom_safe
    # zero out images where denom was <= 0
    c3 = torch.where(denom > 0.0, c3, torch.zeros_like(c3))

    # Stack channels -> (N,3,H,W)
    x = torch.cat([c1, c2, c3], dim=1)

    norm_transform = make_norm_transform(final_h, final_w, to_tensor=False)
    x = norm_transform(x)

    return x
