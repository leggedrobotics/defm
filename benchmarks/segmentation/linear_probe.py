# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Linear-probe semantic segmentation on frozen dense features.

Protocol (DeFM paper, Sec. V-C / rebuttal):
  - frozen encoder, last-layer patch tokens, extracted once and cached (no augmentation)
  - one linear layer per token, logits bilinearly upsampled to the input resolution
  - pixel-wise cross-entropy (ignore 255), AdamW (wd 1e-4), one-cycle LR (linear warm-up, cosine decay),
    batch 32, FP16, grad-norm clip 1.0; peak LR from {1e-3, 5e-3}
  - evaluated once, after the final epoch, on the test split
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from benchmarks.encoders import center_pad
from benchmarks.segmentation.datasets import IGNORE_INDEX, SegmentationDataset


@torch.no_grad()
def cache_features(
    encoder,
    dataset: SegmentationDataset,
    out_dir: Path,
    batch_size: int = 16,
    workers: int = 8,
):
    """Write dense features [N, h*w, C] (fp16) and labels [N, H, W] (uint8) as memory-mapped .npy files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path, label_path = out_dir / "features.npy", out_dir / "labels.npy"
    if (out_dir / "done").exists():
        return np.load(feat_path, mmap_mode="r"), np.load(label_path, mmap_mode="r")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True,
    )
    feats = labels = None
    i, t0 = 0, time.time()
    for x, y in loader:
        x, _ = center_pad(x.cuda(non_blocking=True), encoder.pad_multiple)
        with torch.autocast("cuda", dtype=torch.float16):
            _, dense = encoder(x)
        if feats is None:
            n = len(dataset)
            feats = np.lib.format.open_memmap(
                feat_path, "w+", np.float16, (n, *dense.shape[1:])
            )
            labels = np.lib.format.open_memmap(
                label_path, "w+", np.uint8, (n, *y.shape[1:])
            )
        feats[i : i + len(x)] = dense.half().cpu().numpy()
        labels[i : i + len(x)] = y.numpy().astype(np.uint8)
        i += len(x)
    feats.flush(), labels.flush()
    (out_dir / "done").touch()
    print(
        f"  cached {i} samples {tuple(feats.shape[1:])} in {time.time() - t0:.0f}s -> {out_dir}",
        flush=True,
    )
    return np.load(feat_path, mmap_mode="r"), np.load(label_path, mmap_mode="r")


class _Cached(Dataset):
    def __init__(self, feats, labels):
        self.feats, self.labels = feats, labels

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(self.feats[i])), torch.from_numpy(
            np.asarray(self.labels[i]).astype(np.int64)
        )


class LinearSegHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        image_size,
        patch_size: int,
        pad_multiple: int,
    ):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        h, w = image_size
        self.size = (h, w)
        self.padded = (
            math.ceil(h / pad_multiple) * pad_multiple,
            math.ceil(w / pad_multiple) * pad_multiple,
        )
        self.grid = (self.padded[0] // patch_size, self.padded[1] // patch_size)

    def forward(self, tokens):  # [B, h*w, C] -> logits [B, K, H, W]
        logits = (
            self.head(tokens.float())
            .transpose(1, 2)
            .reshape(len(tokens), -1, *self.grid)
        )
        logits = F.interpolate(
            logits, size=self.padded, mode="bilinear", align_corners=False
        )
        top, left = (
            (self.padded[0] - self.size[0]) // 2,
            (self.padded[1] - self.size[1]) // 2,
        )
        return logits[:, :, top : top + self.size[0], left : left + self.size[1]]


def segmentation_metrics(conf: torch.Tensor) -> dict:
    """mIoU over classes that occur in the ground truth or the predictions (standard), plus per-class IoU."""
    conf = conf.double()
    tp = conf.diag()
    union = conf.sum(0) + conf.sum(1) - tp
    iou = tp / union  # NaN for classes absent from both GT and predictions
    return {
        "miou": 100 * torch.nanmean(iou).item(),
        "miou_absent_as_zero": 100
        * torch.nan_to_num(iou).mean().item(),  # convention of the original paper code
        "pixel_acc": 100 * (tp.sum() / conf.sum()).item(),
        "per_class_iou": [
            None if math.isnan(v) else round(100 * v, 2) for v in iou.tolist()
        ],
    }


def train_and_evaluate(
    train_cache,
    test_cache,
    encoder,
    num_classes,
    image_size,
    *,
    lr,
    epochs,
    warmup_epochs,
    batch_size=32,
    weight_decay=1e-4,
    workers=8,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = LinearSegHead(
        encoder.embed_dim,
        num_classes,
        image_size,
        encoder.patch_size,
        encoder.pad_multiple,
    ).cuda()
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        _Cached(*train_cache),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        _Cached(*test_cache),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=warmup_epochs / epochs,
    )
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        head.train()
        for tokens, y in train_loader:
            tokens, y = tokens.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(head(tokens), y, ignore_index=IGNORE_INDEX)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        print(f"    epoch {epoch + 1}/{epochs}  loss {loss.item():.4f}", flush=True)

    head.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device="cuda")
    with torch.no_grad():
        for tokens, y in test_loader:
            tokens, y = tokens.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = head(tokens).argmax(1)
            keep = y != IGNORE_INDEX
            conf += torch.bincount(
                y[keep] * num_classes + pred[keep], minlength=num_classes**2
            ).view(num_classes, -1)
    return segmentation_metrics(conf)
