# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""ImageNet-1k-Depth: Depth Anything V2 disparity of ImageNet-1k images (Tab. II and III of the DeFM paper).

Disparity -> encoder input follows DeFM pretraining, where monocular-depth data was stored as 8-bit images:
  1. per-image min-max to 8 bits                      (as in the pretraining shards; an 8-bit image for RGB models)
  2. "defm":      pseudo-metric depth 10 * exp(-5 x) and the DeFM 3-channel encoding (float)
     otherwise:   the 8-bit disparity in [0, 1], stacked to 3 channels (near = bright)
  3. Resize(256) on the float tensor, then CenterCrop(224) (RandomCrop(224) + flip for linear-probe training)
  4. encoder mean/std
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import IterableDataset, get_worker_info

from benchmarks.preprocess import normalize

HF_REPO = "leggedrobotics/defm-bench-imagenet-depth"


def resolve_root(data_root: Optional[str]) -> Path:
    if data_root is not None:
        return Path(data_root) / "imagenet_depth"
    from huggingface_hub import snapshot_download

    # gated dataset: accept the ImageNet terms on the dataset page and log in with `hf auth login`
    return Path(
        snapshot_download(
            HF_REPO, repo_type="dataset", allow_patterns=["meta.json", "data/*"]
        )
    )


def three_channel_from_disparity(disp01: np.ndarray) -> np.ndarray:
    """DeFM encoding of min-max normalized relative disparity (pretraining MDE path, defm.utils.preprocess_depth_dav2)."""
    metric = 10.0 * np.exp(-5.0 * disp01)
    log_depth = np.log1p(metric)
    c1 = log_depth / np.log1p(100.0)
    c2 = np.clip(log_depth / np.log1p(9.0), 0, 1)
    lo, hi = np.log1p(metric.min()), np.log1p(metric.max())
    c3 = (log_depth - lo) / (hi - lo) if hi > lo else np.zeros_like(log_depth)
    return np.stack([c1, c2, c3], axis=-1)


def encode_disparity(disp: np.ndarray, input_norm: str) -> torch.Tensor:
    # Min-max normalized in float, as defm.utils.preprocess_depth_dav2 does. The paper quantized to 8 bit here
    # (as in the pretraining shards), which costs DeFM-S ~0.5 kNN top-1.
    disp = disp.astype(np.float64)
    disp01 = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
    x = (
        three_channel_from_disparity(disp01)
        if input_norm == "defm"
        else np.repeat(disp01[..., None], 3, axis=-1)
    )
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).permute(2, 0, 1)


def build_transform(train: bool):
    crop = (
        [T.RandomCrop(224), T.RandomHorizontalFlip()] if train else [T.CenterCrop(224)]
    )
    return T.Compose([T.Resize(256, antialias=True), *crop])


class ImageNetDepth(IterableDataset):
    """Streams Parquet row groups (in a random order when shuffle=True; the shards themselves are pre-shuffled)."""

    def __init__(
        self,
        split: str,
        input_norm: str,
        train_transform: bool,
        data_root: Optional[str] = None,
        shuffle: bool = False,
        seed: int = 0,
    ):
        root = resolve_root(data_root)
        self.meta = json.loads((root / "meta.json").read_text())
        self.files = sorted((root / "data").glob(f"{split}-*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No '{split}' shards in {root / 'data'}")
        self.groups = [
            (f, g) for f in self.files for g in range(pq.ParquetFile(f).num_row_groups)
        ]
        self.length = self.meta["splits"][split]
        self.input_norm, self.transform = input_norm, build_transform(train_transform)
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.length

    def __iter__(self):
        groups = list(self.groups)
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(groups)
        info = get_worker_info()
        if info is not None:
            groups = groups[info.id :: info.num_workers]
        handles = {}
        for path, g in groups:
            pf = handles.setdefault(path, pq.ParquetFile(path))
            table = pf.read_row_group(g, columns=["label", "depth"])
            labels = table.column("label").to_pylist()
            pngs = [d["bytes"] for d in table.column("depth").to_pylist()]
            order = (
                np.random.default_rng(self.seed + self.epoch + g).permutation(
                    len(labels)
                )
                if self.shuffle
                else range(len(labels))
            )
            for i in order:
                disp = cv2.imdecode(
                    np.frombuffer(pngs[i], np.uint8), cv2.IMREAD_UNCHANGED
                )
                x = self.transform(encode_disparity(disp, self.input_norm))
                yield normalize(x, self.input_norm), labels[i]
