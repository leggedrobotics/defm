# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Depth semantic-segmentation benchmarks (Tab. V of the DeFM paper).

All datasets share one on-disk format, produced by the scripts in scripts/datasets/ and used for the copies
hosted on the Hugging Face Hub:

    <root>/meta.json                      name, classes, depth_scale, splits, citation
    <root>/data/<split>-XXXXX-of-XXXXX.parquet
        id     string
        depth  struct{bytes, path}   16-bit PNG, depth_in_meters = value / meta["depth_scale"], 0 = invalid
        label  struct{bytes, path}   8-bit PNG, class ids 0..num_classes-1, 255 = ignore
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from benchmarks.preprocess import encode_metric_depth, normalize

IGNORE_INDEX = 255


@dataclass(frozen=True)
class SegBenchmark:
    size: Tuple[int, int]  # (H, W) every sample is brought to
    fit: str  # "resize" (bilinear depth / nearest labels) or "pad_crop" (center pad + center crop)
    epochs: int
    warmup_epochs: int
    test_split: str
    hf_repo: Optional[
        str
    ]  # None: build it locally with scripts/datasets/convert_<name>.py (license)


BENCHMARKS = {
    "scannet": SegBenchmark((480, 640), "resize", 25, 3, "test", None),
    "sunrgbd": SegBenchmark((530, 730), "pad_crop", 25, 3, "test", None),
    "offsed": SegBenchmark((360, 640), "resize", 200, 20, "test", None),
    "tartanground": SegBenchmark(
        (640, 640), "resize", 25, 3, "test", "leggedrobotics/defm-bench-tartanground"
    ),
    "graspnet": SegBenchmark(
        (360, 640), "resize", 25, 3, "test", "leggedrobotics/defm-bench-graspnet"
    ),
}


def resolve_root(name: str, data_root: Optional[str]) -> Path:
    """Local dataset directory: <data_root>/<name> if given, else a snapshot of the Hub copy."""
    if data_root is not None:
        root = Path(data_root) / name
        if not (root / "meta.json").exists():
            raise FileNotFoundError(
                f"{root}/meta.json not found. Build it with scripts/datasets/convert_{name}.py"
            )
        return root
    repo = BENCHMARKS[name].hf_repo
    if repo is None:
        raise ValueError(
            f"'{name}' cannot be redistributed and is not on the Hub. Download it from the official source, "
            f"run scripts/datasets/convert_{name}.py, and pass --data-root."
        )
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo, repo_type="dataset", allow_patterns=["meta.json", "data/*"]
        )
    )


def _read_split(root: Path, split: str):
    files = sorted((root / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files for split '{split}' in {root / 'data'}"
        )
    ids, depth, label = [], [], []
    for f in files:
        t = pq.read_table(f, columns=["id", "depth", "label"])
        ids += t.column("id").to_pylist()
        depth += [d["bytes"] for d in t.column("depth").to_pylist()]
        label += [d["bytes"] for d in t.column("label").to_pylist()]
    return ids, depth, label


def _decode(png: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)


def _center_pad_crop(
    x: torch.Tensor, size: Tuple[int, int], fill: float
) -> torch.Tensor:
    """Place x [..., H, W] centered on a canvas of `size`, padding with `fill` and cropping as needed."""
    th, tw = size
    h, w = x.shape[-2:]
    ph, pw = max(th - h, 0), max(tw - w, 0)
    x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=fill)
    h, w = x.shape[-2:]
    top, left = (h - th) // 2, (w - tw) // 2
    return x[..., top : top + th, left : left + tw]


class SegmentationDataset(Dataset):
    """Returns (normalized encoder input [3, H, W], labels [H, W] int64) at the benchmark resolution."""

    def __init__(
        self, name: str, split: str, input_norm: str, data_root: Optional[str] = None
    ):
        self.bench = BENCHMARKS[name]
        root = resolve_root(name, data_root)
        self.meta = json.loads((root / "meta.json").read_text())
        self.ids, self._depth, self._label = _read_split(root, split)
        self.input_norm = input_norm
        self.num_classes = self.meta["num_classes"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        depth_m = _decode(self._depth[i]).astype(np.float32) / self.meta["depth_scale"]
        x = encode_metric_depth(depth_m, self.input_norm)
        label = _decode(self._label[i])
        if self.bench.fit == "resize":
            x = F.interpolate(
                x[None],
                size=self.bench.size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
            # PIL nearest (as in the paper's loaders); torch's nearest picks different pixels when downscaling
            label = np.asarray(
                Image.fromarray(label).resize(self.bench.size[::-1], Image.NEAREST)
            )
            y = torch.from_numpy(label.astype(np.int64))
        else:
            x = _center_pad_crop(x, self.bench.size, 0.0)
            y = _center_pad_crop(
                torch.from_numpy(label.astype(np.int64)), self.bench.size, IGNORE_INDEX
            )
        return normalize(x, self.input_norm), y
