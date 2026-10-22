# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Writer for the shared benchmark format (see benchmarks/segmentation/datasets.py).

Parquet shards with Hugging Face `Image` features, so the Hub dataset viewer can render them.
"""

import json
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

IGNORE_INDEX = 255

# ScanNet / SUN RGB-D classes (NYU40 ids 1..40, stored as 0..39; SUN RGB-D uses the first 37)
NYU40 = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor mat",
    "clothes",
    "ceiling",
    "books",
    "refrigerator",
    "television",
    "paper",
    "towel",
    "shower curtain",
    "box",
    "whiteboard",
    "person",
    "night stand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    "otherstructure",
    "otherfurniture",
    "otherprop",
]
_IMAGE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
SCHEMA = pa.schema(
    [("id", pa.string()), ("depth", _IMAGE), ("label", _IMAGE)]
).with_metadata(
    {
        "huggingface": json.dumps(
            {
                "info": {
                    "features": {
                        "id": {"dtype": "string", "_type": "Value"},
                        "depth": {"_type": "Image"},
                        "label": {"_type": "Image"},
                    }
                }
            }
        )
    }
)


def encode_depth(depth_m: np.ndarray, depth_scale: float) -> bytes:
    """Metric depth (m) -> 16-bit PNG with value = round(depth * depth_scale); invalid / NaN -> 0."""
    d = np.nan_to_num(depth_m.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    q = np.clip(np.round(d * depth_scale), 0, 65535).astype(np.uint16)
    ok, buf = cv2.imencode(".png", q)
    assert ok
    return buf.tobytes()


def encode_label(label: np.ndarray) -> bytes:
    assert label.dtype == np.uint8 and label.ndim == 2
    ok, buf = cv2.imencode(".png", label)
    assert ok
    return buf.tobytes()


def write_split(
    out_root: Path,
    split: str,
    samples: Iterable[Tuple[str, bytes, bytes]],
    shard_mb: int = 400,
) -> int:
    """samples: (id, depth_png, label_png). Writes <out_root>/data/<split>-XXXXX-of-XXXXX.parquet."""
    data_dir = Path(out_root) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shards, rows, size, n = [], [], 0, 0

    def flush():
        nonlocal rows, size
        if rows:
            ids, depths, labels = zip(*rows)
            table = pa.table(
                {
                    "id": list(ids),
                    "depth": [{"bytes": b, "path": None} for b in depths],
                    "label": [{"bytes": b, "path": None} for b in labels],
                },
                schema=SCHEMA,
            )
            path = data_dir / f"{split}-{len(shards):05d}.tmp.parquet"
            pq.write_table(table, path, row_group_size=256)
            shards.append(path)
            rows, size = [], 0

    for sid, depth_png, label_png in samples:
        rows.append((sid, depth_png, label_png))
        size += len(depth_png) + len(label_png)
        n += 1
        if size > shard_mb * 2**20:
            flush()
    flush()
    for i, path in enumerate(shards):
        path.rename(data_dir / f"{split}-{i:05d}-of-{len(shards):05d}.parquet")
    return n


def write_meta(out_root: Path, **meta) -> None:
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "meta.json").write_text(json.dumps(meta, indent=2))
