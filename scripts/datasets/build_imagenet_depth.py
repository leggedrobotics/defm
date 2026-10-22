# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Build ImageNet-1k-Depth (Parquet) from Depth Anything V2 disparity maps.

Input: a directory with train_images*/<wnid>_<idx>_<wnid>.npz and val_images/ILSVRC2012_val_<n>_<wnid>.npz, each
holding key 'depth' = uint16(DAv2 ViT-L relative disparity) at the original ImageNet resolution
(see generate_imagenet_depth.py). Output (one shard = one Parquet file, row groups of 256 samples):

    <out>/meta.json
    <out>/data/train-XXXXX-of-XXXXX.parquet        rows in a fixed random order (seed 0)
    <out>/data/validation-XXXXX-of-XXXXX.parquet
        id     string   ImageNet file stem (e.g. n01440764_10026, ILSVRC2012_val_00000001)
        label  int64    0..999, index into the sorted WordNet ids (standard ImageFolder order)
        depth  Image    16-bit PNG of the uint16 disparity (lossless)
"""

import argparse
import glob
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_IMAGE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
SCHEMA = pa.schema(
    [("id", pa.string()), ("label", pa.int64()), ("depth", _IMAGE)]
).with_metadata(
    {
        "huggingface": json.dumps(
            {
                "info": {
                    "features": {
                        "id": {"dtype": "string", "_type": "Value"},
                        "label": {"names": None, "_type": "ClassLabel"},
                        "depth": {"_type": "Image"},
                    }
                }
            }
        )
    }
)


def _sample(path: str):
    stem = os.path.basename(path)[:-4]
    wnid = stem.split("_")[-1]
    image_id = stem[: -len(wnid) - 1]
    disp = np.load(path)["depth"]
    assert disp.dtype == np.uint16, (path, disp.dtype)
    ok, png = cv2.imencode(".png", disp)
    assert ok
    return image_id, wnid, png.tobytes()


def _write_shard(job):
    files, out_path, wnid_to_idx = job
    rows = [_sample(f) for f in files]
    table = pa.table(
        {
            "id": [r[0] for r in rows],
            "label": [wnid_to_idx[r[1]] for r in rows],
            "depth": [{"bytes": r[2], "path": None} for r in rows],
        },
        schema=SCHEMA,
    )
    pq.write_table(table, out_path, row_group_size=256)
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src", required=True, help="directory with train_images*/ and val_images/"
    )
    p.add_argument(
        "--labels",
        default=str(Path(__file__).parent / "imagenet_classes.txt"),
        help="lines 'wnid,class name' (default: the bundled ImageNet-1k class list)",
    )
    p.add_argument("--out", required=True)
    p.add_argument(
        "--shard-size", type=int, default=6400, help="samples per Parquet file"
    )
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    names = dict(
        line.strip().split(",", 1) for line in open(args.labels) if line.strip()
    )
    wnids = sorted(names)
    assert len(wnids) == 1000
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}

    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    splits = {
        "train": np.random.default_rng(0)
        .permutation(sorted(glob.glob(f"{args.src}/train_images*/*.npz")))
        .tolist(),
        "validation": sorted(glob.glob(f"{args.src}/val_images/*.npz")),
    }
    counts = {}
    for split, files in splits.items():
        n_shards = -(-len(files) // args.shard_size)
        jobs = [
            (
                files[i * args.shard_size : (i + 1) * args.shard_size],
                out / "data" / f"{split}-{i:05d}-of-{n_shards:05d}.parquet",
                wnid_to_idx,
            )
            for i in range(n_shards)
        ]
        with Pool(args.workers) as pool:
            counts[split] = sum(pool.imap_unordered(_write_shard, jobs))
        print(f"{split}: {counts[split]} samples in {n_shards} shards", flush=True)

    meta = {
        "name": "imagenet_depth",
        "description": "Depth Anything V2 (ViT-L, input size 518) relative disparity of ImageNet-1k (ILSVRC 2012) "
        "images, stored as uint16 (truncated) at the original image resolution. Larger = closer.",
        "num_classes": 1000,
        "wnids": wnids,
        "class_names": [names[w].strip('"') for w in wnids],
        "splits": counts,
        "citation": json.loads((Path(__file__).parent / "citations.json").read_text())[
            "imagenet_depth"
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
