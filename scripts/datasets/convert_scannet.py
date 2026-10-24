# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Build the ScanNet segmentation benchmark from the official scannet_frames_25k release.

ScanNet's terms of use do not allow redistribution, so download it yourself (after accepting the terms at
http://www.scan-net.org, you receive download-scannet.py):

    python download-scannet.py -o <dir> --preprocessed_frames      # scannet_frames_25k.zip, 5.6 GB
    unzip <dir>/tasks/scannet_frames_25k.zip -d <dir>
    python scripts/datasets/convert_scannet.py --frames <dir>/scannet_frames_25k --out <data_root>/scannet

Splits: train = the official scannetv2_train scenes (1201 scenes, 19466 frames), test = the official scannetv2_val
scenes (312 scenes, 5436 frames); both lists are in splits/ (from github.com/ScanNet/ScanNet, Tasks/Benchmark).
Depth: depth/*.png, uint16 in mm (640x480). Labels: label/*.png, NYU40 ids 1..40 (0 = unannotated) at the color
resolution (1296x968), stored as 0..39 with 255 = ignore.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    IGNORE_INDEX,
    NYU40,
    encode_depth,
    encode_label,
    imread,
    write_meta,
    write_split,
)

SPLITS = {"train": "scannetv2_train.txt", "test": "scannetv2_val.txt"}
CITATION = json.loads((Path(__file__).parent / "citations.json").read_text())["scannet"]


def samples(frames: Path, split: str):
    scenes = (Path(__file__).parent / "splits" / SPLITS[split]).read_text().split()
    for scene in sorted(scenes):
        for color in sorted((frames / scene / "color").glob("*.jpg")):
            frame = color.stem
            depth = imread(frames / scene / "depth" / f"{frame}.png")
            label = imread(frames / scene / "label" / f"{frame}.png")
            assert depth.dtype == np.uint16 and label.dtype == np.uint8, (scene, frame)
            label = label - np.uint8(1)  # NYU40 1..40 -> 0..39, 0 (unannotated) -> 255
            yield (
                f"{scene}_img_{frame}",
                encode_depth(depth.astype(np.float64) / 1000.0, 1000),
                encode_label(label),
            )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--frames",
        required=True,
        type=Path,
        help="the extracted scannet_frames_25k directory",
    )
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    counts = {
        split: write_split(args.out, split, samples(args.frames, split))
        for split in SPLITS
    }
    expected = {"train": 19466, "test": 5436}
    if counts != expected:
        raise RuntimeError(
            f"frame counts {counts} != {expected}: incomplete scannet_frames_25k?"
        )
    write_meta(
        args.out,
        name="scannet",
        num_classes=len(NYU40),
        class_names=NYU40,
        depth_scale=1000,
        ignore_index=IGNORE_INDEX,
        splits=counts,
        license="ScanNet Terms of Use",
        citation=CITATION,
    )
    print(f"scannet: {counts} -> {args.out}")


if __name__ == "__main__":
    main()
