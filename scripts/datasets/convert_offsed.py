# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Build the OFFSED segmentation benchmark from the official release.

OFFSED is published without a redistribution license, so download it yourself from
https://www.dfki.uni-kl.de/~neigel/offsed.html (disparity.zip, 6 GB, and devkit_offsed.zip, 10 MB):

    unzip disparity.zip -d <dir> && unzip devkit_offsed.zip -d <dir>
    python scripts/datasets/convert_offsed.py --disparity <dir>/disparity --devkit <dir>/devkit_offsed \
        --out <data_root>/offsed

Depth: the ZED disparity maps (float32, 2208x1242) are resized to 640x360 (nearest) and converted to depth with
f * B = 126 (1050 px * 0.12 m), in mm. Labels: the colour masks of devkit_offsed/SegmentationClass are mapped to the
labelmap.txt indices, resized the same way and merged into 5 classes. The 203 annotated frames are split 162 / 41
(splits/offsed_{train,test}.txt; OFFSED itself does not define a split).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import (
    IGNORE_INDEX,
    encode_depth,
    encode_label,
    imread,
    write_meta,
    write_split,
)

SIZE = (640, 360)  # (width, height)
FOCAL_TIMES_BASELINE = np.float32(
    126.0
)  # px * m, at the 2208x1242 disparity resolution
CLASSES = ["ground", "structure", "vegetation", "vehicle", "human"]
# labelmap.txt index -> class (255 = ignore): background (0) and sky (16) are ignored, held object (11) is human
FINE_TO_COARSE = {
    0: 255,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 2,
    6: 0,
    7: 0,
    8: 3,
    9: 2,
    10: 1,
    11: 4,
    12: 0,
    13: 0,
    14: 1,
    15: 4,
    16: 255,
    17: 2,
    18: 3,
    19: 1,
}
CITATION = json.loads((Path(__file__).parent / "citations.json").read_text())["offsed"]


def read_labelmap(devkit: Path) -> np.ndarray:
    """Lookup from packed RGB (r << 16 | g << 8 | b) to the 5-class label."""
    rows = [
        line.split(":")
        for line in (devkit / "labelmap.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    lut = {}
    for index, row in enumerate(rows):
        r, g, b = map(int, row[1].split(","))
        lut[(r << 16) | (g << 8) | b] = FINE_TO_COARSE[index]
    return lut


def depth_mm(disparity: np.ndarray) -> np.ndarray:
    disparity = cv2.resize(
        disparity.astype(np.float32), SIZE, interpolation=cv2.INTER_NEAREST
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (FOCAL_TIMES_BASELINE / disparity) * np.float32(1000.0)
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
    return np.clip(depth, 0, 65535).astype(np.uint16)


def label(mask_bgr: np.ndarray, lut: dict) -> np.ndarray:
    rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB).astype(np.int64)
    packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    colours, inverse = np.unique(packed, return_inverse=True)
    unknown = [hex(c) for c in colours if c not in lut]
    if unknown:
        raise ValueError(f"colours not in labelmap.txt: {unknown}")
    coarse = np.array([lut[c] for c in colours], np.uint8)[inverse].reshape(
        packed.shape
    )
    return cv2.resize(coarse, SIZE, interpolation=cv2.INTER_NEAREST)


def samples(disparity_dir: Path, devkit: Path, split: str):
    lut = read_labelmap(devkit)
    for frame in (
        (Path(__file__).parent / "splits" / f"offsed_{split}.txt").read_text().split()
    ):
        depth = (
            depth_mm(np.load(disparity_dir / f"{frame}.npy")).astype(np.float64)
            / 1000.0
        )
        mask = imread(devkit / "SegmentationClass" / f"{frame}.png", cv2.IMREAD_COLOR)
        yield frame, encode_depth(depth, 1000), encode_label(label(mask, lut))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--disparity",
        required=True,
        type=Path,
        help="the extracted disparity/ directory",
    )
    p.add_argument(
        "--devkit",
        required=True,
        type=Path,
        help="the extracted devkit_offsed/ directory",
    )
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    counts = {
        split: write_split(args.out, split, samples(args.disparity, args.devkit, split))
        for split in ("train", "test")
    }
    write_meta(
        args.out,
        name="offsed",
        num_classes=len(CLASSES),
        class_names=CLASSES,
        depth_scale=1000,
        ignore_index=IGNORE_INDEX,
        splits=counts,
        license="see https://www.dfki.uni-kl.de/~neigel/offsed.html",
        citation=CITATION,
    )
    print(f"offsed: {counts} -> {args.out}")


if __name__ == "__main__":
    main()
