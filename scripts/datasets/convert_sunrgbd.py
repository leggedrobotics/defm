# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Build the SUN RGB-D segmentation benchmark (37 classes, 5285 train / 5050 test).

Either source gives byte-identical output:

  official   SUN RGB-D from https://rgbd.cs.princeton.edu (SUNRGBD.zip, 6.9 GB, and SUNRGBDtoolbox.zip, 0.6 GB);
             needs h5py and scipy (pip install h5py scipy)

    python scripts/datasets/convert_sunrgbd.py --source official --sunrgbd <dir>/SUNRGBD \
        --toolbox <dir>/SUNRGBDtoolbox --out <data_root>/sunrgbd

  dformer    the preprocessed copy linked from https://github.com/VCIP-RGBD/DFormer (SUNRGBD/{RGB,Depth,labels})

    python scripts/datasets/convert_sunrgbd.py --source dformer --sunrgbd <dir>/SUNRGBD --out <data_root>/sunrgbd

Split: the toolbox's traintestSUNRGBD/allsplit.mat (sample train_i = alltrain[i], test_i = alltest[i]).
Depth: the raw depth/*.png (16 bit, 3-bit circularly shifted mm). Labels: Metadata/SUNRGBD2Dseg.mat seglabel
(1..37, 0 = unlabeled), stored as 0..36 with 255 = ignore.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import (
    IGNORE_INDEX,
    NYU40,
    encode_depth,
    encode_label,
    write_meta,
    write_split,
)

SPLITS = {"train": 5285, "test": 5050}
CITATION = json.loads((Path(__file__).parent / "citations.json").read_text())["sunrgbd"]


def depth_m(raw: np.ndarray) -> np.ndarray:
    """SUN RGB-D depth PNGs store mm rotated left by 3 bits."""
    assert raw.dtype == np.uint16
    return np.bitwise_or(raw >> 3, raw << 13).astype(np.float64) / 1000.0


def sample(sid: str, raw_depth: np.ndarray, label37: np.ndarray):
    label = label37.astype(np.uint8) - np.uint8(
        1
    )  # 1..37 -> 0..36, 0 (unlabeled) -> 255
    return sid, encode_depth(depth_m(raw_depth), 1000), encode_label(label)


def dformer_samples(root: Path, split: str):
    for i in range(SPLITS[split]):
        sid = f"{split}_{i}"
        yield sample(
            sid,
            cv2.imread(str(root / "Depth" / f"{sid}.png"), cv2.IMREAD_UNCHANGED),
            cv2.imread(str(root / "labels" / f"{sid}.png"), cv2.IMREAD_UNCHANGED),
        )


class Official:
    def __init__(self, root: Path, toolbox: Path):
        import h5py
        import scipy.io as sio

        self.root = root
        split = sio.loadmat(
            toolbox / "traintestSUNRGBD" / "allsplit.mat", squeeze_me=True
        )
        scene = lambda path: path.split("/SUNRGBD/", 1)[1].rstrip("/")  # noqa: E731
        self.scenes = {
            "train": [scene(p) for p in split["alltrain"]],
            "test": [scene(p) for p in split["alltest"]],
        }
        meta = sio.loadmat(
            toolbox / "Metadata" / "SUNRGBDMeta.mat",
            squeeze_me=True,
            struct_as_record=False,
        )["SUNRGBDMeta"]
        self.index = {
            m.sequenceName.split("SUNRGBD/", 1)[-1].rstrip("/"): i
            for i, m in enumerate(meta)
        }
        self.seg = h5py.File(toolbox / "Metadata" / "SUNRGBD2Dseg.mat", "r")

    def samples(self, split: str):
        labels = self.seg["SUNRGBD2Dseg"]["seglabel"]
        for i, scene in enumerate(self.scenes[split]):
            (depth_png,) = (self.root / scene / "depth").glob("*.png")
            label = np.array(
                self.seg[labels[self.index[scene], 0]]
            ).T  # MATLAB stores it column-major
            yield sample(
                f"{split}_{i}", cv2.imread(str(depth_png), cv2.IMREAD_UNCHANGED), label
            )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", required=True, choices=["official", "dformer"])
    p.add_argument(
        "--sunrgbd", required=True, type=Path, help="the extracted SUNRGBD/ directory"
    )
    p.add_argument(
        "--toolbox",
        type=Path,
        help="the extracted SUNRGBDtoolbox/ directory (--source official)",
    )
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    if args.source == "official":
        if args.toolbox is None:
            p.error("--source official needs --toolbox")
        source = Official(args.sunrgbd, args.toolbox).samples
    else:
        source = lambda split: dformer_samples(args.sunrgbd, split)  # noqa: E731

    counts = {split: write_split(args.out, split, source(split)) for split in SPLITS}
    if counts != SPLITS:
        raise RuntimeError(f"sample counts {counts} != {SPLITS}")
    write_meta(
        args.out,
        name="sunrgbd",
        num_classes=37,
        class_names=NYU40[:37],
        depth_scale=1000,
        ignore_index=IGNORE_INDEX,
        splits=counts,
        license="see https://rgbd.cs.princeton.edu",
        citation=CITATION,
    )
    print(f"sunrgbd: {counts} -> {args.out}")


if __name__ == "__main__":
    main()
