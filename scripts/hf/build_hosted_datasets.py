# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Maintainer tool: build the Hub copies of TartanGround and GraspNet-1Billion from the RGB-D archives used for
the paper (users download the result from the Hub, see benchmarks/README.md).

    python scripts/hf/build_hosted_datasets.py --archive-dir <dir with TartanGround_rgbd.zip, graspnet_rgbd.zip> \
        --out <data_root>

The other segmentation datasets cannot be redistributed and are built from their official releases with
scripts/datasets/convert_{scannet,sunrgbd,offsed}.py.
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from common import IGNORE_INDEX, encode_depth, encode_label, write_meta, write_split  # noqa: E402

CITATIONS = json.loads(
    (Path(__file__).resolve().parents[1] / "datasets" / "citations.json").read_text()
)


class Archive:
    def __init__(self, path: Path, prefix: str):
        self.zip, self.prefix = zipfile.ZipFile(path), prefix

    def read(self, rel: str) -> bytes:
        return self.zip.read(f"{self.prefix}/{rel}")

    def lines(self, rel: str):
        return [
            line.split()
            for line in self.read(rel).decode().splitlines()
            if line.strip()
        ]

    def image(self, rel: str) -> np.ndarray:
        return cv2.imdecode(
            np.frombuffer(self.read(rel), np.uint8), cv2.IMREAD_UNCHANGED
        )


def graspnet(a: Archive, split: str):
    mapping = {int(k): int(v) for k, v in json.loads(a.read("id_mapping.json")).items()}
    lut = np.full(256, IGNORE_INDEX, np.uint8)
    for k, v in mapping.items():
        lut[k] = v
    for _, depth_rel, label_rel in a.lines(
        {"train": "train.txt", "test": "val.txt"}[split]
    ):
        sid = depth_rel.replace("/depth/", "_").removesuffix(".png")  # scene_0100_0005
        depth = a.image(depth_rel).astype(np.float64) / 1000.0
        label = lut[
            a.image(label_rel).astype(np.uint8) - np.uint8(1)
        ]  # object id 1.. -> 0.., background -> 255
        yield sid, encode_depth(depth, 1000), encode_label(label)


def graspnet_classes(a: Archive):
    mapping = json.loads(a.read("id_mapping.json"))
    return [
        f"object_{int(k):03d}" for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])
    ]


def tartanground(a: Archive, split: str):
    for _, depth_rel, seg_rel in a.lines(f"{split}.txt"):
        sid = Path(seg_rel).stem.removesuffix("_mapped")
        depth = (
            a.image(depth_rel).astype(np.uint8).view("<f4").squeeze().astype(np.float64)
        )
        depth = np.clip(
            depth, 0, 100
        )  # sky is stored as ~65504 m; the paper clips at 100 m
        yield (
            sid,
            encode_depth(depth, 500),
            encode_label(a.image(seg_rel).astype(np.uint8)),
        )


TARTANGROUND_CLASSES = [
    "ground/floor",
    "buildings/structures",
    "vehicles/machinery",
    "misc (infrastructure/objects)",
    "vegetation",
    "sky/background",
]

DATASETS = {
    # name: (archive, prefix in archive, converter, depth_scale, class names, license)
    "tartanground": (
        "TartanGround_rgbd.zip",
        "TartanGround_rgbd",
        tartanground,
        500,
        lambda a: TARTANGROUND_CLASSES,
        "cc-by-4.0",
    ),
    "graspnet": (
        "graspnet_rgbd.zip",
        "graspnet_rgbd",
        graspnet,
        1000,
        graspnet_classes,
        "cc-by-nc-sa-4.0",
    ),
}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--archive-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS)
    )
    args = p.parse_args()

    for name in args.datasets:
        archive, prefix, convert, scale, classes, license_ = DATASETS[name]
        a = Archive(args.archive_dir / archive, prefix)
        out = args.out / name
        counts = {
            split: write_split(out, split, convert(a, split))
            for split in ("train", "test")
        }
        names = classes(a)
        write_meta(
            out,
            name=name,
            num_classes=len(names),
            class_names=names,
            depth_scale=scale,
            ignore_index=IGNORE_INDEX,
            splits=counts,
            license=license_,
            citation=CITATIONS[name],
        )
        print(f"{name}: {counts} -> {out}", flush=True)


if __name__ == "__main__":
    main()
