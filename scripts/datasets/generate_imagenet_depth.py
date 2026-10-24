# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Generate ImageNet-1k-Depth from your own ImageNet-1k (ILSVRC 2012) copy with Depth Anything V2.

Reproduces the maps used in the paper: DAv2 ViT-L (weights from huggingface.co/depth-anything), input size 518,
raw relative disparity truncated to uint16, saved at the original image resolution. Then run
build_imagenet_depth.py on the output to obtain the benchmark's Parquet format.

    git clone https://github.com/DepthAnything/Depth-Anything-V2
    python scripts/datasets/generate_imagenet_depth.py --dav2-repo Depth-Anything-V2 \
        --imagenet /data/imagenet --out /data/imagenet_depth_npz [--shard 0 --num-shards 8]

Expected ImageNet layout: <imagenet>/train/<wnid>/<wnid>_<idx>.JPEG and <imagenet>/val/<wnid>/ILSVRC2012_val_<n>.JPEG
(the official validation archive unpacks without the <wnid> folders; sort it into them first, e.g. with
https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dav2-repo",
        required=True,
        help="clone of github.com/DepthAnything/Depth-Anything-V2",
    )
    p.add_argument("--imagenet", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    sys.path.insert(0, args.dav2_repo)
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]
    )
    ckpt = hf_hub_download(
        "depth-anything/Depth-Anything-V2-Large", "depth_anything_v2_vitl.pth"
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.cuda().eval()

    out_dirs = {"train": "train_images", "val": "val_images"}
    for split, out_name in out_dirs.items():
        files = sorted(Path(args.imagenet, split).glob("*/*.JPEG"))
        if not files:
            raise SystemExit(
                f"no {args.imagenet}/{split}/<wnid>/*.JPEG files (see the expected layout above)"
            )
        files = files[args.shard :: args.num_shards]
        out = Path(args.out) / out_name
        out.mkdir(parents=True, exist_ok=True)
        for k, f in enumerate(files):
            wnid = f.parent.name
            target = out / f"{f.stem}_{wnid}.npz"  # label encoded in the file name
            if target.exists():
                continue
            disparity = model.infer_image(cv2.imread(str(f)), 518)
            tmp = target.with_name(
                target.name + ".tmp"
            )  # renamed when complete, so a resumed run skips no partial file
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, depth=np.uint16(disparity))
            tmp.rename(target)
            if k % 1000 == 0:
                print(f"{split}: {k}/{len(files)}", flush=True)


if __name__ == "__main__":
    main()
