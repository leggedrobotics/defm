# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Linear-probe depth semantic segmentation.

python -m benchmarks.segmentation --encoder defm_vit_l14 --dataset tartanground
python -m benchmarks.segmentation --encoder dinov3_vitl16 --dataset scannet --data-root /data/defm-bench
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.encoders import build_encoder, list_encoders
from benchmarks.segmentation.datasets import BENCHMARKS, SegmentationDataset
from benchmarks.segmentation.linear_probe import cache_features, train_and_evaluate


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--encoder", required=True, choices=list_encoders())
    p.add_argument("--dataset", required=True, choices=list(BENCHMARKS))
    p.add_argument(
        "--data-root",
        default=None,
        help="directory with locally converted datasets (<root>/<dataset>)",
    )
    p.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=[1e-3, 5e-3],
        help="peak LRs; the best is reported",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="feature cache, kept and reused by later runs; needs up to ~80 GB (ScanNet, ViT-L) "
        "(default: a temporary directory under $TMPDIR, often /tmp, removed at the end)",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", default="results/segmentation")
    args = p.parse_args()

    bench = BENCHMARKS[args.dataset]
    encoder = build_encoder(args.encoder).cuda()
    cache_root = (
        Path(args.cache_dir or tempfile.mkdtemp(prefix="defm_seg_"))
        / args.dataset
        / args.encoder
    )
    print(
        f"[{args.encoder} | {args.dataset}] caching features in {cache_root}",
        flush=True,
    )

    try:
        t0 = time.time()
        caches, num_classes = {}, None
        for split in ("train", bench.test_split):
            ds = SegmentationDataset(
                args.dataset, split, encoder.input_norm, args.data_root
            )
            num_classes = ds.num_classes
            caches[split] = cache_features(
                encoder, ds, cache_root / split, workers=args.workers
            )
        # keep only the feature geometry; free the encoder before probe training
        geometry = SimpleNamespace(
            patch_size=encoder.patch_size,
            pad_multiple=encoder.pad_multiple,
            embed_dim=caches["train"][0].shape[-1],
        )
        del encoder
        torch.cuda.empty_cache()

        runs = {}
        for lr in args.lrs:
            print(f"  training linear probe, lr={lr:g}", flush=True)
            runs[f"{lr:g}"] = train_and_evaluate(
                caches["train"],
                caches[bench.test_split],
                geometry,
                num_classes,
                bench.size,
                lr=lr,
                epochs=bench.epochs,
                warmup_epochs=bench.warmup_epochs,
                workers=args.workers,
            )
            print(f"  lr={lr:g}: mIoU {runs[f'{lr:g}']['miou']:.2f}", flush=True)

        best_lr = max(runs, key=lambda k: runs[k]["miou"])
        result = {
            "encoder": args.encoder,
            "dataset": args.dataset,
            "best_lr": float(best_lr),
            "miou": runs[best_lr]["miou"],
            "runs": runs,
            "minutes": (time.time() - t0) / 60,
        }
        out = Path(args.output) / f"{args.dataset}__{args.encoder}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(
            f"[{args.encoder} | {args.dataset}] mIoU {result['miou']:.2f} (lr {best_lr}) -> {out}",
            flush=True,
        )

    finally:
        if args.cache_dir is None:
            shutil.rmtree(cache_root.parents[1], ignore_errors=True)


if __name__ == "__main__":
    main()
