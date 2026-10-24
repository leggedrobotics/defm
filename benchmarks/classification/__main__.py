# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""ImageNet-1k-Depth classification: weighted kNN and linear probe on frozen features.

python -m benchmarks.classification --encoder defm_vit_l14 --eval knn linear
"""

import argparse
import json
import time
from pathlib import Path

from benchmarks.classification.evaluate import run_knn, run_linear
from benchmarks.encoders import build_encoder, list_encoders


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--encoder", required=True, choices=list_encoders())
    p.add_argument(
        "--eval", nargs="+", default=["knn", "linear"], choices=["knn", "linear"]
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="local copy (<root>/imagenet_depth); default: the Hub dataset",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output", default="results/classification")
    args = p.parse_args()

    encoder = build_encoder(args.encoder).cuda()
    result, minutes = {"encoder": args.encoder}, {}
    if "knn" in args.eval:
        t0 = time.time()
        result["knn"] = run_knn(
            encoder,
            args.data_root,
            args.batch_size,
            args.workers,
            encoder.autocast_dtype,
        )
        print(
            f"[{args.encoder}] kNN k=20: top-1 {result['knn']['k20']['top1']:.2f}  top-5 {result['knn']['k20']['top5']:.2f}",
            flush=True,
        )
        minutes["knn"] = (time.time() - t0) / 60
    if "linear" in args.eval:
        t0 = time.time()
        result["linear"] = run_linear(
            encoder,
            args.data_root,
            batch_size=args.batch_size,
            workers=args.workers,
            dtype=encoder.autocast_dtype,
        )
        print(
            f"[{args.encoder}] linear: top-1 {result['linear']['top1']:.2f}  ({result['linear']['best_head']})",
            flush=True,
        )
        minutes["linear"] = (time.time() - t0) / 60

    out = Path(args.output) / f"imagenet_depth__{args.encoder}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # merge with an earlier run of the other protocol (kNN and linear can run as separate jobs)
    previous = json.loads(out.read_text()) if out.exists() else {}
    earlier_minutes = previous.get("minutes")
    result["minutes"] = {
        **(earlier_minutes if isinstance(earlier_minutes, dict) else {}),
        **minutes,
    }
    out.write_text(json.dumps({**previous, **result}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
