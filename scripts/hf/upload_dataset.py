# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Maintainer tool: upload a converted benchmark dataset to the Hub with a dataset card.

    python scripts/hf/upload_dataset.py --dataset tartanground --root <data_root>/tartanground

The repo (leggedrobotics/defm-bench-<name>) is created private; check it with the benchmark
(`python -m benchmarks.segmentation ...` without --data-root), then make it public, or gated for ImageNet-Depth,
on the Hub. Uploads resume if interrupted.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

REPOS = {
    "tartanground": "leggedrobotics/defm-bench-tartanground",
    "graspnet": "leggedrobotics/defm-bench-graspnet",
    "imagenet_depth": "leggedrobotics/defm-bench-imagenet-depth",
}

SOURCES = {
    "tartanground": (
        "TartanGround",
        "https://tartanair.org/tartanground",
        "Depth semantic segmentation in simulated outdoor environments: 800 train / 400 test frames, "
        "6 classes. Depth is clipped at 100 m (sky).",
    ),
    "graspnet": (
        "GraspNet-1Billion",
        "https://graspnet.net",
        "Depth semantic segmentation of tabletop objects: 2000 train / 600 test frames, 40 object classes "
        "(background is ignored).",
    ),
    "imagenet_depth": (
        "ImageNet-1k (ILSVRC 2012)",
        "https://www.image-net.org",
        "Relative disparity predicted by Depth Anything V2 (ViT-L, input size 518) for every ImageNet-1k "
        "train and validation image, stored losslessly as uint16 at the original image resolution "
        "(larger = closer). Used for kNN and linear-probe classification.",
    ),
}

LICENSES = {
    "tartanground": "license: cc-by-4.0",
    "graspnet": "license: cc-by-nc-sa-4.0",
    "imagenet_depth": "license: other\nlicense_name: imagenet-terms-of-access\n"
    "license_link: https://www.image-net.org/download.php",
}

DEFM_CITATION = """@misc{patel2026defm,
  title={DeFM: Learning Foundation Representations from Depth for Robotics},
  author={Manthan Patel and Jonas Frey and Mayank Mittal and Fan Yang and Alexander Hansson and Amir Bar and Cesar Cadena and Marco Hutter},
  year={2026}, eprint={2601.18923}, archivePrefix={arXiv}, primaryClass={cs.RO}, url={https://arxiv.org/abs/2601.18923}
}"""

GATING = """extra_gated_prompt: >-
  The depth maps are derived from ImageNet images. By requesting access you agree to the ImageNet terms of access
  (https://www.image-net.org/download.php): use only for non-commercial research and educational purposes, no
  redistribution. The disparities were predicted with Depth Anything V2 Large, which is licensed CC-BY-NC-4.0.
extra_gated_fields:
  Name: text
  Affiliation: text
  I agree to use this dataset for non-commercial research only and to the ImageNet terms of access: checkbox
"""


def card(name: str, meta: dict) -> str:
    source, url, summary = SOURCES[name]
    splits = list(meta["splits"])
    data_files = "\n".join(f"  - split: {s}\n    path: data/{s}-*" for s in splits)
    task = "image-classification" if name == "imagenet_depth" else "image-segmentation"
    gating = GATING if name == "imagenet_depth" else ""
    if name == "imagenet_depth":
        columns = (
            "| `id` | string | ImageNet file stem |\n"
            "| `label` | int64 | class index 0..999 (sorted WordNet ids, see `meta.json`) |\n"
            "| `depth` | image | 16-bit PNG, uint16 relative disparity |"
        )
    else:
        columns = (
            "| `id` | string | frame id |\n"
            f"| `depth` | image | 16-bit PNG, depth in meters = value / {meta['depth_scale']}, 0 = invalid |\n"
            f"| `label` | image | 8-bit PNG, class ids 0..{meta['num_classes'] - 1}, 255 = ignore |"
        )
    citation = "\n\n".join([DEFM_CITATION, *meta["citation"].split(" + ")])
    contents = (
        "Class names are"
        if name == "imagenet_depth"
        else "Class names and the depth scale are"
    )
    counts = ", ".join(f"{n:,} {s}" for s, n in meta["splits"].items())
    return f"""---
{LICENSES[name]}
pretty_name: DeFM benchmark - {source}
task_categories:
- {task}
tags:
- depth
- robotics
- defm
configs:
- config_name: default
  data_files:
{data_files}
{gating}---

# DeFM benchmark: {source}

{summary}

This is the version used in the benchmarks of [DeFM: Learning Foundation Representations from Depth for Robotics]\
(https://github.com/leggedrobotics/defm) ({counts}). The original data is from [{source}]({url}); please cite it
(below) together with DeFM.

## Use

```bash
git clone https://github.com/leggedrobotics/defm && cd defm && pip install -e ".[benchmarks]"
python -m benchmarks.{"classification" if name == "imagenet_depth" else "segmentation"} --encoder defm_vit_s14\
{"" if name == "imagenet_depth" else f" --dataset {name}"}
```

The benchmark downloads this dataset automatically. {contents} in `meta.json`.

## Format

`data/<split>-XXXXX-of-XXXXX.parquet`:

| column | type | content |
|---|---|---|
{columns}

## Citation

```bibtex
{citation}
```
"""


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True, choices=list(REPOS))
    p.add_argument(
        "--root",
        required=True,
        type=Path,
        help="converted dataset directory (meta.json + data/)",
    )
    p.add_argument(
        "--card-only",
        action="store_true",
        help="only write README.md next to the data (for review)",
    )
    args = p.parse_args()

    meta = json.loads((args.root / "meta.json").read_text())
    (args.root / "README.md").write_text(card(args.dataset, meta))
    if args.card_only:
        print(f"wrote {args.root / 'README.md'}")
        return

    api, repo = HfApi(), REPOS[args.dataset]
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=args.root,
        allow_patterns=["README.md", "meta.json", "data/*.parquet"],
    )
    print(f"uploaded {args.root} -> https://huggingface.co/datasets/{repo} (private)")


if __name__ == "__main__":
    main()
