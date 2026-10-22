# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Maintainer tool: build the root config.json of huggingface.co/leggedrobotics/defm from defm/configs/*.yaml.

The Hub counts a model download for every request to a root config.json, and create_defm_model() fetches it
before the weights, so this file is what makes DeFM downloads visible on the model page.

    python scripts/hf/build_model_config.py --out config.json
    hf upload leggedrobotics/defm config.json config.json
"""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "defm" / "configs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="config.json")
    args = p.parse_args()

    variants = {
        path.stem: OmegaConf.to_container(OmegaConf.load(path))
        for path in sorted(CONFIG_DIR.glob("*.yaml"))
    }
    config = {
        "model_type": "defm",
        "library_name": "defm",
        "variants": variants,
        "weights": {name: f"{name}.pth" for name in variants},
    }
    Path(args.out).write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(variants)} variants to {args.out}")


if __name__ == "__main__":
    main()
