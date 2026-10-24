# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import json
import torch
import pathlib
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from defm.models import vision_transformer as vits
from defm.models.resnet_bifpn import ResNetBiFPN
from defm.models.regnet_bifpn import RegNetBiFPN
from defm.models.efficientnet_bifpn import EfficientNetBiFPN

# Get the path to the current directory for config loading
BASE_PATH = pathlib.Path(__file__).parent.resolve()
HF_REPO_ID = "leggedrobotics/defm"


def get_defm_config(model_name: str):
    """Load model-specific YAML."""
    model_path = BASE_PATH / "configs" / f"{model_name}.yaml"
    return OmegaConf.load(model_path)


def get_hub_config(model_name: str):
    """Architecture config of a variant from the Hub's root config.json.

    config.json is the file the Hugging Face Hub counts as a model download, so fetching it here is what makes
    downloads of the DeFM weights show up on the model page. Falls back to the packaged YAML when offline.
    """
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename="config.json")
        with open(path) as f:
            return OmegaConf.create(json.load(f)["variants"][model_name])
    except Exception as e:  # offline, or a variant not listed in the Hub config
        print(
            f"Could not read {model_name} from the {HF_REPO_ID} config.json ({e}); using the packaged config."
        )
        return get_defm_config(model_name)


def create_defm_model(model_name, pretrained=False, pretrained_path=None):
    """
    Builds a DeFM model (any variant in defm/configs, e.g. defm_vit_l14, defm_resnet50) and loads its weights:
    from pretrained_path if given, otherwise from the Hugging Face Hub. The CNN backbones are built without
    torchvision's ImageNet weights, so pretrained=False gives a randomly initialized model.
    """
    if not (BASE_PATH / "configs" / f"{model_name}.yaml").exists():
        available = sorted(p.stem for p in (BASE_PATH / "configs").glob("*.yaml"))
        raise ValueError(f"Unknown DeFM model {model_name!r}; available: {available}")
    downloading = pretrained and not pretrained_path
    cfg = get_hub_config(model_name) if downloading else get_defm_config(model_name)

    if "vit" in cfg.arch:
        vit_kwargs = dict(
            img_size=cfg.global_crops_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,  # Usually 3 for your stacked input
            init_values=cfg.layerscale,
            ffn_layer=cfg.ffn_layer,
            block_chunks=cfg.block_chunks,
            qkv_bias=cfg.qkv_bias,
            proj_bias=cfg.proj_bias,
            ffn_bias=cfg.ffn_bias,
            num_register_tokens=cfg.num_register_tokens,
            interpolate_offset=cfg.interpolate_offset,
            interpolate_antialias=cfg.interpolate_antialias,
        )

        # Build model using the architecture specified in YAML (e.g., vit_small, vit_large)
        model = vits.__dict__[cfg.arch](**vit_kwargs)

    elif "resnet" in cfg.arch:
        model = ResNetBiFPN(
            backbone_name=cfg.arch, out_channels=cfg.out_channels, pretrained=False
        )
    elif "regnet" in cfg.arch:
        model = RegNetBiFPN(
            backbone_name=cfg.arch, out_channels=cfg.out_channels, pretrained=False
        )
    elif "efficientnet" in cfg.arch:
        model = EfficientNetBiFPN(
            backbone_name=cfg.arch, out_channels=cfg.out_channels, pretrained=False
        )
    else:
        raise ValueError(f"Unsupported architecture: {cfg.arch}")

    if pretrained:
        if pretrained_path:
            # Use local path if provided
            load_defm_weights(model, pretrained_path)
        else:
            # Automatically download from Hugging Face
            print(f"Downloading {model_name} from Hugging Face...")
            checkpoint_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"{model_name}.pth",  # Matches your HF filenames
            )
            load_defm_weights(model, checkpoint_path)

    return model


def load_defm_weights(model, ckpt_path):
    """Standalone weight loader"""
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "model" in state_dict:
        state_dict = state_dict["model"]

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded DeFM weights from {ckpt_path} with result: {msg}")
