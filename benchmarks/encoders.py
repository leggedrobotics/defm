# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Frozen encoders used in the DeFM benchmarks, all loaded from the Hugging Face Hub.

Every encoder exposes the same interface:
    enc.input_norm    -> "defm" | "imagenet" | "unit"  (which depth encoding / normalization it expects)
    enc.patch_size    -> stride of the dense feature map
    enc.pad_multiple  -> inputs are center-padded to a multiple of this
    enc.embed_dim     -> channel dimension C of the dense features
    enc(x)            -> (global [B, C_g], dense [B, h*w, C]) for an input already padded to pad_multiple
                         (C_g can be larger than C: DeFM CNNs 512 vs 128, C-RADIOv3 3072 vs 1024)
    enc.last_blocks(x, n) -> [(global, dense)] for the last n blocks (linear-probe classification)

DINOv2 and Depth Anything V2 weights are loaded into DeFM's own ViT (the original DINOv2 implementation),
which keeps positional-embedding interpolation identical to the paper. transformers' Dinov2Model uses a
different interpolation and gives different features away from 518x518.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from defm.model_factory import create_defm_model
from defm.models import vision_transformer as vits

# Pinned Hub revisions so results do not change when upstream repos are updated.
HF_REVISIONS = {
    "facebook/dinov2-large": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
    "facebook/dinov2-small": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
    "facebook/dinov3-vits16-pretrain-lvd1689m": "114c1379950215c8b35dfcd4e90a5c251dde0d32",
    "nvidia/C-RADIOv3-L": "9d0413465e8a91e67bbf2c1ad342815478d1b906",
    "depth-anything/Depth-Anything-V2-Large": "cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9",
    "depth-anything/Depth-Anything-V2-Small": "03876f8651c73a60fe4c2c48294e09fcb6838fcf",
    "theaiinstitute/theia-small-patch16-224-cdiv": "b07fb29f3fa084873cb8593b3bc29e0d633080d5",
}


@dataclass(frozen=True)
class EncoderSpec:
    family: str  # defm_vit | defm_cnn | dinov2 | dinov3 | dav2 | radio | theia
    hf_repo: str
    input_norm: str  # defm | imagenet | unit
    arch: Optional[str] = None


ENCODERS = {
    # DeFM (ours)
    "defm_vit_l14": EncoderSpec("defm_vit", "leggedrobotics/defm", "defm"),
    "defm_vit_s14": EncoderSpec("defm_vit", "leggedrobotics/defm", "defm"),
    **{
        name: EncoderSpec("defm_cnn", "leggedrobotics/defm", "defm")
        for name in [
            "defm_resnet18",
            "defm_resnet34",
            "defm_resnet50",
            "defm_regnet_y_400mf",
            "defm_regnet_y_800mf",
            "defm_regnet_y_1_6gf",
            "defm_efficientnet_b0",
            "defm_efficientnet_b2",
            "defm_efficientnet_b4",
            "defm_efficientnet_b6",
        ]
    },
    # RGB foundation models applied to depth
    "dinov2_vitl14": EncoderSpec(
        "dinov2", "facebook/dinov2-large", "imagenet", "vit_large"
    ),
    "dinov2_vits14": EncoderSpec(
        "dinov2", "facebook/dinov2-small", "imagenet", "vit_small"
    ),
    "dinov3_vitl16": EncoderSpec(
        "dinov3", "facebook/dinov3-vitl16-pretrain-lvd1689m", "imagenet"
    ),
    "dinov3_vits16": EncoderSpec(
        "dinov3", "facebook/dinov3-vits16-pretrain-lvd1689m", "imagenet"
    ),
    "dav2_vitl14": EncoderSpec(
        "dav2", "depth-anything/Depth-Anything-V2-Large", "imagenet", "vit_large"
    ),
    "dav2_vits14": EncoderSpec(
        "dav2", "depth-anything/Depth-Anything-V2-Small", "imagenet", "vit_small"
    ),
    "radio_v3_l": EncoderSpec("radio", "nvidia/C-RADIOv3-L", "unit"),
    "theia_s16": EncoderSpec(
        "theia", "theaiinstitute/theia-small-patch16-224-cdiv", "unit"
    ),
}


def list_encoders() -> List[str]:
    return list(ENCODERS)


def build_encoder(name: str) -> nn.Module:
    """Build a frozen encoder in eval mode by registry name (see ENCODERS)."""
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder '{name}'. Available: {', '.join(ENCODERS)}")
    spec = ENCODERS[name]
    builders = {
        "defm_vit": lambda: ViTEncoder(
            create_defm_model(name, pretrained=True), patch_size=14
        ),
        "defm_cnn": lambda: CNNEncoder(create_defm_model(name, pretrained=True)),
        "dinov2": lambda: ViTEncoder(
            _dinov2_from_hf(spec.hf_repo, spec.arch), patch_size=14
        ),
        "dav2": lambda: ViTEncoder(
            _dav2_from_hf(spec.hf_repo, spec.arch), patch_size=14
        ),
        "dinov3": lambda: DINOv3Encoder(spec.hf_repo),
        "radio": lambda: RadioEncoder(spec.hf_repo),
        "theia": lambda: TheiaEncoder(spec.hf_repo),
    }
    encoder = builders[spec.family]()
    encoder.name, encoder.input_norm = name, spec.input_norm
    encoder.is_cnn = spec.family == "defm_cnn"
    # DINOv3 global features degrade in float16 (kNN accuracy collapses), so classification runs it in bfloat16.
    # Segmentation runs every encoder in float16, as in the paper.
    encoder.autocast_dtype = (
        torch.bfloat16 if spec.family == "dinov3" else torch.float16
    )
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.eval()


# --------------------------------------------------------------------------------------------------------------
# DINOv2-architecture ViTs (DeFM ViTs, DINOv2, Depth Anything V2 encoders)
# --------------------------------------------------------------------------------------------------------------

# DINOv2 hub configuration (also used by the Depth Anything V2 encoder)
_DINOV2_KWARGS = dict(
    img_size=518,
    patch_size=14,
    init_values=1.0,
    ffn_layer="mlp",
    block_chunks=0,
    num_register_tokens=0,
    interpolate_antialias=False,
    interpolate_offset=0.1,
)


def _dinov2_from_hf(repo: str, arch: str) -> nn.Module:
    """facebook/dinov2-* safetensors (transformers layout) -> original DINOv2 ViT."""
    from safetensors.torch import load_file

    hf_sd = load_file(
        hf_hub_download(repo, "model.safetensors", revision=HF_REVISIONS[repo])
    )
    model = vits.__dict__[arch](**_DINOV2_KWARGS)
    model.load_state_dict(_convert_hf_dinov2(hf_sd), strict=True)
    return model


def _convert_hf_dinov2(hf_sd: dict) -> dict:
    sd = {
        "cls_token": hf_sd["embeddings.cls_token"],
        "mask_token": hf_sd["embeddings.mask_token"],
        "pos_embed": hf_sd["embeddings.position_embeddings"],
        "patch_embed.proj.weight": hf_sd[
            "embeddings.patch_embeddings.projection.weight"
        ],
        "patch_embed.proj.bias": hf_sd["embeddings.patch_embeddings.projection.bias"],
        "norm.weight": hf_sd["layernorm.weight"],
        "norm.bias": hf_sd["layernorm.bias"],
    }
    depth = 1 + max(
        int(k.split(".")[2]) for k in hf_sd if k.startswith("encoder.layer.")
    )
    for i in range(depth):
        h, o = f"encoder.layer.{i}.", f"blocks.{i}."
        att = h + "attention.attention."
        for kind in ("weight", "bias"):
            sd[o + f"attn.qkv.{kind}"] = torch.cat(
                [hf_sd[att + f"{n}.{kind}"] for n in ("query", "key", "value")]
            )
            sd[o + f"attn.proj.{kind}"] = hf_sd[h + f"attention.output.dense.{kind}"]
            for n in ("norm1", "norm2", "mlp.fc1", "mlp.fc2"):
                sd[o + f"{n}.{kind}"] = hf_sd[h + f"{n}.{kind}"]
        sd[o + "ls1.gamma"] = hf_sd[h + "layer_scale1.lambda1"]
        sd[o + "ls2.gamma"] = hf_sd[h + "layer_scale2.lambda1"]
    return sd


def _dav2_from_hf(repo: str, arch: str) -> nn.Module:
    """Depth Anything V2 checkpoint -> its DINOv2 encoder ('pretrained.*' keys)."""
    filename = {
        "vit_large": "depth_anything_v2_vitl.pth",
        "vit_small": "depth_anything_v2_vits.pth",
    }[arch]
    sd = torch.load(
        hf_hub_download(repo, filename, revision=HF_REVISIONS[repo]), map_location="cpu"
    )
    model = vits.__dict__[arch](**_DINOV2_KWARGS)
    model.load_state_dict(
        {
            k[len("pretrained.") :]: v
            for k, v in sd.items()
            if k.startswith("pretrained.")
        },
        strict=True,
    )
    return model


class ViTEncoder(nn.Module):
    def __init__(self, model: nn.Module, patch_size: int):
        super().__init__()
        self.model = model
        self.patch_size = self.pad_multiple = patch_size
        self.embed_dim = model.embed_dim

    def forward(self, x):
        patch, cls = self.model.get_intermediate_layers(
            x, n=1, return_class_token=True
        )[0]
        return cls, patch

    def last_blocks(self, x, n: int):
        return [
            (cls, patch)
            for patch, cls in self.model.get_intermediate_layers(
                x, n=n, return_class_token=True
            )
        ]


# --------------------------------------------------------------------------------------------------------------
# DeFM CNNs (BiFPN): global = LayerNorm(avgpool(C5)), dense = BiFPN P4 (stride 16)
# --------------------------------------------------------------------------------------------------------------


class CNNEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.patch_size, self.pad_multiple = 16, 32
        self.embed_dim = model.norm_patch.normalized_shape[0]  # dense (P4) channels

    def forward(self, x):
        out = self.model(x)
        return out["global_backbone"], out["dense_bifpn"]["P4"].flatten(2).transpose(
            1, 2
        )

    def last_blocks(self, x, n: int):
        if n != 1:
            raise ValueError("CNN encoders only expose their final features (n=1)")
        return [self(x)]


# --------------------------------------------------------------------------------------------------------------
# DINOv3 (transformers; gated on the Hub: accept the license on the model page and `hf auth login`)
# --------------------------------------------------------------------------------------------------------------


class DINOv3Encoder(nn.Module):
    def __init__(self, repo: str):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(repo, revision=HF_REVISIONS[repo])
        cfg = self.model.config
        self.patch_size = self.pad_multiple = cfg.patch_size
        self.embed_dim = cfg.hidden_size
        self.num_prefix = 1 + cfg.num_register_tokens

    def forward(self, x):
        h = self.model(pixel_values=x).last_hidden_state  # final norm applied
        return h[:, 0], h[:, self.num_prefix :]

    def last_blocks(self, x, n: int):
        hidden = self.model(pixel_values=x, output_hidden_states=True).hidden_states[
            -n:
        ]
        return [
            (h[:, 0], h[:, self.num_prefix :]) for h in map(self.model.norm, hidden)
        ]


# --------------------------------------------------------------------------------------------------------------
# C-RADIOv3 and Theia (remote code pinned to a revision; both take [0, 1] inputs and normalize internally)
# --------------------------------------------------------------------------------------------------------------


RADIO_CODE_REF = "NVlabs/RADIO:4ffaf15d0ff48197c194e306772bdb5fb92db230"


class RadioEncoder(nn.Module):
    """Official RADIO torch.hub loader; it downloads the weights from huggingface.co/nvidia/C-RADIOv3-L.

    (The transformers remote-code path gives identical outputs but only loads with transformers>=5,
    while Theia's remote code needs transformers<5.)
    """

    def __init__(self, repo: str):
        super().__init__()
        version = {"nvidia/C-RADIOv3-L": "c-radio_v3-l"}[repo]
        self.model = torch.hub.load(
            RADIO_CODE_REF,
            "radio_model",
            version=version,
            progress=True,
            skip_validation=True,
        )
        self.patch_size = self.pad_multiple = self.model.patch_size
        self.embed_dim = self.model.embed_dim

    def forward(self, x):
        out = self.model(x)
        return out.summary, out.features

    def last_blocks(self, x, n: int):
        # No per-block outputs: the final features are repeated, as in the paper's evaluation (one forward
        # pass). For n=4 this scales the head's effective learning rate, which matters at the top of the grid.
        return [self(x)] * n


class TheiaEncoder(nn.Module):
    """Theia resizes its input to 224x224 internally; dense tokens are resized back to the input grid."""

    def __init__(self, repo: str):
        super().__init__()
        import sys

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            repo, trust_remote_code=True, revision=HF_REVISIONS[repo]
        )
        self._handle_output = sys.modules[
            type(self.model).__module__
        ].handle_feature_output
        processor = self.model.backbone.processor
        self.register_buffer(
            "mean",
            torch.tensor(processor.image_mean).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor(processor.image_std).view(1, -1, 1, 1), persistent=False
        )
        self.patch_size = self.pad_multiple = 16
        self.embed_dim = self.tokens(torch.rand(1, 3, 224, 224)).shape[-1]

    def tokens(self, x):
        """Theia features [B, 196, C] of x in [0, 1]."""
        # Theia's processor rejects anything outside [0, 1], incl. 1 + 1e-7 from an antialiased resize
        x = x.clamp(0, 1)
        if x.shape[-2:] != (224, 224):
            return self.model.forward_feature(x, do_rescale=False)
        # At 224x224 the processor's PIL resize is a copy, leaving: 8-bit quantization, then (x - mean) / std.
        # Done here on the GPU with the processor's exact arithmetic instead of per image on the CPU.
        q = (x.double() * 255).float().to(torch.uint8)
        x = ((q.double() * (1 / 255)).float() - self.mean) / self.std
        out = self.model.backbone.model(pixel_values=x, interpolate_pos_encoding=None)
        return self._handle_output(
            out.last_hidden_state, num_discard_tokens=self.model.num_reg_tokens
        )

    def forward(self, x):
        tokens = self.tokens(x)  # [B, 196, C] on a 14x14 grid
        b, n, c = tokens.shape
        side = int(math.isqrt(n))
        grid = tokens.transpose(1, 2).reshape(b, c, side, side)
        if x.shape[2] // 16 != side or x.shape[3] // 16 != side:
            grid = F.interpolate(
                grid,
                size=(x.shape[2] // 16, x.shape[3] // 16),
                mode="bilinear",
                align_corners=False,
            )
        return tokens.mean(dim=1), grid.flatten(2).transpose(1, 2)

    def last_blocks(self, x, n: int):
        # No per-block outputs: the final features are repeated, as in the paper's evaluation (one forward
        # pass). For n=4 this scales the head's effective learning rate, which matters at the top of the grid.
        return [self(x)] * n


def center_pad(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Center-pad H and W to a multiple of `multiple`. Returns the padded tensor and the (top, left) offsets."""
    h, w = x.shape[-2:]
    ph, pw = (multiple - h % multiple) % multiple, (multiple - w % multiple) % multiple
    top, left = ph // 2, pw // 2
    return F.pad(x, (left, pw - left, top, ph - top)), (top, left)
