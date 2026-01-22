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

def create_defm_model(model_name, pretrained=False, pretrained_path=None):
    """
    Builds the DeFM model and loads weights.
    Supports ViT-L/14 (307M) and ViT-S/14 (22M) variants
    """
    cfg = get_defm_config(model_name)

    if "vit" in cfg.arch:
        vit_kwargs = dict(
            img_size=cfg.global_crops_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans, # Usually 3 for your stacked input
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
        model = ResNetBiFPN(backbone_name=cfg.arch, out_channels=cfg.out_channels)
    elif "regnet" in cfg.arch:
        model = RegNetBiFPN(backbone_name=cfg.arch, out_channels=cfg.out_channels)
    elif "efficientnet" in cfg.arch:
        model = EfficientNetBiFPN(backbone_name=cfg.arch, out_channels=cfg.out_channels)
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
                filename=f"{model_name}.pth" # Matches your HF filenames
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