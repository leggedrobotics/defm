dependencies = ["torch", "omegaconf", "huggingface_hub", "torchvision", "numpy"]
from defm.model_factory import create_defm_model
import pathlib

# Define the root directory of this repository
ROOT_DIR = pathlib.Path(__file__).parent.resolve()

# Vision Transformer Variants
def defm_vit_s14(pretrained=True, **kwargs):
    """DeFM ViT-S/14"""
    return create_defm_model("defm_vit_s14", pretrained=pretrained, **kwargs)

def defm_vit_l14(pretrained=True, **kwargs):
    """DeFM ViT-L/14 (307M)"""
    return create_defm_model("defm_vit_l14", pretrained=pretrained, **kwargs)

# ResNet Variants
def defm_resnet18(pretrained=True, **kwargs):
    """DeFM ResNet-18 variant """
    return create_defm_model("defm_resnet18", pretrained=pretrained, **kwargs)

def defm_resnet34(pretrained=True, **kwargs):
    """DeFM ResNet-34 variant """
    return create_defm_model("defm_resnet34", pretrained=pretrained, **kwargs)

def defm_resnet50(pretrained=True, **kwargs):
    """DeFM ResNet-50 variant """
    return create_defm_model("defm_resnet50", pretrained=pretrained, **kwargs)

# EfficientNet Variants
def defm_efficientnet_b0(pretrained=True, **kwargs):
    """DeFM EfficientNet-B0 variant """
    return create_defm_model("defm_efficientnet_b0", pretrained=pretrained, **kwargs)

def defm_efficientnet_b2(pretrained=True, **kwargs):
    """DeFM EfficientNet-B2 variant """
    return create_defm_model("defm_efficientnet_b2", pretrained=pretrained, **kwargs)

def defm_efficientnet_b4(pretrained=True, **kwargs):
    """DeFM EfficientNet-B4 variant """
    return create_defm_model("defm_efficientnet_b4", pretrained=pretrained, **kwargs)

def defm_efficientnet_b6(pretrained=True, **kwargs):
    """DeFM EfficientNet-B6 variant """
    return create_defm_model("defm_efficientnet_b6", pretrained=pretrained, **kwargs)

# RegNet Variants
def defm_regnet_y_400mf(pretrained=True, **kwargs):
    """DeFM RegNetY-400MF variant """
    return create_defm_model("defm_regnet_y_400mf", pretrained=pretrained, **kwargs)

def defm_regnet_y_800mf(pretrained=True, **kwargs):
    """DeFM RegNetY-800MF variant """
    return create_defm_model("defm_regnet_y_800mf", pretrained=pretrained, **kwargs)

def defm_regnet_y_1_6gf(pretrained=True, **kwargs):
    """DeFM RegNetY-1.6GF variant """
    return create_defm_model("defm_regnet_y_1_6gf", pretrained=pretrained, **kwargs)