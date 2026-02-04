# DeFM: Learning Foundation Representations from Depth for Robotics

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![uv Managed](https://img.shields.io/badge/uv-managed-blueviolet?style=for-the-badge&logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange?style=for-the-badge)](https://huggingface.co/leggedrobotics/defm)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Arxiv](https://img.shields.io/badge/arXiv-2601.18923v1-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2601.18923v1)
[![Webpage](https://img.shields.io/badge/Webpage-de--fm.github.io-yellow.svg?style=for-the-badge&logo=google-chrome&logoColor=white)](https://de-fm.github.io)
[![GitHub](https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/leggedrobotics/defm)

</div>

---

**DeFM** (Depth Foundation Model) is a vision backbone trained on **60M depth images** via self-distillation. It is engineered for robotic perception, providing metric-aware representations that excel in sim-to-real transfer and cross-sensor generalization.

TL;DR - A DINO-style encoder, but for depth image inputs.

## 🌟 Key Features
- **Large-Scale Pretraining**: We pretrain on our curated dataset of 60 M depth images using self-distillation.
- **Semantic Awareness**: DeFM learns not only robust geometric priors but also semantically rich features from just depth images.
- **Metric-Aware Normalization**: Our novel three channel input normalization preserves metric depth across multiple scales.
- **Compact efficient models**: We distill our DeFM-ViT-L into a family of smaller efficient CNNs as small as 3M params for robot policy learning.
- **Robotics Proven**: Our encoder is proven effective for diverse robotic tasks such as navigation, manipulation and locomotion without task-specific fine-tuning.

---

## 🛠️ Installation

To use DeFM as a backbone in your own projects without cloning this repository, ensure you have the following prerequisites installed:
```bash
pip install torch torchvision numpy huggingface_hub omegaconf
```
and directly jump to [Quick Start](#quick-start). Otherwise, below you can find instructions to install the DeFM module for using the notebooks or for local development.

### Using uv (Fastest)
```bash
# Create and activate environment
uv venv --python 3.10
source .venv/bin/activate

# Install dependencies and DeFM in editable mode
uv pip install -e .
```

### Using standard pip
```bash
pip install -e .
```

---

<a id="quick-start"></a>
## 🚀 Quick Start

### 1. Loading the Model
Load via **TorchHub** for easy integration:

```python
import torch

# Load the 307M Parameter Foundation Model
model = torch.hub.load('leggedrobotics/defm:main', 'defm_vit_l14', pretrained=True)
model.eval().to("cuda")
```

### 2. Preprocessing
DeFM requires depth maps to be processed into our metric-aware 3-channel format.

```python
from defm import preprocess_depth_image

# Depth needs to be in meters (numpy array, tensor or PIL image)
normalized_depth = preprocess_depth_image(metric_depth, target_size=518, patch_size=14)
```

### 3. Inference
```python
with torch.no_grad():
    output = model.get_intermediate_layers(
        normalized_depth, n=1, reshape=True, return_class_token=True)

spatial_tokens = output[0][0] # (B, C, H', W')
class_token = output[0][1] # (B, C)
```

---

## 📂 Project Structure
* `defm/`: Main package containing model factory, architectures, and utils.
* `notebooks/`: Demo notebooks for **Semantic PCA Visualization** and **Inference Scripts** for CNNs and ViTs.
* `classification/`: Scripts and packages to reproduce the classification results [TODO]
* `segmentation/`: Scripts and packages to reproduce the segmentation results [TODO]

---

## 📊 Model Zoo

The following table provides a comprehensive overview of the DeFM model family, including architectural parameters, inference latency across training and deployment hardware (224x224), and performance on the ImageNet-1k-Depth benchmark.

| Model | Params (M) | RTX 4090 BS-128 (ms) | Jetson Orin (ms) | Top-5 KNN (%) | Linear Prob (%) | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeFM ViT-L/14** | 307.0 | 624.91 | 72.82 | 84.79 | 71.72 | [Download](https://huggingface.co/leggedrobotics/defm/resolve/main/defm_vit_l14.pth) |
| **DeFM ViT-S/14** | 22.1 | 63.76 | 11.92 | 78.06 | 61.54 | [Download](https://huggingface.co/leggedrobotics/defm/resolve/main/defm_vit_s14.pth) |
| **DeFM ResNet-50** | 26.2 | 69.39 | 17.79 | 77.63 | 61.54 | [Download](https://huggingface.co/leggedrobotics/defm/resolve/main/defm_resnet50.pth) |
| **DeFM ResNet-34** | 21.8 | 33.08 | 13.54 | 72.72 | 54.39 | [Download](https://huggingface.co/leggedrobotics/defm/resolve/main/defm_resnet34.pth) |
| **DeFM ResNet-18** | 11.7 | 21.06 | 8.67 | 69.69 | 50.58 | [Download](https://huggingface.co/leggedrobotics/defm/resolve/main/defm_resnet18.pth) |
| **DeFM EfficientNet-B6** | 28.98 | 150.98 | 54.11 | 77.81 | 59.23 | [Download](https://huggingface.co/leggedrobotics/defm/resolve/main/defm_efficientnet_b6.pth) |
| **DeFM EfficientNet-B4** | 14.16 | 86.51 | 39.67 | 74.74 | 54.73 | [Download](https://huggingface.co/leggedrobotics/defm/resolve/main/defm_efficientnet_b4.pth) |
| **DeFM EfficientNet-B2** | 4.95 | 46.12 | 28.37 | 71.51 | 50.32 | [Download](https://huggingface.co/leggedrobotics/defm/resolve/main/defm_efficientnet_b2.pth) |
| **DeFM EfficientNet-B0** | 3.01 | 29.39 | 21.04 | 67.98 | 46.17 | [Download](https://huggingface.co/leggedrobotics/defm/resolve/main/defm_efficientnet_b0.pth) |
| **DeFM RegNetY-1.6GF** | 12.4 | 44.25 | 41.82 | 76.21 | 57.28 | [Download](https://huggingface.co/leggedrobotics/defm/resolve/main/defm_regnet_y_1_6gf.pth) |
| **DeFM RegNetY-800MF** | 6.3 | 25.21 | 24.16 | 74.91 | 57.03 | [Download](https://huggingface.co/leggedrobotics/defm/resolve/main/defm_regnet_y_800mf.pth) |
| **DeFM RegNetY-400MF** | 4.1 | 17.27 | 25.17 | 72.87 | 50.51 | [Download](https://huggingface.co/leggedrobotics/defm/resolve/main/defm_regnet_y_400mf.pth) |
---

## 📖 Citation
If you find DeFM useful for your research, please cite our paper:

```
@misc{patel2026defm,
      title={DeFM: Learning Foundation Representations from Depth for Robotics}, 
      author={Manthan Patel and Jonas Frey and Mayank Mittal and Fan Yang and Alexander Hansson and Amir Bar and Cesar Cadena and Marco Hutter},
      year={2026},
      eprint={2601.18923},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.18923}, 
}
```

## Contribution Guidelines

<!-- For documentation, we adopt the [Google Style Guide](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings. Please make sure that your code is well-documented and follows the guidelines. -->

We use the following tools for maintaining code quality:

- [pre-commit](https://pre-commit.com/): Runs a list of formatters and linters over the codebase.
- [ruff](https://github.com/astral-sh/ruff): An extremely fast Python linter and code formatter, written in Rust.

Please check [here](https://pre-commit.com/#install) for instructions to set these up. To run over the entire repository, please execute the following command in the terminal:

```bash
# for installation (only once)
pre-commit install
# for running
pre-commit run --all-files
```

## Acknowledgement

We would like to thank the researchers and developers of the [DINOv2](https://github.com/facebookresearch/dinov2) repository for their excellent open-source work, which served as the foundation for our implementation.

This work was supported as part of the Swiss AI Initiative by a grant from the Swiss National Supercomputing Centre (CSCS) under project ID a144 on Alps. This work was also supported by the Luxembourg National Research Fund (Ref. 18990533), and the Swiss National Science Foundation (SNSF) through projects No. 200021E_229503 and No. 227617.