# DeFM: Depth Foundation Model for Robotics

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![uv Managed](https://img.shields.io/badge/uv-managed-blueviolet?style=for-the-badge&logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Arxiv](https://img.shields.io/badge/arXiv-TODO-B31B1B.svg?style=for-the-badge)](TODO-link)
[![Webpage](https://img.shields.io/badge/Webpage-leggedrobotics.github.io/defm-yellow.svg?style=for-the-badge&logo=google-chrome&logoColor=white)](https://leggedrobotics.github.io/defm)
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
model = torch.hub.load('leggedrobotics/defm', 'defm_vit_l14', pretrained=True)
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

| Model | Params (M) | RTX 4090 (ms) | Jetson Orin (ms) | Top-5 KNN (%) | Linear Prob (%) | Checkpoint |
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
  title         = {DeFM: Learning Foundation Representations from Depth for Robotics},
  author        = {Patel, Manthan and Frey, Jonas and Mittal, Mayank and Yang, Fan and Hansson, Alexander and Bar, Amir and Cadena, Cesar and Hutter, Marco},
  year          = {2026},
  archivePrefix = {arXiv},
  eprint        = {XXXX.XXXXX},
  primaryClass  = {cs.RO}
}
```