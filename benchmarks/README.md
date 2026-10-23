# DeFM benchmarks

Code to reproduce the classification and semantic-segmentation benchmarks of the
[DeFM paper](https://arxiv.org/abs/2601.18923) for DeFM and the baselines, using the public weights on the
Hugging Face Hub. All encoders are frozen; only a linear head is trained.

- **Classification**: kNN and linear probe on ImageNet-1k-Depth (global features)
- **Segmentation**: linear probe on dense features for ScanNet, SUN RGB-D, OFFSED, TartanGround and GraspNet-1Billion

## Installation

```bash
git clone https://github.com/leggedrobotics/defm && cd defm
pip install -e ".[benchmarks]"   # run everything below from the repository root
```

A GPU is required. ViT-L encoders need about 24 GB of GPU memory; the segmentation feature cache needs up to ~80 GB
of disk for ScanNet with a ViT-L (`--cache-dir`). Some third-party weights are gated on the Hub: accept the license
on the model page and log in with `hf auth login` (DINOv3: `facebook/dinov3-*`).

## Encoders

| `--encoder` | Model | Weights |
|---|---|---|
| `defm_vit_l14`, `defm_vit_s14` | DeFM ViT-L/14, ViT-S/14 | [leggedrobotics/defm](https://huggingface.co/leggedrobotics/defm) |
| `defm_resnet50`, `defm_resnet34`, `defm_resnet18` | DeFM ResNets (BiFPN) | [leggedrobotics/defm](https://huggingface.co/leggedrobotics/defm) |
| `defm_efficientnet_b{0,2,4,6}`, `defm_regnet_y_{400mf,800mf,1_6gf}` | DeFM EfficientNets, RegNets | [leggedrobotics/defm](https://huggingface.co/leggedrobotics/defm) |
| `dinov2_vitl14`, `dinov2_vits14` | DINOv2 | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large), [-small](https://huggingface.co/facebook/dinov2-small) |
| `dinov3_vitl16`, `dinov3_vits16` | DINOv3 (LVD-1689M) | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), [vits16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) (gated) |
| `dav2_vitl14`, `dav2_vits14` | Depth Anything V2 encoder | [depth-anything/Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large), [-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) |
| `radio_v3_l` | C-RADIOv3-L | [nvidia/C-RADIOv3-L](https://huggingface.co/nvidia/C-RADIOv3-L) |
| `theia_s16` | Theia (DeiT-S/16) | [theaiinstitute/theia-small-patch16-224-cdiv](https://huggingface.co/theaiinstitute/theia-small-patch16-224-cdiv) |

Third-party weights are pinned to fixed Hub revisions (`benchmarks/encoders.py`). DeFM gets its metric-aware
3-channel input (`defm.utils`); the RGB models get the min-max normalized depth repeated over three channels with
their own normalization.

## Datasets

Every dataset uses the same format (`meta.json` + `data/<split>-*.parquet` with 16-bit depth and 8-bit label PNGs,
see `benchmarks/segmentation/datasets.py`). Datasets whose license allows it are downloaded from the Hub
automatically; the others are built from their official release with a script in `scripts/datasets/`, which
reproduces the data used for the results below exactly.

| Dataset | Classes | Train / test | How to get it |
|---|---|---|---|
| ImageNet-1k-Depth | 1000 | 1,281,167 / 50,000 | automatic: [leggedrobotics/defm-bench-imagenet-depth](https://huggingface.co/datasets/leggedrobotics/defm-bench-imagenet-depth) (gated: accept the ImageNet terms); or generate it from ImageNet, see below |
| TartanGround | 6 | 800 / 400 | automatic: [leggedrobotics/defm-bench-tartanground](https://huggingface.co/datasets/leggedrobotics/defm-bench-tartanground) |
| GraspNet-1Billion | 40 | 2,000 / 600 | automatic: [leggedrobotics/defm-bench-graspnet](https://huggingface.co/datasets/leggedrobotics/defm-bench-graspnet) |
| ScanNet (25k frames) | 40 (NYU40) | 19,466 / 5,436 | `convert_scannet.py` from the official release |
| SUN RGB-D | 37 | 5,285 / 5,050 | `convert_sunrgbd.py` from the official release (or DFormer's copy) |
| OFFSED | 5 | 162 / 41 | `convert_offsed.py` from the official release |

Locally built datasets go to `<data_root>/<name>`; pass `--data-root <data_root>` to the benchmark.

<details>
<summary><b>ScanNet</b></summary>

Accept the [ScanNet terms of use](http://www.scan-net.org) to receive `download-scannet.py`, then

```bash
python download-scannet.py -o <dir> --preprocessed_frames          # scannet_frames_25k.zip, 5.6 GB
unzip <dir>/tasks/scannet_frames_25k.zip -d <dir>
python scripts/datasets/convert_scannet.py --frames <dir>/scannet_frames_25k --out <data_root>/scannet
```

Train: the official `scannetv2_train` scenes, test: the official `scannetv2_val` scenes (`scripts/datasets/splits/`).
</details>

<details>
<summary><b>SUN RGB-D</b></summary>

From the [official release](https://rgbd.cs.princeton.edu) (`SUNRGBD.zip` and `SUNRGBDtoolbox.zip`):

```bash
python scripts/datasets/convert_sunrgbd.py --source official --sunrgbd <dir>/SUNRGBD \
    --toolbox <dir>/SUNRGBDtoolbox --out <data_root>/sunrgbd
```

or from the preprocessed copy linked in [DFormer](https://github.com/VCIP-RGBD/DFormer) (identical output):

```bash
python scripts/datasets/convert_sunrgbd.py --source dformer --sunrgbd <dir>/SUNRGBD --out <data_root>/sunrgbd
```
</details>

<details>
<summary><b>OFFSED</b></summary>

Download `disparity.zip` (6 GB) and `devkit_offsed.zip` from the [OFFSED page](https://www.dfki.uni-kl.de/~neigel/offsed.html), then

```bash
unzip disparity.zip -d <dir> && unzip devkit_offsed.zip -d <dir>
python scripts/datasets/convert_offsed.py --disparity <dir>/disparity --devkit <dir>/devkit_offsed \
    --out <data_root>/offsed
```

OFFSED does not define a split; ours is in `scripts/datasets/splits/offsed_{train,test}.txt`.
</details>

<details>
<summary><b>ImageNet-1k-Depth from ImageNet</b></summary>

Instead of downloading the gated copy, the depth maps can be regenerated from your ImageNet-1k copy with Depth
Anything V2 (one forward pass per image; `--shard i --num-shards n` splits the work over GPUs):

```bash
python scripts/datasets/generate_imagenet_depth.py --dav2-repo <Depth-Anything-V2 clone> \
    --imagenet <imagenet> --out <dir>/imagenet_depth_npz
python scripts/datasets/build_imagenet_depth.py --src <dir>/imagenet_depth_npz --out <data_root>/imagenet_depth
```
</details>

## Running

```bash
# segmentation: caches features once, trains the probe for each peak LR, reports the best
python -m benchmarks.segmentation --encoder defm_vit_l14 --dataset tartanground
python -m benchmarks.segmentation --encoder dinov3_vitl16 --dataset scannet --data-root <data_root> --cache-dir <fast disk>

# classification
python -m benchmarks.classification --encoder defm_vit_s14 --eval knn linear
```

Results are written as JSON to `results/{segmentation,classification}/` (per-LR runs, per-class IoU, kNN for
k = 10/20/100/200).

## Protocol

**Segmentation.** Inputs are brought to a fixed size per dataset (ScanNet 480x640, SUN RGB-D 530x730 by center
pad/crop, OFFSED 360x640, TartanGround 640x640, GraspNet 360x640). Last-layer patch tokens are extracted once
(no augmentation); a linear layer maps each token to class logits, which are bilinearly upsampled. Training:
pixel-wise cross-entropy (ignore 255), AdamW (weight decay 1e-4), one-cycle schedule with linear warm-up and cosine
decay, batch 32, FP16, gradient clipping 1.0; 25 epochs with 3 warm-up epochs (OFFSED: 200 / 20). The probe is
trained with peak LR 1e-3 and 5e-3, evaluated after the last epoch, and the better mIoU is reported. mIoU is computed
from a confusion matrix accumulated over the whole test set.

**Classification.** Center-cropped 224x224 inputs. kNN: cosine similarity on L2-normalized global features,
temperature 0.07, k = 20 reported (DINOv2 protocol). Linear: linear heads on the class token (+ average-pooled patch
tokens) of the last 1 or 4 blocks (DeFM CNNs: last only; C-RADIOv3 and Theia have no per-block outputs, so their
final features are repeated) over a grid of 10 learning rates, SGD with cosine schedule, 3 epochs, random-crop
and flip augmentation; the best head on the validation set is reported. The relative disparity is min-max normalized
per image and converted with DeFM's MDE preprocessing (`defm.utils.preprocess_depth_dav2`).

## Results

Segmentation, test mIoU (%):

| Model | Encoder | ScanNet | SUN RGB-D | OFFSED | TartanGround | GraspNet-1B |
|---|---|---|---|---|---|---|
| ViT-L/14 | DINOv2 | 29.91 | 32.71 | 53.98 | 65.88 | 24.12 |
| ViT-L/16 | C-RADIOv3 | 31.17 | 33.83 | 55.40 | 70.46 | 24.65 |
| ViT-L/16 | DINOv3 | 34.86 | **39.23** | 54.32 | 68.61 | 25.52 |
| ViT-L/14 | Depth Anything V2 | 29.91 | 32.68 | 55.87 | 67.25 | 25.43 |
| ViT-L/14 | **DeFM** | **38.26** | 36.88 | **57.15** | **73.22** | **27.57** |
| DeiT-S/16 | Theia | 19.56 | 17.48 | 41.56 | 55.13 | 11.11 |
| ViT-S/14 | DINOv2 | 22.90 | 23.77 | 48.06 | 61.83 | 15.92 |
| ViT-S/16 | DINOv3 | 26.31 | 26.55 | 56.08 | 63.59 | 18.04 |
| ViT-S/14 | Depth Anything V2 | 22.28 | 22.87 | 47.63 | 63.98 | 14.22 |
| ViT-S/14 | **DeFM** | **33.79** | **33.27** | **58.10** | **70.82** | **20.53** |
| ResNet-50 | DeFM | 35.34 | 32.19 | 52.55 | 70.89 | 31.95 |
| ResNet-34 | DeFM | 33.87 | 30.36 | 52.21 | 70.01 | 27.86 |
| ResNet-18 | DeFM | 28.01 | 25.97 | 46.87 | 65.49 | 23.27 |

ImageNet-1k-Depth classification, accuracy (%), kNN with k = 20:

| Model | Encoder | kNN top-1 | kNN top-5 | Linear top-1 |
|---|---|---|---|---|
| ViT-L/14 | DINOv2 | 61.32 | 81.51 | 68.90 |
| ViT-L/16 | C-RADIOv3 | 55.87 | 77.57 | 63.99 |
| ViT-L/16 | DINOv3 | **64.79** | 84.00 | 69.66 |
| ViT-L/14 | **DeFM** | 64.76 | **85.42** | **72.76** |
| DeiT-S/16 | Theia | 28.82 | 48.45 | 33.36 |
| ViT-S/14 | DINOv2 | 48.10 | 69.63 | 54.97 |
| ViT-S/16 | DINOv3 | 49.45 | 70.81 | 50.95 |
| ViT-S/14 | **DeFM** | **57.02** | **79.37** | **62.74** |
| ResNet-50 | DeFM | 54.87 | 78.93 | 62.56 |
| ResNet-34 | DeFM | 49.20 | 73.84 | 54.78 |
| ResNet-18 | DeFM | 46.73 | 70.93 | 51.57 |
| RegNetY-1.6GF | DeFM | 54.81 | 77.96 | 58.88 |
| RegNetY-800MF | DeFM | 53.33 | 76.43 | 58.62 |
| RegNetY-400MF | DeFM | 50.92 | 74.43 | 51.80 |
| EfficientNet-B6 | DeFM | 54.77 | 79.30 | 60.70 |
| EfficientNet-B4 | DeFM | 51.96 | 76.45 | 56.24 |
| EfficientNet-B2 | DeFM | 48.90 | 72.89 | 51.64 |
| EfficientNet-B0 | DeFM | 45.81 | 69.54 | 47.17 |

Re-running gives the same numbers up to run-to-run variation (within ~0.5 mIoU; up to ~1-2 on OFFSED, whose test
set has 41 images).

## Citation

If you use the benchmarks, please cite DeFM (see the [main README](../README.md#-citation)) and the datasets you use;
their BibTeX entries are in `scripts/datasets/citations.json` and in each dataset's `meta.json`.
