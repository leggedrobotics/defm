# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Fast CPU checks of the package and the benchmark code (no downloads).

    pip install -e ".[benchmarks,test]" && pytest tests

DEFM_TEST_NETWORK=1 also builds every benchmark encoder from the Hub and runs a forward pass (downloads ~5 GB;
DINOv3 needs `hf auth login` with access to the gated facebook/dinov3-* repos).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "datasets"))

network = pytest.mark.skipif(
    os.environ.get("DEFM_TEST_NETWORK") != "1", reason="set DEFM_TEST_NETWORK=1"
)


# ---------------------------------------------------------------------------------------------- package


def test_top_level_exports():
    from defm import (
        convert_to_3channel_metric,
        preprocess_depth_batch,
        preprocess_depth_dav2,
        preprocess_depth_image,
    )

    assert all(
        map(
            callable,
            [
                convert_to_3channel_metric,
                preprocess_depth_batch,
                preprocess_depth_dav2,
                preprocess_depth_image,
            ],
        )
    )


def test_hub_config_matches_packaged(monkeypatch):
    """The Hub config.json is generated from the packaged YAMLs; offline, get_hub_config falls back to them."""
    from defm import model_factory

    monkeypatch.setattr(
        model_factory,
        "hf_hub_download",
        lambda **_: (_ for _ in ()).throw(OSError("offline")),
    )
    for path in (ROOT / "defm" / "configs").glob("*.yaml"):
        assert model_factory.get_hub_config(path.stem) == model_factory.get_defm_config(
            path.stem
        )


def test_build_model_config(tmp_path):
    from omegaconf import OmegaConf

    sys.path.insert(0, str(ROOT / "scripts" / "hf"))
    import build_model_config

    sys.argv = ["build_model_config.py", "--out", str(tmp_path / "config.json")]
    build_model_config.main()
    config = json.loads((tmp_path / "config.json").read_text())
    for name, variant in config["variants"].items():
        assert OmegaConf.create(variant) == OmegaConf.load(
            ROOT / "defm" / "configs" / f"{name}.yaml"
        )
        assert config["weights"][name] == f"{name}.pth"


# ---------------------------------------------------------------------------------------------- preprocessing


def test_defm_encoding_matches_package():
    from benchmarks.preprocess import encode_metric_depth
    from defm.utils.utils import convert_to_3channel_metric

    depth = np.random.default_rng(0).uniform(0.1, 30.0, (48, 64)).astype(np.float32)
    depth[0, 0] = 0.0
    x = encode_metric_depth(depth, "defm")
    assert x.shape == (3, 48, 64)
    np.testing.assert_allclose(
        x.permute(1, 2, 0).numpy(), convert_to_3channel_metric(depth), rtol=1e-6
    )


@pytest.mark.parametrize("norm", ["imagenet", "unit"])
def test_rgb_baseline_encoding(norm):
    from benchmarks.preprocess import encode_metric_depth, normalize

    depth = np.linspace(1, 5, 12, dtype=np.float32).reshape(3, 4)
    x = encode_metric_depth(depth, norm)
    assert (
        x.shape == (3, 3, 4)
        and float(x.min()) == 0.0
        and abs(float(x.max()) - 1.0) < 1e-5
    )
    assert torch.equal(x[0], x[1]) and torch.equal(x[1], x[2])
    assert normalize(x, norm).shape == x.shape


def test_disparity_is_not_quantized():
    from benchmarks.classification.data import encode_disparity

    disp = np.arange(0, 60000, 3, dtype=np.uint16).reshape(100, 200)
    x = encode_disparity(disp, "unit")
    assert len(torch.unique(x[0])) > 256  # float16 path: no 8-bit quantization


# ---------------------------------------------------------------------------------------------- data format


def _write_toy_dataset(root: Path, name: str, size, num_classes: int):
    from common import encode_depth, encode_label, write_meta, write_split

    rng = np.random.default_rng(0)

    def samples(n):
        for i in range(n):
            depth = rng.uniform(0.5, 10.0, size)
            label = rng.integers(0, num_classes, size).astype(np.uint8)
            label[0, :] = 255
            yield f"s{i}", encode_depth(depth, 1000), encode_label(label)

    counts = {
        "train": write_split(root / name, "train", samples(4)),
        "test": write_split(root / name, "test", samples(2)),
    }
    write_meta(
        root / name,
        name=name,
        num_classes=num_classes,
        class_names=[str(c) for c in range(num_classes)],
        depth_scale=1000,
        ignore_index=255,
        splits=counts,
        license="test",
        citation="",
    )
    return counts


@pytest.mark.parametrize("name, size", [("offsed", (90, 160)), ("sunrgbd", (400, 500))])
def test_segmentation_dataset(tmp_path, name, size):
    from benchmarks.segmentation.datasets import BENCHMARKS, SegmentationDataset

    counts = _write_toy_dataset(tmp_path, name, size, num_classes=5)
    ds = SegmentationDataset(name, "train", "defm", data_root=str(tmp_path))
    assert len(ds) == counts["train"] == 4
    x, y = ds[0]
    assert (
        x.shape == (3, *BENCHMARKS[name].size)
        and y.shape == BENCHMARKS[name].size
        and y.dtype == torch.int64
    )
    assert set(torch.unique(y).tolist()) <= set(range(5)) | {255}


def test_unhosted_dataset_needs_data_root():
    from benchmarks.segmentation.datasets import resolve_root

    with pytest.raises(ValueError, match="convert_scannet"):
        resolve_root("scannet", None)


# ---------------------------------------------------------------------------------------------- probes and metrics


def test_metrics_count_beyond_float32():
    """The confusion matrix must not saturate at 2**24 pixels per cell (the bug fixed in the v2 results)."""
    from benchmarks.segmentation.linear_probe import segmentation_metrics

    conf = torch.tensor([[2**25 + 1, 2**24], [0, 2**24]], dtype=torch.long)
    iou0 = (2**25 + 1) / (2**25 + 1 + 2**24)
    iou1 = 2**24 / (2**24 + 2**24)
    m = segmentation_metrics(conf)
    assert abs(m["miou"] - 100 * (iou0 + iou1) / 2) < 1e-9


def test_metrics_ignore_absent_classes():
    from benchmarks.segmentation.linear_probe import segmentation_metrics

    m = segmentation_metrics(torch.tensor([[5, 0, 0], [0, 5, 0], [0, 0, 0]]))
    assert (
        m["miou"] == 100.0
        and abs(m["miou_absent_as_zero"] - 200 / 3) < 1e-9
        and m["per_class_iou"][2] is None
    )


@pytest.mark.parametrize(
    "size, patch, pad", [((360, 640), 14, 14), ((530, 730), 16, 32)]
)
def test_linear_head_output_size(size, patch, pad):
    from benchmarks.segmentation.linear_probe import LinearSegHead

    head = LinearSegHead(8, 5, size, patch, pad)
    tokens = torch.randn(2, head.grid[0] * head.grid[1], 8)
    assert head(tokens).shape == (2, 5, *size)


def test_center_pad():
    from benchmarks.encoders import center_pad

    x, (top, left) = center_pad(torch.ones(1, 3, 360, 640), 14)
    assert (
        x.shape[-2:] == (364, 644)
        and (top, left) == (2, 2)
        and x[..., 2:362, 2:642].min() == 1
    )


def test_knn_separable():
    from benchmarks.classification.evaluate import knn_classify

    centers = torch.nn.functional.normalize(torch.randn(6, 16), dim=1)
    train_y, val_y = torch.arange(6).repeat(20), torch.arange(6).repeat(5)
    noise = lambda n: 0.01 * torch.randn(n, 16)  # noqa: E731
    train_f = torch.nn.functional.normalize(centers[train_y] + noise(120), dim=1)
    val_f = torch.nn.functional.normalize(centers[val_y] + noise(30), dim=1)
    acc = knn_classify(train_f, train_y, val_f, val_y, ks=(10, 20), num_classes=6)
    assert acc["k20"]["top1"] == 100.0


# ---------------------------------------------------------------------------------------------- encoders (network)


def test_encoder_registry():
    from benchmarks.encoders import ENCODERS, HF_REVISIONS

    assert {
        "defm_vit_l14",
        "defm_vit_s14",
        "dinov2_vitl14",
        "dinov3_vitl16",
        "radio_v3_l",
        "theia_s16",
    } <= set(ENCODERS)
    for spec in ENCODERS.values():
        assert spec.input_norm in {"defm", "imagenet", "unit"}
        assert (
            spec.family.startswith("defm") or spec.hf_repo in HF_REVISIONS
        )  # third-party weights are pinned


@network
@pytest.mark.parametrize(
    "name", __import__("benchmarks.encoders", fromlist=["ENCODERS"]).ENCODERS
)
def test_encoder_forward(name):
    from benchmarks.encoders import build_encoder, center_pad
    from benchmarks.preprocess import encode_metric_depth, normalize

    enc = build_encoder(name)
    depth = np.random.default_rng(0).uniform(0.5, 8.0, (100, 130)).astype(np.float32)
    x = normalize(encode_metric_depth(depth, enc.input_norm), enc.input_norm)[None]
    x, _ = center_pad(x, enc.pad_multiple)
    with torch.no_grad():
        g, dense = enc(x)
    h, w = x.shape[-2] // enc.patch_size, x.shape[-1] // enc.patch_size
    # the global feature can be wider than the dense one (DeFM CNNs: 512 vs 128, RADIO summary: 3072 vs 1024)
    assert g.dim() == 2 and len(g) == 1 and dense.shape == (1, h * w, enc.embed_dim)
    assert torch.isfinite(g).all() and torch.isfinite(dense).all()
