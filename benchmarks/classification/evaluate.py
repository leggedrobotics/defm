# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""kNN and linear-probe classification on frozen global features (DINOv2 protocol)."""

import itertools
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from benchmarks.classification.data import ImageNetDepth

LINEAR_LRS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]


def _loader(ds, batch_size, workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    )


@torch.no_grad()
def extract_global(encoder, ds, batch_size, workers, dtype):
    # Filled in place on the CPU and moved to the GPU once: concatenating batches would briefly hold two copies
    # (2 x 16 GB for RADIO's 3072-d features on ImageNet train).
    feats, labels, n, t0 = None, torch.empty(len(ds), dtype=torch.long), 0, time.time()
    for i, (x, y) in enumerate(_loader(ds, batch_size, workers)):
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            g, _ = encoder(x.cuda(non_blocking=True))
        if feats is None:
            feats = torch.empty(len(ds), g.shape[1])
        feats[n : n + len(y)] = F.normalize(g.float(), dim=1).cpu()
        labels[n : n + len(y)] = y
        n += len(y)
        if i % 500 == 0:
            print(
                f"    {i * batch_size}/{len(ds)}  {(i + 1) * batch_size / (time.time() - t0):.0f} img/s",
                flush=True,
            )
    assert n == len(ds), (n, len(ds))
    return feats.cuda(), labels.cuda()


@torch.no_grad()
def knn_classify(
    train_f,
    train_y,
    val_f,
    val_y,
    ks=(10, 20, 100, 200),
    temperature=0.07,
    num_classes=1000,
    chunk=512,
):
    """Weighted kNN of DINO/DINOv2: each of the top-k neighbours votes with exp(cos_sim / T)."""
    correct = {k: [0, 0] for k in ks}
    for s in range(0, len(val_f), chunk):
        sims, idx = (val_f[s : s + chunk] @ train_f.T).topk(max(ks), dim=1)
        neighbours = train_y[idx]
        y = val_y[s : s + chunk, None]
        for k in ks:
            votes = torch.zeros(len(y), num_classes, device=y.device).scatter_add_(
                1, neighbours[:, :k], (sims[:, :k] / temperature).exp()
            )
            top5 = votes.topk(5, dim=1).indices
            correct[k][0] += (top5[:, :1] == y).any(1).sum().item()
            correct[k][1] += (top5 == y).any(1).sum().item()
    return {
        f"k{k}": {"top1": 100 * c1 / len(val_f), "top5": 100 * c5 / len(val_f)}
        for k, (c1, c5) in correct.items()
    }


def run_knn(encoder, data_root, batch_size=256, workers=16, dtype=torch.float16):
    print("  kNN: extracting train features", flush=True)
    train = ImageNetDepth(
        "train", encoder.input_norm, train_transform=False, data_root=data_root
    )
    train_f, train_y = extract_global(encoder, train, batch_size, workers, dtype)
    print("  kNN: extracting validation features", flush=True)
    val = ImageNetDepth(
        "validation", encoder.input_norm, train_transform=False, data_root=data_root
    )
    val_f, val_y = extract_global(encoder, val, batch_size, workers, dtype)
    return knn_classify(train_f, train_y, val_f, val_y)


# ------------------------------------------------------------------------------------------------- linear probe


def _linear_input(blocks, n, avgpool):
    """Concatenate the global tokens of the last n blocks (+ the mean of the last block's dense tokens)."""
    feats = [g for g, _ in blocks[-n:]]
    if avgpool:
        feats.append(blocks[-1][1].mean(dim=1))
    return torch.cat(feats, dim=-1).float()


class LinearHeads(nn.Module):
    """All (n_blocks, avgpool, lr) heads, trained at once on the same frozen features."""

    def __init__(self, dims: dict, num_classes: int, lrs, batch_size: int):
        super().__init__()
        self.configs = [(n, avg, lr) for (n, avg), lr in itertools.product(dims, lrs)]
        self.heads = nn.ModuleList(
            nn.Linear(dims[(n, avg)], num_classes) for n, avg, _ in self.configs
        )
        for h in self.heads:
            nn.init.normal_(h.weight, std=0.01)
            nn.init.zeros_(h.bias)
        self.lr_scale = batch_size / 256

    def param_groups(self):
        return [
            {"params": h.parameters(), "lr": lr * self.lr_scale}
            for h, (_, _, lr) in zip(self.heads, self.configs)
        ]

    def forward(self, blocks):
        cache = {}
        outs = []
        for head, (n, avg, _) in zip(self.heads, self.configs):
            if (n, avg) not in cache:
                cache[(n, avg)] = _linear_input(blocks, n, avg)
            outs.append(head(cache[(n, avg)]))
        return outs


def run_linear(
    encoder,
    data_root,
    epochs=3,
    batch_size=256,
    workers=16,
    dtype=torch.float16,
    lrs=LINEAR_LRS,
):
    n_options = [1, 4] if encoder.per_block else [1]
    train = ImageNetDepth(
        "train",
        encoder.input_norm,
        train_transform=True,
        data_root=data_root,
        shuffle=True,
    )
    val = ImageNetDepth(
        "validation", encoder.input_norm, train_transform=False, data_root=data_root
    )

    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
    ):
        probe = encoder.last_blocks(
            torch.zeros(2, 3, 224, 224, device="cuda"), max(n_options)
        )
    dims = {
        (n, avg): _linear_input(probe, n, avg).shape[-1]
        for n in n_options
        for avg in (False, True)
    }

    heads = LinearHeads(dims, 1000, lrs, batch_size).cuda()
    optimizer = torch.optim.SGD(heads.param_groups(), momentum=0.9, weight_decay=0.0)
    steps = epochs * math.ceil(len(train) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=0
    )
    max_n = max(n_options)

    step, t0 = 0, time.time()
    for epoch in range(epochs):
        train.set_epoch(epoch)
        for x, y in _loader(train, batch_size, workers):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
            ):
                blocks = encoder.last_blocks(x, max_n)
            loss = sum(F.cross_entropy(o, y) for o in heads(blocks))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 500 == 0:
                print(
                    f"    step {step}/{steps}  loss/head {loss.item() / len(heads.heads):.3f}  "
                    f"{step * batch_size / (time.time() - t0):.0f} img/s",
                    flush=True,
                )

    correct = torch.zeros(len(heads.heads), device="cuda")
    with torch.no_grad():
        for x, y in _loader(val, batch_size, workers):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                blocks = encoder.last_blocks(x, max_n)
            correct += torch.stack(
                [(o.argmax(1) == y).sum() for o in heads(blocks)]
            ).float()
    acc = (100 * correct / len(val)).tolist()
    results = [
        {"n_blocks": n, "avgpool": avg, "lr": lr, "top1": a}
        for (n, avg, lr), a in zip(heads.configs, acc)
    ]
    best = max(results, key=lambda r: r["top1"])
    return {"top1": best["top1"], "best_head": best, "heads": results}
