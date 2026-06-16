"""
train_cnn.py
============

Multi-task CNN baseline (Part II).

A single ResNet18 backbone produces 512-dim features that feed four
classification heads, one per task:

    modulation type      (16 classes)
    phase noise level    ( 3 classes: none / mild / severe)
    IQ imbalance level   ( 3 classes: none / mild / severe)
    SNR bin              ( 4 classes; the held-out bin is excluded
                           from training and used for OOD evaluation)

Outputs (under results/):
    checkpoints/cnn_best.pt          best model on validation set
    cnn_metrics.json                 per-task accuracy on test and ood_test,
                                     model size, training time, inference latency
    05_cnn_confusion_modulation.png  modulation confusion matrix
    06_cnn_accuracy_vs_snr.png       per-task accuracy as function of SNR (dB)
    07_cnn_ood_vs_test.png           OOD vs in-distribution accuracy bar chart
    08_cnn_confusion_other.png       phase / IQ / SNR confusion matrices
    09_cnn_severity_sensitivity.png  modulation accuracy vs impairment severity

Run:
    python train_cnn.py                       # full training
    python train_cnn.py --epochs 3            # quick smoke test
    python train_cnn.py --batch_size 64       # tune for available memory
    python train_cnn.py --device cpu          # force CPU
"""
from __future__ import annotations

import os
import json
import time
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from signals import resolve_path

from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------------
# Label encoding   (consistent ordering for plots and reports)
# ---------------------------------------------------------------------------
MODULATIONS = [
    "4-ASK", "8-ASK", "BPSK", "QPSK",
    "4-HQAM", "16-HQAM", "64-HQAM",
    "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM",
    "16-APSK", "32-APSK", "64-APSK", "128-APSK",
]
PHASE_LEVELS = ["none", "mild", "severe"]
IQ_LEVELS    = ["none", "mild", "severe"]
SNR_BINS     = ["low", "low_mid", "mid", "high_mid", "high"]

# `held_out_snr_bin` excluded from training; encoding compact for the SNR head.
HELD_OUT_SNR_BIN = "low_mid"
SNR_BINS_TRAIN   = [b for b in SNR_BINS if b != HELD_OUT_SNR_BIN]

LABEL_TO_IDX = {
    "modulation": {m: i for i, m in enumerate(MODULATIONS)},
    "phase":      {p: i for i, p in enumerate(PHASE_LEVELS)},
    "iq":         {q: i for i, q in enumerate(IQ_LEVELS)},
    "snr":        {b: i for i, b in enumerate(SNR_BINS_TRAIN)},
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ConstellationDataset(Dataset):
    """Reads images and integer-encoded labels from a labels.csv DataFrame."""

    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(resolve_path(row["image_path"])).convert("RGB")
        img = self.transform(img)

        # SNR may be the held-out bin in `ood_test`. We map it to -1 there;
        # the loss/accuracy code masks SNR predictions in that case.
        snr_idx = LABEL_TO_IDX["snr"].get(row["snr_bin"], -1)

        return img, {
            "modulation": LABEL_TO_IDX["modulation"][row["modulation"]],
            "phase":      LABEL_TO_IDX["phase"][row["phase_noise_level"]],
            "iq":         LABEL_TO_IDX["iq"][row["iq_imbalance_level"]],
            "snr":        snr_idx,
            "snr_db":     float(row["snr_db"]),
        }


def make_loaders(labels_csv: str, batch_size: int, num_workers: int = 2):
    """Build train / val / test / ood_test loaders from labels.csv."""
    df = pd.read_csv(labels_csv)

    # ImageNet normalization — backbone is pretrained on ImageNet.
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    eval_tf = train_tf  # no augmentation; the image already represents many symbols.

    loaders = {}
    for split in ("train", "val", "test", "ood_test"):
        split_df = df[df["split"] == split]
        if len(split_df) == 0:
            continue
        ds = ConstellationDataset(split_df, eval_tf if split != "train" else train_tf)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders, df


# ---------------------------------------------------------------------------
# Multi-head CNN
# ---------------------------------------------------------------------------
class MultiHeadCNN(nn.Module):
    """ResNet18 backbone with four independent classification heads."""

    def __init__(self):
        super().__init__()
        try:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            backbone = torchvision.models.resnet18(weights=weights)
        except Exception:
            backbone = torchvision.models.resnet18(weights=None)  # offline fallback
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # remove fc
        self.dropout = nn.Dropout(0.2)
        feat_dim = 512
        self.head_modulation = nn.Linear(feat_dim, len(MODULATIONS))
        self.head_phase      = nn.Linear(feat_dim, len(PHASE_LEVELS))
        self.head_iq         = nn.Linear(feat_dim, len(IQ_LEVELS))
        self.head_snr        = nn.Linear(feat_dim, len(SNR_BINS_TRAIN))

    def forward(self, x):
        f = self.features(x).flatten(1)
        f = self.dropout(f)
        return {
            "modulation": self.head_modulation(f),
            "phase":      self.head_phase(f),
            "iq":         self.head_iq(f),
            "snr":        self.head_snr(f),
        }


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
TASKS = ("modulation", "phase", "iq", "snr")


def compute_loss(logits: dict, targets: dict) -> tuple[torch.Tensor, dict]:
    """Sum of CE losses over the four tasks. SNR may be masked (idx == -1)."""
    losses = {}
    for task in TASKS:
        y = targets[task]
        if task == "snr":
            mask = y >= 0
            if mask.any():
                losses[task] = F.cross_entropy(logits[task][mask], y[mask])
            else:
                losses[task] = torch.tensor(0.0, device=logits[task].device)
        else:
            losses[task] = F.cross_entropy(logits[task], y)
    total = sum(losses.values())
    return total, {k: v.item() for k, v in losses.items()}


def step(model, loader, device, optimizer=None):
    """One epoch of train (optimizer != None) or eval (optimizer == None)."""
    is_train = optimizer is not None
    model.train(is_train)
    sums = defaultdict(float)
    counts = defaultdict(int)
    correct = defaultdict(int)
    seen = defaultdict(int)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y_t = {k: torch.as_tensor(y[k]).to(device, non_blocking=True)
                   for k in TASKS}

            logits = model(x)
            total, per_task = compute_loss(logits, y_t)

            if is_train:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()

            sums["total"] += total.item() * x.size(0)
            counts["total"] += x.size(0)
            for k in TASKS:
                sums[k] += per_task[k] * x.size(0)
                # accuracy (only on valid samples for SNR)
                pred = logits[k].argmax(1)
                if k == "snr":
                    m = y_t[k] >= 0
                    correct[k] += int((pred[m] == y_t[k][m]).sum())
                    seen[k] += int(m.sum())
                else:
                    correct[k] += int((pred == y_t[k]).sum())
                    seen[k] += x.size(0)

    return {
        "loss":     sums["total"] / max(counts["total"], 1),
        "loss_per_task": {k: sums[k] / max(counts["total"], 1) for k in TASKS},
        "accuracy": {k: correct[k] / max(seen[k], 1) for k in TASKS},
    }


def collect_predictions(model, loader, device) -> dict:
    """Run inference, return per-sample predictions and labels."""
    model.eval()
    preds = defaultdict(list)
    truths = defaultdict(list)
    snr_dbs = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            for k in TASKS:
                p = logits[k].argmax(1).cpu().numpy()
                t = torch.as_tensor(y[k]).numpy()
                preds[k].append(p); truths[k].append(t)
            snr_dbs.append(np.asarray(y["snr_db"]))
    return {
        "preds":   {k: np.concatenate(preds[k]) for k in TASKS},
        "truths":  {k: np.concatenate(truths[k]) for k in TASKS},
        "snr_dbs": np.concatenate(snr_dbs),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_confusion_modulation(preds, truths, outpath):
    cm = confusion_matrix(truths, preds, labels=list(range(len(MODULATIONS))))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=MODULATIONS, yticklabels=MODULATIONS,
                cbar_kws={"label": "row-normalized"},
                annot_kws={"size": 6}, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Modulation classification — confusion matrix (test split)")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_snr(results, outpath, snr_bin_edges=None):
    """Accuracy per SNR bucket, one line per task."""
    if snr_bin_edges is None:
        snr_bin_edges = np.arange(0, 32, 3)
    centers = 0.5 * (snr_bin_edges[:-1] + snr_bin_edges[1:])
    snr_dbs = results["snr_dbs"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for task, label in [("modulation", "modulation"),
                        ("phase",      "phase noise"),
                        ("iq",         "IQ imbalance")]:
        accs = []
        for lo, hi in zip(snr_bin_edges[:-1], snr_bin_edges[1:]):
            mask = (snr_dbs >= lo) & (snr_dbs < hi)
            if mask.sum() == 0:
                accs.append(np.nan); continue
            p = results["preds"][task][mask]
            t = results["truths"][task][mask]
            accs.append((p == t).mean())
        ax.plot(centers, accs, "o-", lw=1.6, ms=6, label=label)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3); ax.legend()
    ax.set_title("CNN per-task accuracy vs SNR (test split)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_ood_vs_test(metrics_test, metrics_ood, outpath):
    """Side-by-side bar chart for the three non-SNR tasks."""
    tasks = ["modulation", "phase", "iq"]
    test_acc = [metrics_test["accuracy"][t] for t in tasks]
    ood_acc  = [metrics_ood["accuracy"][t]  for t in tasks]
    x = np.arange(len(tasks)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, test_acc, w, label="in-distribution (test)", color="steelblue")
    ax.bar(x + w/2, ood_acc,  w, label=f"OOD (held-out SNR={HELD_OUT_SNR_BIN})",
           color="darkorange")
    for xi, v in zip(x - w/2, test_acc): ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + w/2, ood_acc):  ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
    ax.legend(); ax.set_title("In-distribution vs OOD accuracy")
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_other_tasks(preds_test: dict, outpath: str):
    """Three small confusion matrices: phase, IQ imbalance, SNR bin."""
    panels = [
        ("phase", "Phase noise level", PHASE_LEVELS),
        ("iq",    "IQ imbalance level", IQ_LEVELS),
        ("snr",   "SNR bin", SNR_BINS_TRAIN),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (task, title, labels) in zip(axes, panels):
        p = preds_test["preds"][task]
        t = preds_test["truths"][task]
        if task == "snr":
            mask = t >= 0
            p, t = p[mask], t[mask]
        cm = confusion_matrix(t, p, labels=list(range(len(labels))))
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    cbar=False, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title)
    fig.suptitle("CNN — confusion matrices for impairment tasks", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def compute_severity_sensitivity(preds_test: dict, test_df: pd.DataFrame) -> dict:
    """
    Modulation classification accuracy broken down by impairment severity.
    Returns nested dict {"phase": {level: acc}, "iq": {level: acc}}.
    """
    mod_pred  = preds_test["preds"]["modulation"]
    mod_truth = preds_test["truths"]["modulation"]
    correct = (mod_pred == mod_truth)
    out = {"phase": {}, "iq": {}}
    for level in PHASE_LEVELS:
        mask = (test_df["phase_noise_level"].values == level)
        out["phase"][level] = float(correct[mask].mean()) if mask.sum() else float("nan")
    for level in IQ_LEVELS:
        mask = (test_df["iq_imbalance_level"].values == level)
        out["iq"][level] = float(correct[mask].mean()) if mask.sum() else float("nan")
    return out


def plot_severity_sensitivity(sens: dict, outpath: str):
    """Two bar charts: modulation accuracy by phase severity and by IQ severity."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, kind, levels, color, title in [
        (axes[0], "phase", PHASE_LEVELS, "steelblue",  "Modulation acc vs phase noise severity"),
        (axes[1], "iq",    IQ_LEVELS,    "darkorange", "Modulation acc vs I/Q imbalance severity"),
    ]:
        vals = [sens[kind][lvl] for lvl in levels]
        bars = ax.bar(levels, vals, color=color, width=0.6)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, v + .01, f"{v:.2f}",
                        ha="center", fontsize=10)
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
        ax.set_title(title); ax.set_ylabel("Accuracy" if kind == "phase" else "")
    fig.suptitle("CNN — sensitivity to impairment severity (test split)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Computational complexity helpers
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> dict[str, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def measure_inference_latency(model: nn.Module, loader: DataLoader,
                              device: str, n_warmup: int = 2) -> dict[str, float]:
    """Average forward-pass latency per image, after a few warmup batches."""
    model.eval()
    # warmup
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_warmup: break
            _ = model(x.to(device))
    # measure
    n_imgs = 0
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device))
            n_imgs += x.size(0)
    if device == "cuda": torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {
        "total_seconds":      elapsed,
        "images":             n_imgs,
        "ms_per_image":       1000.0 * elapsed / max(n_imgs, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", 
                        default=os.path.join(script_dir, "data", "labels.csv"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--save_dir", 
                        default=os.path.join(script_dir, "results"))
    args = parser.parse_args()

    # Ensure absolute paths if relative ones were provided via CLI
    if not os.path.isabs(args.labels_csv):
        args.labels_csv = os.path.join(script_dir, args.labels_csv)
    if not os.path.isabs(args.save_dir):
        args.save_dir = os.path.join(script_dir, args.save_dir)

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
    else:
        device = args.device
    print(f"device: {device}")

    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- data ----
    loaders, df = make_loaders(args.labels_csv, args.batch_size, args.num_workers)
    print(f"sizes: " + ", ".join(f"{k}={len(loaders[k].dataset)}" for k in loaders))

    # ---- model ----
    model = MultiHeadCNN().to(device)
    param_counts = count_parameters(model)
    print(f"params: total={param_counts['total']:,}   "
          f"trainable={param_counts['trainable']:,}")
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ---- training loop ----
    best_val_avg = -1.0
    history = []
    best_path = os.path.join(ckpt_dir, "cnn_best.pt")
    training_start = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        tr = step(model, loaders["train"], device, optimizer)
        va = step(model, loaders["val"],   device, optimizer=None)
        scheduler.step()
        avg_val_acc = np.mean(list(va["accuracy"].values()))
        history.append({"epoch": epoch,
                        "train_loss": tr["loss"], "val_loss": va["loss"],
                        "val_acc": va["accuracy"],
                        "lr": scheduler.get_last_lr()[0],
                        "seconds": time.time() - ep_t0})
        msg = (f"epoch {epoch:02d}  "
               f"train_loss={tr['loss']:.3f}  val_loss={va['loss']:.3f}  "
               f"val_acc=" + "/".join(f"{va['accuracy'][k]:.2f}" for k in TASKS)
               + f"  ({time.time()-ep_t0:.1f}s)")
        if avg_val_acc > best_val_avg:
            best_val_avg = avg_val_acc
            torch.save(model.state_dict(), best_path)
            msg += "  *"
        print(msg)
    total_training_time = time.time() - training_start
    print(f"\ntotal training time: {total_training_time:.1f}s  "
          f"({total_training_time/args.epochs:.1f}s / epoch)")

    # ---- load best and evaluate ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    metrics_test = step(model, loaders["test"], device, optimizer=None)
    metrics_ood  = step(model, loaders["ood_test"], device, optimizer=None)
    preds_test = collect_predictions(model, loaders["test"], device)

    print("\n=== TEST (in-distribution) ===")
    for k in TASKS:
        print(f"  {k:<11} acc = {metrics_test['accuracy'][k]:.3f}")
    print("=== OOD TEST (held-out SNR bin) ===")
    for k in TASKS:
        if k == "snr": continue
        print(f"  {k:<11} acc = {metrics_ood['accuracy'][k]:.3f}")

    # ---- inference latency on test split ----
    latency = measure_inference_latency(model, loaders["test"], device)
    print(f"\ninference latency: {latency['ms_per_image']:.2f} ms/image "
          f"({latency['images']} images in {latency['total_seconds']:.2f}s)")

    # ---- sensitivity to impairment severity ----
    test_df = loaders["test"].dataset.df.reset_index(drop=True)
    severity = compute_severity_sensitivity(preds_test, test_df)
    print("\nsensitivity to impairment severity (modulation accuracy):")
    for kind in ("phase", "iq"):
        cells = "  ".join(f"{lvl}={severity[kind][lvl]:.3f}"
                          for lvl in (PHASE_LEVELS if kind == "phase" else IQ_LEVELS))
        print(f"  by {kind:<5}:  {cells}")

    # ---- save metrics ----
    out = {
        "device":           device,
        "epochs":           args.epochs,
        "history":          history,
        "test":             {"loss": metrics_test["loss"],
                             "accuracy": metrics_test["accuracy"]},
        "ood_test":         {"loss": metrics_ood["loss"],
                             "accuracy": metrics_ood["accuracy"]},
        "best_val_avg_acc": best_val_avg,
        "params":           param_counts,
        "training_time_s":  total_training_time,
        "training_time_per_epoch_s": total_training_time / args.epochs,
        "inference_latency": latency,
        "severity_sensitivity": severity,
    }
    with open(os.path.join(args.save_dir, "cnn_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.save_dir}/cnn_metrics.json")

    # ---- plots ----
    plot_confusion_modulation(
        preds_test["preds"]["modulation"],
        preds_test["truths"]["modulation"],
        os.path.join(args.save_dir, "05_cnn_confusion_modulation.png"))
    plot_accuracy_vs_snr(
        preds_test,
        os.path.join(args.save_dir, "06_cnn_accuracy_vs_snr.png"))
    plot_ood_vs_test(
        metrics_test, metrics_ood,
        os.path.join(args.save_dir, "07_cnn_ood_vs_test.png"))
    plot_confusion_other_tasks(
        preds_test,
        os.path.join(args.save_dir, "08_cnn_confusion_other.png"))
    plot_severity_sensitivity(
        severity,
        os.path.join(args.save_dir, "09_cnn_severity_sensitivity.png"))
    for name in ("05_cnn_confusion_modulation.png", "06_cnn_accuracy_vs_snr.png",
                 "07_cnn_ood_vs_test.png", "08_cnn_confusion_other.png",
                 "09_cnn_severity_sensitivity.png"):
        print(f"saved -> {args.save_dir}/{name}")


if __name__ == "__main__":
    main()
