"""
compare.py
==========

Side-by-side comparison of the CNN and VLM models trained in Part II.

Reads:
    results/cnn_metrics.json
    results/vlm_metrics.json

Produces:
    results/11_comparison_per_task.png       grouped bar chart of per-task acc
    results/12_comparison_ood_gap.png        in-distribution vs OOD per model
    results/14_comparison_complexity.png     computational complexity comparison
    results/15_comparison_intermediate.png   intermediate-OOD comparison (optional)
    results/comparison_table.csv             numeric accuracy table for the report
    results/comparison_table.md              same as above, markdown formatted
    results/comparison_complexity.csv        param counts, training time, latency
    results/comparison_summary.txt           plain-text summary printed to stdout

Run:
    python compare.py                              # use default paths
    python compare.py --cnn other/cnn_metrics.json # custom paths
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Tasks that BOTH models predict (CNN does not learn the binary 'jamming' task,
# so we compare on the four shared ones).
SHARED_TASKS = ["modulation", "phase", "iq", "snr"]
OOD_TASKS    = ["modulation", "phase", "iq"]   # SNR is held-out by construction

CNN_COLOR = "steelblue"
VLM_COLOR = "seagreen"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_metrics(path: str) -> dict | None:
    if not os.path.exists(path):
        print(f"!! missing: {path}  (will be skipped)")
        return None
    with open(path) as f:
        return json.load(f)


def extract_accuracy(metrics: dict, kind: str) -> dict:
    """
    Return {task: accuracy} for `kind` in {"test", "ood_test"}.

    Handles both layouts produced by train_cnn.py and train_vlm.py.
    """
    if metrics is None:
        return {}

    # CNN layout:  metrics["test"]["accuracy"]
    # VLM layout:  metrics["test_accuracy"]
    if kind in metrics and isinstance(metrics[kind], dict) and "accuracy" in metrics[kind]:
        return metrics[kind]["accuracy"]
    key = f"{kind}_accuracy"
    if key in metrics:
        return metrics[key]
    return {}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_per_task(cnn_test: dict, vlm_test: dict, outpath: str):
    """Grouped bar chart: CNN vs VLM accuracy per task (in-distribution test)."""
    tasks_present = [t for t in SHARED_TASKS if t in cnn_test or t in vlm_test]
    cnn_vals = [cnn_test.get(t, np.nan) for t in tasks_present]
    vlm_vals = [vlm_test.get(t, np.nan) for t in tasks_present]

    x = np.arange(len(tasks_present)); w = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - w / 2, cnn_vals, w, label="CNN (multi-head ResNet18)", color=CNN_COLOR)
    ax.bar(x + w / 2, vlm_vals, w, label="VLM (SmolVLM-256M + LoRA)",  color=VLM_COLOR)

    for xi, v in zip(x - w / 2, cnn_vals):
        if not np.isnan(v):
            ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + w / 2, vlm_vals):
        if not np.isnan(v):
            ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(tasks_present)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
    ax.set_title("Per-task accuracy on test split (in-distribution)")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_ood_gap(cnn_test, cnn_ood, vlm_test, vlm_ood, outpath: str):
    """
    Two grouped subplots, one per model: in-distribution vs OOD per task.
    Lets you read off the generalization gap visually.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    panels = [
        ("CNN",  cnn_test, cnn_ood, CNN_COLOR),
        ("VLM",  vlm_test, vlm_ood, VLM_COLOR),
    ]
    for ax, (name, t_acc, o_acc, color) in zip(axes, panels):
        if not t_acc and not o_acc:
            ax.text(0.5, 0.5, f"{name}: no metrics found",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        tasks = OOD_TASKS
        in_v  = [t_acc.get(t, np.nan) for t in tasks]
        ood_v = [o_acc.get(t, np.nan) for t in tasks]
        x = np.arange(len(tasks)); w = 0.35
        ax.bar(x - w/2, in_v,  w, label="in-distribution", color=color, alpha=0.95)
        ax.bar(x + w/2, ood_v, w, label="OOD (held-out SNR)",
               color=color, alpha=0.45, hatch="//")
        for xi, v in zip(x - w/2, in_v):
            if not np.isnan(v): ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=9)
        for xi, v in zip(x + w/2, ood_v):
            if not np.isnan(v): ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(tasks)
        ax.set_ylim(0, 1.1); ax.set_title(name); ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Generalization gap: in-distribution vs OOD (held-out SNR bin)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Numeric tables
# ---------------------------------------------------------------------------
def build_table(cnn_test, cnn_ood, vlm_test, vlm_ood) -> pd.DataFrame:
    rows = []
    for t in SHARED_TASKS:
        rows.append({
            "task":           t,
            "CNN test":       cnn_test.get(t, np.nan),
            "VLM test":       vlm_test.get(t, np.nan),
            "CNN OOD":        cnn_ood.get(t, np.nan) if t != "snr" else np.nan,
            "VLM OOD":        vlm_ood.get(t, np.nan) if t != "snr" else np.nan,
            "CNN gap (in-OOD)": ((cnn_test.get(t, np.nan) - cnn_ood.get(t, np.nan))
                                  if t != "snr" else np.nan),
            "VLM gap (in-OOD)": ((vlm_test.get(t, np.nan) - vlm_ood.get(t, np.nan))
                                  if t != "snr" else np.nan),
        })
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids the tabulate dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.3f}" if isinstance(v, float) and not np.isnan(v)
                         else ("—" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Computational complexity comparison
# ---------------------------------------------------------------------------
def build_complexity_table(cnn_m: dict | None, vlm_m: dict | None) -> pd.DataFrame:
    """Side-by-side table: params, training time, inference latency, etc."""
    def get(d, *keys, default=None):
        if d is None: return default
        for k in keys:
            if d is None or k not in d: return default
            d = d[k]
        return d

    rows = [
        ("Total parameters",         get(cnn_m, "params", "total"),
                                     get(vlm_m, "params", "total")),
        ("Trainable parameters",     get(cnn_m, "params", "trainable"),
                                     get(vlm_m, "params", "trainable")),
        ("Trainable fraction (%)",
            (100.0 * get(cnn_m, "params", "trainable") /
                     get(cnn_m, "params", "total")) if get(cnn_m, "params", "total") else None,
            (100.0 * get(vlm_m, "params", "trainable") /
                     get(vlm_m, "params", "total")) if get(vlm_m, "params", "total") else None),
        ("Training time (s)",        get(cnn_m, "training_time_s"),
                                     get(vlm_m, "training_time_s")),
        ("Training time / epoch (s)",get(cnn_m, "training_time_per_epoch_s"),
                                     get(vlm_m, "training_time_per_epoch_s")),
        ("Inference latency (ms/image)",
                                     get(cnn_m, "inference_latency", "ms_per_image"),
                                     get(vlm_m, "inference_latency", "ms_per_image")),
        ("Device",                   get(cnn_m, "device"), get(vlm_m, "device")),
        ("Epochs",                   get(cnn_m, "epochs"), get(vlm_m, "epochs")),
    ]
    return pd.DataFrame(rows, columns=["Metric", "CNN", "VLM"])


def plot_complexity(complexity_df: pd.DataFrame, outpath: str):
    """Three horizontal bar panels: params, training time, inference latency."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))

    def get_row(name):
        row = complexity_df[complexity_df["Metric"] == name]
        if row.empty: return None, None
        return row.iloc[0]["CNN"], row.iloc[0]["VLM"]

    panels = [
        ("Trainable parameters", "Trainable parameters",            "log",    ""),
        ("Training time (s)",    "Training time (seconds)",         "linear", "s"),
        ("Inference latency (ms/image)",
                                 "Inference latency (ms / image)",  "linear", "ms"),
    ]
    for ax, (key, title, scale, suffix) in zip(axes, panels):
        cnn_v, vlm_v = get_row(key)
        names, vals, colors = [], [], []
        if cnn_v is not None: names.append("CNN"); vals.append(cnn_v); colors.append("steelblue")
        if vlm_v is not None: names.append("VLM"); vals.append(vlm_v); colors.append("seagreen")
        bars = ax.barh(names, vals, color=colors)
        for b, v in zip(bars, vals):
            if v is None: continue
            fmt = (f"{int(v):,}" if scale == "log"
                   else (f"{v:.1f} {suffix}" if isinstance(v, (int, float)) else str(v)))
            ax.text(v, b.get_y() + b.get_height()/2, "  " + fmt,
                    va="center", fontsize=10)
        ax.set_title(title)
        if scale == "log" and vals:
            ax.set_xscale("log")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Computational complexity: CNN vs VLM", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_intermediate_ood(metrics_path: str, outpath: str) -> bool:
    """Optional: side-by-side bar chart for intermediate-impairment-level eval."""
    if not os.path.exists(metrics_path):
        return False
    with open(metrics_path) as f:
        m = json.load(f)
    cnn_acc = m.get("cnn_accuracy") or {}
    vlm_acc = m.get("vlm_accuracy") or {}
    tasks = ["modulation", "phase", "iq"]
    cnn_vals = [cnn_acc.get(t, np.nan) for t in tasks]
    vlm_vals = [vlm_acc.get(t, np.nan) for t in tasks]
    x = np.arange(len(tasks)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, cnn_vals, w, label="CNN", color="steelblue")
    ax.bar(x + w/2, vlm_vals, w, label="VLM", color="seagreen")
    for xi, v in zip(x - w/2, cnn_vals):
        if not np.isnan(v): ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + w/2, vlm_vals):
        if not np.isnan(v): ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
    ax.set_title("Generalization to intermediate impairment levels (OOD)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Plain-text summary
# ---------------------------------------------------------------------------
def write_summary(cnn_metrics, vlm_metrics, table: pd.DataFrame,
                  complexity_df: pd.DataFrame, outpath: str):
    cnn_avg = np.nanmean([table.loc[i, "CNN test"]
                          for i in range(len(table)) if not np.isnan(table.loc[i, "CNN test"])])
    vlm_avg = np.nanmean([table.loc[i, "VLM test"]
                          for i in range(len(table)) if not np.isnan(table.loc[i, "VLM test"])])
    cnn_gap = np.nanmean(table["CNN gap (in-OOD)"].dropna())
    vlm_gap = np.nanmean(table["VLM gap (in-OOD)"].dropna())

    lines = []
    lines.append("=" * 70)
    lines.append("CNN vs VLM — final comparison")
    lines.append("=" * 70)
    lines.append("")

    lines.append("Models")
    lines.append("-" * 70)
    if cnn_metrics:
        lines.append(f"  CNN: multi-head ResNet18 ({cnn_metrics.get('epochs', '?')} epochs"
                     f", device={cnn_metrics.get('device', '?')})")
    else:
        lines.append("  CNN: metrics not found")
    if vlm_metrics:
        lines.append(f"  VLM: {vlm_metrics.get('model_id', '?')} + LoRA "
                     f"({vlm_metrics.get('epochs', '?')} epochs"
                     f", device={vlm_metrics.get('device', '?')})")
    else:
        lines.append("  VLM: metrics not found")
    lines.append("")

    lines.append("Per-task accuracy")
    lines.append("-" * 70)
    lines.append(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    lines.append("")

    lines.append("Computational complexity")
    lines.append("-" * 70)
    def fmt_cell(v):
        if v is None: return "—"
        if isinstance(v, (int,)) or (isinstance(v, float) and v.is_integer()):
            return f"{int(v):,}"
        if isinstance(v, float):  return f"{v:.2f}"
        return str(v)
    cdf = complexity_df.copy()
    cdf["CNN"] = cdf["CNN"].map(fmt_cell)
    cdf["VLM"] = cdf["VLM"].map(fmt_cell)
    lines.append(cdf.to_string(index=False))
    lines.append("")

    lines.append("Aggregate")
    lines.append("-" * 70)
    lines.append(f"  Mean in-distribution accuracy : CNN={cnn_avg:.3f}   VLM={vlm_avg:.3f}")
    lines.append(f"  Mean generalization gap (OOD) : CNN={cnn_gap:+.3f}  VLM={vlm_gap:+.3f}")
    lines.append("  (positive gap = drop on OOD)")
    lines.append("")

    lines.append("Interpretation notes for the report")
    lines.append("-" * 70)
    if not np.isnan(cnn_avg) and not np.isnan(vlm_avg):
        better = "CNN" if cnn_avg > vlm_avg else "VLM"
        lines.append(f"  - On in-distribution test, the {better} achieves the higher mean accuracy")
        lines.append(f"    across the {len(SHARED_TASKS)} shared tasks.")
    if not np.isnan(cnn_gap) and not np.isnan(vlm_gap):
        robust = "CNN" if cnn_gap < vlm_gap else "VLM"
        lines.append(f"  - On the held-out SNR bin, the {robust} shows the smaller mean drop,")
        lines.append(f"    suggesting better generalization to unseen impairment conditions.")
    lines.append("  - The modulation task is the most challenging (16 classes vs 3 for")
    lines.append("    phase/IQ); confusion is highest among high-order QAM/APSK at low SNR.")
    lines.append("=" * 70)

    text = "\n".join(lines)
    with open(outpath, "w") as f:
        f.write(text)
    print("\n" + text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn", 
                        default=os.path.join(script_dir, "results", "cnn_metrics.json"))
    parser.add_argument("--vlm", 
                        default=os.path.join(script_dir, "results", "vlm_metrics.json"))
    parser.add_argument("--save_dir", 
                        default=os.path.join(script_dir, "results"))
    args = parser.parse_args()

    # Ensure absolute paths if relative ones were provided via CLI
    for attr in ["cnn", "vlm", "save_dir"]:
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.join(script_dir, val))

    cnn_metrics = load_metrics(args.cnn)
    vlm_metrics = load_metrics(args.vlm)
    if cnn_metrics is None and vlm_metrics is None:
        print("No metrics to compare. Run train_cnn.py and train_vlm.py first.")
        return

    cnn_test = extract_accuracy(cnn_metrics, "test")
    cnn_ood  = extract_accuracy(cnn_metrics, "ood_test")
    vlm_test = extract_accuracy(vlm_metrics, "test")
    vlm_ood  = extract_accuracy(vlm_metrics, "ood_test")

    os.makedirs(args.save_dir, exist_ok=True)
    table = build_table(cnn_test, cnn_ood, vlm_test, vlm_ood)
    table.to_csv(os.path.join(args.save_dir, "comparison_table.csv"), index=False)
    with open(os.path.join(args.save_dir, "comparison_table.md"), "w") as f:
        f.write(df_to_markdown(table))
    print(f"saved -> {args.save_dir}/comparison_table.csv")
    print(f"saved -> {args.save_dir}/comparison_table.md")

    # ---- computational complexity ----
    complexity_df = build_complexity_table(cnn_metrics, vlm_metrics)
    complexity_df.to_csv(os.path.join(args.save_dir, "comparison_complexity.csv"),
                         index=False)
    print(f"saved -> {args.save_dir}/comparison_complexity.csv")

    plot_per_task(cnn_test, vlm_test,
                  os.path.join(args.save_dir, "11_comparison_per_task.png"))
    plot_ood_gap(cnn_test, cnn_ood, vlm_test, vlm_ood,
                 os.path.join(args.save_dir, "12_comparison_ood_gap.png"))
    plot_complexity(complexity_df,
                    os.path.join(args.save_dir, "14_comparison_complexity.png"))
    print(f"saved -> {args.save_dir}/11_comparison_per_task.png")
    print(f"saved -> {args.save_dir}/12_comparison_ood_gap.png")
    print(f"saved -> {args.save_dir}/14_comparison_complexity.png")

    # ---- intermediate-OOD (optional, if ood_intermediate.py was run) ----
    interm_path = os.path.join(args.save_dir, "ood_intermediate_metrics.json")
    if plot_intermediate_ood(interm_path,
            os.path.join(args.save_dir, "15_comparison_intermediate.png")):
        print(f"saved -> {args.save_dir}/15_comparison_intermediate.png")
    else:
        print(f"   (skip 15_comparison_intermediate.png — run ood_intermediate.py)")

    write_summary(cnn_metrics, vlm_metrics, table, complexity_df,
                  os.path.join(args.save_dir, "comparison_summary.txt"))


if __name__ == "__main__":
    main()
