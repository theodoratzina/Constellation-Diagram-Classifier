"""
ood_intermediate.py
===================

Out-of-distribution evaluation at INTERMEDIATE impairment levels.

The labelled dataset uses three discrete severity levels per impairment
(none / mild / severe). The project spec asks us to also test generalization
at "slightly different impairment levels". This script generates a fresh
small dataset where:

  - phase noise sigma_phi is drawn from {1, 3, 4, 5} deg
    (training values were {0, 2, 6})
  - IQ amplitude eps is drawn from {0.025, 0.08, 0.10, 0.12}
    (training values were {0.0, 0.05, 0.15})
  - IQ phase psi is drawn from {1, 4, 6} deg
    (training values were {0, 2, 8})

For each sample, the ground-truth severity label is the NEAREST training
level (none / mild / severe). The script then loads both trained models
and reports their modulation, phase, and IQ accuracies on this set.

Outputs:
    results/ood_intermediate_metrics.json
    results/13_ood_intermediate.png       CNN vs VLM accuracy bar chart

Run:
    python ood_intermediate.py                # uses default checkpoints
    python ood_intermediate.py --n_samples 320
    python ood_intermediate.py --skip_vlm     # CNN-only (no VLM trained yet)
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
import torch
from PIL import Image

from signals import generate_symbols, apply_channel, render_constellation, resolve_path


# ---------------------------------------------------------------------------
# Label discretization
# ---------------------------------------------------------------------------
PHASE_LEVELS = ["none", "mild", "severe"]
IQ_LEVELS    = ["none", "mild", "severe"]
MODULATIONS = [
    "4-ASK", "8-ASK", "BPSK", "QPSK",
    "4-HQAM", "16-HQAM", "64-HQAM",
    "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM",
    "16-APSK", "32-APSK", "64-APSK", "128-APSK",
]

# midpoints between training severity values, used as boundaries
PHASE_BOUNDS = (1.0, 4.0)        # <1 -> none, <4 -> mild, else severe
IQ_EPS_BOUNDS = (0.025, 0.10)    # similar partition for IQ amplitude


def phase_level_from_deg(sigma_deg: float) -> str:
    if sigma_deg < PHASE_BOUNDS[0]: return "none"
    if sigma_deg < PHASE_BOUNDS[1]: return "mild"
    return "severe"


def iq_level_from_eps(eps: float) -> str:
    if eps < IQ_EPS_BOUNDS[0]: return "none"
    if eps < IQ_EPS_BOUNDS[1]: return "mild"
    return "severe"


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate_intermediate_dataset(n_samples: int,
                                  out_dir: str,
                                  seed: int = 1234) -> pd.DataFrame:
    """Generate `n_samples` images at intermediate impairment values."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    PHASE_DEG_CHOICES = [1.0, 3.0, 4.0, 5.0]
    EPS_CHOICES       = [0.025, 0.08, 0.10, 0.12]
    PSI_DEG_CHOICES   = [1.0, 4.0, 6.0]

    rows = []
    print(f"Generating {n_samples} intermediate-OOD samples...")
    for idx in range(n_samples):
        mod        = MODULATIONS[idx % len(MODULATIONS)]
        snr_db     = rng.uniform(10.0, 20.0)               # in-distribution SNR
        sigma_deg  = float(rng.choice(PHASE_DEG_CHOICES))
        eps        = float(rng.choice(EPS_CHOICES))
        psi_deg    = float(rng.choice(PSI_DEG_CHOICES))

        s = generate_symbols(mod, 2048, rng=rng)
        rx = apply_channel(
            s, snr_db=snr_db,
            phase_noise_sigma_rad=np.deg2rad(sigma_deg),
            iq_eps=eps, iq_psi_rad=np.deg2rad(psi_deg),
            rng=rng,
        )
        img = render_constellation(rx, mode="histogram",
                                   image_size=224, view_limit=2.5)
        fn = f"ood_intermediate_{idx:05d}.png"
        path = os.path.join(out_dir, fn)
        img.save(path)

        rows.append({
            "image_path":          path,
            "modulation":          mod,
            "snr_db":              round(snr_db, 2),
            "phase_noise_std_deg": sigma_deg,
            "iq_eps":              eps,
            "iq_psi_deg":          psi_deg,
            "phase_noise_level":   phase_level_from_deg(sigma_deg),
            "iq_imbalance_level":  iq_level_from_eps(eps),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "intermediate_labels.csv"), index=False)
    print(f"saved {len(df)} images + labels to {out_dir}")
    return df


# ---------------------------------------------------------------------------
# CNN evaluation
# ---------------------------------------------------------------------------
def evaluate_cnn(df: pd.DataFrame, checkpoint: str, device: str) -> dict:
    """Run the trained CNN on the intermediate-OOD dataset."""
    from train_cnn import MultiHeadCNN, LABEL_TO_IDX
    from torchvision import transforms

    print(f"\nLoading CNN checkpoint {checkpoint}...")
    model = MultiHeadCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    idx_to_mod   = {v: k for k, v in LABEL_TO_IDX["modulation"].items()}
    idx_to_phase = {v: k for k, v in LABEL_TO_IDX["phase"].items()}
    idx_to_iq    = {v: k for k, v in LABEL_TO_IDX["iq"].items()}

    preds = {"modulation": [], "phase": [], "iq": []}
    print(f"Evaluating CNN on {len(df)} samples...")
    with torch.no_grad():
        for _, row in df.iterrows():
            img = Image.open(resolve_path(row["image_path"])).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            logits = model(x)
            preds["modulation"].append(idx_to_mod[int(logits["modulation"].argmax(1))])
            preds["phase"].append(idx_to_phase[int(logits["phase"].argmax(1))])
            preds["iq"].append(idx_to_iq[int(logits["iq"].argmax(1))])

    truths = {
        "modulation": df["modulation"].tolist(),
        "phase":      df["phase_noise_level"].tolist(),
        "iq":         df["iq_imbalance_level"].tolist(),
    }
    accuracy = {t: float(np.mean(np.asarray(preds[t]) == np.asarray(truths[t])))
                for t in preds}
    print("CNN accuracy:")
    for t, a in accuracy.items():
        print(f"  {t:<11}  {a:.3f}")
    return {"preds": preds, "truths": truths, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# VLM evaluation
# ---------------------------------------------------------------------------
def evaluate_vlm(df: pd.DataFrame, adapter_dir: str, base_model_id: str,
                 device: str, eval_batch_size: int = 4) -> dict:
    """Run the LoRA-fine-tuned VLM on the intermediate-OOD dataset."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    from train_vlm import VLMConstellationDataset, VLMCollator, parse_answer
    from torch.utils.data import DataLoader

    print(f"\nLoading VLM ({base_model_id}) + adapter from {adapter_dir}...")
    processor = AutoProcessor.from_pretrained(adapter_dir)
    base = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    model.eval()

    # build the same eval interface
    df_eval = df.rename(columns={}).copy()
    df_eval["snr_bin"] = "ood_intermediate"
    ds = VLMConstellationDataset(df_eval, mode="eval",
                                 tasks=["modulation", "phase", "iq"])
    loader = DataLoader(ds, batch_size=eval_batch_size, num_workers=0,
                        collate_fn=VLMCollator(processor, mode="eval"))

    preds  = {"modulation": [], "phase": [], "iq": []}
    truths = {"modulation": [], "phase": [], "iq": []}

    print(f"Evaluating VLM (this generates {len(ds)} answers)...")
    with torch.no_grad():
        for batch in loader:
            tasks   = batch.pop("_tasks")
            answers = batch.pop("_answers")
            batch.pop("_snr_dbs", None)
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            gen = model.generate(**batch, max_new_tokens=16, do_sample=False)
            in_len = batch["input_ids"].shape[1]
            texts = processor.batch_decode(gen[:, in_len:], skip_special_tokens=True)
            for raw, gt, task in zip(texts, answers, tasks):
                preds[task].append(parse_answer(raw, task))
                truths[task].append(gt)

    accuracy = {t: float(np.mean(np.asarray(preds[t]) == np.asarray(truths[t])))
                for t in preds}
    print("VLM accuracy:")
    for t, a in accuracy.items():
        print(f"  {t:<11}  {a:.3f}")
    return {"preds": preds, "truths": truths, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_results(cnn_acc: dict | None, vlm_acc: dict | None, outpath: str):
    tasks = ["modulation", "phase", "iq"]
    cnn_vals = [cnn_acc[t] for t in tasks] if cnn_acc else [np.nan]*3
    vlm_vals = [vlm_acc[t] for t in tasks] if vlm_acc else [np.nan]*3
    x = np.arange(len(tasks)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    if cnn_acc is not None:
        ax.bar(x - w/2, cnn_vals, w, label="CNN", color="steelblue")
        for xi, v in zip(x - w/2, cnn_vals):
            ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=9)
    if vlm_acc is not None:
        ax.bar(x + w/2, vlm_vals, w, label="VLM", color="seagreen")
        for xi, v in zip(x + w/2, vlm_vals):
            ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
    ax.set_title("Generalization to intermediate impairment levels")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=320,
                        help="20 per modulation by default")
    parser.add_argument("--out_dir", 
                        default=os.path.join(script_dir, "data", "intermediate"))
    parser.add_argument("--cnn_ckpt", 
                        default=os.path.join(script_dir, "results", "checkpoints", "cnn_best.pt"))
    parser.add_argument("--vlm_adapter", 
                        default=os.path.join(script_dir, "results", "checkpoints", "vlm_lora"))
    parser.add_argument("--vlm_base_id",
                        default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--save_dir", 
                        default=os.path.join(script_dir, "results"))
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--skip_cnn", action="store_true")
    parser.add_argument("--skip_vlm", action="store_true")
    args = parser.parse_args()

    # Ensure absolute paths if relative ones were provided via CLI
    for attr in ["out_dir", "cnn_ckpt", "vlm_adapter", "save_dir"]:
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.join(script_dir, val))

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- generate intermediate-OOD dataset ----
    df = generate_intermediate_dataset(args.n_samples, args.out_dir)

    # ---- evaluate CNN ----
    cnn_res = None
    if not args.skip_cnn:
        if os.path.exists(args.cnn_ckpt):
            cnn_res = evaluate_cnn(df, args.cnn_ckpt, device)
        else:
            print(f"!! CNN checkpoint not found at {args.cnn_ckpt} — skipping CNN")

    # ---- evaluate VLM ----
    vlm_res = None
    if not args.skip_vlm:
        if os.path.isdir(args.vlm_adapter):
            try:
                vlm_res = evaluate_vlm(df, args.vlm_adapter, args.vlm_base_id, device)
            except Exception as e:
                print(f"!! VLM evaluation failed: {type(e).__name__}: {e}")
        else:
            print(f"!! VLM adapter dir not found at {args.vlm_adapter} — skipping VLM")

    # ---- save ----
    metrics = {
        "device":   device,
        "n_samples": args.n_samples,
        "cnn_accuracy": cnn_res["accuracy"] if cnn_res else None,
        "vlm_accuracy": vlm_res["accuracy"] if vlm_res else None,
        "notes": ("Intermediate-OOD: phase_noise_std_deg in {1,3,4,5}, "
                  "iq_eps in {.025,.08,.10,.12}, iq_psi_deg in {1,4,6}. "
                  "Training values: phase {0,2,6}, eps {0,.05,.15}, psi {0,2,8}. "
                  "Ground-truth severity = nearest training level."),
    }
    with open(os.path.join(args.save_dir, "ood_intermediate_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nsaved -> {args.save_dir}/ood_intermediate_metrics.json")

    plot_results(
        cnn_res["accuracy"] if cnn_res else None,
        vlm_res["accuracy"] if vlm_res else None,
        os.path.join(args.save_dir, "13_ood_intermediate.png"))
    print(f"saved -> {args.save_dir}/13_ood_intermediate.png")


if __name__ == "__main__":
    main()
