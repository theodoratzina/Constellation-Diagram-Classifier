"""
train_vlm.py
============

Parameter-efficient fine-tuning of a lightweight vision-language model
on the constellation Q&A dataset built by build_dataset.py.

Default backbone: HuggingFaceTB/SmolVLM-256M-Instruct (~256M params).
Trained with LoRA adapters on the language-model attention projections,
leaving the vision encoder frozen. This typically reduces the trainable
parameter count to under 1% of the full model.

Outputs (under results/):
    checkpoints/vlm_lora/             trained LoRA adapter weights + processor
    vlm_metrics.json                  per-task accuracy on test and ood_test,
                                      model size, training time, inference latency
    08_vlm_confusion_modulation.png   modulation confusion matrix
    09_vlm_accuracy_vs_snr.png        per-task accuracy as function of SNR
    10_vlm_ood_vs_test.png            in-distribution vs OOD bar chart
    11_vlm_confusion_other.png        phase / IQ confusion matrices
    12_vlm_severity_sensitivity.png   modulation accuracy vs impairment severity

Run:
    python train_vlm.py                                # full training
    python train_vlm.py --epochs 1 --batch_size 2      # quick smoke test
    python train_vlm.py --model_id <other-vlm-id>      # try another backbone
    python train_vlm.py --device cpu                   # force CPU (slow)
"""
from __future__ import annotations

import os
import json
import time
import argparse
import random
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
from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import confusion_matrix

from signals import resolve_path


# ---------------------------------------------------------------------------
# Task definitions (must mirror those in build_dataset.py:make_qa_pairs)
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

HELD_OUT_SNR_BIN = "low_mid"

QUESTIONS = {
    "modulation": "What modulation is used?",
    "phase":      "What is the level of phase noise?",
    "iq":         "What is the level of I/Q imbalance?",
    "jamming":    "Is there external interference?",
    "snr":        "What is the SNR range?",
}

# Valid answer vocabularies, used for constrained matching at inference.
VALID_ANSWERS = {
    "modulation": MODULATIONS,
    "phase":      PHASE_LEVELS,
    "iq":         IQ_LEVELS,
    "jamming":    ["yes", "no"],
    "snr":        SNR_BINS,
}


def label_to_question_answer(row: pd.Series, task: str) -> tuple[str, str]:
    """Pull the (question, answer) pair for `task` from a labels.csv row."""
    q = QUESTIONS[task]
    if task == "modulation":     a = row["modulation"]
    elif task == "phase":        a = row["phase_noise_level"]
    elif task == "iq":           a = row["iq_imbalance_level"]
    elif task == "jamming":      a = "yes" if int(row["jamming"]) == 1 else "no"
    elif task == "snr":          a = row["snr_bin"]
    else: raise ValueError(task)
    return q, a


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VLMConstellationDataset(Dataset):
    """
    For each row in labels.csv, sample one task at random and emit
    (image, question, answer). This expands one image into many Q/A pairs
    across epochs without storing them all in memory.

    For evaluation we iterate deterministically over all five tasks for
    every image, so accuracy can be reported per task.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 tasks: list[str] | None = None,
                 mode: str = "train",
                 seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.tasks = tasks or list(QUESTIONS.keys())
        self.mode = mode
        if mode == "eval":
            # one row per (image, task)
            entries = []
            for i in range(len(self.df)):
                for t in self.tasks:
                    entries.append((i, t))
            self.entries = entries
        else:
            self.rng = random.Random(seed)

    def __len__(self):
        return len(self.entries) if self.mode == "eval" else len(self.df)

    def __getitem__(self, i):
        if self.mode == "eval":
            row_idx, task = self.entries[i]
        else:
            row_idx = i
            task = self.rng.choice(self.tasks)
        row = self.df.iloc[row_idx]
        q, a = label_to_question_answer(row, task)
        return {
            "image":    Image.open(resolve_path(row["image_path"])).convert("RGB"),
            "question": q,
            "answer":   a,
            "task":     task,
            "snr_db":   float(row["snr_db"]),
            "snr_bin":  row["snr_bin"],
        }


# ---------------------------------------------------------------------------
# Collator — builds the chat messages and tokenizes them with the processor
# ---------------------------------------------------------------------------
class VLMCollator:
    """
    Build training tensors. The training target is the answer text;
    everything before it (system prompt + image + question) is masked
    in the loss so the model only learns to produce the answer.
    """

    def __init__(self, processor, mode: str = "train"):
        self.processor = processor
        self.mode = mode

    def _build_messages(self, question: str, answer: str | None):
        user_content = [{"type": "image"}, {"type": "text", "text": question}]
        msgs = [{"role": "user", "content": user_content}]
        if answer is not None:
            msgs.append({"role": "assistant",
                         "content": [{"type": "text", "text": answer}]})
        return msgs

    def __call__(self, batch: list[dict]):
        images   = [b["image"]    for b in batch]
        questions= [b["question"] for b in batch]
        answers  = [b["answer"]   for b in batch]

        if self.mode == "train":
            # full chat including the answer
            prompts = [
                self.processor.apply_chat_template(
                    self._build_messages(q, a), add_generation_prompt=False)
                for q, a in zip(questions, answers)
            ]
            # also build a "prompt-only" version so we can mask its tokens
            prompts_only = [
                self.processor.apply_chat_template(
                    self._build_messages(q, None), add_generation_prompt=True)
                for q in questions
            ]
            enc = self.processor(text=prompts, images=[[im] for im in images],
                                 return_tensors="pt", padding=True)
            # mask labels: keep only the answer tokens
            labels = enc["input_ids"].clone()
            for i, p_only in enumerate(prompts_only):
                p_len = len(self.processor.tokenizer(
                    p_only, add_special_tokens=False)["input_ids"])
                labels[i, :p_len] = -100
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            enc["labels"] = labels
            return enc
        else:
            # prompt-only for generation; ground truth carried separately
            prompts = [
                self.processor.apply_chat_template(
                    self._build_messages(q, None), add_generation_prompt=True)
                for q in questions
            ]
            enc = self.processor(text=prompts, images=[[im] for im in images],
                                 return_tensors="pt", padding=True)
            enc["_answers"] = answers
            enc["_tasks"]   = [b["task"]    for b in batch]
            enc["_snr_dbs"] = [b["snr_db"]  for b in batch]
            return enc


# ---------------------------------------------------------------------------
# Constrained answer parsing
# ---------------------------------------------------------------------------
def parse_answer(raw: str, task: str) -> str:
    """
    Pick the valid label that best matches `raw`. Strategy:
      1. case-insensitive exact substring match
      2. case-insensitive starts-with
      3. fallback: stripped raw text
    """
    candidates = VALID_ANSWERS[task]
    raw_clean = raw.strip().lower()
    # exact substring (longest match wins so '64-QAM' beats '4-QAM' etc.)
    hits = [c for c in candidates if c.lower() in raw_clean]
    if hits:
        return max(hits, key=len)
    for c in candidates:
        if raw_clean.startswith(c.lower()):
            return c
    return raw.strip()


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_loop(model, processor, loader, optimizer, device, accum_steps: int = 1):
    model.train()
    running = 0.0
    n = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / accum_steps
        loss.backward()
        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        running += out.loss.item() * batch["input_ids"].size(0)
        n       += batch["input_ids"].size(0)
    return running / max(n, 1)


@torch.no_grad()
def generate_and_score(model, processor, loader, device,
                       max_new_tokens: int = 16) -> dict:
    """
    Run inference on every (image, task) pair in the eval loader and return
    per-sample predictions, ground truths and SNR values.
    """
    model.eval()
    preds  = defaultdict(list)
    truths = defaultdict(list)
    snr_dbs = defaultdict(list)
    for batch in tqdm(loader, desc="generate", leave=False):
        tasks    = batch.pop("_tasks")
        answers  = batch.pop("_answers")
        snr_list = batch.pop("_snr_dbs")
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        gen = model.generate(**batch, max_new_tokens=max_new_tokens,
                             do_sample=False)
        # take only the newly generated portion
        in_len = batch["input_ids"].shape[1]
        out_tokens = gen[:, in_len:]
        texts = processor.batch_decode(out_tokens, skip_special_tokens=True)
        for raw, gt, task, snr in zip(texts, answers, tasks, snr_list):
            pred = parse_answer(raw, task)
            preds[task].append(pred)
            truths[task].append(gt)
            snr_dbs[task].append(snr)
    return {
        "preds":   {t: preds[t]   for t in preds},
        "truths":  {t: truths[t]  for t in truths},
        "snr_dbs": {t: snr_dbs[t] for t in snr_dbs},
    }


def accuracy_per_task(results: dict) -> dict[str, float]:
    out = {}
    for task in results["preds"]:
        p = np.asarray(results["preds"][task])
        t = np.asarray(results["truths"][task])
        out[task] = float((p == t).mean()) if len(p) else 0.0
    return out


# ---------------------------------------------------------------------------
# Plots (mirror the CNN plotting style for direct comparison)
# ---------------------------------------------------------------------------
def plot_confusion_modulation(preds: list, truths: list, outpath: str):
    labels_present = sorted(set(truths) | set(preds), key=lambda x: (
        MODULATIONS.index(x) if x in MODULATIONS else 999))
    cm = confusion_matrix(truths, preds, labels=labels_present)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=labels_present, yticklabels=labels_present,
                cbar_kws={"label": "row-normalized"},
                annot_kws={"size": 6}, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("VLM modulation classification — confusion matrix (test split)")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_snr(results: dict, outpath: str,
                         snr_bin_edges: np.ndarray | None = None):
    if snr_bin_edges is None:
        snr_bin_edges = np.arange(0, 32, 3)
    centers = 0.5 * (snr_bin_edges[:-1] + snr_bin_edges[1:])
    fig, ax = plt.subplots(figsize=(8, 5))
    for task in ["modulation", "phase", "iq"]:
        if task not in results["preds"]: continue
        snr = np.asarray(results["snr_dbs"][task])
        p   = np.asarray(results["preds"][task])
        t   = np.asarray(results["truths"][task])
        accs = []
        for lo, hi in zip(snr_bin_edges[:-1], snr_bin_edges[1:]):
            m = (snr >= lo) & (snr < hi)
            accs.append((p[m] == t[m]).mean() if m.sum() else np.nan)
        ax.plot(centers, accs, "o-", lw=1.6, ms=6,
                label=task if task != "iq" else "IQ imbalance")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("VLM per-task accuracy vs SNR (test split)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_ood_vs_test(acc_test: dict, acc_ood: dict, outpath: str):
    tasks = [t for t in ["modulation", "phase", "iq"] if t in acc_test]
    test_acc = [acc_test[t] for t in tasks]
    ood_acc  = [acc_ood.get(t, 0.0) for t in tasks]
    x = np.arange(len(tasks)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, test_acc, w, label="in-distribution (test)", color="seagreen")
    ax.bar(x + w/2, ood_acc,  w, label=f"OOD (held-out SNR={HELD_OUT_SNR_BIN})",
           color="darkorange")
    for xi, v in zip(x - w/2, test_acc): ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + w/2, ood_acc):  ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
    ax.legend(); ax.set_title("VLM: in-distribution vs OOD accuracy")
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_other_tasks(results: dict, outpath: str):
    """Confusion matrices for phase and IQ tasks (3 classes each)."""
    panels = [
        ("phase", "Phase noise level",  PHASE_LEVELS),
        ("iq",    "IQ imbalance level", IQ_LEVELS),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (task, title, labels) in zip(axes, panels):
        if task not in results["preds"]:
            ax.set_visible(False); continue
        preds  = results["preds"][task]
        truths = results["truths"][task]
        cm = confusion_matrix(truths, preds, labels=labels)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                    xticklabels=labels, yticklabels=labels,
                    cbar=False, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title)
    fig.suptitle("VLM — confusion matrices for impairment tasks", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def compute_severity_sensitivity(results: dict, test_df: pd.DataFrame) -> dict:
    """
    VLM modulation accuracy broken down by impairment severity.

    The eval loader iterated over all tasks for each image, so the entries
    in results["preds"]["modulation"] map 1:1 to the rows of test_df.
    """
    if "modulation" not in results["preds"]:
        return {"phase": {}, "iq": {}}
    mod_pred  = np.asarray(results["preds"]["modulation"])
    mod_truth = np.asarray(results["truths"]["modulation"])
    correct = (mod_pred == mod_truth)
    out = {"phase": {}, "iq": {}}
    for lvl in PHASE_LEVELS:
        m = (test_df["phase_noise_level"].values == lvl)
        out["phase"][lvl] = float(correct[m].mean()) if m.sum() else float("nan")
    for lvl in IQ_LEVELS:
        m = (test_df["iq_imbalance_level"].values == lvl)
        out["iq"][lvl] = float(correct[m].mean()) if m.sum() else float("nan")
    return out


def plot_severity_sensitivity(sens: dict, outpath: str):
    """Bar charts: VLM modulation acc by phase severity and by IQ severity."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, kind, levels, color, title in [
        (axes[0], "phase", PHASE_LEVELS, "seagreen",   "Modulation acc vs phase noise severity"),
        (axes[1], "iq",    IQ_LEVELS,    "darkorange", "Modulation acc vs I/Q imbalance severity"),
    ]:
        vals = [sens[kind].get(lvl, float("nan")) for lvl in levels]
        bars = ax.bar(levels, vals, color=color, width=0.6)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, v + .01, f"{v:.2f}",
                        ha="center", fontsize=10)
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
        ax.set_title(title); ax.set_ylabel("Accuracy" if kind == "phase" else "")
    fig.suptitle("VLM — sensitivity to impairment severity (test split)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Computational complexity helpers
# ---------------------------------------------------------------------------
def count_parameters(model) -> dict:
    """Counts the underlying model parameters; LoRA wraps mark only adapter trainable."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable),
            "trainable_fraction": float(trainable) / max(total, 1)}


def measure_vlm_inference_latency(model, processor, loader, device,
                                  max_new_tokens: int = 16,
                                  n_warmup_batches: int = 1) -> dict:
    """Average per-image generation latency on the eval loader."""
    model.eval()
    # warmup
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_warmup_batches: break
            for k in ("_tasks", "_answers", "_snr_dbs"):
                batch.pop(k, None)
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            _ = model.generate(**batch, max_new_tokens=max_new_tokens,
                               do_sample=False)
    # measure
    n_imgs = 0
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            for k in ("_tasks", "_answers", "_snr_dbs"):
                batch.pop(k, None)
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            _ = model.generate(**batch, max_new_tokens=max_new_tokens,
                               do_sample=False)
            n_imgs += batch["input_ids"].size(0)
    if device == "cuda": torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {"total_seconds":  elapsed,
            "generations":    n_imgs,
            "ms_per_image":   1000.0 * elapsed / max(n_imgs, 1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", 
                        default=os.path.join(script_dir, "data", "labels.csv"))
    parser.add_argument("--model_id", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--max_eval_per_split", type=int, default=400,
                        help="cap eval images per split for speed (set 0 = all)")
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
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"device: {device}")
    print(f"model:  {args.model_id}")

    save_dir = args.save_dir
    ckpt_dir = os.path.join(save_dir, "checkpoints", "vlm_lora")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- processor & model ----
    processor = AutoProcessor.from_pretrained(args.model_id)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )
    # freeze the vision encoder
    for n, p in base_model.named_parameters():
        if "vision" in n.lower():
            p.requires_grad = False

    # Enable gradient checkpointing to reduce activation memory.
    # Critical on free-tier GPUs (T4 / 16GB) for VLMs.
    base_model.gradient_checkpointing_enable()
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.to(device)
    model.print_trainable_parameters()

    # ---- data ----
    df = pd.read_csv(args.labels_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)
    ood_df   = df[df["split"] == "ood_test"].reset_index(drop=True)
    if args.max_eval_per_split and len(test_df) > args.max_eval_per_split:
        test_df = test_df.sample(n=args.max_eval_per_split, random_state=0)
    if args.max_eval_per_split and len(ood_df) > args.max_eval_per_split:
        ood_df = ood_df.sample(n=args.max_eval_per_split, random_state=0)

    # OOD: drop SNR question (the right answer is the held-out class)
    eval_tasks_in  = ["modulation", "phase", "iq", "jamming", "snr"]
    eval_tasks_ood = ["modulation", "phase", "iq", "jamming"]

    train_ds = VLMConstellationDataset(train_df, mode="train", seed=0)
    test_ds  = VLMConstellationDataset(test_df,  mode="eval", tasks=eval_tasks_in)
    ood_ds   = VLMConstellationDataset(ood_df,   mode="eval", tasks=eval_tasks_ood)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=VLMCollator(processor, mode="train"))
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size,
                             num_workers=args.num_workers,
                             collate_fn=VLMCollator(processor, mode="eval"))
    ood_loader  = DataLoader(ood_ds, batch_size=args.eval_batch_size,
                             num_workers=args.num_workers,
                             collate_fn=VLMCollator(processor, mode="eval"))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )

    # ---- parameter counts ----
    param_counts = count_parameters(model)
    print(f"params: total={param_counts['total']:,}   "
          f"trainable={param_counts['trainable']:,}   "
          f"({100*param_counts['trainable_fraction']:.2f}% of total)")

    # ---- training ----
    history = []
    training_start = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        loss = train_loop(model, processor, train_loader,
                          optimizer, device, accum_steps=args.accum_steps)
        history.append({"epoch": epoch, "train_loss": loss,
                        "seconds": time.time() - ep_t0})
        print(f"epoch {epoch:02d}  train_loss={loss:.4f}  "
              f"({time.time()-ep_t0:.1f}s)")
    total_training_time = time.time() - training_start
    print(f"\ntotal training time: {total_training_time:.1f}s  "
          f"({total_training_time/args.epochs:.1f}s / epoch)")

    # ---- save adapter ----
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    print(f"saved adapter -> {ckpt_dir}")

    # ---- eval ----
    print("\nEvaluating on test split...")
    test_results = generate_and_score(model, processor, test_loader, device)
    test_acc = accuracy_per_task(test_results)
    print("=== TEST (in-distribution) ===")
    for t, a in test_acc.items():
        print(f"  {t:<11} acc = {a:.3f}")

    print("\nEvaluating on OOD split...")
    ood_results = generate_and_score(model, processor, ood_loader, device)
    ood_acc = accuracy_per_task(ood_results)
    print("=== OOD TEST (held-out SNR bin) ===")
    for t, a in ood_acc.items():
        print(f"  {t:<11} acc = {a:.3f}")

    # ---- inference latency (per-image generation, modulation question only) ----
    print("\nMeasuring inference latency...")
    latency_ds = VLMConstellationDataset(
        test_df.head(64), mode="eval", tasks=["modulation"])
    latency_loader = DataLoader(
        latency_ds, batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=VLMCollator(processor, mode="eval"))
    latency = measure_vlm_inference_latency(model, processor, latency_loader, device)
    print(f"inference latency: {latency['ms_per_image']:.1f} ms/generation "
          f"({latency['generations']} generations in {latency['total_seconds']:.1f}s)")

    # ---- severity sensitivity ----
    severity = compute_severity_sensitivity(test_results, test_df.reset_index(drop=True))
    print("\nsensitivity to impairment severity (modulation accuracy):")
    for kind in ("phase", "iq"):
        if not severity[kind]: continue
        cells = "  ".join(f"{lvl}={severity[kind][lvl]:.3f}"
                          for lvl in (PHASE_LEVELS if kind == "phase" else IQ_LEVELS))
        print(f"  by {kind:<5}:  {cells}")

    # ---- save metrics ----
    metrics = {
        "device":   device,
        "model_id": args.model_id,
        "epochs":   args.epochs,
        "history":  history,
        "test_accuracy":     test_acc,
        "ood_test_accuracy": ood_acc,
        "lora_config": {
            "r": args.lora_r, "alpha": args.lora_alpha,
            "dropout": args.lora_dropout, "targets": args.lora_targets,
        },
        "params":           param_counts,
        "training_time_s":  total_training_time,
        "training_time_per_epoch_s": total_training_time / args.epochs,
        "inference_latency": latency,
        "severity_sensitivity": severity,
    }
    with open(os.path.join(save_dir, "vlm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved -> {save_dir}/vlm_metrics.json")

    # ---- plots ----
    if "modulation" in test_results["preds"]:
        plot_confusion_modulation(
            test_results["preds"]["modulation"],
            test_results["truths"]["modulation"],
            os.path.join(save_dir, "08_vlm_confusion_modulation.png"))
        print(f"saved -> {save_dir}/08_vlm_confusion_modulation.png")
    plot_accuracy_vs_snr(test_results,
        os.path.join(save_dir, "09_vlm_accuracy_vs_snr.png"))
    print(f"saved -> {save_dir}/09_vlm_accuracy_vs_snr.png")
    plot_ood_vs_test(test_acc, ood_acc,
        os.path.join(save_dir, "10_vlm_ood_vs_test.png"))
    print(f"saved -> {save_dir}/10_vlm_ood_vs_test.png")
    plot_confusion_other_tasks(test_results,
        os.path.join(save_dir, "11_vlm_confusion_other.png"))
    print(f"saved -> {save_dir}/11_vlm_confusion_other.png")
    plot_severity_sensitivity(severity,
        os.path.join(save_dir, "12_vlm_severity_sensitivity.png"))
    print(f"saved -> {save_dir}/12_vlm_severity_sensitivity.png")


if __name__ == "__main__":
    main()
