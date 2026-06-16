"""
evaluate_vlm.py
===============
Standalone VLM evaluation script.
Loads the saved LoRA adapter and runs evaluation only.
"""
import os, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

from train_vlm import (
    VLMConstellationDataset, VLMCollator,
    generate_and_score, accuracy_per_task,
    compute_severity_sensitivity, measure_vlm_inference_latency,
    plot_confusion_modulation, plot_accuracy_vs_snr,
    plot_ood_vs_test, plot_confusion_other_tasks, plot_severity_sensitivity,
)

parser = argparse.ArgumentParser()
parser.add_argument("--labels_csv",  default="data/labels.csv")
parser.add_argument("--adapter_dir", default="/kaggle/working/results/checkpoints/vlm_lora")
parser.add_argument("--base_model",  default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--save_dir",    default="results")
parser.add_argument("--eval_batch",  type=int, default=1)
parser.add_argument("--max_eval",    type=int, default=200)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
if device == "cuda":
    torch.cuda.empty_cache()
    print(f"GPU memory free: {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB")

print(f"\nLoading processor + base model + LoRA adapter...")
print(f"  adapter: {args.adapter_dir}")

processor = AutoProcessor.from_pretrained(args.adapter_dir)
base = AutoModelForImageTextToText.from_pretrained(
    args.base_model,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)
model = PeftModel.from_pretrained(base, args.adapter_dir).to(device)
model.eval()
print("Model loaded.")

df = pd.read_csv(args.labels_csv)
test_df = df[df["split"] == "test"].reset_index(drop=True)
ood_df  = df[df["split"] == "ood_test"].reset_index(drop=True)

if args.max_eval and len(test_df) > args.max_eval:
    test_df = test_df.sample(n=args.max_eval, random_state=0).reset_index(drop=True)
if args.max_eval and len(ood_df) > args.max_eval:
    ood_df = ood_df.sample(n=args.max_eval, random_state=0).reset_index(drop=True)
print(f"\nEval sets: test={len(test_df)} ood_test={len(ood_df)}")

eval_tasks_in  = ["modulation", "phase", "iq", "jamming", "snr"]
eval_tasks_ood = ["modulation", "phase", "iq", "jamming"]

collator = VLMCollator(processor, mode="eval")
test_loader = DataLoader(
    VLMConstellationDataset(test_df, mode="eval", tasks=eval_tasks_in),
    batch_size=args.eval_batch, num_workers=0, collate_fn=collator)
ood_loader = DataLoader(
    VLMConstellationDataset(ood_df, mode="eval", tasks=eval_tasks_ood),
    batch_size=args.eval_batch, num_workers=0, collate_fn=collator)

print("\n=== TEST (in-distribution) ===")
test_results = generate_and_score(model, processor, test_loader, device)
test_acc = accuracy_per_task(test_results)
for t, a in test_acc.items():
    print(f"  {t:<11} acc = {a:.3f}")

print("\n=== OOD TEST (held-out SNR bin) ===")
ood_results = generate_and_score(model, processor, ood_loader, device)
ood_acc = accuracy_per_task(ood_results)
for t, a in ood_acc.items():
    print(f"  {t:<11} acc = {a:.3f}")

print("\nMeasuring inference latency...")
latency_ds = VLMConstellationDataset(test_df.head(32), mode="eval", tasks=["modulation"])
latency_loader = DataLoader(latency_ds, batch_size=args.eval_batch, num_workers=0, collate_fn=collator)
latency = measure_vlm_inference_latency(model, processor, latency_loader, device)
print(f"inference latency: {latency['ms_per_image']:.1f} ms/generation")

severity = compute_severity_sensitivity(test_results, test_df.reset_index(drop=True))

os.makedirs(args.save_dir, exist_ok=True)
metrics = {
    "device": device, "model_id": args.base_model, "epochs": 1,
    "history": [{"epoch": 1, "train_loss": 4.7904, "seconds": 13721.4}],
    "test_accuracy": test_acc,
    "ood_test_accuracy": ood_acc,
    "lora_config": {"r": 8, "alpha": 16, "dropout": 0.05,
                    "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
    "params": {"total": 257848896, "trainable": 1363968,
               "trainable_fraction": 1363968/257848896},
    "training_time_s": 13721.4,
    "training_time_per_epoch_s": 13721.4,
    "inference_latency": latency,
    "severity_sensitivity": severity,
}
with open(os.path.join(args.save_dir, "vlm_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nsaved -> {args.save_dir}/vlm_metrics.json")

if "modulation" in test_results["preds"]:
    plot_confusion_modulation(
        test_results["preds"]["modulation"],
        test_results["truths"]["modulation"],
        os.path.join(args.save_dir, "08_vlm_confusion_modulation.png"))
plot_accuracy_vs_snr(test_results, os.path.join(args.save_dir, "09_vlm_accuracy_vs_snr.png"))
plot_ood_vs_test(test_acc, ood_acc, os.path.join(args.save_dir, "10_vlm_ood_vs_test.png"))
plot_confusion_other_tasks(test_results, os.path.join(args.save_dir, "11_vlm_confusion_other.png"))
plot_severity_sensitivity(severity, os.path.join(args.save_dir, "12_vlm_severity_sensitivity.png"))

print("\nDONE")
