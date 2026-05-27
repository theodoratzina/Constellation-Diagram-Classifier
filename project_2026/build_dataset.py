"""
build_dataset.py
================

Reads config.yaml and produces three artefacts under data/:

    images/NNNNNN.png    one constellation image per example
    labels.csv           structured labels (one row per image)
    qa_pairs.jsonl       (image_path, question, answer) triples for VLM training

The held-out SNR bin defined in config.yaml is automatically routed
to a separate `ood_test` split, leaving the other bins to be partitioned
into train / val / test according to the configured ratios.

Run:
    python build_dataset.py                       # full dataset
    python build_dataset.py --max_samples 200     # quick smoke test
    python build_dataset.py --config other.yaml   # alternative config
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from signals import generate_symbols, apply_channel, render_constellation


# ---------------------------------------------------------------------------
# QA template — natural-language phrasing keeps every answer in its own
# distinct vocabulary, so the VLM never has to disambiguate "mild" between
# tasks. The CNN ignores these and reads labels.csv directly.
# ---------------------------------------------------------------------------
def make_qa_pairs(image_path: str,
                  modulation: str,
                  phase_noise_level: str,
                  iq_imbalance_level: str,
                  jamming: bool,
                  snr_bin: str) -> list[dict]:
    return [
        {"image": image_path,
         "question": "What modulation is used?",
         "answer": modulation},
        {"image": image_path,
         "question": "What is the level of phase noise?",
         "answer": phase_noise_level},
        {"image": image_path,
         "question": "What is the level of I/Q imbalance?",
         "answer": iq_imbalance_level},
        {"image": image_path,
         "question": "Is there external interference?",
         "answer": "yes" if jamming else "no"},
        {"image": image_path,
         "question": "What is the SNR range?",
         "answer": snr_bin},
    ]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
def build_dataset(config_path: str = "config.yaml",
                  max_samples: int | None = None) -> None:

    # If the file isn't found at the provided path, try looking relative to this script
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ---- paths ----
    # script_dir is used internally to find files regardless of where Python
    # was launched from. But paths written to labels.csv stay RELATIVE so the
    # dataset is portable across machines (Windows local, Colab, Linux).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    def resolve(p):
        """Convert relative path to absolute, using script directory as anchor."""
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(script_dir, p))

    images_dir_abs = resolve(cfg["images_dir"])      # absolute, for file I/O
    images_dir_rel = cfg["images_dir"]                # relative, for CSV
    labels_csv     = resolve(cfg["labels_csv"])
    qa_jsonl       = resolve(cfg["qa_jsonl"])

    os.makedirs(images_dir_abs, exist_ok=True)
    os.makedirs(os.path.dirname(labels_csv) or ".", exist_ok=True)

    # ---- generation params ----
    n_symbols  = cfg["n_symbols"]
    image_size = cfg["image_size"]
    view_limit = cfg["view_limit"]
    log_scale  = cfg["log_scale"]

    modulations    = cfg["modulations"]
    snr_bins       = cfg["snr_bins_db"]
    held_out_bin   = cfg["held_out_snr_bin"]
    phase_levels   = cfg["phase_noise_deg"]
    iq_levels      = cfg["iq_imbalance"]

    jam_enable      = cfg["jamming_enable"]
    jam_prob        = cfg["jamming_probability"]
    jam_sir_range   = cfg["jamming_sir_range_db"]
    jam_freq_range  = cfg["jamming_freq_range"]

    samples_per_combo = cfg["samples_per_combination"]
    rng = np.random.default_rng(cfg["seed"])
    split_cfg = cfg["split"]
    train_thr = split_cfg["train"]
    val_thr   = train_thr + split_cfg["val"]

    total = (len(modulations) * len(snr_bins) *
             len(phase_levels) * len(iq_levels) *
             samples_per_combo)
    if max_samples is not None:
        total = min(total, max_samples)

    # ---- main loop ----
    rows, qa_rows = [], []
    idx = 0
    pbar = tqdm(total=total, desc="Building dataset")

    for mod in modulations:
        for snr_name, (snr_lo, snr_hi) in snr_bins.items():
            for ph_name, ph_deg in phase_levels.items():
                for iq_name, iq_par in iq_levels.items():
                    for _ in range(samples_per_combo):
                        if max_samples is not None and idx >= max_samples:
                            break

                        # ----- sample impairment values -----
                        snr_db        = rng.uniform(snr_lo, snr_hi)
                        sigma_phi_rad = np.deg2rad(ph_deg)
                        eps           = iq_par["eps"]
                        psi_rad       = np.deg2rad(iq_par["psi_deg"])

                        if jam_enable and rng.random() < jam_prob:
                            jamming    = True
                            sir_db     = rng.uniform(*jam_sir_range)
                            freq_norm  = rng.uniform(*jam_freq_range)
                        else:
                            jamming, sir_db, freq_norm = False, None, 0.0

                        # ----- generate + render -----
                        s  = generate_symbols(mod, n_symbols, rng=rng)
                        rx = apply_channel(
                            s, snr_db=snr_db,
                            phase_noise_sigma_rad=sigma_phi_rad,
                            iq_eps=eps, iq_psi_rad=psi_rad,
                            jammer_sir_db=sir_db,
                            jammer_freq_norm=freq_norm,
                            rng=rng,
                        )
                        img = render_constellation(
                            rx, mode="histogram",
                            image_size=image_size,
                            view_limit=view_limit,
                            log_scale=log_scale,
                        )

                        filename = f"{idx:06d}.png"
                        img.save(os.path.join(images_dir_abs, filename))

                        # ----- split assignment -----
                        if snr_name == held_out_bin:
                            split = "ood_test"
                        else:
                            r = rng.random()
                            split = ("train" if r < train_thr
                                     else "val" if r < val_thr
                                     else "test")

                        # ----- record row -----
                        # Use forward slashes always, so the CSV works on
                        # both Windows and Linux (PIL.Image.open accepts both).
                        rel_path = f"{images_dir_rel}/{filename}".replace("\\", "/")
                        rows.append({
                            "image_id":            idx,
                            "filename":            filename,
                            "image_path":          rel_path,
                            "modulation":          mod,
                            "snr_db":              round(snr_db, 2),
                            "snr_bin":             snr_name,
                            "phase_noise_level":   ph_name,
                            "phase_noise_std_deg": ph_deg,
                            "iq_imbalance_level":  iq_name,
                            "iq_eps":              eps,
                            "iq_psi_deg":          iq_par["psi_deg"],
                            "jamming":             int(jamming),
                            "jam_sir_db":          round(sir_db, 2) if sir_db is not None else "",
                            "split":               split,
                        })
                        qa_rows.extend(make_qa_pairs(
                            rel_path, mod, ph_name, iq_name, jamming, snr_name))

                        idx += 1
                        pbar.update(1)
                if max_samples is not None and idx >= max_samples: break
            if max_samples is not None and idx >= max_samples: break
        if max_samples is not None and idx >= max_samples: break
    pbar.close()

    # ---- write outputs ----
    df = pd.DataFrame(rows)
    df.to_csv(labels_csv, index=False)

    with open(qa_jsonl, "w") as f:
        for r in qa_rows:
            f.write(json.dumps(r) + "\n")

    # ---- summary ----
    print(f"\nGenerated {len(df)} images in {os.path.abspath(images_dir_abs)}/")
    print(f"Labels:    {os.path.abspath(labels_csv)}")
    print(f"QA pairs:  {os.path.abspath(qa_jsonl)}  ({len(qa_rows)} rows)")
    print("\nSplit distribution:")
    print(df["split"].value_counts().to_string())
    print("\nPer-modulation counts (top 5 rows):")
    print(df["modulation"].value_counts().head().to_string())
    print("\nSNR distribution within training split (mean ± std):")
    train_df = df[df["split"] == "train"]
    if len(train_df):
        print(f"  {train_df['snr_db'].mean():.1f} ± {train_df['snr_db'].std():.1f} dB")
    print(f"\nOOD (held-out '{held_out_bin}') split: "
          f"{(df['split']=='ood_test').sum()} images")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="cap total examples (for quick smoke tests)")
    args = parser.parse_args()
    build_dataset(args.config, args.max_samples)
