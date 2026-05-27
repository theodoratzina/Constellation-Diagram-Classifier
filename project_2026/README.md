# Seeing Signals — Constellation Diagram Interpretation

Project for **Τηλεπικοινωνιακά Συστήματα ΙΙ** (2026).

## Layout

```
project_2026/
├── README.md
├── requirements.txt
├── config.yaml          all parameters (SNR bins, severities, dataset size)
│
├── signals.py           modulations + impairments + image rendering
├── build_dataset.py     generates data/images/*.png + data/labels.csv
├── sep_vs_snr.py        Monte-Carlo SEP-vs-SNR for QAM with phase noise
├── train_cnn.py         multi-head CNN training + evaluation
├── train_vlm.py         LoRA fine-tuning of a lightweight VLM
├── ood_intermediate.py  generalization test at unseen impairment levels
├── compare.py           final comparison (accuracy + complexity + plots)
│
├── data/                auto-generated:  images/  labels.csv  qa_pairs.jsonl
└── results/             auto-generated:  figures, checkpoints, metrics
```

## Run order

```bash
pip install -r requirements.txt

python signals.py            # sanity check (3 figures)
python build_dataset.py      # Part I: dataset
python sep_vs_snr.py         # Part I: SEP-vs-SNR analysis

python train_cnn.py          # Part II: CNN baseline
python train_vlm.py          # Part II: VLM fine-tuning (LoRA)
python ood_intermediate.py   # Part II: OOD at intermediate impairments

python compare.py            # Part II: final comparison
```

## What you get in `results/`

After running the full pipeline you'll have:

- `00`–`02` visual sanity checks for signals and rendering
- `03` random samples from the generated dataset
- `04` SEP-vs-SNR curves for QAM with phase noise
- `05`–`09` CNN evaluation figures (modulation confusion, accuracy vs SNR,
  OOD bar chart, impairment-task confusion matrices, severity sensitivity)
- `08`*–`12`* VLM evaluation figures (same set)
- `13` intermediate-OOD bar chart (single model view)
- `11`*, `12`*, `14`, `15` CNN vs VLM head-to-head (accuracy, OOD gap,
  computational complexity, intermediate-OOD)
- `cnn_metrics.json`, `vlm_metrics.json` raw numbers
- `comparison_table.{csv,md}`, `comparison_complexity.csv`,
  `comparison_summary.txt` tabular outputs for the report

## Modulations (from [R2])

4-ASK, 8-ASK, BPSK, QPSK, 4-HQAM, 16-HQAM, 64-HQAM, 16-QAM, 32-QAM,
64-QAM, 128-QAM, 256-QAM, 16-APSK, 32-APSK, 64-APSK, 128-APSK.

## Impairments

- AWGN with controlled SNR
- Phase noise (wrapped Gaussian, as in [R2])
- IQ imbalance (amplitude + phase mismatch)
- Optional CW jamming
- Optional Saleh AM/AM amplitude distortion

## References

- [R1] Zou et al., *RF-GPT: Teaching AI to See the Wireless World*, 2026.
- [R2] Oikonomou et al., *CNN-Based Automatic Modulation Classification Under
  Phase Imperfections*, IEEE Wireless Comm. Letters, 13(5), May 2024.
