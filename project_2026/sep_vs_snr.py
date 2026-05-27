"""
sep_vs_snr.py
=============

Monte Carlo Symbol Error Probability (SEP) versus SNR curves for square QAM
constellations under wrapped Gaussian phase noise, as required by Part I.

The simulated curves are compared with the closed-form expression for square
M-QAM in pure AWGN (no phase noise) — this validates the simulation at
sigma_phi = 0 and highlights the irreducible error floor that phase noise
introduces at high SNR.

Run:
    python sep_vs_snr.py                # full simulation (~1-2 minutes)
    python sep_vs_snr.py --quick        # fast version (~10 seconds)
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc
from tqdm import tqdm

from signals import get_constellation, apply_channel


# ---------------------------------------------------------------------------
# Theoretical SEP for square M-QAM in AWGN  (used as sanity reference curve)
# ---------------------------------------------------------------------------
def qfunc(x: np.ndarray) -> np.ndarray:
    """Q(x) = 0.5 * erfc(x / sqrt(2))."""
    return 0.5 * erfc(x / np.sqrt(2))


def theoretical_sep_qam(snr_db: np.ndarray, M: int) -> np.ndarray:
    """
    SEP for unit-energy square M-QAM in AWGN:

        SEP = 1 - (1 - P_sqrt)^2,
        P_sqrt = 2 * (1 - 1/sqrt(M)) * Q(sqrt(3 * SNR / (M - 1)))
    """
    snr_lin = 10.0 ** (np.asarray(snr_db) / 10.0)
    sqrt_M = np.sqrt(M)
    P_sqrt = 2.0 * (1.0 - 1.0 / sqrt_M) * qfunc(np.sqrt(3.0 * snr_lin / (M - 1)))
    return 1.0 - (1.0 - P_sqrt) ** 2


# ---------------------------------------------------------------------------
# Monte Carlo simulation with ML (nearest-neighbour) detector
# ---------------------------------------------------------------------------
def simulate_sep(scheme: str,
                 snr_db: float,
                 sigma_phi_rad: float,
                 n_symbols: int,
                 rng: np.random.Generator,
                 batch_size: int = 50_000) -> float:
    """
    Monte Carlo SEP estimate for `scheme` at given SNR and phase noise sigma.

    ML decision is performed in symbol space (nearest neighbour in the
    constellation), since the receiver in this project does not implement
    phase tracking — exactly the scenario in [R2].
    """
    alphabet = get_constellation(scheme)
    M = len(alphabet)

    total_errors = 0
    remaining = n_symbols
    while remaining > 0:
        batch = min(batch_size, remaining)
        indices = rng.integers(0, M, size=batch)
        s = alphabet[indices]
        rx = apply_channel(s, snr_db=snr_db,
                           phase_noise_sigma_rad=sigma_phi_rad,
                           rng=rng)
        # Nearest-neighbour ML detection.
        # dist shape: (batch, M).
        dist = np.abs(rx[:, None] - alphabet[None, :])
        decisions = np.argmin(dist, axis=1)
        total_errors += int(np.sum(decisions != indices))
        remaining -= batch

    return total_errors / n_symbols


# ---------------------------------------------------------------------------
# Full sweep across modulations, SNR values and phase-noise sigmas
# ---------------------------------------------------------------------------
def run_simulation(modulations: list[str],
                   snr_values_db: list[float],
                   sigma_phi_deg_list: list[float],
                   n_symbols: int,
                   seed: int = 42) -> dict:
    """Returns dict keyed by (mod, sigma_deg) -> list of SEP per SNR."""
    rng = np.random.default_rng(seed)
    results: dict = {}
    total = len(modulations) * len(sigma_phi_deg_list) * len(snr_values_db)
    pbar = tqdm(total=total, desc="SEP sweep")
    for mod in modulations:
        for sigma_deg in sigma_phi_deg_list:
            sigma_rad = np.deg2rad(sigma_deg)
            seps = []
            for snr_db in snr_values_db:
                sep = simulate_sep(mod, snr_db, sigma_rad, n_symbols, rng)
                seps.append(sep)
                pbar.update(1)
            results[(mod, sigma_deg)] = seps
    pbar.close()
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(snr_values_db: list[float],
                 results: dict,
                 modulations: list[str],
                 sigma_phi_deg_list: list[float],
                 outpath: str) -> None:
    """One subplot per modulation, log-y axis, curve per sigma_phi value."""
    fig, axes = plt.subplots(1, len(modulations),
                             figsize=(5 * len(modulations), 5),
                             sharey=True)
    if len(modulations) == 1:
        axes = [axes]

    cmap = plt.cm.viridis(np.linspace(0.0, 0.85, len(sigma_phi_deg_list)))

    for ax, mod in zip(axes, modulations):
        M = len(get_constellation(mod))

        # ----- theoretical reference curve -----
        snr_fine = np.linspace(min(snr_values_db), max(snr_values_db), 200)
        ax.semilogy(snr_fine, theoretical_sep_qam(snr_fine, M),
                    "k--", lw=1.5, alpha=0.7,
                    label="Theory (no phase noise)")

        # ----- simulated curves -----
        for color, sigma_deg in zip(cmap, sigma_phi_deg_list):
            seps = np.asarray(results[(mod, sigma_deg)])
            seps_plot = np.maximum(seps, 1e-7)         # avoid log(0)
            ax.semilogy(snr_values_db, seps_plot,
                        "o-", color=color, lw=1.5, ms=5,
                        label=fr"$\sigma_\varphi$ = {sigma_deg}°")

        ax.set_xlabel("SNR (dB)")
        ax.set_title(f"{mod}  (M={M})")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(1e-5, 1.0)
        ax.legend(fontsize=8, loc="lower left")

    axes[0].set_ylabel("Symbol Error Probability")
    fig.suptitle("SEP vs SNR for square QAM under wrapped Gaussian phase noise",
                 fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {outpath}")


def save_results_csv(snr_values_db: list[float],
                     results: dict,
                     outpath: str) -> None:
    rows = []
    for (mod, sigma_deg), seps in results.items():
        for snr_db, sep in zip(snr_values_db, seps):
            rows.append({"modulation": mod,
                         "sigma_phi_deg": sigma_deg,
                         "snr_db": snr_db,
                         "sep": sep})
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"saved -> {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="fewer symbols / coarser SNR grid for fast tests")
    args = parser.parse_args()

    modulations        = ["16-QAM", "64-QAM", "256-QAM"]
    sigma_phi_deg_list = [0.0, 2.0, 4.0, 6.0]

    if args.quick:
        snr_values_db = list(range(0, 31, 4))   # 8 points
        n_symbols     = 20_000
    else:
        snr_values_db = list(range(0, 31, 2))   # 16 points
        n_symbols     = 100_000

    print(f"Running SEP sweep: {len(modulations)} mods × "
          f"{len(sigma_phi_deg_list)} sigmas × {len(snr_values_db)} SNR points "
          f"× {n_symbols} symbols per point")

    results = run_simulation(modulations, snr_values_db,
                             sigma_phi_deg_list, n_symbols, seed=42)

    # Resolve output paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    
    png_path = os.path.join(results_dir, "04_sep_vs_snr.png")
    csv_path = os.path.join(results_dir, "sep_vs_snr.csv")

    plot_results(snr_values_db, results, modulations, sigma_phi_deg_list, png_path)
    save_results_csv(snr_values_db, results, csv_path)
