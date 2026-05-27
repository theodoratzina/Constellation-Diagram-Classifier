"""
signals.py
==========

Everything needed for Part I signal generation, in one file:

    Section 1: Modulation alphabets (16 schemes from [R2])
    Section 2: Transceiver impairments (AWGN, phase noise, IQ imbalance, jammer)
    Section 3: Constellation-image rendering (2-D histogram, fixed size)
    Section 4: Self-test (run `python signals.py`)

All alphabets are normalized so that E[|s|^2] = 1, which keeps every SNR
figure interpretable across modulations.
"""
from __future__ import annotations

import io
import os
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# =============================================================================
# Section 1 — Modulation alphabets
# =============================================================================

ALL_MODULATIONS: List[str] = [
    "4-ASK", "8-ASK",
    "BPSK", "QPSK",
    "4-HQAM", "16-HQAM", "64-HQAM",
    "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM",
    "16-APSK", "32-APSK", "64-APSK", "128-APSK",
]


def _normalize_unit_energy(c: np.ndarray) -> np.ndarray:
    """Scale alphabet so E[|s|^2] = 1."""
    return c / np.sqrt(np.mean(np.abs(c) ** 2))


def _rect_qam(side: int) -> np.ndarray:
    """Rectangular M = side*side QAM (16, 64, 256)."""
    levels = np.arange(-(side - 1), side, 2)
    re, im = np.meshgrid(levels, levels)
    return (re + 1j * im).flatten().astype(complex)


def _cross_qam(M: int) -> np.ndarray:
    """
    Cross-shaped 32- or 128-QAM.

    32-QAM:  6x6 grid minus 4 single corner points    -> 32
    128-QAM: 12x12 grid minus 4 corner 2x2 blocks (16) -> 128
    """
    if M == 32:
        side, cut = 6, 1
    elif M == 128:
        side, cut = 12, 2
    else:
        raise ValueError(f"Cross-QAM defined only for 32 and 128, got {M}")

    levels = np.arange(-(side - 1), side, 2)
    re, im = np.meshgrid(levels, levels)
    pts = (re + 1j * im).flatten()
    lim = levels[-1] - 2 * (cut - 1)
    mask = ~((np.abs(pts.real) >= lim) & (np.abs(pts.imag) >= lim))
    pts = pts[mask]
    assert len(pts) == M
    return pts.astype(complex)


def _hex_qam(M: int) -> np.ndarray:
    """Hexagonal-lattice QAM: pick the M lattice points closest to origin."""
    n = int(np.ceil(np.sqrt(M))) + 2
    pts = []
    for a in range(-n, n + 1):
        for b in range(-n, n + 1):
            x = a + 0.5 * b
            y = (np.sqrt(3) / 2) * b
            pts.append(complex(x, y))
    pts = np.array(pts)
    return pts[np.argsort(np.abs(pts))[:M]].astype(complex)


def _apsk(num_per_ring: List[int],
          radii: List[float],
          phase_offsets_deg: List[float] | None = None) -> np.ndarray:
    """Build APSK from concentric rings (radii, points/ring, phase offsets)."""
    if phase_offsets_deg is None:
        phase_offsets_deg = [0.0] * len(num_per_ring)
    pts: List[complex] = []
    for n, r, ph0 in zip(num_per_ring, radii, phase_offsets_deg):
        ang = np.deg2rad(ph0) + 2 * np.pi * np.arange(n) / n
        pts.extend((r * np.exp(1j * ang)).tolist())
    return np.array(pts, dtype=complex)


def get_constellation(scheme: str) -> np.ndarray:
    """Return the normalized complex alphabet for `scheme` (unit avg. energy)."""
    s = scheme.upper()

    if s == "4-ASK":
        c = np.array([-3, -1, 1, 3], dtype=complex)
    elif s == "8-ASK":
        c = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=complex)
    elif s == "BPSK":
        c = np.array([1.0, -1.0], dtype=complex)
    elif s == "QPSK":
        c = (1 / np.sqrt(2)) * np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=complex)

    elif s == "16-QAM":
        c = _rect_qam(4)
    elif s == "64-QAM":
        c = _rect_qam(8)
    elif s == "256-QAM":
        c = _rect_qam(16)

    elif s == "32-QAM":
        c = _cross_qam(32)
    elif s == "128-QAM":
        c = _cross_qam(128)

    elif s == "4-HQAM":
        c = _hex_qam(4)
    elif s == "16-HQAM":
        c = _hex_qam(16)
    elif s == "64-HQAM":
        c = _hex_qam(64)

    elif s == "16-APSK":
        c = _apsk([4, 12], [1.0, 2.85], [45.0, 15.0])
    elif s == "32-APSK":
        c = _apsk([4, 12, 16], [1.0, 2.84, 5.27], [45.0, 15.0, 0.0])
    elif s == "64-APSK":
        c = _apsk([4, 12, 20, 28], [1.0, 2.6, 4.3, 6.0], [45.0, 15.0, 9.0, 0.0])
    elif s == "128-APSK":
        c = _apsk([8, 16, 20, 28, 28, 28],
                  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                  [22.5, 11.25, 9.0, 0.0, 6.43, 0.0])
    else:
        raise ValueError(f"Unknown modulation scheme: {scheme}")

    return _normalize_unit_energy(c)


def generate_symbols(scheme: str,
                     n_symbols: int,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw `n_symbols` uniformly random symbols from `scheme`'s alphabet."""
    rng = rng if rng is not None else np.random.default_rng()
    alphabet = get_constellation(scheme)
    return alphabet[rng.integers(0, len(alphabet), size=n_symbols)]


# =============================================================================
# Section 2 — Transceiver impairments (operate on E[|s|^2] = 1 signals)
# =============================================================================

def add_awgn(s: np.ndarray,
             snr_db: float,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """Add complex AWGN, assuming E[|s|^2] = 1."""
    rng = rng if rng is not None else np.random.default_rng()
    sigma = np.sqrt(10.0 ** (-snr_db / 10.0) / 2.0)
    n = sigma * (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape))
    return s + n


def add_phase_noise(s: np.ndarray,
                    sigma_phi_rad: float,
                    rng: np.random.Generator | None = None,
                    correlated: bool = False) -> np.ndarray:
    """
    Multiply by exp(j*theta).
    correlated=False -> i.i.d. wrapped Gaussian per symbol (matches [R2]).
    correlated=True  -> Wiener (random-walk) phase noise.
    """
    rng = rng if rng is not None else np.random.default_rng()
    if sigma_phi_rad == 0.0:
        return s.copy()
    if correlated:
        theta = np.cumsum(rng.normal(0.0, sigma_phi_rad, size=s.shape))
    else:
        theta = rng.normal(0.0, sigma_phi_rad, size=s.shape)
    return s * np.exp(1j * theta)


def apply_iq_imbalance(s: np.ndarray,
                       eps: float,
                       psi_rad: float) -> np.ndarray:
    """
    Standard transmit-side IQ imbalance:  y = alpha*s + beta*conj(s),
    with the mismatch split symmetrically across I and Q branches.
    """
    if eps == 0.0 and psi_rad == 0.0:
        return s.copy()
    g_I, g_Q = 1.0 + eps / 2.0, 1.0 - eps / 2.0
    phi_I, phi_Q = -psi_rad / 2.0, +psi_rad / 2.0
    alpha = 0.5 * (g_I * np.exp(-1j * phi_I) + g_Q * np.exp(-1j * phi_Q))
    beta  = 0.5 * (g_I * np.exp(-1j * phi_I) - g_Q * np.exp(-1j * phi_Q))
    return alpha * s + beta * np.conj(s)


def add_cw_jammer(s: np.ndarray,
                  sir_db: float,
                  freq_norm: float = 0.0,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Add a single-tone interferer at the given signal-to-interference ratio."""
    rng = rng if rng is not None else np.random.default_rng()
    jam_amp = np.sqrt(10.0 ** (-sir_db / 10.0))
    n = len(s)
    phase0 = rng.uniform(0.0, 2.0 * np.pi)
    jammer = jam_amp * np.exp(1j * (2 * np.pi * freq_norm * np.arange(n) + phase0))
    return s + jammer


def apply_amplitude_distortion(s: np.ndarray,
                               alpha_a: float = 2.0,
                               beta_a: float = 1.0) -> np.ndarray:
    """Saleh AM/AM:  |y| = alpha_a * |s| / (1 + beta_a * |s|^2). Phase preserved."""
    r = np.abs(s)
    return alpha_a * r / (1.0 + beta_a * r ** 2) * np.exp(1j * np.angle(s))


def apply_channel(s: np.ndarray,
                  snr_db: float,
                  phase_noise_sigma_rad: float = 0.0,
                  iq_eps: float = 0.0,
                  iq_psi_rad: float = 0.0,
                  jammer_sir_db: float | None = None,
                  jammer_freq_norm: float = 0.0,
                  amp_distortion: bool = False,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Apply impairments in canonical order:
        s --[IQ imbalance]--> --[amp distortion]--> --[phase noise]-->
          --[jammer]--> --[AWGN]--> r
    """
    rng = rng if rng is not None else np.random.default_rng()
    y = apply_iq_imbalance(s, iq_eps, iq_psi_rad)
    if amp_distortion:
        y = apply_amplitude_distortion(y)
    y = add_phase_noise(y, phase_noise_sigma_rad, rng=rng)
    if jammer_sir_db is not None:
        y = add_cw_jammer(y, jammer_sir_db, jammer_freq_norm, rng=rng)
    y = add_awgn(y, snr_db, rng=rng)
    return y


# =============================================================================
# Section 3 — Constellation image rendering
# =============================================================================

# ---- Path helper used by training/eval scripts ----
# The dataset CSV stores RELATIVE paths (e.g. "data/images/000000.png") so
# the dataset is portable across machines (Windows local <-> Colab/Linux).
# At runtime, we resolve them against this script's directory, so the code
# works no matter where Python is launched from.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def resolve_path(p: str) -> str:
    """
    Turn a path from labels.csv into a full filesystem path.

    - If `p` is already absolute, returns it unchanged.
    - If `p` is relative, joins it with the project root (the folder
      containing signals.py).
    - Handles both forward slashes and backslashes so legacy CSVs from
      Windows builds keep working on Linux.
    """
    if not isinstance(p, str):
        return p
    p_norm = p.replace("\\", "/")
    if os.path.isabs(p_norm):
        return p_norm
    return os.path.normpath(os.path.join(_PROJECT_ROOT, p_norm))

def iq_to_histogram_image(iq: np.ndarray,
                          image_size: int = 224,
                          view_limit: float = 2.5,
                          log_scale: bool = True) -> np.ndarray:
    """2-D density histogram -> uint8 grayscale array of shape (H, W)."""
    re = np.clip(iq.real, -view_limit, view_limit)
    im = np.clip(iq.imag, -view_limit, view_limit)
    edges = np.linspace(-view_limit, view_limit, image_size + 1)
    H, _, _ = np.histogram2d(im, re, bins=[edges, edges])
    H = np.flipud(H)                                # positive Q is up
    if log_scale:
        H = np.log1p(H)
    if H.max() > 0:
        H = H / H.max()
    return (255.0 * H).astype(np.uint8)


def render_constellation(iq: np.ndarray,
                         mode: str = "histogram",
                         image_size: int = 224,
                         view_limit: float = 2.5,
                         log_scale: bool = True) -> Image.Image:
    """
    Return an RGB PIL.Image (image_size x image_size).
    `mode` = "histogram" (recommended for models) or "scatter" (matplotlib).
    """
    if mode == "histogram":
        arr = iq_to_histogram_image(iq, image_size, view_limit, log_scale)
        return Image.fromarray(arr, mode="L").convert("RGB")

    if mode == "scatter":
        dpi = 100
        figsize = image_size / dpi
        fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
        ax.scatter(iq.real, iq.imag, s=1, c="#1f77b4", alpha=0.4)
        ax.set_xlim(-view_limit, view_limit); ax.set_ylim(-view_limit, view_limit)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, pad_inches=0)
        plt.close(fig); buf.seek(0)
        return Image.open(buf).convert("RGB").resize((image_size, image_size))

    raise ValueError(f"Unknown render mode: {mode}")


# =============================================================================
# Section 4 — Self-test  ($ python signals.py)
# =============================================================================

def _selftest():
    """Build all alphabets, apply impairments, render images. Save 3 figures."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    # ---- (a) energy sanity ----
    print(f"{'Scheme':<12}  M    mean|s|^2   peak|s|^2")
    print("-" * 45)
    for sch in ALL_MODULATIONS:
        c = get_constellation(sch)
        print(f"{sch:<12}  {len(c):<4} "
              f"{np.mean(np.abs(c)**2):.4f}      {np.max(np.abs(c)**2):.4f}")

    # ---- (b) clean alphabets figure ----
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    for ax, sch in zip(axes.flat, ALL_MODULATIONS):
        c = get_constellation(sch)
        ax.scatter(c.real, c.imag, s=18, c="tab:blue")
        ax.set_title(f"{sch}  (M={len(c)})", fontsize=10)
        ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Clean constellation alphabets", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "00_clean_constellations.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---- (c) impairment scenarios ----
    showcase = ["QPSK", "16-QAM", "64-QAM", "16-APSK"]
    scenarios = [
        ("clean (SNR=30)",        dict(snr_db=30.0)),
        ("AWGN (SNR=10)",         dict(snr_db=10.0)),
        ("phase noise 6 deg",     dict(snr_db=20.0, phase_noise_sigma_rad=np.deg2rad(6))),
        ("IQ imb. eps=.15 psi=8", dict(snr_db=20.0, iq_eps=0.15, iq_psi_rad=np.deg2rad(8))),
        ("CW jammer SIR=5",       dict(snr_db=20.0, jammer_sir_db=5.0, jammer_freq_norm=0.07)),
        ("all impairments",       dict(snr_db=15.0, phase_noise_sigma_rad=np.deg2rad(4),
                                       iq_eps=0.10, iq_psi_rad=np.deg2rad(5),
                                       jammer_sir_db=10.0, jammer_freq_norm=0.05)),
    ]
    fig, axes = plt.subplots(len(scenarios), len(showcase),
                             figsize=(3.2*len(showcase), 3.0*len(scenarios)))
    for r, (lbl, par) in enumerate(scenarios):
        for c, sch in enumerate(showcase):
            ax = axes[r, c]
            s = generate_symbols(sch, 4000, rng=rng)
            rx = apply_channel(s, rng=rng, **par)
            ax.scatter(rx.real, rx.imag, s=2, c="tab:red", alpha=0.4)
            ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.25)
            if r == 0: ax.set_title(sch, fontsize=11)
            if c == 0: ax.set_ylabel(lbl, fontsize=9)
    fig.suptitle("Received constellations under different impairments", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "01_impaired_examples.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---- (d) histogram-image renders ----
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax, sch in zip(axes.flat, ALL_MODULATIONS):
        s = generate_symbols(sch, 2048, rng=rng)
        rx = apply_channel(s, snr_db=18.0,
                           phase_noise_sigma_rad=np.deg2rad(2),
                           iq_eps=0.05, iq_psi_rad=np.deg2rad(2),
                           rng=rng)
        img = render_constellation(rx, mode="histogram")
        ax.imshow(np.array(img), cmap="gray")
        ax.set_title(sch, fontsize=10); ax.axis("off")
    fig.suptitle("Histogram constellation images (SNR=18 dB, mild impairments)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "02_histogram_images.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:")
    print(f"  {os.path.join(results_dir, '00_clean_constellations.png')}")
    print(f"  {os.path.join(results_dir, '01_impaired_examples.png')}")
    print(f"  {os.path.join(results_dir, '02_histogram_images.png')}")


if __name__ == "__main__":
    _selftest()
