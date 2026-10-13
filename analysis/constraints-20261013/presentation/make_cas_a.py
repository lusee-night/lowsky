"""Build slide-ready current-model Cas A validation figures (PNG and SVG)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FIT = ROOT / "analysis/constraints-20261013/calibration/cas_a/fit.json"
MEASUREMENTS = ROOT / "analysis/constraints-20261013/objects_bright_sources/measurements.json"
HARMONIC = ROOT / "outputs_our_sky_gum_gaussian_final/counterfactual_ulsa_harmonic.npz"
TEAL = "#087f8c"
ORANGE = "#e2762d"
INK = "#172b36"


def current_cas(nu: np.ndarray, model: dict) -> np.ndarray:
    return (
        model["new_s50_jy"]
        * (nu / 50) ** model["fixed_alpha"]
        * np.exp(
            -model["new_tau10"] * ((nu / 10) ** -2.1 - 5**-2.1)
            - model["foreground_tau_at_50mhz"] * (nu / 50) ** -2.1
        )
    )


def finish(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"{stem}.{ext}", facecolor="white")
    plt.close(fig)


def main() -> None:
    fit = json.loads(FIT.read_text())["model"]
    records = json.loads(MEASUREMENTS.read_text())["observational_records"]
    with np.load(HARMONIC, allow_pickle=False) as data:
        fgrid = np.asarray(data["frequency_mhz"], dtype=float)
        a00 = np.asarray(data["ateam_cyg_a"][:, 0].real)
    cyg_jy = (
        a00 * np.sqrt(4 * np.pi) * (2 * 1.380649e-23 * (fgrid * 1e6) ** 2 / 299792458.0**2) * 1e26
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 17,
            "axes.titlesize": 27,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "savefig.dpi": 180,
        }
    )

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.86, bottom=0.20)
    nu = np.linspace(8, 50, 600)
    ax.plot(nu, current_cas(nu, fit) / 1000, color=TEAL, lw=4, label="Current lowsky Cas A")
    for row in records:
        if row.get("object") != "CAS_A" or row.get("kind") != "fit_derived_peak":
            continue
        epoch = row["epoch"]
        ax.errorbar(
            row["frequency_mhz"],
            row["value"] / 1000,
            xerr=0.2 if epoch == "2023.5" else 0.5,
            yerr=row["reported_pm"] / 1000,
            fmt="o",
            ms=12,
            lw=2,
            capsize=5,
            color=ORANGE,
            markeredgecolor="white",
            markeredgewidth=1.2,
            label=f"Observed fitted peak ({epoch})",
        )
    ax.set(
        xlim=(8, 50), ylim=(0, 43), xlabel="Frequency (MHz)", ylabel="Cas A integrated flux (kJy)"
    )
    ax.grid(color="#cfd7da", alpha=0.65)
    ax.legend(frameon=False, fontsize=16, loc="upper right")
    ax.set_title("Cas A low-frequency turnover", color=INK, weight="bold", pad=20)
    fig.text(
        0.5,
        0.065,
        "2023.5 fitted peak calibrates the model; it is not independent validation. 2019 is an epoch-mismatched check.\n"
        "Data: Stanislavsky et al., A&A 670 A157 (2019 GURT); arXiv:2312.09764 (2023.5 NenuFAR/URAN-2).",
        ha="center",
        va="center",
        fontsize=12,
        color="#4c5c63",
    )
    finish(fig, "cas_a_spectrum")

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.86, bottom=0.20)
    nu = np.linspace(34.979, 50, 500)
    model_ratio = current_cas(nu, fit) / np.interp(nu, fgrid, cyg_jy)
    ax.plot(nu, model_ratio, color=TEAL, lw=4, label="Current lowsky Cas A / Cyg A")
    selected = sorted(
        (
            r
            for r in records
            if r.get("source_id") == "lwa1_llfss_2017"
            and r.get("object") == "CAS_A"
            and r["frequency_mhz_nominal"] <= 50
        ),
        key=lambda r: r["frequency_mhz_nominal"],
    )
    ax.errorbar(
        [r["frequency_mhz_center"] for r in selected],
        [r["ratio_to_cyg_a"] for r in selected],
        yerr=[r["reported_pm"] for r in selected],
        fmt="o",
        ms=11,
        lw=2,
        capsize=5,
        color=ORANGE,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="LWA1 measured ratio (2013)",
    )
    ax.set(
        xlim=(33.8, 51),
        ylim=(0.55, 1.95),
        xlabel="Frequency (MHz)",
        ylabel="Cas A / Cyg A flux ratio",
    )
    ax.grid(color="#cfd7da", alpha=0.65)
    ax.legend(frameon=False, fontsize=16, loc="upper right")
    ax.set_title("Cas A relative flux: LWA1 35–50 MHz", color=INK, weight="bold", pad=20)
    fig.text(
        0.5,
        0.065,
        "Source-only model versus LWA1 beam/annulus photometry; 50.005-MHz band uses nominal 50 MHz.\n"
        "Data: Dowell et al., MNRAS 2017, Table 2; model: lowsky, this work.",
        ha="center",
        va="center",
        fontsize=12,
        color="#4c5c63",
    )
    finish(fig, "cas_a_ratio")


if __name__ == "__main__":
    main()
