"""Reproduce the conditional Cas A peak calibration and before/after plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
C = 299792458.0
K = 1.380649e-23


def spectrum(nu: np.ndarray, s50: float, tau10: float, alpha: float, foreground_tau50: float) -> np.ndarray:
    """Same Cas A flux law as ateam_integrated_k_sr, at positive frequencies <=50 MHz."""
    return s50 * (nu / 50) ** alpha * np.exp(
        -tau10 * ((nu / 10) ** -2.1 - 5 ** -2.1)
        -foreground_tau50 * (nu / 50) ** -2.1
    )


def main() -> None:
    fit = json.loads((HERE / "fit.json").read_text())
    m = fit["model"]
    measurements = json.loads((ROOT / "analysis/constraints-20261013/objects_bright_sources/measurements.json").read_text())
    artifact = ROOT / "outputs_our_sky_gum_gaussian_final/counterfactual_ulsa_harmonic.npz"
    with np.load(artifact, allow_pickle=False) as data:
        frequencies = np.asarray(data["frequency_mhz"], float)
        unit = 2 * K * (frequencies * 1e6) ** 2 / C**2 * 1e26
        cas_saved = data["ateam_cas_a"][:, 0].real * np.sqrt(4 * np.pi) * unit
        cyg_saved = data["ateam_cyg_a"][:, 0].real * np.sqrt(4 * np.pi) * unit
    alpha = m["fixed_alpha"]
    foreground = m["foreground_tau_at_50mhz"]
    fine = np.linspace(8, 50, 1000)
    before = spectrum(fine, m["old_s50_jy"], m["old_tau10"], alpha, foreground)
    after = spectrum(fine, m["new_s50_jy"], m["new_tau10"], alpha, foreground)
    before_grid = spectrum(frequencies, m["old_s50_jy"], m["old_tau10"], alpha, foreground)
    assert np.allclose(before_grid[frequencies >= 8], cas_saved[frequencies >= 8], rtol=2e-5)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 180})
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(fine, before / 1000, color="#555f69", lw=2, ls="--", label="Before: saved source prescription")
    ax.plot(fine, after / 1000, color="#176f80", lw=2.5, label="After: calibrated to 2023.5 peak")
    points = []
    for row in measurements["observational_records"]:
        if row.get("object") != "CAS_A" or row.get("kind") != "fit_derived_peak":
            continue
        f = row["frequency_mhz"]
        ax.errorbar(f, row["value"] / 1000, xerr=.2 if row["epoch"] == "2023.5" else .5,
                    yerr=row["reported_pm"] / 1000, marker="o", ms=7, capsize=3, lw=1.4,
                    color="#ae482c" if row["epoch"] == "2023.5" else "#4b75a8",
                    label=f"{row['epoch']} fitted peak")
        points.append({"epoch": row["epoch"], "frequency_mhz": f, "observed_peak_jy": row["value"],
                       "before_jy_at_frequency": float(spectrum(np.array([f]), m["old_s50_jy"], m["old_tau10"], alpha, foreground)[0]),
                       "after_jy_at_frequency": float(spectrum(np.array([f]), m["new_s50_jy"], m["new_tau10"], alpha, foreground)[0])})
    ax.set(xlim=(8, 50), ylim=(0, 73), xlabel="Frequency (MHz)", ylabel="Integrated Cas A flux (kJy)")
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Cas A: conditional 2023.5 turnover calibration", weight="bold")
    fig.text(.5, -.02, "Fit-derived peak summaries share Cyg A calibration; lowsky has no epoch or matched interferometer response.\n"
             "The 2023.5 peak sets both updated parameters and is calibration, not validation.", ha="center", va="top", fontsize=8.5)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"before_after_turnover.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    nu = np.linspace(35, 50, 200)
    cyg = np.interp(nu, frequencies, cyg_saved)
    ax.plot(nu, spectrum(nu, m["old_s50_jy"], m["old_tau10"], alpha, foreground) / cyg,
            ls="--", color="#555f69", lw=2, label="Before source-only ratio")
    ax.plot(nu, spectrum(nu, m["new_s50_jy"], m["new_tau10"], alpha, foreground) / cyg,
            color="#176f80", lw=2.5, label="After source-only ratio")
    ratio_rows = []
    for row in measurements["observational_records"]:
        if row.get("source_id") != "lwa1_llfss_2017" or row["object"] != "CAS_A":
            continue
        x = row["frequency_mhz_center"]
        ax.errorbar(x, row["ratio_to_cyg_a"], yerr=row["reported_pm"], fmt="o",
                    color="#ae482c" if x <= 50.01 else "#9b9b9b", ms=5.5, capsize=2.5)
        if x <= 50.01:
            f = min(x, 50.0)
            denom = float(np.interp(f, frequencies, cyg_saved))
            ratio_rows.append({"actual_center_mhz": x, "compared_frequency_mhz": f,
                               "observed_ratio": row["ratio_to_cyg_a"], "reported_pm": row["reported_pm"],
                               "before_ratio": float(spectrum(np.array([f]), m["old_s50_jy"], m["old_tau10"], alpha, foreground)[0] / denom),
                               "after_ratio": float(spectrum(np.array([f]), m["new_s50_jy"], m["new_tau10"], alpha, foreground)[0] / denom)})
    ax.axvspan(50.02, 82, color="#e9ebee", zorder=-2)
    ax.set(xlim=(33, 82), ylim=(.55, 1.9), xlabel="Frequency (MHz)", ylabel="Cas A / Cyg A flux ratio")
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("LWA1 ratio diagnostic: turnover fit changes the 35–50 MHz prediction", weight="bold")
    fig.text(.5, -.02, "LWA1 ratios were not fitted. Model lines are integrated source-only fluxes, not LWA1 annulus/Gaussian estimates.\n"
             "Gray points lie above 50 MHz and have no model prediction; 50.005-MHz center uses nominal 50 MHz.",
             ha="center", va="top", fontsize=8.5)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"lwa1_ratio_tradeoff.{ext}", bbox_inches="tight")
    plt.close(fig)
    (HERE / "fit_comparison.json").write_text(json.dumps({"peak_checks": points, "lwa1_ratio_checks": ratio_rows,
        "model_fifty_mhz_jy": {"before": float(cas_saved[-1]),
                                "after": float(spectrum(np.array([50.]), m["new_s50_jy"], m["new_tau10"], alpha, foreground)[0])},
        "note": "2023.5 peak is fitted calibration; all other points are contextual diagnostics, with epoch/measurement-operator mismatch"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
