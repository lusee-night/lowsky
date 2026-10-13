"""Plot published bright-source constraints against saved lowsky A-team maps.

The source-only diagnostic maps integrate to the canonical source amplitudes,
but this script does not mimic a survey beam, background annulus, or epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "analysis/constraints-20261013/objects_bright_sources/measurements.json"
DEFAULT_ARTIFACT = ROOT / "outputs_our_sky_gum_gaussian_final/ateam_components.npz"
DEFAULT_OUTPUT = ROOT / "analysis/constraints-20261013/comparisons/bright_sources"
SOURCES = ("CAS_A", "CYG_A", "TAU_A", "VIR_A")
C_LIGHT = 299_792_458.0
K_BOLTZMANN = 1.380_649e-23


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integrated_flux_jy(map_k: np.ndarray, frequency_mhz: np.ndarray) -> np.ndarray:
    """Integrate a full-sky source-only Kelvin map and invert Rayleigh-Jeans."""
    if map_k.ndim != 2 or frequency_mhz.shape != (map_k.shape[0],):
        raise ValueError("Expected a (frequency, pixel) map and matching frequencies")
    if not np.all(np.isfinite(map_k)) or not np.all(frequency_mhz > 0):
        raise ValueError("Maps must be finite and frequencies positive")
    omega_pix = 4.0 * np.pi / map_k.shape[1]
    integrated_k_sr = np.sum(map_k, axis=1) * omega_pix
    return integrated_k_sr * 2.0 * K_BOLTZMANN * (frequency_mhz * 1e6) ** 2 / C_LIGHT**2 * 1e26


def load_flux(artifact: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(artifact, allow_pickle=False) as data:
        frequency = np.asarray(data["frequency_mhz"], dtype=float)
        if frequency.ndim != 1 or np.any(np.diff(frequency) <= 0):
            raise ValueError("Frequency grid must be strictly increasing")
        flux = {key: integrated_flux_jy(np.asarray(data[key]), frequency) for key in SOURCES}
    return frequency, flux


def prepare_comparisons(measurements: dict, frequency: np.ndarray, flux: dict) -> dict:
    ratio_rows = []
    for record in measurements["observational_records"]:
        if record.get("source_id") != "lwa1_llfss_2017":
            continue
        nominal = record["frequency_mhz_nominal"]
        # The center of the nominal 50-MHz LLFSS band is 50.005 MHz, 5 kHz
        # outside the saved 1--50 MHz grid. Evaluate at exactly 50 MHz.
        actual_center = float(record["frequency_mhz_center"])
        approximate_nominal_50 = nominal == 50 and np.isclose(actual_center, 50.005)
        in_range = bool(float(frequency[0]) <= actual_center <= float(frequency[-1]))
        comparison_frequency = float(frequency[-1]) if approximate_nominal_50 else actual_center
        if not in_range and not approximate_nominal_50:
            prediction = None
        else:
            numerator = np.interp(comparison_frequency, frequency, flux[record["object"]])
            denominator = np.interp(comparison_frequency, frequency, flux["CYG_A"])
            prediction = float(numerator / denominator)
        observed = record["ratio_to_cyg_a"]
        ratio_rows.append(
            {
                "object": record["object"],
                "nominal_mhz": nominal,
                "actual_center_mhz": record["frequency_mhz_center"],
                "comparison_mhz": comparison_frequency if prediction is not None else None,
                "observed_ratio": observed,
                "reported_pm": record["reported_pm"],
                "model_ratio_source_only": prediction,
                "model_minus_observed": None if prediction is None else prediction - observed,
                "model_minus_observed_over_reported_pm": (
                    None if prediction is None else (prediction - observed) / record["reported_pm"]
                ),
                "in_saved_model_range": in_range,
                "comparison_available": prediction is not None,
                "nominal_frequency_approximation": bool(approximate_nominal_50),
                "note": (
                    "50.005-MHz band evaluated at nominal 50 MHz"
                    if approximate_nominal_50
                    else None
                ),
            }
        )
    cas_rows = []
    for record in measurements["observational_records"]:
        if record.get("object") != "CAS_A" or record.get("kind") != "fit_derived_peak":
            continue
        freq = record["frequency_mhz"]
        pred = float(np.interp(freq, frequency, flux["CAS_A"]))
        cas_rows.append(
            {
                "epoch": record["epoch"],
                "observed_peak_frequency_mhz": freq,
                "frequency_reported_pm_mhz": 0.2 if record["epoch"] == "2023.5" else 0.5,
                "observed_peak_jy": record["value"],
                "reported_pm_jy": record["reported_pm"],
                "model_total_flux_jy_at_reported_peak": pred,
                "model_minus_observed_jy": pred - record["value"],
                "model_minus_observed_over_reported_pm": (pred - record["value"])
                / record["reported_pm"],
                "warning": "Fit-derived peak, shared Cyg A flux scale, no model epoch correction or matched aperture",
            }
        )
    return {"lwa1_ratios": ratio_rows, "cas_peak_summaries": cas_rows}


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
        }
    )


def plot_ratios(out: Path, frequency: np.ndarray, flux: dict, comparisons: dict) -> None:
    colors = {"CAS_A": "#b4442a", "TAU_A": "#245b92", "VIR_A": "#23755b"}
    labels = {"CAS_A": "Cas A / Cyg A", "TAU_A": "Tau A / Cyg A", "VIR_A": "Vir A / Cyg A"}
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.1), sharex=True, constrained_layout=True)
    rows = comparisons["lwa1_ratios"]
    for ax, source in zip(axes, colors):
        color = colors[source]
        ax.axvspan(50.02, 82, color="#e9ebee", zorder=0)
        selected = sorted(
            (r for r in rows if r["object"] == source), key=lambda r: r["nominal_mhz"]
        )
        x = np.array([r["actual_center_mhz"] for r in selected])
        y = np.array([r["observed_ratio"] for r in selected])
        err = np.array([r["reported_pm"] for r in selected])
        ax.errorbar(
            x[x <= 50.02],
            y[x <= 50.02],
            yerr=err[x <= 50.02],
            fmt="o",
            color=color,
            ms=5.2,
            capsize=2.5,
            lw=1.2,
            label="LWA1 overlap / nominal 50 MHz",
        )
        ax.errorbar(
            x[x > 50.02],
            y[x > 50.02],
            yerr=err[x > 50.02],
            fmt="o",
            mfc="white",
            mec=color,
            ecolor=color,
            color=color,
            ms=5.2,
            capsize=2.5,
            lw=1.2,
            label="LWA1 above saved model range",
        )
        mask = (frequency >= 30) & (frequency <= 50)
        ax.plot(
            frequency[mask],
            flux[source][mask] / flux["CYG_A"][mask],
            color="#202530",
            lw=2.2,
            label="saved pre-calibration source-only ratio",
        )
        ax.text(
            0.02, 0.93, labels[source], transform=ax.transAxes, ha="left", va="top", weight="bold"
        )
        ax.grid(alpha=0.18, lw=0.6)
        ax.set_ylabel("Flux ratio")
        ax.set_xlim(31.5, 82)
    axes[0].legend(frameon=False, fontsize=8.4, loc="upper right")
    axes[-1].set_xlabel("Frequency (MHz)")
    fig.suptitle(
        "Bright-source ratios: LWA1 and saved pre-calibration baseline", fontsize=13, weight="bold"
    )
    fig.text(
        0.5,
        -0.01,
        "Model curves integrate source-only maps; LWA1 used a beam, local annulus, and Gaussian fit.\n"
        "Cyg A sets the LWA1 flux scale; points share calibration/systematic errors. Shading: above 50 MHz model grid.",
        ha="center",
        va="top",
        fontsize=8.7,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"lwa1_source_ratios.{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_cas(out: Path, frequency: np.ndarray, flux: dict, comparisons: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    mask = (frequency >= 8) & (frequency <= 50)
    ax.plot(
        frequency[mask],
        flux["CAS_A"][mask] / 1000,
        color="#202530",
        lw=2.3,
        label="Saved pre-calibration Cas A, total source flux",
    )
    colors = {"2019": "#4579a3", "2023.5": "#b4442a"}
    for row in comparisons["cas_peak_summaries"]:
        ax.errorbar(
            row["observed_peak_frequency_mhz"],
            row["observed_peak_jy"] / 1000,
            xerr=row["frequency_reported_pm_mhz"],
            yerr=row["reported_pm_jy"] / 1000,
            fmt="o",
            color=colors[row["epoch"]],
            ms=7,
            lw=1.5,
            capsize=3,
            label=f"{row['epoch']} observed fitted peak",
        )
    ax.set(
        xlim=(8, 51),
        ylim=(0, max(65, np.max(flux["CAS_A"][mask]) / 1000 * 1.08)),
        xlabel="Frequency (MHz)",
        ylabel="Integrated flux density (kJy)",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle(
        "Cas A turnover: observed peak versus saved pre-calibration baseline",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.5,
        -0.035,
        "Observed symbols are fit-derived peak summaries, not sampled spectrum points. GURT (2019) and\n"
        "NenuFAR/URAN-2 (2023.5) use Cyg A calibration; lowsky has no epoch evolution or matched interferometric response.",
        ha="center",
        va="top",
        fontsize=8.7,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"cas_a_turnover.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact, measurements_path, out = (
        args.artifact.resolve(),
        args.measurements.resolve(),
        args.output_dir.resolve(),
    )
    out.mkdir(parents=True, exist_ok=True)
    measurements = json.loads(measurements_path.read_text())
    frequency, flux = load_flux(artifact)
    if frequency[0] != 1 or frequency[-1] != 50:
        raise ValueError("Expected saved 1--50 MHz model artifact")
    comparisons = prepare_comparisons(measurements, frequency, flux)
    metadata = {
        "source_artifact": str(artifact),
        "source_artifact_sha256": sha256(artifact),
        "artifact_pre_cas_calibration": True,
        "measurement_manifest": str(measurements_path),
        "measurement_manifest_sha256": sha256(measurements_path),
        "run_manifest": str(artifact.with_name("run_manifest.json")),
        "run_manifest_sha256": sha256(artifact.with_name("run_manifest.json")),
        "conversion": "S_Jy = sum_pixels(T_K * 4pi/Npix) * 2 k_B (nu_MHz*1e6)^2 / c^2 * 1e26",
        "model_range_mhz": [float(frequency[0]), float(frequency[-1])],
        "comparison_type": "source-only integrated model flux versus observed ratios and fit-derived peaks; no beam/annulus/epoch matching",
        "uncertainty_warning": "Reported +/- used for visual error bars only; ratios across sources and frequencies share Cyg A calibration and systematics; no chi-square computed.",
        "crab_shell_peak_excluded": "Published 2025 27.5-MHz point is shell-only; lowsky Tau A is total-source and cannot be compared directly.",
    }
    (out / "comparison_values.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "model_flux_jy": {
                    "frequency_mhz": frequency.tolist(),
                    **{key: value.tolist() for key, value in flux.items()},
                },
                **comparisons,
            },
            indent=2,
        )
        + "\n"
    )
    style()
    plot_ratios(out, frequency, flux, comparisons)
    plot_cas(out, frequency, flux, comparisons)


if __name__ == "__main__":
    main()
