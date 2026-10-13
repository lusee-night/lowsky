"""Slide-ready Tau A and Vir A / Cyg A ratios, 35--50 MHz only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples"))
from compare_bright_sources import load_flux  # noqa: E402

MEASUREMENTS = ROOT / "analysis/constraints-20261013/objects_bright_sources/measurements.json"
ARTIFACT = ROOT / "outputs_our_sky_gum_gaussian_final/ateam_components.npz"
OUTPUT = Path(__file__).resolve().parent


def main() -> None:
    records = json.loads(MEASUREMENTS.read_text())["observational_records"]
    frequency, flux = load_flux(ARTIFACT)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 21,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True)
    teal, orange = "#087d80", "#d36a32"
    used = []
    for ax, (object_id, label) in zip(
        axes, [("TAU_A", "Tau A / Cyg A"), ("VIR_A", "Vir A / Cyg A")]
    ):
        chosen = sorted(
            (
                row
                for row in records
                if row.get("source_id") == "lwa1_llfss_2017"
                and row.get("object") == object_id
                and row.get("frequency_mhz_center", 1e9) <= 50.01
            ),
            key=lambda row: row["frequency_mhz_center"],
        )
        if len(chosen) != 5:
            raise ValueError(f"Expected five in-band LWA1 rows for {object_id}; got {len(chosen)}")
        xx = np.array([r["frequency_mhz_center"] for r in chosen], dtype=float)
        yy = np.array([r["ratio_to_cyg_a"] for r in chosen], dtype=float)
        ee = np.array([r["reported_pm"] for r in chosen], dtype=float)
        grid = np.linspace(34.5, 50, 250)
        predicted = np.interp(grid, frequency, flux[object_id]) / np.interp(
            grid, frequency, flux["CYG_A"]
        )
        ax.plot(grid, predicted, color=teal, lw=3.5, label="lowsky source-only ratio", zorder=2)
        ax.errorbar(
            xx,
            yy,
            yerr=ee,
            fmt="o",
            ms=9,
            capsize=4.5,
            capthick=2,
            elinewidth=2,
            color=orange,
            mec="white",
            mew=1,
            label="LWA1 measured ratio ± reported error",
            zorder=3,
        )
        ax.set_title(label, loc="left", weight="bold", pad=14)
        ax.set_xlim(33, 51.5)
        ax.set_xticks([35, 40, 45, 50])
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Flux ratio to Cyg A")
        ax.grid(alpha=0.17, lw=1)
        low = min(np.min(predicted), np.min(yy - ee))
        high = max(np.max(predicted), np.max(yy + ee))
        ax.set_ylim(max(0, low - 0.04), high + 0.05)
        used.extend(
            {
                "object": object_id,
                "frequency_mhz_center": float(r["frequency_mhz_center"]),
                "ratio_to_cyg_a": float(r["ratio_to_cyg_a"]),
                "reported_pm": float(r["reported_pm"]),
            }
            for r in chosen
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=2,
        frameon=False,
        fontsize=15,
    )
    fig.suptitle(
        "Lowsky and observed low-frequency source ratios", fontsize=26, weight="bold", y=0.97
    )
    fig.text(
        0.5,
        0.065,
        "LWA1 LLFSS (2017), MNRAS 469, 4537. Ratios share Cyg A calibration; model uses integrated source-only flux,\n"
        "while LWA1 uses a beam/annulus/Gaussian estimator. The 50.005 MHz band is shown against the 50 MHz model endpoint.",
        ha="center",
        va="center",
        fontsize=12,
        color="#3e4c57",
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.72, bottom=0.17, wspace=0.22)
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / "tau_vir_lwa1_ratios.png", dpi=180)
    fig.savefig(OUTPUT / "tau_vir_lwa1_ratios.svg")
    plt.close(fig)
    metadata = {
        "title": "Tau A and Vir A ratios to Cyg A, observed versus lowsky",
        "source": "https://academic.oup.com/mnras/article/469/4/4537/3815529",
        "measurements_path": str(MEASUREMENTS.relative_to(ROOT)),
        "artifact_path": str(ARTIFACT.relative_to(ROOT)),
        "artifact_sha256": hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
        "model_status": "saved baseline; Tau A, Vir A and Cyg A source parameters unchanged by the Cas A update",
        "measurement_operator": "LWA1 beam/annulus/Gaussian flux ratio; displayed lowsky line is integrated source-only ratio",
        "nominal_50_mhz_note": "Measured band center is 50.005 MHz; compared visually with 50 MHz model endpoint",
        "records": used,
    }
    (OUTPUT / "tau_vir_lwa1_ratios.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
