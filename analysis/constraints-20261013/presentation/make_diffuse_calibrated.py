"""Slide-ready survey-calibrated diffuse sky versus observed regional profiles.

Run after compare_radio_regions.py writes calibration/diffuse_survey_conditioned/comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "analysis/constraints-20261013/calibration/diffuse_survey_conditioned/comparisons/radio_regions.json"
)
COLORS = {"10.03": "#6550a4", "22": "#137d75", "45south": "#d66e29"}
LABELS = {"10.03": "10.03 MHz", "22": "22 MHz", "45south": "45 MHz (south)"}
REGIONS = [
    ("anticenter_north", "Anticenter north", "160°–200° longitude · 15°–40° latitude"),
    ("orion_south", "Orion–Eridanus south", "205°–220° longitude · −45° to −20° latitude"),
]


def main() -> None:
    data = json.loads(SOURCE.read_text())
    if not any("diffuse_survey_conditioned" in path for path in data["inputs"]):
        raise ValueError("Comparison JSON does not reference the survey-conditioned FITS")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.titlesize": 21,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "svg.fonttype": "none",
            "savefig.dpi": 180,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor("white")
    for ax, (key, title, bounds) in zip(axes, REGIONS):
        region = data["regions"][key]
        for freq_id, series in region["by_frequency"].items():
            rows = series["longitude_bins"]
            x = np.array([0.5 * (r["lon_min_deg"] + r["lon_max_deg"]) for r in rows])
            measured = np.array([r["survey_K_q16_q50_q84"][1] for r in rows]) / 1000
            model = np.array([r["model_K_q16_q50_q84"][1] for r in rows]) / 1000
            color = COLORS[freq_id]
            ax.plot(
                x,
                measured,
                color=color,
                lw=3.4,
                marker="o",
                ms=10,
                label=f"{LABELS[freq_id]} measured",
            )
            ax.plot(
                x,
                model,
                color=color,
                lw=3.3,
                ls="--",
                marker="s",
                ms=8,
                label=f"{LABELS[freq_id]} current lowsky",
            )
        ax.set_yscale("log")
        ax.set_title(f"{title}\n{bounds}", pad=18, color="#172b36")
        ax.set_xlabel("Galactic longitude (degrees)", labelpad=10)
        ax.set_ylabel("Brightness temperature (thousand K)", labelpad=10)
        ax.grid(alpha=0.2, which="both")
        ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#dddddd", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Survey-calibrated diffuse sky", fontsize=29, y=0.95, fontweight="bold", color="#172b36"
    )
    fig.text(
        0.5,
        0.078,
        "Data: Caswell et al. 10 MHz · Roger et al. 22 MHz · Guzmán et al. southern 45 MHz",
        ha="center",
        fontsize=13,
        color="#4c5c63",
    )
    fig.text(
        0.5,
        0.05,
        "These surveys calibrate the smooth diffuse correction; profiles are conditional checks, not independent validation.",
        ha="center",
        fontsize=12,
        color="#4c5c63",
    )
    fig.text(
        0.5,
        0.024,
        "Common 12° smoothing and source masks; 22 MHz scale shares Haslam lineage. Profiles show medians, not error bars.",
        ha="center",
        fontsize=11,
        color="#4c5c63",
    )
    fig.subplots_adjust(left=0.115, right=0.975, bottom=0.19, top=0.81, wspace=0.25)
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"diffuse_calibrated.{ext}", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
