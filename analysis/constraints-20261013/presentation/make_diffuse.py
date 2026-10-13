"""Make slide-ready baseline radio-survey validation plots from the matched comparison JSON.

Run: .venv/bin/python analysis/constraints-20261013/presentation/make_diffuse.py
"""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "comparisons/radio_regions/radio_regions.json"
PALETTE = {"10.03": "#6A3D9A", "22": "#157F66", "45south": "#E07B28"}
LABELS = {"10.03": "10.03 MHz", "22": "22 MHz", "45south": "45 MHz (south)"}
REGIONS = [
    ("anticenter_north", "Anticenter north", "160°–200° longitude, 15°–40° latitude"),
    ("orion_south", "Orion–Eridanus south", "205°–220° longitude, −45° to −20° latitude"),
]


def main():
    data = json.loads(SOURCE.read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("white")
    for ax, (key, title, subtitle) in zip(axes, REGIONS):
        ax.set_facecolor("white")
        region = data["regions"][key]
        for freq_id, series in region["by_frequency"].items():
            rows = series["longitude_bins"]
            x = np.array([0.5 * (r["lon_min_deg"] + r["lon_max_deg"]) for r in rows])
            observed = np.array([r["survey_K_q16_q50_q84"][1] for r in rows]) / 1000
            model = np.array([r["model_K_q16_q50_q84"][1] for r in rows]) / 1000
            color = PALETTE[freq_id]
            ax.plot(
                x,
                observed,
                color=color,
                lw=3.2,
                marker="o",
                ms=9,
                label=f"{LABELS[freq_id]} measured",
            )
            ax.plot(
                x,
                model,
                color=color,
                lw=3.0,
                ls="--",
                marker="s",
                ms=8,
                label=f"{LABELS[freq_id]} lowsky",
            )
        ax.set_yscale("log")
        ax.set_title(title + "\n" + subtitle, pad=17)
        ax.set_xlabel("Galactic longitude (degrees)", labelpad=11)
        ax.set_ylabel("Brightness temperature (thousand K)", labelpad=11)
        ax.grid(alpha=0.18, which="both")
        ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#dddddd", ncol=1)
        # Common cell counts are retained in the source JSON, not overlaid on profiles.
    fig.suptitle("Diffuse sky: lowsky and radio surveys", fontsize=27, y=0.955, fontweight="bold")
    fig.text(
        0.5,
        0.065,
        "Surveys: Caswell et al. 10 MHz · Roger et al. 22 MHz · Guzmán et al. southern 45 MHz",
        ha="center",
        fontsize=12,
        color="#444444",
    )
    fig.text(
        0.5,
        0.038,
        "Shared 12° smoothing; no gain/offset fit or error bars. 22 MHz scale uses Haslam; survey/model inputs overlap.",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.15, top=0.82, wspace=0.22)
    for suffix in ["png", "svg"]:
        fig.savefig(HERE / f"diffuse_baseline.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    print(HERE / "diffuse_baseline.png")


if __name__ == "__main__":
    main()
