#!/usr/bin/env python3
"""Compare source-free and analytic point-source angular power.

The input is the canonical, un-beamed harmonic NPZ produced by
``counterfactual_ulsa.py``.  No map synthesis or pixel window is involved in
this analysis: C_ell is evaluated directly from the packed healpy alms.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


def power_from_alms(alms: np.ndarray) -> np.ndarray:
    """Return one auto-spectrum per leading row of packed healpy alms."""
    alms = np.asarray(alms)
    if alms.ndim != 2:
        raise ValueError(f"expected a 2-D (frequency, alm) array, got {alms.shape}")
    return np.stack([hp.alm2cl(row) for row in alms])


def first_sustained_dominance(
    ratio: np.ndarray, *, ell_min: int = 1, ell_max: int | None = None, consecutive: int = 10
) -> int | None:
    """First ell beginning ``consecutive`` bins with source/diffuse >= 1."""
    ratio = np.asarray(ratio)
    if ratio.ndim != 1:
        raise ValueError("ratio must be one-dimensional")
    stop = ratio.size if ell_max is None else min(ratio.size, ell_max + 1)
    for ell in range(ell_min, stop - consecutive + 1):
        if np.all(np.isfinite(ratio[ell : ell + consecutive])) and np.all(
            ratio[ell : ell + consecutive] >= 1.0
        ):
            return ell
    return None


def choose_panel_indices(frequencies_mhz: np.ndarray, count: int = 6) -> np.ndarray:
    """Select approximately log-spaced frequencies, preserving endpoints."""
    targets = np.geomspace(frequencies_mhz[0], frequencies_mhz[-1], count)
    indices = np.asarray([np.argmin(abs(frequencies_mhz - target)) for target in targets])
    return np.unique(indices)


def analyze(input_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    with np.load(input_path, allow_pickle=False) as product:
        required = {"no_sources", "ateam", "frequency_mhz", "lmax"}
        missing = required.difference(product.files)
        if missing:
            raise KeyError(f"missing required arrays: {sorted(missing)}")
        diffuse_alms = np.asarray(product["no_sources"])
        source_alms = np.asarray(product["ateam"])
        frequencies = np.asarray(product["frequency_mhz"], dtype=float)
        lmax = int(product["lmax"])
        input_nside = int(product["input_nside"]) if "input_nside" in product.files else -1
        source_component_keys = sorted(
            key for key in product.files if key.startswith("ateam_") and key != "ateam"
        )
        source_component_alms = [np.asarray(product[key]) for key in source_component_keys]

    expected_nalm = hp.Alm.getsize(lmax)
    expected_shape = (frequencies.size, expected_nalm)
    if diffuse_alms.shape != expected_shape or source_alms.shape != expected_shape:
        raise ValueError(
            f"alm arrays must have shape {expected_shape}; got "
            f"{diffuse_alms.shape} and {source_alms.shape}"
        )

    diffuse_cl = power_from_alms(diffuse_alms)
    source_cl = power_from_alms(source_alms)
    shot_noise_cl = np.sum([power_from_alms(alm) for alm in source_component_alms], axis=0)
    ell = np.arange(lmax + 1)
    validated_ell_max = input_nside
    ratio = np.divide(
        source_cl,
        diffuse_cl,
        out=np.full_like(source_cl, np.nan),
        where=diffuse_cl > 0,
    )
    crossings = np.asarray(
        [
            -1
            if (
                cross := first_sustained_dominance(row, ell_max=validated_ell_max)
            )
            is None
            else cross
            for row in ratio
        ],
        dtype=int,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    spectra_path = output_dir / "diffuse_vs_point_source_cl.npz"
    np.savez_compressed(
        spectra_path,
        frequency_mhz=frequencies,
        ell=ell,
        diffuse_cl_k2=diffuse_cl,
        point_source_cl_k2=source_cl,
        incoherent_shot_noise_cl_k2=shot_noise_cl,
        point_source_to_diffuse_ratio=ratio,
        first_sustained_source_dominance_ell=crossings,
        sustained_bins=np.int32(10),
        beam_applied=np.bool_(False),
        diffuse_input_nside=np.int32(input_nside),
        validated_diffuse_ell_max=np.int32(validated_ell_max),
    )

    summary_path = output_dir / "diffuse_vs_point_source_crossover.csv"
    sample_ells = [10, 20, 50, 100, lmax]
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["frequency_mhz", "first_sustained_source_dominance_ell"]
            + [f"source_to_diffuse_at_ell_{value}" for value in sample_ells]
        )
        for fi, frequency in enumerate(frequencies):
            writer.writerow(
                [
                    f"{frequency:.8g}",
                    "" if crossings[fi] < 0 else str(crossings[fi]),
                    *[f"{ratio[fi, value]:.8g}" for value in sample_ells],
                ]
            )

    panel_indices = choose_panel_indices(frequencies)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), sharex=True)
    for ax, fi in zip(axes.flat, panel_indices):
        valid = ell >= 1
        ax.loglog(ell[valid], diffuse_cl[fi, valid], color="#276FBF", lw=2.0, label="Extended / source-free")
        ax.loglog(ell[valid], source_cl[fi, valid], color="#D1495B", lw=1.8, label="A-team deltas")
        ax.loglog(
            ell[valid],
            shot_noise_cl[fi, valid],
            color="#D1495B",
            lw=1.1,
            ls="--",
            label="Incoherent shot noise",
        )
        ax.fill_between(
            ell[valid],
            diffuse_cl[fi, valid],
            source_cl[fi, valid],
            where=source_cl[fi, valid] >= diffuse_cl[fi, valid],
            color="#D1495B",
            alpha=0.10,
        )
        cross = crossings[fi]
        if cross >= 0:
            ax.axvline(cross, color="0.35", ls=":", lw=1)
            ax.text(
                0.04,
                0.07,
                rf"10-bin dominance: $\ell\geq {cross}$",
                transform=ax.transAxes,
                fontsize=8.5,
                color="0.25",
            )
        if validated_ell_max > 0 and validated_ell_max < lmax:
            ax.axvspan(validated_ell_max, lmax, color="0.2", alpha=0.035, hatch="//", lw=0)
        ax.set_title(f"{frequencies[fi]:g} MHz")
        ax.grid(which="both", alpha=0.18)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$C_\ell$ [K$^2$]")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Multipole $\ell$")
    axes.flat[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Canonical un-beamed harmonic sky: diffuse emission vs analytic A-team sources", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Shading marks individual multipoles where point-source power exceeds diffuse power; "
        "vertical lines require 10 consecutive bins. Combined-source interference makes its Cℓ non-flat. "
        "The dashed curve is the flat incoherent shot-noise sum. "
        "Hatching marks the range beyond the validated diffuse limit (ℓ=64).",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    plot_path = output_dir / "diffuse_vs_point_source_cl.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path, spectra_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs_our_sky_harmonic/counterfactual_ulsa_harmonic.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.input.parent
    for path in analyze(args.input, output_dir):
        print(path)


if __name__ == "__main__":
    main()
