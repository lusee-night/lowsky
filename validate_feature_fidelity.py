#!/usr/bin/env python3
"""Quantitative literature and numerical-fidelity audit for the conditioned sky."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import matplotlib
import numpy as np
from astropy.io import fits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from counterfactual_ulsa import local_bubble_boundary_kpc


SOURCES = {
    "local_bubble": "https://arxiv.org/abs/2403.04961",
    "loop_i_low_frequency": "https://arxiv.org/abs/2501.00431",
    "loop_i_geometry": "https://arxiv.org/abs/0704.0276",
    "radio_loop_spectra": "https://academic.oup.com/mnras/article/376/2/634/1074743",
    "gum_em": "https://academic.oup.com/mnras/article/315/2/241/981357",
    "orion_nested_shells": "https://arxiv.org/abs/1909.10083",
    "cygnus_filaments": "https://arxiv.org/abs/2205.09193",
    "shell_power_projection": "https://arxiv.org/abs/2012.03975",
}


def angular_mask(nside: int, l0: float, b0: float, radius_deg: float) -> np.ndarray:
    l, b = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    vectors = np.asarray(hp.ang2vec(l, b, lonlat=True))
    center = np.asarray(hp.ang2vec(l0, b0, lonlat=True))
    separation = np.degrees(np.arccos(np.clip(vectors @ center, -1.0, 1.0)))
    return separation < radius_deg


def percentiles(values: np.ndarray) -> dict[str, float]:
    q = np.percentile(values, [5, 25, 50, 75, 95, 99, 100])
    return dict(zip(["p05", "p25", "p50", "p75", "p95", "p99", "max"], map(float, q)))


def audit(output_dir: Path) -> dict[str, object]:
    fits_path = output_dir / "counterfactual_ulsa_nside32_1_50mhz.fits"
    harmonic_path = output_dir / "counterfactual_ulsa_harmonic.npz"
    catalog_path = output_dir / "local_shell_catalog.json"
    convergence_path = output_dir / "shell_resolution_convergence.npz"

    with fits.open(fits_path, memmap=False) as hdus:
        anchor_em = np.asarray(hdus["ANCHOR_EM_TOTAL"].data, dtype=float)
    nside = hp.npix2nside(anchor_em.size)
    regions = {
        "gum": anchor_em[angular_mask(nside, 264.0, -4.0, 35.0)],
        "orion_eridanus": anchor_em[angular_mask(nside, 202.0, -38.0, 35.0)],
        "cygnus_x": anchor_em[angular_mask(nside, 80.0, 0.0, 4.0)],
    }
    regional_em = {name: percentiles(values) for name, values in regions.items()}

    x, y, z = hp.pix2vec(64, np.arange(hp.nside2npix(64)))
    bubble_pc = 1_000.0 * np.asarray(
        local_bubble_boundary_kpc(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z))
    )
    bubble = percentiles(bubble_pc)
    bubble["mean"] = float(np.mean(bubble_pc))
    bubble["wall_transition_pc"] = 35.0

    with catalog_path.open() as handle:
        catalog = json.load(handle)
    loop_checks = {
        "centers_within_6deg": bool(
            np.allclose(catalog["center_l_deg"], [346, 347, 100, 124, 315], atol=6.0)
            and np.allclose(catalog["center_b_deg"], [3, 37, -32.5, 15.5, 48.5], atol=6.0)
        ),
        "contrast_408_k_in_3_9": bool(
            np.all((np.asarray(catalog["excess_408_k"]) >= 3.0) & (np.asarray(catalog["excess_408_k"]) <= 9.0))
        ),
        "temperature_index_in_2p55_3p03": bool(
            np.all((-np.asarray(catalog["beta_temperature"]) >= 2.55) & (-np.asarray(catalog["beta_temperature"]) <= 3.03))
        ),
        "low_frequency_flattens": bool(
            np.all(np.asarray(catalog["low_frequency_beta_temperature"]) > np.asarray(catalog["beta_temperature"]))
        ),
        "loop_i_wolleben_scale": bool(
            np.all((np.asarray(catalog["distance_kpc"])[:2] >= 0.065) & (np.asarray(catalog["distance_kpc"])[:2] <= 0.115))
            and np.all((np.asarray(catalog["radius_kpc"])[:2] >= 0.060) & (np.asarray(catalog["radius_kpc"])[:2] <= 0.105))
        ),
    }

    with np.load(harmonic_path) as harmonic:
        additivity_residual = harmonic["total"] - (
            harmonic["no_sources"] + harmonic["ateam"]
        )
        harmonic_checks = {
            "all_finite": bool(
                all(
                    np.isfinite(harmonic[key]).all()
                    for key in ["no_sources", "ateam", "total", "local_shells"]
                )
            ),
            "total_additivity_max_abs": float(
                np.max(np.abs(additivity_residual))
            ),
            "total_additivity_relative": float(
                np.max(np.abs(additivity_residual))
                / np.max(np.abs(harmonic["total"]))
            ),
            "lmax": int(harmonic["lmax"]),
            "input_nside": int(harmonic["input_nside"]),
        }

    convergence: dict[str, float] = {}
    if convergence_path.exists():
        with np.load(convergence_path) as conv:
            ell = conv["ell"]
            lo, hi = conv["cl_nside64"], conv["cl_nside128_to64"]
            convergence = {
                "map_correlation": float(conv["correlation"]),
                "relative_map_rms": float(conv["relative_rms"]),
            }
            for lower, upper in [(2, 30), (31, 64), (65, 100)]:
                mask = (ell >= lower) & (ell <= upper)
                weight = 2.0 * ell[mask] + 1.0
                ratio = np.sum(weight * lo[mask]) / np.sum(weight * hi[mask])
                convergence[f"bandpower_ratio_{lower}_{upper}"] = float(ratio)

    checks = {
        "local_bubble_mean_140_200pc": 140.0 <= bubble["mean"] <= 200.0,
        "local_bubble_has_70_600pc_span": bubble["p05"] <= 120.0 and bubble["p99"] >= 600.0,
        "gum_bright_em_220_470": 220.0 <= regional_em["gum"]["p95"] <= 470.0,
        "gum_diffuse_em_present": 8.0 <= regional_em["gum"]["p50"] <= 150.0,
        "orion_em_not_uniform_or_saturated": regional_em["orion_eridanus"]["p95"] < 150.0,
        "cygnus_reaches_observed_low_end": regional_em["cygnus_x"]["p50"] >= 1_000.0,
        "shell_map_converged": convergence.get("map_correlation", 0.0) > 0.999,
        "shell_bandpower_converged_through_ell64": abs(convergence.get("bandpower_ratio_31_64", 0.0) - 1.0) < 0.05,
        **loop_checks,
        "harmonic_finite": harmonic_checks["all_finite"],
        "harmonic_machine_precision_additivity": harmonic_checks["total_additivity_relative"] < 1.0e-14,
    }
    return {
        "status": "pass_with_scope_limit" if all(checks.values()) else "needs_attention",
        "validated_ell_max": 64,
        "checks": checks,
        "local_bubble_pc": bubble,
        "regional_anchor_em_pc_cm6": regional_em,
        "loop_catalog": catalog,
        "harmonic": harmonic_checks,
        "shell_resolution_convergence": convergence,
        "scope_limits": [
            "Diffuse morphology is numerically validated through ell=64; higher multipoles require a finer native ray grid.",
            "Loops II-IV have weak distance constraints and are conditioned analogues, not claimed 3D reconstructions.",
            "Cygnus X 4.3-pc filaments are sub-grid at NSIDE=64 and represented only in beam-averaged opacity.",
            "No Haslam, LWA, H-alpha, dust, or other survey pixels are used as morphology templates.",
        ],
        "literature": SOURCES,
    }


def plot_dashboard(report: dict[str, object], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bubble = report["local_bubble_pc"]
    axes[0, 0].bar(["p05", "median", "mean", "p95", "p99"], [bubble["p05"], bubble["p50"], bubble["mean"], bubble["p95"], bubble["p99"]])
    axes[0, 0].axhline(170, color="k", ls="--", label="observed mean ~170 pc")
    axes[0, 0].set(title="Local Bubble radial surface", ylabel="distance [pc]")
    axes[0, 0].legend(fontsize=8)

    regional = report["regional_anchor_em_pc_cm6"]
    labels = ["Gum", "Orion–Eridanus", "Cygnus X"]
    for key, label in zip(["gum", "orion_eridanus", "cygnus_x"], labels):
        q = regional[key]
        axes[0, 1].plot([5, 25, 50, 75, 95, 99], [q["p05"], q["p25"], q["p50"], q["p75"], q["p95"], q["p99"]], marker="o", label=label)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(title="Regional anchor EM distributions", xlabel="percentile", ylabel=r"EM [pc cm$^{-6}$]")
    axes[0, 1].legend(fontsize=8)

    catalog = report["loop_catalog"]
    index = -np.asarray(catalog["beta_temperature"])
    low_index = -np.asarray(catalog["low_frequency_beta_temperature"])
    x = np.arange(len(index))
    axes[1, 0].plot(x, index, "o-", label="408-MHz-side index")
    axes[1, 0].plot(x, low_index, "s-", label="low-frequency index")
    axes[1, 0].axhspan(2.55, 3.03, color="0.9", label="observed loop range")
    axes[1, 0].set_xticks(x, catalog["name"], rotation=25, ha="right")
    axes[1, 0].set(title="Loop spectral priors", ylabel="positive temperature index")
    axes[1, 0].legend(fontsize=8)

    conv = report["shell_resolution_convergence"]
    bands = ["2–30", "31–64", "65–100"]
    ratios = [conv.get("bandpower_ratio_2_30", np.nan), conv.get("bandpower_ratio_31_64", np.nan), conv.get("bandpower_ratio_65_100", np.nan)]
    axes[1, 1].bar(bands, ratios)
    axes[1, 1].axhline(1, color="k", ls="--")
    axes[1, 1].axhspan(0.95, 1.05, color="tab:green", alpha=0.15)
    axes[1, 1].set(title="NSIDE 64 / NSIDE 128 shell bandpower", xlabel=r"multipole band $\ell$", ylabel="bandpower ratio")
    fig.suptitle("Counterfactual low-frequency sky: feature-fidelity audit")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_markdown(report: dict[str, object], path: Path) -> None:
    checks = report["checks"]
    lines = ["# Feature-fidelity validation", "", f"Overall status: **{report['status']}**", "", "## Checks", ""]
    lines += [f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in checks.items()]
    lines += ["", "## Validated scope", "", f"Diffuse/shell morphology is validated through **ell = {report['validated_ell_max']}**.", ""]
    lines += [f"- {item}" for item in report["scope_limits"]]
    lines += ["", "## Primary literature", ""]
    lines += [f"- [{name.replace('_', ' ')}]({url})" for name, url in report["literature"].items()]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = audit(args.output_dir)
    json_path = args.output_dir / "feature_fidelity_validation.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output_dir / "feature_fidelity_validation.md")
    plot_dashboard(report, args.output_dir / "feature_fidelity_validation.png")
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
