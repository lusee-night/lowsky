"""Diagnose a bounded global diffuse-synchrotron correction.

This produces a rejected sensitivity diagnostic, not a production preset.
Run with CPU threads limited::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
        examples/calibrate_diffuse.py \
        --output-dir analysis/constraints-20261013/calibration/diffuse
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import compare_radio_regions as radio
from lowsky.calibration import calibrated_no_sources_from_components

COMPONENTS = ("STOCHASTIC_SYNCH", "LOCAL_SHELLS", "FREEFREE", "EXTRAGALACTIC")
PLOT_GROUPS = (
    ("anticenter_north", "10.03", "#7b3294"),
    ("anticenter_north", "22", "#008837"),
    ("orion_south", "22", "#2c7fb8"),
    ("orion_south", "45south", "#d95f02"),
)


def interpolate_component(cube: np.ndarray, frequency: float) -> np.ndarray:
    lower = int(np.floor(frequency))
    if lower == frequency:
        return np.asarray(cube[lower - 1], dtype=np.float64)
    first = np.asarray(cube[lower - 1], dtype=np.float64)
    second = np.asarray(cube[lower], dtype=np.float64)
    if np.any(first <= 0) or np.any(second <= 0):
        raise ValueError("Nonpositive component prevents log interpolation")
    fraction = np.log(frequency / lower) / np.log((lower + 1) / lower)
    return np.exp((1 - fraction) * np.log(first) + fraction * np.log(second))


def log_residual(parameter: np.ndarray, rows: list[dict]) -> np.ndarray:
    log_ratio, delta = parameter
    ratio = np.exp(log_ratio)
    predicted = np.array(
        [
            calibrated_no_sources_from_components(
                row["synch"],
                row["shell"],
                row["freefree"],
                row["extra"],
                row["frequency"],
                ratio,
                delta,
            )
            for row in rows
        ]
    )
    observed = np.array([row["observed"] for row in rows])
    return np.log(predicted / observed)


def bin_rows(
    surveys: dict, components: dict[str, np.ndarray]
) -> tuple[list[dict], dict[str, int]]:
    longitude, latitude = radio.sky_grid(32)
    source_ok = radio.source_mask(longitude, latitude)
    entries = []
    pixel_counts = {}
    for region_name, region in radio.REGIONS.items():
        lon_min, lon_max, lat_min, lat_max = region["bounds"]
        inside = (
            (longitude >= lon_min)
            & (longitude <= lon_max)
            & (latitude >= lat_min)
            & (latitude <= lat_max)
            & source_ok
        )
        for survey_key in region["surveys"]:
            survey_map, survey_weight, _path, _native_mask = surveys[survey_key]
            inside &= np.isfinite(survey_map) & (
                survey_weight > radio.MIN_VALID_WEIGHT
            )
        if inside.sum() < 30:
            raise RuntimeError(f"Too few matched cells: {region_name}")
        pixel_counts[region_name] = int(inside.sum())
        longitude_edges = np.linspace(lon_min, lon_max, 7)
        for survey_key in region["surveys"]:
            frequency = radio.SURVEYS[survey_key][2]
            observed = surveys[survey_key][0]
            native_mask = surveys[survey_key][3]
            mapped = {}
            for component_name in COMPONENTS:
                component = interpolate_component(components[component_name], frequency)
                at_nside64 = hp.ud_grade(
                    component, 64, order_in="RING", order_out="RING"
                )
                mapped[component_name] = radio.smooth_weighted(
                    at_nside64, native_mask
                )[0]
            for bin_index, (left, right) in enumerate(
                zip(longitude_edges[:-1], longitude_edges[1:])
            ):
                selected = inside & (longitude >= left) & (longitude < right)
                if selected.sum() < 3:
                    continue
                entries.append(
                    {
                        "region": region_name,
                        "survey": survey_key,
                        "frequency": frequency,
                        "bin": bin_index,
                        "split": (
                            "train" if bin_index % 2 == 0 else "held_out_longitude_bin"
                        ),
                        "n_pixels": int(selected.sum()),
                        "lon_mid": float((left + right) / 2),
                        "observed": float(np.mean(observed[selected])),
                        "synch": float(np.mean(mapped["STOCHASTIC_SYNCH"][selected])),
                        "shell": float(np.mean(mapped["LOCAL_SHELLS"][selected])),
                        "freefree": float(np.mean(mapped["FREEFREE"][selected])),
                        "extra": float(np.mean(mapped["EXTRAGALACTIC"][selected])),
                    }
                )
    return entries, pixel_counts


def evaluate(rows: list[dict], ratio: float, delta: float) -> dict:
    residuals = log_residual([np.log(ratio), delta], rows)
    return {
        "median_fractional_model_over_survey_minus_1": float(
            np.median(np.expm1(residuals))
        ),
        "median_absolute_log_residual": float(np.median(np.abs(residuals))),
        "rms_log_residual": float(np.sqrt(np.mean(residuals**2))),
        "n_bins": len(rows),
    }


def two_amplitude_log_residual(
    amplitudes: np.ndarray, rows: list[dict], include_shell: bool
) -> np.ndarray:
    synch_ratio, extra_ratio = amplitudes
    predicted = np.array(
        [
            synch_ratio * (row["synch"] + (row["shell"] if include_shell else 0))
            + (0 if include_shell else row["shell"])
            + row["freefree"]
            + extra_ratio * row["extra"]
            for row in rows
        ]
    )
    observed = np.array([row["observed"] for row in rows])
    return np.log(predicted / observed)


def alternative_trials(train: list[dict], held_out: list[dict]) -> dict:
    alternatives = {}
    for name, include_shell in (
        ("volume_only_plus_EGB", False),
        ("volume_and_shell_plus_EGB", True),
    ):
        trial = least_squares(
            two_amplitude_log_residual,
            [0.7, 0.7],
            args=(train, include_shell),
            bounds=([0.05, 0.05], [2, 2]),
            loss="soft_l1",
            f_scale=0.15,
        )
        orion_22 = [
            row
            for row in held_out
            if row["region"] == "orion_south" and row["survey"] == "22"
        ]
        held_residual = two_amplitude_log_residual(
            trial.x, held_out, include_shell
        )
        orion_residual = two_amplitude_log_residual(
            trial.x, orion_22, include_shell
        )
        alternatives[name] = {
            "synch_ratio": float(trial.x[0]),
            "extragalactic_ratio": float(trial.x[1]),
            "active_bounds": trial.active_mask.tolist(),
            "held_out_rms_log_residual": float(
                np.sqrt(np.mean(held_residual**2))
            ),
            "orion_south_22_held_out_model_over_survey_percent": [
                float(value) for value in 100 * np.expm1(orion_residual)
            ],
            "status": (
                "rejected_bound_saturated"
                if any(trial.active_mask)
                else "rejected_orion_residual_and_loop_prior"
            ),
        }
    return alternatives


def plot_diagnostic(entries: list[dict], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for axis, split in zip(axes, ("train", "held_out_longitude_bin")):
        for region, survey_key, color in PLOT_GROUPS:
            rows = [
                row
                for row in entries
                if row["split"] == split
                and row["region"] == region
                and row["survey"] == survey_key
            ]
            longitude = np.array([row["lon_mid"] for row in rows])
            before = np.array(
                [100 * (row["before_K"] / row["observed"] - 1) for row in rows]
            )
            after = np.array(
                [100 * (row["after_K"] / row["observed"] - 1) for row in rows]
            )
            axis.plot(
                longitude,
                before,
                color=color,
                ls="--",
                marker="x",
                label=f"{region} {survey_key} before",
            )
            axis.plot(
                longitude,
                after,
                color=color,
                ls="-",
                marker="o",
                label=f"{region} {survey_key} after",
            )
        axis.axhline(0, color="black", lw=0.8)
        axis.set_title(split.replace("_", " "))
        axis.set_xlabel("Galactic longitude (deg)")
        axis.set_ylabel("(model / survey − 1) × 100 (%)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.suptitle(
        "REJECTED diagnostic fit (bounds saturated); fixed thermal opacity and shells"
    )
    figure.subplots_adjust(top=0.82, bottom=0.19, left=0.08, right=0.98, wspace=0.25)
    figure.text(
        0.5,
        0.035,
        "Sensitivity only: no parameters adopted; held-out bins share survey calibration.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(output_dir / "before_after.png", dpi=180)
    figure.savefig(output_dir / "before_after.svg")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=radio.MODEL)
    parser.add_argument("--survey-dir", type=Path, default=radio.SURVEY_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis/constraints-20261013/calibration/diffuse",
    )
    args = parser.parse_args()
    model_path = args.model.resolve()
    survey_dir = args.survey_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    surveys = {key: radio.survey_map(key, survey_dir) for key in radio.SURVEYS}
    with fits.open(model_path, memmap=True) as hdul:
        header = hdul[0].header
        if (
            header["NSIDE"] != 32
            or header["ORDERING"].strip() != "RING"
            or header["COORDSYS"].strip() != "GALACTIC"
            or header["BUNIT"].strip() != "K"
        ):
            raise ValueError("Unexpected model grid or unit")
        old_scale = float(header["EMISSCAL"])
        old_beta = float(header["BETADLT"])
        components = {
            name: np.asarray(hdul[name].data, dtype=np.float64)
            for name in COMPONENTS
        }
    entries, pixel_counts = bin_rows(surveys, components)
    train = [row for row in entries if row["split"] == "train"]
    held_out = [row for row in entries if row["split"] != "train"]
    fit = least_squares(
        log_residual,
        x0=[np.log(0.7), 0.0],
        args=(train,),
        bounds=([np.log(0.3), -0.25], [np.log(1.2), 0.25]),
        loss="soft_l1",
        f_scale=0.15,
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    ratio = float(np.exp(fit.x[0]))
    delta = float(fit.x[1])
    for row in entries:
        arguments = (
            row["synch"],
            row["shell"],
            row["freefree"],
            row["extra"],
            row["frequency"],
        )
        row["before_K"] = float(
            calibrated_no_sources_from_components(*arguments, 1.0, 0.0)
        )
        row["after_K"] = float(
            calibrated_no_sources_from_components(*arguments, ratio, delta)
        )
    summary_metrics = {
        "train_before": evaluate(train, 1, 0),
        "train_after": evaluate(train, ratio, delta),
        "held_out_before": evaluate(held_out, 1, 0),
        "held_out_after": evaluate(held_out, ratio, delta),
    }
    preset = {
        "emissivity_scale": old_scale * ratio,
        "beta_offset": old_beta + delta,
        "emission_measure_scale": "unchanged_from_original_run",
        "emissivity_ratio": ratio,
        "beta_offset_delta": delta,
        "original_emissivity_scale": old_scale,
        "original_beta_offset": old_beta,
    }
    bound_saturated = bool(
        np.any(fit.active_mask != 0) or ratio <= 0.300001 or delta >= 0.249999
    )
    result = {
        "status": (
            "rejected_bound_saturated_diagnostic_only"
            if bound_saturated
            else "experimental_candidate_pending_physical_review"
        ),
        "preset_status": "do_not_integrate" if bound_saturated else "candidate_only",
        "adopted_parameters": None,
        "preset": preset,
        "metrics": summary_metrics,
        "regions_common_nside32_cells": pixel_counts,
        "bins": entries,
        "operator": "same survey masks and ell<=64 weighted 12-degree smoothing as examples/compare_radio_regions.py; bin means for exact linear component algebra; alternate longitude bins held out within same survey, not an independent instrument",
        "caveats": [
            "Spatial holdout shares instrument calibration and model conditioning",
            "10.03/22/45-MHz survey calibration/zero-level systematics not in objective",
            "Native survey beams not exactly matched at 12 degrees",
            "No local-shell, EM, free-free, or source reoptimization",
            "No claim object-specific Orion or Gum correction",
        ],
        "input_sha256": {
            str(path): radio.sha256(path)
            for path in [model_path] + [item[2] for item in surveys.values()]
        },
        "alternative_two_amplitude_trials": alternative_trials(train, held_out),
        "rejection_reason": (
            "Both amplitude/tilt fit parameters saturated bounds; volume-only/EGB trial saturates synch lower bound; volume+shell/EGB interior trial leaves Orion 22 MHz high and suppresses loop priors"
            if bound_saturated
            else None
        ),
    }
    (output_dir / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_diagnostic(entries, output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "preset_status": result["preset_status"],
                "metrics": summary_metrics,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
