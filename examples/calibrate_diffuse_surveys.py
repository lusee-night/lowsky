"""Create a survey-conditioned low-ell map without changing the original model.

Run from the repository root with CPU threading limited::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
        examples/calibrate_diffuse_surveys.py \
        --output-dir analysis/constraints-20261013/calibration/diffuse_survey_conditioned
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import healpy as hp
import numpy as np
from astropy.io import fits
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import compare_radio_regions as radio

ANCHORS = [("10.03", 10.03), ("22", 22.0), ("45south", 45.0)]
BASIS_NSIDE = 2
BASIS_SIGMA_DEG = 20.0
REGULARIZATION = 0.02
MODEL_NSIDE = 32
COMPONENT_NAMES = (
    "STOCHASTIC_SYNCH",
    "SMOOTH_SYNCH",
    "EXTRAGALACTIC",
    "LOCAL_SHELLS",
    "FREEFREE",
    "ATEAM",
)


def basis_maps() -> np.ndarray:
    """Return 48 smooth, nonnegative NSIDE=32 basis maps summing to one."""
    lon, lat = radio.sky_grid(MODEL_NSIDE)
    sky_vectors = hp.ang2vec(lon, lat, lonlat=True)
    center_lon, center_lat = hp.pix2ang(
        BASIS_NSIDE, np.arange(hp.nside2npix(BASIS_NSIDE)), lonlat=True
    )
    center_vectors = hp.ang2vec(center_lon, center_lat, lonlat=True)
    cosine = sky_vectors @ center_vectors.T
    distance = np.arccos(np.clip(cosine, -1.0, 1.0))
    basis = np.exp(-0.5 * (distance / np.deg2rad(BASIS_SIGMA_DEG)) ** 2)
    basis /= basis.sum(axis=1, keepdims=True)
    assert np.allclose(basis.sum(axis=1), 1, rtol=0, atol=1e-12)
    return basis.T.astype(np.float64)


def interp_cube(cube: np.ndarray, frequency_mhz: float) -> np.ndarray:
    """Log-interpolate the positive component between saved integer MHz maps."""
    lower = int(np.floor(frequency_mhz))
    if lower == frequency_mhz:
        return np.asarray(cube[lower - 1], dtype=np.float64)
    first = np.asarray(cube[lower - 1], dtype=np.float64)
    second = np.asarray(cube[lower], dtype=np.float64)
    if np.any(first <= 0) or np.any(second <= 0):
        raise ValueError("Nonpositive component for log-frequency interpolation")
    fraction = np.log(frequency_mhz / lower) / np.log((lower + 1) / lower)
    return np.exp((1 - fraction) * np.log(first) + fraction * np.log(second))


def survey_operator(model_map: np.ndarray, survey_weight: np.ndarray) -> np.ndarray:
    """Apply the same mask-weighted low-ell operator as the observed map."""
    at_nside64 = hp.ud_grade(
        np.asarray(model_map, dtype=np.float64),
        64,
        order_in="RING",
        order_out="RING",
    )
    return radio.smooth_weighted(at_nside64, survey_weight)[0]


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    residual = (predicted - observed) / observed
    return {
        "n_cells": int(len(residual)),
        "median_model_over_survey_minus_1": float(np.median(residual)),
        "rms_fractional": float(np.sqrt(np.mean(residual * residual))),
        "q16_q50_q84_fractional": [
            float(value) for value in np.quantile(residual, [0.16, 0.5, 0.84])
        ],
    }


def read_model_components(model_path: Path) -> dict[str, np.ndarray]:
    with fits.open(model_path, memmap=True) as hdul:
        header = hdul[0].header
        if (
            header["NSIDE"] != MODEL_NSIDE
            or header["ORDERING"].strip() != "RING"
            or header["COORDSYS"].strip() != "GALACTIC"
            or header["BUNIT"].strip() != "K"
        ):
            raise ValueError("Unexpected model FITS metadata")
        components = {
            name: np.asarray(hdul[name].data, dtype=np.float64)
            for name in COMPONENT_NAMES
        }
    expected_shape = (50, hp.nside2npix(MODEL_NSIDE))
    if any(component.shape != expected_shape for component in components.values()):
        raise ValueError("Unexpected component shape")
    return components


def fit_anchor(
    survey_key: str,
    frequency_mhz: float,
    survey_dir: Path,
    components: dict[str, np.ndarray],
    basis: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    source_ok: np.ndarray,
    holdout: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    survey, survey_weight, _path, native_mask = radio.survey_map(
        survey_key, survey_dir
    )
    diffuse = (
        interp_cube(components["STOCHASTIC_SYNCH"], frequency_mhz)
        + interp_cube(components["EXTRAGALACTIC"], frequency_mhz)
    )
    fixed = (
        interp_cube(components["LOCAL_SHELLS"], frequency_mhz)
        + interp_cube(components["FREEFREE"], frequency_mhz)
    )
    fixed_smoothed = survey_operator(fixed, native_mask)
    design = np.stack(
        [survey_operator(diffuse * sky_basis, native_mask) for sky_basis in basis],
        axis=1,
    )
    valid = (
        np.isfinite(survey)
        & np.isfinite(fixed_smoothed)
        & np.all(np.isfinite(design), axis=1)
        & (survey_weight > 0.98)
        & (np.abs(latitude) > 10)
        & source_ok
        & (survey > 0)
    )
    train = valid & ~holdout
    held_out = valid & holdout
    if train.sum() < 100 or held_out.sum() < 30:
        raise RuntimeError(
            f"Insufficient spatial split {survey_key}: "
            f"{train.sum()} / {held_out.sum()}"
        )

    # Equal relative weight per valid survey pixel, plus a ridge toward unity.
    row_scale = survey[train] * np.sqrt(train.sum())
    design_train = design[train] / row_scale[:, None]
    target_train = (survey[train] - fixed_smoothed[train]) / row_scale
    ridge_weight = np.sqrt(REGULARIZATION / len(basis))
    augmented_design = np.vstack(
        [design_train, ridge_weight * np.eye(len(basis))]
    )
    augmented_target = np.concatenate(
        [target_train, ridge_weight * np.ones(len(basis))]
    )
    fit = lsq_linear(
        augmented_design,
        augmented_target,
        bounds=(0.1, 2.0),
        tol=1e-10,
        max_iter=100,
    )
    if not fit.success:
        raise RuntimeError(f"Fit failed at {frequency_mhz}: {fit.message}")
    coefficients = fit.x
    before = fixed_smoothed + design.sum(axis=1)
    after = fixed_smoothed + design @ coefficients
    anchor_scale = coefficients @ basis
    result = {
        "frequency_mhz": frequency_mhz,
        "survey_id": survey_key,
        "train": {
            "before": metrics(survey[train], before[train]),
            "after": metrics(survey[train], after[train]),
        },
        "held_out_longitude_sectors": {
            "before": metrics(survey[held_out], before[held_out]),
            "after": metrics(survey[held_out], after[held_out]),
        },
        "coefficient_bounds_active": int(
            np.sum((coefficients < 0.100001) | (coefficients > 1.999999))
        ),
        "coefficients": [float(value) for value in coefficients],
    }
    for region_name, region in radio.REGIONS.items():
        if survey_key not in region["surveys"]:
            continue
        lon_min, lon_max, lat_min, lat_max = region["bounds"]
        selected = (
            valid
            & (longitude >= lon_min)
            & (longitude <= lon_max)
            & (latitude >= lat_min)
            & (latitude <= lat_max)
        )
        result.setdefault("regions", {})[region_name] = (
            {
                "before": metrics(survey[selected], before[selected]),
                "after": metrics(survey[selected], after[selected]),
            }
            if selected.sum() > 0
            else None
        )
    return coefficients, anchor_scale.astype(np.float32), result


def interpolated_scales(coefficients: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Interpolate within anchors and freeze the endpoint scale outside them."""
    anchor_frequencies = np.array([frequency for _, frequency in ANCHORS])
    scales = []
    for frequency in np.arange(1, 51, dtype=float):
        if frequency <= anchor_frequencies[0]:
            current = coefficients[0]
        elif frequency >= anchor_frequencies[-1]:
            current = coefficients[-1]
        else:
            lower = np.searchsorted(anchor_frequencies, frequency) - 1
            fraction = np.log(frequency / anchor_frequencies[lower]) / np.log(
                anchor_frequencies[lower + 1] / anchor_frequencies[lower]
            )
            current = (
                (1 - fraction) * coefficients[lower]
                + fraction * coefficients[lower + 1]
            )
        scales.append(current @ basis)
    sky_scales = np.asarray(scales, dtype=np.float64)
    if np.any(sky_scales <= 0) or np.any(~np.isfinite(sky_scales)):
        raise RuntimeError("Invalid corrected scale")
    return sky_scales


def corrected_components(
    components: dict[str, np.ndarray], scales: np.ndarray
) -> dict[str, np.ndarray]:
    corrected = {name: component.copy() for name, component in components.items()}
    for name in ("STOCHASTIC_SYNCH", "SMOOTH_SYNCH", "EXTRAGALACTIC"):
        corrected[name] *= scales
    corrected["BASELINE"] = (
        corrected["STOCHASTIC_SYNCH"]
        + corrected["FREEFREE"]
        + corrected["EXTRAGALACTIC"]
    )
    corrected["NO_SOURCES"] = corrected["BASELINE"] + corrected["LOCAL_SHELLS"]
    corrected["TOTAL"] = corrected["NO_SOURCES"] + corrected["ATEAM"]
    return corrected


def write_corrected_fits(
    model_path: Path, output_path: Path, corrected: dict[str, np.ndarray]
) -> None:
    with fits.open(model_path, memmap=True) as source:
        hdus = fits.HDUList([hdu.copy() for hdu in source])
    hdus[0].data = corrected["TOTAL"].astype(np.float32)
    for name in (
        "STOCHASTIC_SYNCH",
        "SMOOTH_SYNCH",
        "EXTRAGALACTIC",
        "BASELINE",
        "NO_SOURCES",
    ):
        hdus[name].data = corrected[name].astype(np.float32)
    header = hdus[0].header
    header["PRODUCT"] = "SURVEY-CONDITIONED"
    header["CALSTAT"] = "EMPIRIC"
    header["CALFIT"] = "REL-LSQ"
    header["CALMODEL"] = "BROAD48"
    header["CALSURV"] = "10.03,22,45"
    header["CALDOM"] = "10.03-45"
    header["EXTRAP"] = "CONST-END"
    header["ATEAMVER"] = "SAVED"
    header.add_history(
        "Empirical survey-conditioned diffuse scale; original physical model unchanged."
    )
    header.add_history(
        "Volume synch + EGB rescaled; local shells/freefree/A-team unchanged."
    )
    header.add_history(
        "ATEAM is copied from saved historical realization, not newer Cas update."
    )
    header.add_history(
        "Outside 10.03-45 MHz, endpoint empirical scale is extrapolated."
    )
    header.add_history(
        "EMISSCAL/BETADLT are inherited baseline physical parameters; empirical scale is separate."
    )
    hdus.writeto(output_path, overwrite=True, checksum=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=radio.MODEL)
    parser.add_argument("--survey-dir", type=Path, default=radio.SURVEY_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis/constraints-20261013/calibration/diffuse_survey_conditioned",
    )
    args = parser.parse_args()
    model_path = args.model.resolve()
    survey_dir = args.survey_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    components = read_model_components(model_path)
    basis = basis_maps()
    longitude, latitude = radio.sky_grid(MODEL_NSIDE)
    source_ok = radio.source_mask(longitude, latitude)
    holdout = (np.floor(longitude / 30).astype(int) % 3) == 0
    coefficient_sets = []
    anchor_maps = {}
    results = {}
    for survey_key, frequency in ANCHORS:
        coefficients, anchor_scale, result = fit_anchor(
            survey_key,
            frequency,
            survey_dir,
            components,
            basis,
            longitude,
            latitude,
            source_ok,
            holdout,
        )
        coefficient_sets.append(coefficients)
        anchor_maps[survey_key] = anchor_scale
        results[survey_key] = result

    coefficient_sets = np.asarray(coefficient_sets)
    scales = interpolated_scales(coefficient_sets, basis)
    corrected = corrected_components(components, scales)
    fits_path = output_dir / "survey_conditioned_nside32_1_50mhz.fits"
    write_corrected_fits(model_path, fits_path, corrected)
    anchor_path = output_dir / "anchor_scale_maps.npz"
    np.savez_compressed(
        anchor_path,
        frequency_mhz=np.array([frequency for _, frequency in ANCHORS]),
        galactic_ring_nside32=32,
        **anchor_maps,
    )

    manifest = {
        "status": "survey_conditioned_empirical_realization_not_physical_parameter_fit",
        "method": {
            "basis": "48 NSIDE=2 HEALPix Gaussian sky centers, sigma 20 deg, partition of unity",
            "coefficient_bounds": [0.1, 2.0],
            "regularization_to_one_per_coefficient": REGULARIZATION / len(basis),
            "fitted_components": ["STOCHASTIC_SYNCH", "EXTRAGALACTIC"],
            "unchanged_components": ["LOCAL_SHELLS", "FREEFREE", "ATEAM"],
            "survey_operator": "same 12 degree masked weighted smoothing and lmax64 as compare_radio_regions.py",
            "selection": "all valid survey pixels at |b|>10 deg; A-team 8 deg exclusions; 45 MHz southern declination <-2 deg",
            "spatial_holdout": "floor(Galactic_longitude_deg/30) % 3 == 0; correlated sectors within same survey",
            "frequency_interpolation": "log-frequency interpolation of 48 coefficients between 10.03, 22, 45 MHz; endpoint coefficient freeze outside; 1--10.03 and 45--50 MHz extrapolated from nearest anchor",
        },
        "results": results,
        "attempt1_summary": "attempt1_summary.json",
        "input_sha256": {
            str(model_path): radio.sha256(model_path),
            **{
                str(survey_dir / radio.SURVEYS[key][0]): radio.sha256(
                    survey_dir / radio.SURVEYS[key][0]
                )
                for key in radio.SURVEYS
            },
        },
        "output_fits": str(fits_path),
        "output_fits_sha256": radio.sha256(fits_path),
        "anchor_scale_maps": "anchor_scale_maps.npz",
        "anchor_scale_maps_sha256": radio.sha256(anchor_path),
        "warnings": [
            "Training survey pixels are conditioning data, not validation holdouts",
            "Longitude-sector holdouts share instrument calibration and foreground ancestry",
            "Survey gains/zero levels not separately fitted; broad empirical scale may absorb those errors",
            "Native survey beams only approximately matched at 12 degrees",
            "Model frequencies outside 10.03--45 MHz use extrapolated endpoint sky-scale shapes",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fits": str(fits_path),
                "sha256": manifest["output_fits_sha256"],
                "held_out_rms": {
                    key: result["held_out_longitude_sectors"]["after"]["rms_fractional"]
                    for key, result in results.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
