"""Matched, regional low-frequency survey comparison; descriptive, not a fit.

Run: OMP_NUM_THREADS=1 .venv/bin/python examples/compare_radio_regions.py \
    --output-dir analysis/constraints-20261013/comparisons/radio_regions
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
SURVEY_DIR = ROOT / "analysis/constraints-20261013/radio_surveys/data"
MODEL = ROOT / "outputs_our_sky_gum_gaussian_final/counterfactual_ulsa_nside32_1_50mhz.fits"
SOURCES = {
    "Cas A": (111.735, -2.129),
    "Cyg A": (76.19, 5.76),
    "Tau A": (184.557, -5.784),
    "Vir A": (283.778, 74.49),
}
SURVEYS = {
    "10.03": ("DRAO_10MHz_hpx_r7.fits", 128, 10.03),
    "22": ("lambda_DRAO_22MHz_1_256.fits", 256, 22.0),
    "45south": ("MAIPU_MU_1_64.fits", 64, 45.0),
}
REGIONS = {
    "anticenter_north": {"bounds": (160, 200, 15, 40), "surveys": ("10.03", "22")},
    "orion_south": {"bounds": (205, 220, -45, -20), "surveys": ("22", "45south")},
}
SMOOTH_FWHM_DEG = (
    12.0  # Additional convolution for all maps; varying native beams remain approximate.
)
MIN_VALID_WEIGHT = 0.98
NOUT = 32


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sky_grid(nside):
    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    return lon, lat


def source_mask(lon, lat):
    valid = np.ones(len(lon), dtype=bool)
    for sl, sb in SOURCES.values():
        cosang = np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(sb)) + np.cos(
            np.deg2rad(lat)
        ) * np.cos(np.deg2rad(sb)) * np.cos(np.deg2rad(lon - sl))
        valid &= np.rad2deg(np.arccos(np.clip(cosang, -1, 1))) > 8.0
    return valid


def smooth_weighted(values64, weight64):
    """Identical masked, band-limited operator for a survey and its model."""
    values64 = np.asarray(values64, dtype=np.float64)
    weight64 = np.asarray(weight64, dtype=np.float64)
    if values64.shape != weight64.shape or len(values64) != hp.nside2npix(64):
        raise ValueError("Expected paired NSIDE=64 HEALPix arrays")
    if (
        np.any(~np.isfinite(values64))
        or np.any(~np.isfinite(weight64))
        or np.any((weight64 < 0) | (weight64 > 1))
    ):
        raise ValueError("Invalid values or weights")
    kernel = np.deg2rad(SMOOTH_FWHM_DEG)
    num = hp.smoothing(values64 * weight64, fwhm=kernel, lmax=64)
    den = hp.smoothing(weight64, fwhm=kernel, lmax=64)
    num = hp.ud_grade(num, NOUT, order_in="RING", order_out="RING")
    den = hp.ud_grade(den, NOUT, order_in="RING", order_out="RING")
    out = np.full(len(den), np.nan)
    np.divide(num, den, out=out, where=den > MIN_VALID_WEIGHT)
    return out, den


def survey_map(key, survey_dir):
    name, nside, _ = SURVEYS[key]
    path = survey_dir / name
    with fits.open(path, memmap=True) as hdul:
        header = hdul[1].header
        if (
            header.get("NSIDE") != nside
            or header.get("ORDERING", "").strip().upper() != "NESTED"
            or header.get("COORDSYS", "").strip().upper() != "G"
        ):
            raise ValueError(f"Unexpected survey HEALPix metadata: {path}")
    a = hp.reorder(hp.read_map(path, field=0, nest=True), n2r=True)
    lon, lat = sky_grid(nside)
    valid = np.isfinite(a) & (a > -1e20) & (a != -32768) & source_mask(lon, lat)
    if key == "45south":
        dec = SkyCoord(l=lon * u.deg, b=lat * u.deg, frame="galactic").icrs.dec.deg
        valid &= dec < -2.0
    weights = hp.ud_grade(valid.astype(float), 64, order_in="RING", order_out="RING")
    numerator = hp.ud_grade(np.where(valid, a, 0.0), 64, order_in="RING", order_out="RING")
    val64 = np.zeros_like(numerator)
    np.divide(numerator, weights, out=val64, where=weights > 0)
    smoothed, output_weight = smooth_weighted(val64, weights)
    return smoothed, output_weight, path, weights


def model_map(freq, model_path):
    if not 1 <= freq <= 50:
        raise ValueError("Model frequency must be within 1--50 MHz")
    with fits.open(model_path, memmap=True) as hdul:
        header = hdul[0].header
        if (
            header.get("NSIDE") != 32
            or header.get("ORDERING", "").strip().upper() != "RING"
            or header.get("COORDSYS", "").strip().upper() != "GALACTIC"
            or header.get("BUNIT", "").strip().upper() != "K"
        ):
            raise ValueError("Unexpected model HEALPix metadata")
        data = hdul["NO_SOURCES"].data
        if data.shape != (50, hp.nside2npix(32)):
            raise ValueError("Unexpected model data shape")
        lower = int(np.floor(freq))
        if lower == freq:
            a = np.asarray(data[lower - 1], dtype=np.float64)
        else:
            t0 = np.asarray(data[lower - 1], dtype=np.float64)
            t1 = np.asarray(data[lower], dtype=np.float64)
            if not np.all(np.isfinite(t0) & np.isfinite(t1) & (t0 > 0) & (t1 > 0)):
                raise ValueError(
                    "Cannot log-interpolate nonpositive or nonfinite model temperatures"
                )
            frac = np.log(freq / lower) / np.log((lower + 1) / lower)
            a = np.exp(np.log(t0) * (1 - frac) + np.log(t1) * frac)
    if not np.all(np.isfinite(a)):
        raise ValueError("Nonfinite model temperature")
    return a


def quantiles(a):
    return [float(x) for x in np.quantile(a, [0.16, 0.50, 0.84])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis/constraints-20261013/comparisons/radio_regions",
    )
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--survey-dir", type=Path, default=SURVEY_DIR)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sdata = {k: survey_map(k, args.survey_dir.resolve()) for k in SURVEYS}
    mdata = {
        k: smooth_weighted(
            hp.ud_grade(
                model_map(v[2], args.model.resolve()), 64, order_in="RING", order_out="RING"
            ),
            sdata[k][3],
        )[0]
        for k, v in SURVEYS.items()
    }
    lon, lat = sky_grid(NOUT)
    result = {
        "method": {
            "comparison": "descriptive direct survey versus saved no-A-team lowsky model; no fitted gain or offset",
            "coordinates": "Galactic RING NSIDE=32",
            "additional_gaussian_fwhm_deg": SMOOTH_FWHM_DEG,
            "shared_operator": "survey and model each convolved with the same survey validity/source mask and lmax=64, then degraded to NSIDE=32",
            "native_model_beam_fwhm_deg": 2.0,
            "survey_native_beams": "position-dependent, anisotropic 1--4.6 degrees; common effective beam only approximate",
            "valid_weight_minimum": MIN_VALID_WEIGHT,
            "bright_source_aperture_radius_deg": 8,
            "45_MHz_component": "equatorial declination <-2 degrees, southern Maipu 45 MHz only",
            "10_03_MHz_model_interpolation": "log-temperature versus log-frequency between integer 10 and 11 MHz maps",
            "scatter_definition": "16th--84th percentile of spatial pixels, not statistical measurement uncertainty",
            "calibration_caveats": [
                "10.03 MHz absolute zero-level and pixel-error model not verified",
                "22 MHz declination gain tied to Haslam 408 MHz; possible ~5 kK zero-level systematic",
                "45 MHz hosted composite lacks EDGES-derived gain/offset correction",
                "Lowsky was tuned to ULSA and some same low-frequency survey lineages; residuals are not independent validation",
            ],
            "frequency_range_mhz": [1, 50],
            "model_validated_ell_max": 64,
        },
        "inputs": {
            str(p): sha256(p) for p in [args.model.resolve()] + [x[2] for x in sdata.values()]
        },
        "regions": {},
    }
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.89, hspace=0.42, wspace=0.24)
    colors = {"10.03": "#7b3294", "22": "#008837", "45south": "#d95f02"}
    for row, (rname, spec) in enumerate(REGIONS.items()):
        l0, l1, b0, b1 = spec["bounds"]
        inside = (lon >= l0) & (lon <= l1) & (lat >= b0) & (lat <= b1) & source_mask(lon, lat)
        for key in spec["surveys"]:
            inside &= (
                np.isfinite(sdata[key][0])
                & np.isfinite(mdata[key])
                & (sdata[key][1] > MIN_VALID_WEIGHT)
            )
        r = {
            "bounds_galactic_deg": spec["bounds"],
            "survey_ids": list(spec["surveys"]),
            "common_pixel_count": int(inside.sum()),
            "by_frequency": {},
        }
        if inside.sum() < 10:
            raise RuntimeError(f"Too few common pixels in {rname}: {inside.sum()}")
        bins = np.linspace(l0, l1, 7)
        for key in spec["surveys"]:
            obs = sdata[key][0][inside]
            mod = mdata[key][inside]
            residual = (obs - mod) / mod
            entry = {
                "frequency_mhz": SURVEYS[key][2],
                "survey_temperature_K_q16_q50_q84": quantiles(obs),
                "model_temperature_K_q16_q50_q84": quantiles(mod),
                "fractional_residual_q16_q50_q84": quantiles(residual),
                "longitude_bins": [],
            }
            centers, obins, mbins, rbins = [], [], [], []
            for left, right in zip(bins[:-1], bins[1:]):
                select = (lon[inside] >= left) & (lon[inside] < right)
                if select.sum() < 3:
                    continue
                centers.append((left + right) / 2)
                oq, mq, rq = (
                    quantiles(obs[select]),
                    quantiles(mod[select]),
                    quantiles(residual[select]),
                )
                obins.append(oq)
                mbins.append(mq)
                rbins.append(rq)
                entry["longitude_bins"].append(
                    {
                        "lon_min_deg": float(left),
                        "lon_max_deg": float(right),
                        "pixel_count": int(select.sum()),
                        "survey_K_q16_q50_q84": oq,
                        "model_K_q16_q50_q84": mq,
                        "fractional_residual_q16_q50_q84": rq,
                    }
                )
            label = f"{SURVEYS[key][2]:g} MHz"
            color = colors[key]
            axs[row, 0].plot(
                centers, [x[1] for x in obins], color=color, marker="o", label=label + " survey"
            )
            axs[row, 0].plot(
                centers,
                [x[1] for x in mbins],
                color=color,
                linestyle="--",
                marker="s",
                label=label + " lowsky",
            )
            axs[row, 1].plot(
                centers, [x[1] * 100 for x in rbins], color=color, marker="o", label=label
            )
            axs[row, 1].fill_between(
                centers,
                [x[0] * 100 for x in rbins],
                [x[2] * 100 for x in rbins],
                color=color,
                alpha=0.13,
            )
            r["by_frequency"][key] = entry
        result["regions"][rname] = r
        axs[row, 0].set_yscale("log")
        axs[row, 0].set_ylabel("Brightness temperature (K)")
        axs[row, 1].set_ylabel("(survey − lowsky) / lowsky (%)")
        axs[row, 1].axhline(0, color="black", lw=0.8)
        for ax in axs[row]:
            ax.set_xlabel("Galactic longitude (deg)")
            ax.set_title(f"{rname.replace('_', ' ')}: {b0}° < b < {b1}°; N={inside.sum()} cells")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle(
        "Regional low-frequency survey comparison (additional 12° smoothing; no gain/offset fit)"
    )
    fig.text(
        0.5,
        0.025,
        "Bands: spatial 16–84% pixel spread, not measurement error. Shared survey/model calibration priors; native beams vary.",
        ha="center",
        fontsize=9,
    )
    for ext in ["png", "svg"]:
        fig.savefig(out / f"radio_regions.{ext}", dpi=180)
    plt.close(fig)
    (out / "radio_regions.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output_dir": str(out),
                "pixel_counts": {k: v["common_pixel_count"] for k, v in result["regions"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
