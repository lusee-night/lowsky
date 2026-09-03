"""One-lunar-night point-source separation experiment with Lowsky + LuSEEpy.

This stops before the electrical/receiver forward model. It beam-convolves the
canonical Lowsky harmonics, saves diffuse-only and source-only waterfalls, and
fits free source spectra using known transit templates plus an unknown smooth
temporal foreground. The diffuse-only waterfall is used only to score the fit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import lusee


SOURCE_KEYS = ("ateam_cas_a", "ateam_cyg_a", "ateam_tau_a", "ateam_vir_a")


class HarmonicCubeSky:
    """Minimal LuSEEpy gridded-sky interface for a packed alm cube."""

    def __init__(self, alms: np.ndarray, frequency_mhz: np.ndarray, nside: int):
        self.mapalm = jnp.asarray(alms)
        self.freq = np.asarray(frequency_mhz, dtype=np.float64)
        self.frame = "galactic"
        self.Nside = int(nside)

    def get_alm(self, indices, freq=None):
        del freq
        return self.mapalm[jnp.atleast_1d(indices)]


def truncate_alms(alms: np.ndarray, lmax: int) -> np.ndarray:
    """Truncate packed healpy alms whose original lmax is >= ``lmax``."""
    alms = np.asarray(alms)
    input_lmax = hp.Alm.getlmax(alms.shape[-1])
    if input_lmax < lmax:
        raise ValueError(f"input lmax={input_lmax} is below requested lmax={lmax}")
    ell, emm = hp.Alm.getlm(lmax)
    source_indices = hp.Alm.getidx(input_lmax, ell, emm)
    return np.asarray(alms[:, source_indices], dtype=np.complex128)


def fourier_nuisance_basis(nt: int, max_mode: int) -> np.ndarray:
    """Real orthonormal constant/sine/cosine time basis through max_mode."""
    phase = 2.0 * np.pi * np.arange(nt, dtype=float) / nt
    cols = [np.ones(nt)]
    for mode in range(1, max_mode + 1):
        cols.extend((np.cos(mode * phase), np.sin(mode * phase)))
    return np.linalg.qr(np.column_stack(cols))[0]


def recover_spectra(total_waterfall, unit_source_templates, foreground_modes):
    """Fit source amplitudes per channel after Fourier nuisance projection."""
    nt, nf = total_waterfall.shape
    ns = unit_source_templates.shape[0]
    nuisance = fourier_nuisance_basis(nt, foreground_modes)
    projector = np.eye(nt) - nuisance @ nuisance.T
    amplitudes = np.empty((ns, nf))
    condition = np.empty(nf)
    reconstructed = np.empty_like(total_waterfall)
    for fi in range(nf):
        design = projector @ unit_source_templates[:, :, fi].T
        target = projector @ total_waterfall[:, fi]
        amplitudes[:, fi], *_ = np.linalg.lstsq(design, target, rcond=1e-10)
        condition[fi] = np.linalg.cond(design)
        reconstructed[:, fi] = unit_source_templates[:, :, fi].T @ amplitudes[:, fi]
    return amplitudes, reconstructed, condition


def fft_power(waterfall: np.ndarray) -> np.ndarray:
    centered = waterfall - waterfall.mean(axis=0, keepdims=True)
    return np.abs(np.fft.rfft(centered, axis=0).T) ** 2


def simulate(simulator, sky: HarmonicCubeSky) -> np.ndarray:
    return np.asarray(simulator.simulate(sky=sky))[:, 0, :]


def make_summary_figure(output: Path, data: dict, metrics: dict) -> None:
    freq, hours = data["frequency_mhz"], data["elapsed_hours"]
    diffuse, sources = data["diffuse_waterfall_k"], data["source_waterfall_k"]
    truth, fit = data["true_source_spectra_k_sr"], data["recovered_source_spectra_k_sr"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, image, title in (
        (axes[0, 0], diffuse, "Diffuse sky (source-free truth)"),
        (axes[0, 1], sources, "A-team sources (truth)"),
    ):
        im = ax.pcolormesh(freq, hours / 24, np.log10(np.maximum(image, 1e-12)), shading="auto")
        ax.set(xlabel="Frequency [MHz]", ylabel="Elapsed days", title=title + " — log10 K")
        fig.colorbar(im, ax=ax)
    ratio = fft_power(sources) / np.maximum(fft_power(diffuse), np.finfo(float).tiny)
    im = axes[1, 0].pcolormesh(np.arange(ratio.shape[1]), freq,
                               np.log10(np.maximum(ratio, 1e-20)), shading="auto")
    axes[1, 0].set(xlabel="Temporal Fourier mode", ylabel="Frequency [MHz]",
                   title="log10(source/diffuse FFT power)")
    fig.colorbar(im, ax=axes[1, 0])
    for si, name in enumerate(data["source_names"]):
        axes[1, 1].loglog(freq, truth[si], label=f"{name} truth")
        axes[1, 1].loglog(freq, np.abs(fit[si]), "--", label=f"{name} fit")
    axes[1, 1].set(xlabel="Frequency [MHz]", ylabel="Integrated source T [K sr]",
                   title=f"Blind fit; median |fractional error|={metrics['median_absolute_fractional_error']:.2g}")
    axes[1, 1].legend(ncol=2, fontsize=8)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harmonics", type=Path, help="counterfactual_ulsa_harmonic.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("one-night-source-separation"))
    parser.add_argument("--lunar-cycle", type=int, default=2500)
    parser.add_argument("--cadence-hours", type=float, default=2.0)
    parser.add_argument("--frequency-min", type=float, default=5.0)
    parser.add_argument("--frequency-max", type=float, default=50.0)
    parser.add_argument("--lmax", type=int, default=32)
    parser.add_argument("--beam-sigma-deg", type=float, default=45.0)
    parser.add_argument("--foreground-modes", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    product = np.load(args.harmonics)
    if args.lmax > int(product["lmax"]):
        raise ValueError("requested lmax exceeds the harmonic product")
    all_freq = np.asarray(product["frequency_mhz"])
    select = (all_freq >= args.frequency_min) & (all_freq <= args.frequency_max)
    freq, nside = all_freq[select], int(product["input_nside"])

    def sky(key):
        return HarmonicCubeSky(truncate_alms(product[key][select], args.lmax), freq, nside)

    obs = lusee.Observation(args.lunar_cycle, deltaT_sec=args.cadence_hours * 3600,
                            lun_lat_deg=-23.814, lun_long_deg=182.258)
    # LunarCalendar integers delimit noon-to-noon cycles. LuSEE-Night operates
    # over the contiguous local interval for which the Sun is below the horizon.
    solar_altitude, _ = obs.get_track_solar("sun")
    night = np.asarray(solar_altitude) < 0
    if not np.any(night) or not np.all(np.diff(np.flatnonzero(night)) == 1):
        raise RuntimeError("could not identify one contiguous local lunar night")
    obs.times = obs.times[night]
    beam = lusee.BeamGauss(alt_deg=90, az_deg=0, sigma_deg=args.beam_sigma_deg,
                           one_over_freq_scaling=False, freq_min=float(freq[0]),
                           freq_max=float(freq[-1]), Nfreq=len(freq), id="diagnostic_zenith")
    simulator = lusee.CroSimulator(obs, [beam], sky("no_sources"), Tground=0,
                                   combinations=[(0, 0)], freq=freq, lmax=args.lmax)

    print("Simulating source-free and A-team waterfalls...")
    diffuse, sources = simulate(simulator, sky("no_sources")), simulate(simulator, sky("ateam"))
    total = diffuse + sources
    print("Simulating four known-location source templates...")
    individual = np.stack([simulate(simulator, sky(key)) for key in SOURCE_KEYS])
    source_alms = np.stack([truncate_alms(product[key][select], args.lmax) for key in SOURCE_KEYS])
    truth = np.real(source_alms[:, :, 0]) * np.sqrt(4 * np.pi)
    unit_templates = individual / truth[:, None, :]
    recovered, reconstructed, condition = recover_spectra(total, unit_templates, args.foreground_modes)
    fractional_error = (recovered - truth) / truth
    mode_scan = {}
    for modes in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        if 2 * modes + len(SOURCE_KEYS) + 1 >= len(obs.times):
            continue
        scan_fit, _, scan_condition = recover_spectra(total, unit_templates, modes)
        scan_error = np.abs((scan_fit - truth) / truth)
        mode_scan[str(modes)] = {
            "median_absolute_fractional_error": float(np.median(scan_error)),
            "p90_absolute_fractional_error": float(np.percentile(scan_error, 90)),
            "maximum_design_condition_number": float(np.max(scan_condition)),
        }
    metrics = {
        "median_absolute_fractional_error": float(np.median(np.abs(fractional_error))),
        "p90_absolute_fractional_error": float(np.percentile(np.abs(fractional_error), 90)),
        "maximum_design_condition_number": float(np.max(condition)),
        "source_to_diffuse_rms_ratio": float(np.std(sources) / np.std(diffuse)),
        "source_to_total_rms_ratio": float(np.std(sources) / np.std(total)),
        "utc_start": str(obs.times[0].isot),
        "utc_end": str(obs.times[-1].isot),
        "samples": int(len(obs.times)),
        "foreground_mode_scan": mode_scan,
    }
    output = {
        "frequency_mhz": freq,
        "elapsed_hours": np.arange(len(obs.times)) * obs.deltaT_sec / 3600,
        "diffuse_waterfall_k": diffuse,
        "source_waterfall_k": sources,
        "total_waterfall_k": total,
        "individual_source_waterfalls_k": individual,
        "unit_source_templates_k_per_k_sr": unit_templates,
        "true_source_spectra_k_sr": truth,
        "recovered_source_spectra_k_sr": recovered,
        "reconstructed_source_waterfall_k": reconstructed,
        "source_names": np.asarray([key.removeprefix("ateam_") for key in SOURCE_KEYS]),
        "design_condition_number": condition,
        "fractional_error": fractional_error,
    }
    np.savez_compressed(args.output_dir / "one_night_waterfalls.npz", **output)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    make_summary_figure(args.output_dir / "summary.png", output, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
