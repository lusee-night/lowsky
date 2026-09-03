"""One-lunar-night point-source separation experiment with Lowsky + LuSEEpy.

This contracts the canonical Lowsky harmonics with the generated four-port
instrument response, applies receiver loading, saves diffuse-only and
source-only waterfalls, and fits free source spectra using known transit
templates plus an unknown smooth temporal foreground. The diffuse-only
waterfall is used only to score the fit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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
    """Fit common source spectra across channels after per-channel projection."""
    total_waterfall = np.asarray(total_waterfall)
    unit_source_templates = np.asarray(unit_source_templates)
    if total_waterfall.ndim == 2:
        total_waterfall = total_waterfall[:, None, :]
        unit_source_templates = unit_source_templates[:, :, None, :]
    nt, nc, nf = total_waterfall.shape
    ns = unit_source_templates.shape[0]
    nuisance = fourier_nuisance_basis(nt, foreground_modes)
    projector = np.eye(nt) - nuisance @ nuisance.T
    amplitudes = np.empty((ns, nf))
    condition = np.empty(nf)
    reconstructed = np.empty_like(total_waterfall)
    for fi in range(nf):
        # Each channel gets its own arbitrary low-order foreground coefficients,
        # while a source has one shared spectral amplitude across all channels.
        design = np.einsum(
            "tu,suc->tcs", projector, unit_source_templates[:, :, :, fi]
        )
        target = np.einsum("tu,uc->tc", projector, total_waterfall[:, :, fi])
        design = design.reshape(nt * nc, ns)
        target = target.reshape(nt * nc)
        amplitudes[:, fi], *_ = np.linalg.lstsq(design, target, rcond=1e-10)
        condition[fi] = np.linalg.cond(design)
        reconstructed[:, :, fi] = np.einsum(
            "stc,s->tc", unit_source_templates[:, :, :, fi], amplitudes[:, fi]
        )
    if nc == 1:
        reconstructed = reconstructed[:, 0, :]
    return amplitudes, reconstructed, condition


def fft_power(waterfall: np.ndarray) -> np.ndarray:
    if waterfall.ndim == 3:
        waterfall = np.sqrt(np.mean(np.abs(waterfall) ** 2, axis=1))
    centered = waterfall - waterfall.mean(axis=0, keepdims=True)
    return np.abs(np.fft.rfft(centered, axis=0).T) ** 2


def simulate(simulator, sky: HarmonicCubeSky) -> np.ndarray:
    return np.asarray(simulator.simulate(sky=sky))


def make_summary_figure(output: Path, data: dict, metrics: dict) -> None:
    freq, hours = data["frequency_mhz"], data["elapsed_hours"]
    diffuse = data["diffuse_waterfall_v2_per_hz"]
    sources = data["source_waterfall_v2_per_hz"]
    diffuse_display = np.sqrt(np.mean(np.abs(diffuse) ** 2, axis=1))
    source_display = np.sqrt(np.mean(np.abs(sources) ** 2, axis=1))
    truth, fit = data["true_source_spectra_k_sr"], data["recovered_source_spectra_k_sr"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, image, title in (
        (axes[0, 0], diffuse_display, "Diffuse sky (channel RMS)"),
        (axes[0, 1], source_display, "A-team sources (channel RMS)"),
    ):
        im = ax.pcolormesh(
            freq,
            hours / 24,
            np.log10(np.maximum(image, np.finfo(float).tiny)),
            shading="auto",
        )
        ax.set(
            xlabel="Frequency [MHz]",
            ylabel="Elapsed days",
            title=title + " — log10(V²/Hz)",
        )
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
    parser.add_argument(
        "--response-file",
        type=Path,
        help=(
            "validated four-port InstrumentResponse FITS; defaults to "
            "$LUSEE_RESPONSE_FILE or the local generated V16 product"
        ),
    )
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
    response_file = args.response_file
    if response_file is None and os.environ.get("LUSEE_RESPONSE_FILE"):
        response_file = Path(os.environ["LUSEE_RESPONSE_FILE"])
    if response_file is None:
        candidate = Path("/local/zack/receive_matrix/lusee_bgl_v16_response_v3.fits")
        if candidate.exists():
            response_file = candidate
    if response_file is None or not response_file.exists():
        raise FileNotFoundError(
            "pass --response-file or set LUSEE_RESPONSE_FILE to a validated "
            "four-port response FITS"
        )
    response = lusee.InstrumentResponse(response_file, require_validated=True)
    receiver = lusee.receiver_from_config({"model": "jfet"})
    # Component waterfalls must be strictly additive. The generated response
    # still supplies the coherent layered-regolith/lander response, while
    # thermal Moon and antenna emission are intentionally excluded here.
    simulator = lusee.FullStokesCroSimulator(
        obs,
        response,
        sky("no_sources"),
        receiver,
        T_moon=0.0,
        T_ant=0.0,
        products="all",
        freq=freq,
        lmax=args.lmax,
    )

    print("Simulating source-free and A-team waterfalls...")
    diffuse = simulate(simulator, sky("no_sources"))
    channel_labels = tuple(simulator.product_labels)
    port_names = tuple(str(response.header["PORTNAMES"]).split(","))
    channel_names = [
        f"{port_names[int(label[0])]}{port_names[int(label[1])]}_"
        f"{'real' if label[2] == 'R' else 'imag'}"
        for label in channel_labels
    ]
    sources = simulate(simulator, sky("ateam"))
    total = diffuse + sources
    print("Simulating four known-location source templates...")
    individual = np.stack([simulate(simulator, sky(key)) for key in SOURCE_KEYS])
    source_alms = np.stack([truncate_alms(product[key][select], args.lmax) for key in SOURCE_KEYS])
    truth = np.real(source_alms[:, :, 0]) * np.sqrt(4 * np.pi)
    unit_templates = individual / truth[:, None, None, :]
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
        "response_file": str(response_file.resolve()),
        "response_content_hash": response.content_hash,
        "response_simulation": str(response.header.get("SIMULATION", "unknown")),
        "receiver_model": type(receiver).__name__,
        "thermal_moon_and_antenna_included": False,
        "channels": channel_names,
        "foreground_mode_scan": mode_scan,
    }
    output = {
        "frequency_mhz": freq,
        "elapsed_hours": np.arange(len(obs.times)) * obs.deltaT_sec / 3600,
        "diffuse_waterfall_v2_per_hz": diffuse,
        "source_waterfall_v2_per_hz": sources,
        "total_waterfall_v2_per_hz": total,
        "individual_source_waterfalls_v2_per_hz": individual,
        "unit_source_templates_v2_per_hz_per_k_sr": unit_templates,
        "true_source_spectra_k_sr": truth,
        "recovered_source_spectra_k_sr": recovered,
        "reconstructed_source_waterfall_v2_per_hz": reconstructed,
        "source_names": np.asarray([key.removeprefix("ateam_") for key in SOURCE_KEYS]),
        "channel_names": np.asarray(channel_names),
        "design_condition_number": condition,
        "fractional_error": fractional_error,
    }
    np.savez_compressed(args.output_dir / "one_night_waterfalls.npz", **output)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    make_summary_figure(args.output_dir / "summary.png", output, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
