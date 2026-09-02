"""Generate a fresh, physics-based ultra-low-frequency sky realization.

This deliberately preserves the radiative-transfer ingredients of Cong et al.
(2021) while avoiding a pixel-by-pixel Haslam residual template.  JAX performs
the three-dimensional line-of-sight model, transfer, and gradient-based tuning;
healpy is used only for HEALPix geometry and spherical-harmonic synthesis.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from astropy.io import fits
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from lusee.SkyModels import HarmonicPointSourceSky

from .model import (
    SkyInputs,
    SkyParameters,
    optical_depth_coefficient,
    smooth_broken_power_law,
    transfer_frequency,
)


KPC_TO_PC = 1_000.0
MHZ_TO_GHZ = 1.0e-3
C_LIGHT = 299_792_458.0
K_BOLTZMANN = 1.380649e-23


@dataclass(frozen=True)
class SkyConfig:
    nside: int = 32
    sky_mode: str = "random"
    solar_radius_kpc: float = 8.20
    solar_height_kpc: float = 0.015
    ray_oversample: int = 2
    n_distance: int = 192
    max_distance_kpc: float = 50.0
    seed: int = 20260901
    n_shells: int = 8
    synch_fluctuation_sigma: float = 0.28
    em_fluctuation_sigma: float = 0.32
    beta_fluctuation_sigma: float = 0.055
    output_beam_fwhm_deg: float = 2.0
    harmonic_lmax: int | None = None
    electron_temperature_k: float = 8_000.0
    local_bubble_radius_kpc: float = 0.20
    local_bubble_em_factor: float = 0.15
    mean_local_shells: float = 4.0
    min_local_shells: int = 3
    max_local_shells: int = 6
    shell_distance_bins: int = 8
    shell_quadrature_steps: int = 128
    shell_surface_fluctuation_sigma: float = 0.30
    shell_radius_corrugation_fraction: float = 0.07
    shell_width_modulation_fraction: float = 0.35
    shell_spectral_break_mhz: float = 30.0
    shell_spectral_smoothness: float = 2.0


@dataclass(frozen=True)
class TunedParameters:
    emissivity_scale: float
    emission_measure_scale: float
    beta_offset: float
    objective: float


@dataclass(frozen=True)
class LocalShellCatalog:
    name: np.ndarray
    center_l_deg: np.ndarray
    center_b_deg: np.ndarray
    distance_kpc: np.ndarray
    radius_kpc: np.ndarray
    thickness_kpc: np.ndarray
    excess_408_k: np.ndarray
    beta_temperature: np.ndarray
    low_frequency_beta_temperature: np.ndarray
    magnetic_axis: np.ndarray
    density_gradient_axis: np.ndarray
    density_gradient_strength: np.ndarray
    filament_axes: np.ndarray
    filament_wavenumber: np.ndarray
    filament_phase: np.ndarray
    emitting_fraction: np.ndarray
    patch_softness: np.ndarray


@dataclass(frozen=True)
class PreparedSky:
    """Prepared model fields plus metadata needed by the full pipeline."""

    inputs: SkyInputs
    geometry: dict[str, np.ndarray]
    local_catalog: LocalShellCatalog
    anchor_emission_measure_rate: jax.Array


ATEAM = {
    # Galactic (l,b), S_50MHz [Jy], flux spectral index, distance [kpc or inf],
    # phenomenological internal turnover optical depth at 10 MHz.
    "CAS_A": (111.7376, -2.1345, 27_104.0, -0.77, 3.4, 0.35),
    "CYG_A": (76.1899, 5.7554, 22_146.0, -0.78, np.inf, 0.08),
    "TAU_A": (184.5547, -5.7833, 2_008.0, -0.27, 2.0, 0.12),
    "VIR_A": (283.7778, 74.4912, 2_635.0, -0.85, np.inf, 0.04),
}


def _unit_gaussian_field(
    nside: int,
    rng: np.random.Generator,
    slope: float,
    ell_min: int = 2,
    ell_taper: float | None = None,
) -> np.ndarray:
    """Return a reproducible, zero-mean unit-rms isotropic Gaussian field."""
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1, dtype=float)
    cl = np.zeros_like(ell)
    good = ell >= ell_min
    cl[good] = (ell[good] / 30.0) ** slope
    if ell_taper is not None:
        cl *= np.exp(-((ell / ell_taper) ** 4))
    # synalm accepts NumPy's global RNG, so pass explicitly generated complex
    # coefficients instead of mutating global random state.
    size = hp.Alm.getsize(lmax)
    l_of_alm, m_of_alm = hp.Alm.getlm(lmax, np.arange(size))
    variance = cl[l_of_alm]
    real = rng.normal(size=size)
    imag = rng.normal(size=size)
    alm = (real + 1j * imag) * np.sqrt(variance / 2.0)
    alm[m_of_alm == 0] = real[m_of_alm == 0] * np.sqrt(variance[m_of_alm == 0])
    field = hp.alm2map(alm, nside=nside, lmax=lmax)
    field -= np.mean(field)
    field /= np.std(field)
    return field


def make_random_fields(config: SkyConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    ray_nside = config.nside * config.ray_oversample
    synch_shells = np.stack(
        [_unit_gaussian_field(ray_nside, rng, -2.7, ell_taper=80.0) for _ in range(config.n_shells)]
    )
    em_shells = np.stack(
        [_unit_gaussian_field(ray_nside, rng, -11.0 / 3.0, ell_taper=65.0) for _ in range(config.n_shells)]
    )
    beta = _unit_gaussian_field(ray_nside, rng, -3.2, ell_taper=30.0)
    return {"synch_shells": synch_shells, "em_shells": em_shells, "beta": beta}


def _random_unit_vectors(rng: np.random.Generator, count: int) -> np.ndarray:
    vectors = rng.normal(size=(count, 3))
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def make_local_shell_catalog(config: SkyConfig) -> LocalShellCatalog:
    """Draw a catalog of nearby radio-loop/superbubble analogues.

    The first object is conditioned to be Loop-I-like in physical scale, but
    its direction is random. Remaining objects follow broad priors from the
    classical radio loops. Nothing is copied from the observed sky.
    """
    if config.sky_mode == "ours":
        return make_our_sky_shell_catalog(config)
    rng = np.random.default_rng(config.seed + 104_729)
    count = int(np.clip(rng.poisson(config.mean_local_shells), config.min_local_shells, config.max_local_shells))
    l_deg = rng.uniform(0.0, 360.0, count)
    b_deg = np.clip(rng.normal(0.0, 25.0, count), -65.0, 65.0)
    distance = np.clip(rng.lognormal(np.log(0.15), np.log(2.0), count), 0.06, 0.60)
    radius = np.clip(rng.lognormal(np.log(0.080), np.log(1.5), count), 0.045, 0.180)
    thickness_fraction = rng.uniform(0.08, 0.20, count)
    excess = np.clip(rng.lognormal(np.log(7.0), 0.35, count), 2.0, 16.0)
    beta = np.clip(rng.normal(2.85, 0.16, count), 2.50, 3.15)
    beta_low = np.clip(rng.normal(2.55, 0.10, count), 2.35, 2.72)

    # A mandatory large, nearby analogue reflects the strong local-selection
    # condition that the Sun lies next to a Loop-I-scale superbubble.
    distance[0] = np.clip(rng.normal(0.24, 0.035), 0.16, 0.32)
    radius[0] = np.clip(rng.normal(0.22, 0.025), 0.17, 0.27)
    thickness_fraction[0] = np.clip(rng.normal(0.10, 0.02), 0.06, 0.16)
    excess[0] = np.clip(rng.normal(8.4, 1.0), 5.0, 12.0)
    beta[0] = np.clip(rng.normal(2.64, 0.10), 2.45, 2.85)
    beta_low[0] = np.clip(rng.normal(2.52, 0.08), 2.35, 2.68)

    return LocalShellCatalog(
        name=np.asarray([f"RANDOM_LOOP_{i + 1}" for i in range(count)]),
        center_l_deg=l_deg,
        center_b_deg=b_deg,
        distance_kpc=distance,
        radius_kpc=radius,
        thickness_kpc=radius * thickness_fraction,
        excess_408_k=excess,
        beta_temperature=-beta,
        low_frequency_beta_temperature=-beta_low,
        magnetic_axis=_random_unit_vectors(rng, count),
        density_gradient_axis=_random_unit_vectors(rng, count),
        density_gradient_strength=rng.uniform(0.4, 1.3, count),
        filament_axes=_random_unit_vectors(rng, count * 3).reshape(count, 3, 3),
        filament_wavenumber=rng.uniform(2.5, 7.0, (count, 3)),
        filament_phase=rng.uniform(0.0, 2.0 * np.pi, (count, 3)),
        emitting_fraction=rng.uniform(0.25, 0.70, count),
        patch_softness=rng.uniform(0.10, 0.20, count),
    )


def _normal_at_shell_surface(
    center_l_deg: float,
    center_b_deg: float,
    distance_kpc: float,
    radius_kpc: float,
    target_l_deg: float,
    target_b_deg: float,
) -> np.ndarray:
    def direction(l_deg: float, b_deg: float) -> np.ndarray:
        l = np.radians(l_deg)
        b = np.radians(b_deg)
        return np.array([np.cos(b) * np.sin(l), -np.cos(b) * np.cos(l), np.sin(b)])

    center = distance_kpc * direction(center_l_deg, center_b_deg)
    ray = direction(target_l_deg, target_b_deg)
    projection = float(ray @ center)
    perpendicular2 = max(float(center @ center) - projection**2, 0.0)
    offset = np.sqrt(max(radius_kpc**2 - perpendicular2, 0.0))
    point = max(projection + offset, 0.0) * ray
    normal = point - center
    return normal / max(np.linalg.norm(normal), 1e-12)


def make_our_sky_shell_catalog(config: SkyConfig) -> LocalShellCatalog:
    """Condition local-loop physics on measured Milky Way directions.

    Centers are anchored, but distances, sizes, brightnesses, spectra, shock
    coverage, and surface fluctuations remain uncertain seeded draws.
    """
    rng = np.random.default_rng(config.seed + 104_729)
    names = np.asarray(["LOOP_I_S1", "LOOP_I_S2", "LOOP_II", "LOOP_III", "LOOP_IV"])
    l0 = np.asarray([346.0, 347.0, 100.0, 124.0, 315.0])
    b0 = np.asarray([3.0, 37.0, -32.5, 15.5, 48.5])
    d0 = np.asarray([0.078, 0.095, 0.097, 0.150, 0.180])
    r0 = np.asarray([0.0815, 0.075, 0.069, 0.081, 0.061])
    # These are total-intensity synchrotron envelope widths, not the thinner
    # geometric/polarized walls fitted by ideal-shell models.  Broad envelopes
    # are needed because the observed loops are diffuse, overlapping ridges.
    w0 = np.asarray([0.036, 0.042, 0.027, 0.031, 0.025])
    t408 = np.asarray([5.0, 5.0, 8.0, 8.5, 3.0])
    beta_high = np.asarray([2.74, 2.74, 2.88, 2.68, 2.90])
    beta_low = np.asarray([2.64, 2.64, 2.60, 2.63, 2.77])
    target_l = np.asarray([25.0, 345.0, 160.0, 120.0, 305.0])
    target_b = np.asarray([50.0, 58.0, -35.0, 40.0, 65.0])
    count = len(names)

    l_deg = l0 + rng.normal(0.0, 1.5, count)
    b_deg = b0 + rng.normal(0.0, 1.5, count)
    distance = d0 * rng.lognormal(0.0, 0.10, count)
    radius = r0 * rng.lognormal(0.0, 0.10, count)
    thickness = w0 * rng.lognormal(0.0, 0.12, count)
    excess = np.clip(
        t408 + rng.normal(0.0, [0.7, 0.7, 1.0, 1.0, 0.7]), 3.0, 9.0
    )
    # Truncated observational priors prevent rare seeded draws from creating
    # spectra outside the measured loop population.  The low-frequency branch
    # must flatten, rather than accidentally steepen, below the break.
    bh = np.clip(beta_high + rng.normal(0.0, 0.08, count), 2.55, 3.03)
    bl = np.clip(beta_low + rng.normal(0.0, 0.06, count), 2.40, bh - 0.03)
    gradient_axes = np.stack(
        [
            _normal_at_shell_surface(l_deg[i], b_deg[i], distance[i], radius[i], target_l[i], target_b[i])
            for i in range(count)
        ]
    )
    magnetic_axes = _random_unit_vectors(rng, count)
    # Ensure the conditioned bright surface is not suppressed by a parallel
    # shock/field geometry.
    magnetic_axes -= np.sum(magnetic_axes * gradient_axes, axis=1)[:, None] * gradient_axes
    magnetic_axes /= np.maximum(np.linalg.norm(magnetic_axes, axis=1)[:, None], 1e-12)
    return LocalShellCatalog(
        name=names,
        center_l_deg=l_deg,
        center_b_deg=b_deg,
        distance_kpc=distance,
        radius_kpc=radius,
        thickness_kpc=thickness,
        excess_408_k=excess,
        beta_temperature=-bh,
        low_frequency_beta_temperature=-bl,
        magnetic_axis=magnetic_axes,
        density_gradient_axis=gradient_axes,
        density_gradient_strength=rng.uniform(0.8, 1.4, count),
        filament_axes=_random_unit_vectors(rng, count * 3).reshape(count, 3, 3),
        filament_wavenumber=rng.uniform(2.5, 7.0, (count, 3)),
        filament_phase=rng.uniform(0.0, 2.0 * np.pi, (count, 3)),
        emitting_fraction=np.asarray([0.55, 0.50, 0.45, 0.55, 0.35]) + rng.normal(0.0, 0.04, count),
        patch_softness=rng.uniform(0.10, 0.18, count),
    )


def shell_ray_segments(
    catalog: LocalShellCatalog, geometry: dict[str, np.ndarray], config: SkyConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Project continuous 3D shells into distance-ordered emission layers.

    A Gaussian radial emissivity profile avoids the tangent cusp of a top-hat
    hollow sphere. Fine local quadrature is accumulated into a small number of
    distance layers, retaining foreground-opacity ordering without putting the
    entire local grid into the main Galactic ray calculation.
    """
    l = np.radians(geometry["l_deg"])
    b = np.radians(geometry["b_deg"])
    directions = np.column_stack((np.cos(b) * np.sin(l), -np.cos(b) * np.cos(l), np.sin(b)))
    nfeature = len(catalog.distance_kpc)
    npix = len(l)
    nbins = config.shell_distance_bins
    nquad = config.shell_quadrature_steps
    if nquad % nbins:
        raise ValueError("shell_quadrature_steps must be divisible by shell_distance_bins")
    segment_408 = np.zeros((nfeature, nbins, npix), dtype=np.float64)
    distance_index = np.zeros((nfeature, nbins, npix), dtype=np.int32)
    chunk_size = 4096

    for i in range(nfeature):
        lc = np.radians(catalog.center_l_deg[i])
        bc = np.radians(catalog.center_b_deg[i])
        center_direction = np.array([np.cos(bc) * np.sin(lc), -np.cos(bc) * np.cos(lc), np.sin(bc)])
        center = catalog.distance_kpc[i] * center_direction
        radial_sigma = catalog.thickness_kpc[i] / np.sqrt(8.0 * np.log(2.0))
        max_radius = catalog.radius_kpc[i] * (1.0 + config.shell_radius_corrugation_fraction)
        max_sigma = radial_sigma * (1.0 + config.shell_width_modulation_fraction)
        s_max = catalog.distance_kpc[i] + max_radius + 5.0 * max_sigma
        ds_local = s_max / nquad
        sample_s = (np.arange(nquad) + 0.5) * ds_local
        samples_per_bin = nquad // nbins
        for j in range(nbins):
            mid_distance = np.mean(sample_s[j * samples_per_bin : (j + 1) * samples_per_bin])
            distance_index[i, j] = np.clip(
                int(np.floor(mid_distance / geometry["ds"])), 0, config.n_distance - 1
            )

        for start in range(0, npix, chunk_size):
            stop = min(start + chunk_size, npix)
            ray = directions[start:stop]
            displacement = ray[:, None, :] * sample_s[None, :, None] - center[None, None, :]
            radial_distance = np.linalg.norm(displacement, axis=2)
            normal = displacement / np.maximum(radial_distance[:, :, None], 1e-10)
            bdot = np.einsum("qsd,d->qs", normal, catalog.magnetic_axis[i])
            obliquity = 0.20 + 0.80 * np.maximum(1.0 - bdot**2, 0.0) ** 1.5
            gradient_dot = np.einsum("qsd,d->qs", normal, catalog.density_gradient_axis[i])
            gradient = np.exp(catalog.density_gradient_strength[i] * (gradient_dot - 1.0))
            modes = np.zeros_like(radial_distance)
            for mode in range(3):
                phase_coordinate = np.einsum(
                    "qsd,d->qs", normal, catalog.filament_axes[i, mode]
                )
                modes += np.sin(
                    catalog.filament_wavenumber[i, mode] * phase_coordinate
                    + catalog.filament_phase[i, mode]
                )
            modes /= np.sqrt(3.0)
            # Old superbubble walls are neither spherical nor uniformly thin.
            # Use coherent surface modes to warp their radius and width.  The
            # deformation is tied to the same physical surface field as the
            # emissivity, so it produces broad, irregular spurs rather than
            # adding uncorrelated angular texture.
            corrugation = np.tanh(modes)
            local_radius = catalog.radius_kpc[i] * (
                1.0 + config.shell_radius_corrugation_fraction * corrugation
            )
            width_pattern = np.tanh(0.70 * modes + 0.30 * gradient_dot)
            local_sigma = radial_sigma * (
                1.0 + config.shell_width_modulation_fraction * width_pattern
            )
            radial_profile = np.exp(
                -0.5 * ((radial_distance - local_radius) / local_sigma) ** 2
            )
            filament = np.exp(
                config.shell_surface_fluctuation_sigma * modes
                - 0.5 * config.shell_surface_fluctuation_sigma**2
            )
            patch_coordinate = gradient_dot + 0.35 * modes
            patch_threshold = 1.0 - 2.0 * catalog.emitting_fraction[i]
            activation = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(
                        (patch_coordinate - patch_threshold) / catalog.patch_softness[i],
                        -30.0,
                        30.0,
                    )
                )
            )
            emissivity = radial_profile * obliquity * gradient * filament * activation
            for j in range(nbins):
                lo = j * samples_per_bin
                hi = (j + 1) * samples_per_bin
                segment_408[i, j, start:stop] = np.sum(emissivity[:, lo:hi], axis=1) * ds_local

        # Normalize the projected bright rim, rather than the voxel emissivity,
        # to the measured background-subtracted 408-MHz loop contrast.
        projected = np.sum(segment_408[i], axis=0)
        bright_rim = np.percentile(projected[projected > 0.0], 99.0)
        segment_408[i] *= catalog.excess_408_k[i] / max(bright_rim, 1e-12)
    return segment_408, distance_index


def prepare_geometry(config: SkyConfig) -> dict[str, np.ndarray]:
    ray_nside = config.nside * config.ray_oversample
    npix = hp.nside2npix(ray_nside)
    l_deg, b_deg = hp.pix2ang(ray_nside, np.arange(npix), lonlat=True)
    l = np.radians(l_deg)[:, None]
    b = np.radians(b_deg)[:, None]
    ds = config.max_distance_kpc / config.n_distance
    s = (np.arange(config.n_distance) + 0.5) * ds
    cos_b = np.cos(b)
    # NE2001 right-handed coordinates: Sun=(0,Rsun,0), +x toward l=90.
    x = s[None, :] * cos_b * np.sin(l)
    y = config.solar_radius_kpc - s[None, :] * cos_b * np.cos(l)
    z = config.solar_height_kpc + s[None, :] * np.sin(b)
    radius = np.sqrt(x * x + y * y)
    theta = np.mod(np.arctan2(-x, y), 2.0 * np.pi)
    shell_edges = np.linspace(0.0, config.max_distance_kpc, config.n_shells + 1)
    shell_index = np.clip(np.digitize(s, shell_edges) - 1, 0, config.n_shells - 1)
    return {
        "nside": ray_nside,
        "l_deg": l_deg,
        "b_deg": b_deg,
        "s": s,
        "ds": ds,
        "x": x,
        "y": y,
        "z": z,
        "radius": radius,
        "theta": theta,
        "shell_index": shell_index,
    }


def _sech2(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / jnp.cosh(jnp.clip(x, -20.0, 20.0)) ** 2


def local_bubble_boundary_kpc(
    ux: jnp.ndarray, uy: jnp.ndarray, uz: jnp.ndarray
) -> jnp.ndarray:
    """Irregular Local-Bubble wall radius conditioned on modern 3D dust work."""
    bubble_mode = (
        0.55 * jnp.sin(2.1 * ux + 1.3 * uy + 0.4)
        + 0.35 * jnp.sin(2.7 * uy - 1.6 * uz + 1.1)
        + 0.25 * jnp.sin(3.2 * uz + 1.4 * ux - 0.7)
    )
    boundary = 0.125 * (1.0 + 0.55 * jnp.tanh(bubble_mode))
    boundary += 0.47 / (1.0 + jnp.exp(-(uz - 0.86) / 0.035))
    boundary += 0.18 / (1.0 + jnp.exp(-(-uz - 0.93) / 0.030))
    return jnp.clip(boundary, 0.070, 0.650)


def _spiral_arm_density(radius: jnp.ndarray, theta: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Differentiable approximation to the five NE2001 logarithmic arms."""
    a = jnp.asarray([4.25, 4.25, 4.89, 4.89, 4.57])
    rmin = jnp.asarray([3.48, 3.48, 4.90, 3.76, 8.10])
    thmin = jnp.asarray([0.0, 3.141, 2.525, 4.24, 5.847])
    extent = jnp.asarray([6.0, 6.0, 6.0, 6.0, 0.55])
    # Wainscoat-to-TC93 remapping factors from the released NE2001 source.
    density_factor = jnp.asarray([0.5, 1.3, 1.0, 1.2, 0.25])
    width_factor = jnp.asarray([1.0, 1.0, 0.8, 1.5, 1.0])
    height_factor = jnp.asarray([1.0, 1.3, 1.5, 0.8, 1.0])
    candidates = jnp.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    total = jnp.zeros_like(radius)
    for i in range(5):
        th = theta[..., None] + 2.0 * jnp.pi * candidates
        valid = (th >= thmin[i]) & (th <= thmin[i] + extent[i])
        arm_r = rmin[i] * jnp.exp((th - thmin[i]) / a[i])
        radial_delta = jnp.where(valid, jnp.abs(radius[..., None] - arm_r), 1.0e3)
        distance = jnp.min(radial_delta, axis=-1)
        gauss = jnp.exp(-(distance / (0.65 * width_factor[i])) ** 2)
        radial_cut = jnp.where(radius > 10.5, _sech2((radius - 10.5) / 2.0), 1.0)
        total += (
            0.028
            * density_factor[i]
            * gauss
            * radial_cut
            * _sech2(z / (0.23 * height_factor[i]))
        )
    return total


@jax.jit
def physical_fields(
    radius: jnp.ndarray,
    theta: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    s: jnp.ndarray,
    em_multiplier: jnp.ndarray,
    local_bubble_radius: float,
    local_bubble_factor: float,
    solar_radius_kpc: float,
    solar_height_kpc: float,
    conditioned_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return 408-MHz emissivity and differential EM in pc cm^-6/kpc."""
    warp = 0.10 * jnp.clip(radius - 9.0, 0.0, 6.0) * jnp.sin(theta - jnp.radians(18.0))
    z_disk = z - conditioned_weight * warp
    # Cong et al. (2021), Eq. 17 and Table 1.
    emissivity_408 = (
        43.10
        * ((radius + 0.1) / 3.41) ** 0.46
        * jnp.exp(-radius / 3.41)
        * jnp.exp(-(jnp.abs(z_disk) / 1.12) ** 1.23)
    )

    # NE2001 large-scale components and fluctuation factors.
    suncos = jnp.cos(0.5 * jnp.pi * solar_radius_kpc / 17.5)
    thick_radial = jnp.where(radius < 17.5, jnp.cos(0.5 * jnp.pi * radius / 17.5) / suncos, 0.0)
    ne_thick = (0.033 / 0.97) * thick_radial * _sech2(z_disk / 0.97)
    ne_thin = 0.08 * jnp.exp(-((radius - 3.8) / 1.8) ** 2) * _sech2(z_disk / 0.15)
    ne_arms = _spiral_arm_density(radius, theta, z_disk)
    ne_gc = 10.0 * jnp.exp(-(((x + 0.01) / 0.145) ** 2 + (y / 0.145) ** 2 + ((z + 0.02) / 0.026) ** 2))

    # Material arms trace low-energy CRE/B structure only weakly after
    # kiloparsec-scale diffusion. Add a conservative contrast, plus a modest
    # bar-shaped central enhancement in the our-sky-conditioned mode.
    arm_modulation = 1.0 + conditioned_weight * 0.25 * jnp.clip(ne_arms / 0.028, 0.0, 1.5)
    bar_angle = jnp.radians(30.0)
    bar_x = x * jnp.cos(bar_angle) + y * jnp.sin(bar_angle)
    bar_y = -x * jnp.sin(bar_angle) + y * jnp.cos(bar_angle)
    bar = jnp.exp(-((jnp.abs(bar_x) / 5.0) ** 4 + (jnp.abs(bar_y) / 1.2) ** 4 + (jnp.abs(z_disk) / 0.30) ** 2))
    emissivity_408 *= arm_modulation * (1.0 + conditioned_weight * 0.25 * bar)

    # Eq. 21: effective unresolved cloud/turbulence enhancement.  The
    # absolute normalization is subsequently tuned to ULSA's global spectrum.
    em_rate = KPC_TO_PC * (
        0.18 * ne_thick**2
        + 120.0 * ne_thin**2
        + 5.0 * ne_arms**2
        + 6.0e4 * ne_gc**2
    )
    spherical_local = jnp.where(s[None, :] < local_bubble_radius, local_bubble_factor, 1.0)
    # O'Neill et al. (2024)-conditioned Local Bubble: a low-order irregular
    # radial surface with narrow polar blowouts, rather than an ellipsoid.
    # Its all-sky mean is close to 170 pc, ordinary walls span roughly
    # 70--220 pc, and the northern chimney reaches beyond 600 pc.  The 8-pc
    # logistic scale corresponds to a roughly 35-pc 10--90% wall transition.
    ray_radius = jnp.maximum(s[None, :], 1.0e-6)
    ux = x / ray_radius
    uy = (y - solar_radius_kpc) / ray_radius
    uz = (z - solar_height_kpc) / ray_radius
    bubble_boundary = local_bubble_boundary_kpc(ux, uy, uz)
    bubble_inside = 1.0 / (
        1.0 + jnp.exp(jnp.clip((ray_radius - bubble_boundary) / 0.008, -30.0, 30.0))
    )
    conditioned_local = 1.0 - (1.0 - 0.20) * bubble_inside
    local_factor = (1.0 - conditioned_weight) * spherical_local + conditioned_weight * conditioned_local
    em_rate = em_rate * local_factor * em_multiplier

    def center_from_lbd(l_deg: float, b_deg: float, distance: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ll = jnp.radians(l_deg)
        bb = jnp.radians(b_deg)
        return (
            distance * jnp.cos(bb) * jnp.sin(ll),
            solar_radius_kpc - distance * jnp.cos(bb) * jnp.cos(ll),
            solar_height_kpc + distance * jnp.sin(bb),
        )

    cx, cy, cz = center_from_lbd(80.0, 0.0, 1.45)
    # At our ~degree native resolution, the observed 4.3-pc Cygnus filaments
    # are necessarily a beam-averaged sub-grid complex.  The normalization
    # reaches the observed low end (EM ~ 10^3 pc cm^-6) without pretending to
    # resolve their arcminute morphology.
    cygnus_em = 14_000.0 * jnp.exp(
        -0.5 * (((x - cx) / 0.10) ** 2 + ((y - cy) / 0.10) ** 2 + ((z - cz) / 0.065) ** 2)
    )
    # Nearby thin ionized shells are integrated separately below.  Putting
    # them on this coarse Galactic distance grid artificially broadens their
    # projected absorption footprints.
    anchor_em_rate = conditioned_weight * cygnus_em
    return emissivity_408, em_rate + anchor_em_rate, anchor_em_rate


def make_local_partial_screens(
    config: SkyConfig,
    geometry: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray-trace nearby ionized shells as partial-cover screens.

    Gum's roughly 15--20 pc wall and Orion--Eridanus's 5--13 pc filaments are
    both narrower than the global radial sampling.  A dedicated 2-pc grid
    avoids numerically inflating their angular sizes.  Emission measure inside
    a clump and projected covering fraction remain separate because
    ``1-f+f*exp(-tau)`` is not ``exp(-f*tau)``.
    """

    npix = hp.nside2npix(int(geometry["nside"]))
    if config.sky_mode != "ours":
        zeros = np.zeros((3, npix), dtype=np.float64)
        return zeros, zeros, np.full((3, npix), 0.25, dtype=np.float64)

    l = np.radians(np.asarray(geometry["l_deg"], dtype=np.float64))
    b = np.radians(np.asarray(geometry["b_deg"], dtype=np.float64))
    directions = np.stack(
        [np.cos(b) * np.sin(l), -np.cos(b) * np.cos(l), np.sin(b)], axis=1
    )
    local_s = np.arange(0.001, 0.651, 0.002, dtype=np.float64)
    ds = 0.002
    # l, b, center distance, radius, FWHM, axis stretches, phase,
    # n_e^2-equivalent rate, min/max sub-beam covering, selector threshold
    # and softness. Gum is first; the two nested Orion interfaces follow.
    specifications = (
        # Purcell et al. (2015): D~450 pc, R~160 pc and an 18.5-pc wall.
        # Purcell's azimuthal profile is diffuse: EM is about 80 pc cm^-6 in
        # the interior, peaks near 220 at the limb, and falls below 30 outside.
        # The rate is the in-clump value; the measured f~0.3 is applied below.
        (258.0, -6.6, 0.45, 0.160, 0.0185, (1.00, 1.00, 1.00), 0.4, 12_000.0, 0.20, 0.98, 0.20, 0.20),
        (198.0, -32.0, 0.25, 0.080, 0.008, (1.20, 0.86, 1.08), 2.1, 2_200.0, 0.00, 0.45, 0.95, 0.08),
        (205.0, -43.0, 0.33, 0.100, 0.012, (1.14, 0.90, 1.16), 4.0, 1_700.0, 0.00, 0.38, 0.95, 0.08),
    )
    emission_measure = np.zeros((len(specifications), npix), dtype=np.float64)
    covering = np.zeros_like(emission_measure)
    distance = np.zeros_like(emission_measure)

    for screen_index, spec in enumerate(specifications):
        (
            l_deg, b_deg, center_distance, radius, fwhm, stretch, phase,
            peak_rate, fmin, fmax, cover_threshold, cover_softness,
        ) = spec
        ll, bb = np.radians(l_deg), np.radians(b_deg)
        center = center_distance * np.asarray(
            [np.cos(bb) * np.sin(ll), -np.cos(bb) * np.cos(ll), np.sin(bb)]
        )
        sx, sy, sz = stretch
        sigma = fwhm / np.sqrt(8.0 * np.log(2.0))
        for start in range(0, npix, 4096):
            stop = min(start + 4096, npix)
            xyz = directions[start:stop, :, None] * local_s[None, None, :]
            delta = xyz - center[None, :, None]
            rr = np.sqrt(
                (delta[:, 0] / sx) ** 2
                + (delta[:, 1] / sy) ** 2
                + (delta[:, 2] / sz) ** 2
            )
            radial = np.exp(-0.5 * ((rr - radius) / sigma) ** 2)
            weight = np.sum(radial, axis=1)
            em = peak_rate * weight * ds
            hit_distance = np.sum(radial * local_s[None, :], axis=1) / np.maximum(weight, 1.0e-30)

            hit_xyz = directions[start:stop] * hit_distance[:, None]
            hit_delta = hit_xyz - center[None, :]
            hit_rr = np.sqrt(
                (hit_delta[:, 0] / sx) ** 2
                + (hit_delta[:, 1] / sy) ** 2
                + (hit_delta[:, 2] / sz) ** 2
            )
            nx = hit_delta[:, 0] / np.maximum(hit_rr * sx, 1.0e-12)
            ny = hit_delta[:, 1] / np.maximum(hit_rr * sy, 1.0e-12)
            nz = hit_delta[:, 2] / np.maximum(hit_rr * sz, 1.0e-12)
            surface_mode = (
                0.65 * np.sin(2.3 * nx + 1.7 * ny + phase)
                + 0.45 * np.sin(3.1 * ny - 1.4 * nz + 0.7 * phase)
                + 0.30 * np.sin(2.6 * nz + 1.2 * nx - 0.4 * phase)
            )
            if screen_index == 0:
                # The fitted warm-gas filling factor is 0.3 (+0.3/-0.1), not
                # evidence for a hand-selected 70--98% covering sector. Keep
                # only gentle cloud-to-cloud modulation and taper continuously
                # with the shell column. This preserves the observed bright
                # limb without drawing a sharp geometric bite at 1 MHz.
                cloud_modulation = np.clip(
                    1.0 + 0.16 * np.tanh(surface_mode), 0.8, 1.2
                )
                column_presence = -np.expm1(-em / 25.0)
                arc_fraction = 0.30 * cloud_modulation * column_presence
            else:
                arc_fraction = fmin + (fmax - fmin) / (
                    1.0
                    + np.exp(
                        -np.clip(
                            (surface_mode - cover_threshold) / cover_softness,
                            -30.0,
                            30.0,
                        )
                    )
                )
                arc_fraction *= 1.0 / (
                    1.0 + np.exp(-np.clip((em - 2.0) / 0.7, -30.0, 30.0))
                )
            emission_measure[screen_index, start:stop] = em
            covering[screen_index, start:stop] = arc_fraction
            distance[screen_index, start:stop] = np.where(
                weight > 1.0e-8, hit_distance, center_distance
            )

    return emission_measure, covering, distance


def make_orion_partial_screens(
    config: SkyConfig,
    geometry: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility helper returning only the two Orion--Eridanus screens."""

    values = make_local_partial_screens(config, geometry)
    return tuple(value[1:] for value in values)


def beta_map(config: SkyConfig, geometry: dict[str, np.ndarray], beta_random: np.ndarray) -> np.ndarray:
    b = np.radians(geometry["b_deg"])
    # ULSA Fig. 15: plane is shallower, high latitude steeper, south slightly shallower.
    beta = -2.43 - 0.20 * np.abs(np.sin(b)) - 0.035 * np.sin(b)
    beta += config.beta_fluctuation_sigma * beta_random
    return np.clip(beta, -2.85, -2.25)


def shell_multiplier(shell_fields: np.ndarray, shell_index: np.ndarray, sigma: float) -> np.ndarray:
    # [shell,pixel] -> [pixel,distance]
    selected = shell_fields[shell_index].T
    return np.exp(sigma * selected - 0.5 * sigma**2)


def _tau_coefficient(freq_mhz: jnp.ndarray, electron_temperature_k: float) -> jnp.ndarray:
    """Compatibility wrapper around the differentiable core implementation."""

    return optical_depth_coefficient(freq_mhz, electron_temperature_k)


def _frequency_beta(beta: jnp.ndarray, freq_mhz: float) -> jnp.ndarray:
    # ULSA treats its frequency-dependent (Eq. 28) and direction-dependent
    # (Eq. 30) indices as alternative models.  The mounted comparison cube is
    # the direction-dependent variant, so do not combine the two here.
    del freq_mhz
    return beta


def _smooth_broken_power_law_scale(
    freq_mhz: jnp.ndarray,
    beta_high: jnp.ndarray,
    beta_low: jnp.ndarray,
    break_mhz: float,
    smoothness: float,
) -> jnp.ndarray:
    """Compatibility wrapper around the differentiable core implementation."""

    return smooth_broken_power_law(freq_mhz, beta_high, beta_low, break_mhz, smoothness)


@jax.jit
def transfer_one_frequency(
    freq_mhz: float,
    emissivity_408: jnp.ndarray,
    em_rate: jnp.ndarray,
    synch_multiplier: jnp.ndarray,
    beta: jnp.ndarray,
    shell_segment_408: jnp.ndarray,
    shell_foreground_em: jnp.ndarray,
    shell_beta: jnp.ndarray,
    shell_beta_low: jnp.ndarray,
    shell_spectral_break_mhz: float,
    shell_spectral_smoothness: float,
    ds_kpc: float,
    emissivity_scale: float,
    em_scale: float,
    beta_offset: float,
    electron_temperature_k: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compatibility wrapper used by the full preparation pipeline."""

    inputs = SkyInputs(
        emissivity_408=emissivity_408,
        emission_measure_rate=em_rate,
        synchrotron_multiplier=synch_multiplier,
        spectral_index=beta,
        shell_emission_408=shell_segment_408,
        shell_foreground_emission_measure=shell_foreground_em,
        shell_spectral_index=shell_beta,
        shell_low_frequency_spectral_index=shell_beta_low,
        shell_distance_kpc=jnp.zeros_like(shell_segment_408),
        partial_screen_emission_measure=jnp.zeros(
            (0, emissivity_408.shape[0]), dtype=emissivity_408.dtype
        ),
        partial_screen_covering_fraction=jnp.zeros(
            (0, emissivity_408.shape[0]), dtype=emissivity_408.dtype
        ),
        partial_screen_distance_kpc=jnp.zeros(
            (0, emissivity_408.shape[0]), dtype=emissivity_408.dtype
        ),
        distance_midpoint_kpc=(
            jnp.arange(emissivity_408.shape[1], dtype=emissivity_408.dtype) + 0.5
        )
        * ds_kpc,
        distance_step_kpc=ds_kpc,
    )
    parameters = SkyParameters(
        emissivity_scale=emissivity_scale,
        emission_measure_scale=em_scale,
        spectral_index_offset=beta_offset,
        electron_temperature_k=electron_temperature_k,
        shell_spectral_break_mhz=shell_spectral_break_mhz,
        shell_spectral_smoothness=shell_spectral_smoothness,
    )
    return transfer_frequency(freq_mhz, inputs, parameters)


def source_templates(config: SkyConfig) -> dict[str, np.ndarray]:
    npix = hp.nside2npix(config.nside)
    omega_pix = hp.nside2pixarea(config.nside)
    templates: dict[str, np.ndarray] = {}
    sigma = np.radians(config.output_beam_fwhm_deg) / np.sqrt(8.0 * np.log(2.0))
    all_vec = np.asarray(hp.pix2vec(config.nside, np.arange(npix))).T
    for name, (l_deg, b_deg, *_rest) in ATEAM.items():
        source_vec = np.asarray(hp.ang2vec(l_deg, b_deg, lonlat=True))
        angle = np.arccos(np.clip(all_vec @ source_vec, -1.0, 1.0))
        profile = np.exp(-0.5 * (angle / sigma) ** 2)
        profile /= np.sum(profile) * omega_pix
        templates[name] = profile
    return templates


def add_ateam(
    freq_mhz: np.ndarray,
    total: np.ndarray,
    em_rate: np.ndarray,
    geometry: dict[str, np.ndarray],
    em_scale: float,
    config: SkyConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    templates = source_templates(config)
    omega_component = np.zeros_like(total)
    individual: dict[str, np.ndarray] = {}
    amplitudes = ateam_integrated_k_sr(freq_mhz, em_rate, geometry, em_scale, config)
    for name in ATEAM:
        component = amplitudes[name][:, None] * templates[name][None, :]
        individual[name] = component
        omega_component += component
    return total + omega_component, omega_component, individual


def ateam_integrated_k_sr(
    freq_mhz: np.ndarray,
    em_rate: np.ndarray,
    geometry: dict[str, np.ndarray],
    em_scale: float,
    config: SkyConfig,
) -> dict[str, np.ndarray]:
    """Return attenuated source amplitudes integrated over solid angle."""
    amplitudes: dict[str, np.ndarray] = {}
    cumulative_em = np.cumsum(em_rate * em_scale * geometry["ds"], axis=1)
    for name, (l_deg, b_deg, s50, alpha, distance_kpc, tau10) in ATEAM.items():
        pix = hp.ang2pix(geometry["nside"], l_deg, b_deg, lonlat=True)
        if np.isfinite(distance_kpc):
            di = min(np.searchsorted(geometry["s"], distance_kpc), config.n_distance - 1)
        else:
            di = config.n_distance - 1
        em_source = cumulative_em[pix, di]
        tau_mw = (
            3.28e-7
            * (config.electron_temperature_k / 1.0e4) ** -1.35
            * (freq_mhz * MHZ_TO_GHZ) ** -2.1
            * em_source
        )
        flux_jy = s50 * (freq_mhz / 50.0) ** alpha
        internal_tau = tau10 * ((freq_mhz / 10.0) ** -2.1 - (50.0 / 10.0) ** -2.1)
        flux_jy *= np.exp(-np.maximum(internal_tau, 0.0) - tau_mw)
        # Integrated Rayleigh-Jeans temperature in K sr.
        integrated_k_sr = flux_jy * 1.0e-26 * C_LIGHT**2 / (2.0 * K_BOLTZMANN * (freq_mhz * 1.0e6) ** 2)
        amplitudes[name] = integrated_k_sr
    return amplitudes


def exact_ateam_alms(
    freq_mhz: np.ndarray,
    amplitudes_k_sr: dict[str, np.ndarray],
    lmax: int,
) -> dict[str, np.ndarray]:
    """Construct LuSEEpy analytic delta-function alms for every A-team source."""
    frequency_indices = np.arange(len(freq_mhz))
    result: dict[str, np.ndarray] = {}
    for name, (l_deg, b_deg, *_rest) in ATEAM.items():
        source = HarmonicPointSourceSky(
            lmax=lmax,
            freq=freq_mhz,
            T=amplitudes_k_sr[name],
            l_deg=l_deg,
            b_deg=b_deg,
        )
        result[name] = np.asarray(source.get_alm(frequency_indices), dtype=np.complex128)
    return result


def map_cube_to_alm(cube: np.ndarray, lmax: int) -> np.ndarray:
    """Transform an extended high-resolution component to packed healpy alms."""
    return np.stack([hp.map2alm(m, lmax=lmax, iter=3) for m in cube])


def beam_convolved_maps(
    alm_cube: np.ndarray,
    input_nside: int,
    output_nside: int,
    lmax: int,
    beam_fwhm_deg: float,
) -> np.ndarray:
    """Create a visualization map from canonical alms after applying a beam."""
    beam = hp.gauss_beam(np.radians(beam_fwhm_deg), lmax=lmax)
    maps = np.stack(
        [
            hp.alm2map(hp.almxfl(alm, beam), input_nside, lmax=lmax)
            for alm in alm_cube
        ]
    )
    if input_nside != output_nside:
        maps = np.stack(
            [
                hp.ud_grade(m, nside_out=output_nside, order_in="RING", order_out="RING", power=0)
                for m in maps
            ]
        )
    return np.maximum(maps, 0.0)


def tune_to_ulsa(
    mounted_maps: np.ndarray,
    anchor_freqs: np.ndarray,
    emissivity_408: jnp.ndarray,
    em_rate: jnp.ndarray,
    synch_multiplier: jnp.ndarray,
    beta: jnp.ndarray,
    shell_segment_408: jnp.ndarray,
    shell_foreground_em: jnp.ndarray,
    shell_beta: jnp.ndarray,
    shell_beta_low: jnp.ndarray,
    config: SkyConfig,
    ds: float,
) -> TunedParameters:
    target = jnp.asarray(np.mean(mounted_maps[anchor_freqs.astype(int) - 1], axis=1))
    freqs = jnp.asarray(anchor_freqs)

    def objective_jax(raw: jnp.ndarray) -> jnp.ndarray:
        emissivity_scale = jnp.exp(raw[0])
        em_scale = jnp.exp(raw[1])
        beta_offset = raw[2]

        def one(f: float) -> jnp.ndarray:
            result = transfer_one_frequency(
                f,
                emissivity_408,
                em_rate,
                synch_multiplier,
                beta,
                shell_segment_408,
                shell_foreground_em,
                shell_beta,
                shell_beta_low,
                config.shell_spectral_break_mhz,
                config.shell_spectral_smoothness,
                ds,
                emissivity_scale,
                em_scale,
                beta_offset,
                config.electron_temperature_k,
            )
            # ULSA masks Loop I/NPS when fitting its smooth emissivity. Match
            # that baseline rather than dimming it to compensate for the new
            # independently normalized local-shell component.
            baseline_without_local_shells = result[0] - result[3]
            return jnp.mean(baseline_without_local_shells)

        prediction = jax.vmap(one)(freqs)
        residual = jnp.log(prediction) - jnp.log(target)
        regularization = 0.002 * (raw[0] ** 2 + raw[1] ** 2) + 0.02 * raw[2] ** 2
        return jnp.mean(residual**2) + regularization

    vg = jax.jit(jax.value_and_grad(objective_jax))

    def scipy_fun(raw: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = vg(jnp.asarray(raw))
        return float(value), np.asarray(grad, dtype=float)

    fit = minimize(scipy_fun, np.zeros(3), jac=True, method="L-BFGS-B", bounds=[(-2.0, 2.0), (-4.0, 4.0), (-0.5, 0.5)])
    return TunedParameters(
        emissivity_scale=float(np.exp(fit.x[0])),
        emission_measure_scale=float(np.exp(fit.x[1])),
        beta_offset=float(fit.x[2]),
        objective=float(fit.fun),
    )


def prepare_sky(config: SkyConfig) -> PreparedSky:
    """Build a deterministic realization for the differentiable core.

    This is intentionally a pipeline operation: random-field synthesis,
    HEALPix geometry, and local-shell catalog construction happen here, before
    :func:`lowsky.model.generate_sky` is traced or differentiated.
    """

    geometry = prepare_geometry(config)
    random_fields = make_random_fields(config)
    synchrotron_multiplier = shell_multiplier(
        random_fields["synch_shells"],
        geometry["shell_index"],
        config.synch_fluctuation_sigma,
    )
    emission_measure_multiplier = shell_multiplier(
        random_fields["em_shells"],
        geometry["shell_index"],
        config.em_fluctuation_sigma,
    )
    spectral_index = beta_map(config, geometry, random_fields["beta"])

    emissivity_408, emission_measure_rate, anchor_emission_measure_rate = physical_fields(
        jnp.asarray(geometry["radius"]),
        jnp.asarray(geometry["theta"]),
        jnp.asarray(geometry["x"]),
        jnp.asarray(geometry["y"]),
        jnp.asarray(geometry["z"]),
        jnp.asarray(geometry["s"]),
        jnp.asarray(emission_measure_multiplier),
        config.local_bubble_radius_kpc,
        config.local_bubble_em_factor,
        config.solar_radius_kpc,
        config.solar_height_kpc,
        1.0 if config.sky_mode == "ours" else 0.0,
    )
    emissivity_408.block_until_ready()
    local_catalog = make_local_shell_catalog(config)
    shell_emission_408, shell_distance_index = shell_ray_segments(
        local_catalog, geometry, config
    )
    cumulative_emission_measure = np.cumsum(
        np.asarray(emission_measure_rate) * geometry["ds"], axis=1
    )
    shell_foreground_emission_measure = cumulative_emission_measure[
        np.arange(cumulative_emission_measure.shape[0])[None, None, :],
        shell_distance_index,
    ]
    local_screen_emission_measure, local_screen_covering, local_screen_distance = (
        make_local_partial_screens(config, geometry)
    )
    inputs = SkyInputs(
        emissivity_408=emissivity_408,
        emission_measure_rate=emission_measure_rate,
        synchrotron_multiplier=jnp.asarray(synchrotron_multiplier),
        spectral_index=jnp.asarray(spectral_index),
        shell_emission_408=jnp.asarray(shell_emission_408),
        shell_foreground_emission_measure=jnp.asarray(
            shell_foreground_emission_measure
        ),
        shell_spectral_index=jnp.asarray(local_catalog.beta_temperature),
        shell_low_frequency_spectral_index=jnp.asarray(
            local_catalog.low_frequency_beta_temperature
        ),
        shell_distance_kpc=jnp.asarray(geometry["s"][shell_distance_index]),
        partial_screen_emission_measure=jnp.asarray(local_screen_emission_measure),
        partial_screen_covering_fraction=jnp.asarray(local_screen_covering),
        partial_screen_distance_kpc=jnp.asarray(local_screen_distance),
        distance_midpoint_kpc=jnp.asarray(geometry["s"]),
        distance_step_kpc=jnp.asarray(geometry["ds"]),
    )
    return PreparedSky(
        inputs=inputs,
        geometry=geometry,
        local_catalog=local_catalog,
        anchor_emission_measure_rate=anchor_emission_measure_rate,
    )


def prepare_sky_inputs(config: SkyConfig) -> SkyInputs:
    """Return realistic, untuned inputs for :func:`lowsky.generate_sky`."""

    return prepare_sky(config).inputs


def generate(
    config: SkyConfig, mounted_path: Path
) -> tuple[
    dict[str, np.ndarray],
    TunedParameters,
    dict[str, np.ndarray],
    LocalShellCatalog,
    dict[str, np.ndarray],
]:
    mounted_maps = fits.getdata(mounted_path).astype(np.float64)
    prepared = prepare_sky(config)
    geometry = prepared.geometry
    local_catalog = prepared.local_catalog
    anchor_em_rate = prepared.anchor_emission_measure_rate
    inputs = prepared.inputs
    emissivity_408 = inputs.emissivity_408
    em_rate = inputs.emission_measure_rate
    synch_multiplier = inputs.synchrotron_multiplier
    beta = inputs.spectral_index
    shell_segment_408_jax = inputs.shell_emission_408
    shell_foreground_em_jax = inputs.shell_foreground_emission_measure
    shell_beta = inputs.shell_spectral_index
    shell_beta_low = inputs.shell_low_frequency_spectral_index
    anchor_freqs = np.asarray([1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    tuned = tune_to_ulsa(
        mounted_maps,
        anchor_freqs,
        emissivity_408,
        em_rate,
        jnp.asarray(synch_multiplier),
        jnp.asarray(beta),
        shell_segment_408_jax,
        shell_foreground_em_jax,
        shell_beta,
        shell_beta_low,
        config,
        geometry["ds"],
    )

    freqs = np.arange(1.0, 51.0)

    parameters = SkyParameters(
        emissivity_scale=tuned.emissivity_scale,
        emission_measure_scale=tuned.emission_measure_scale,
        spectral_index_offset=tuned.beta_offset,
        electron_temperature_k=config.electron_temperature_k,
        shell_spectral_break_mhz=config.shell_spectral_break_mhz,
        shell_spectral_smoothness=config.shell_spectral_smoothness,
    )

    def one(f: float):
        return transfer_frequency(f, inputs, parameters)

    transferred = jax.vmap(one)(jnp.asarray(freqs))
    _no_sources, smooth_synch, stochastic_synch, local_shells, freefree, extragalactic = [
        np.asarray(x) for x in transferred
    ]
    ray_nside = geometry["nside"]
    lmax = 3 * ray_nside - 1 if config.harmonic_lmax is None else config.harmonic_lmax
    if lmax > 3 * ray_nside - 1:
        raise ValueError(f"harmonic_lmax={lmax} exceeds the NSIDE={ray_nside} sampling limit")

    # Canonical representation: transform every extended component on the
    # native ray grid.  No display beam or low-resolution pixelization enters
    # these coefficients.
    extended_highres = {
        "smooth_synch": smooth_synch,
        "stochastic_synch": stochastic_synch,
        "local_shells": local_shells,
        "freefree": freefree,
        "extragalactic": extragalactic,
    }
    harmonic_products = {
        name: map_cube_to_alm(cube, lmax) for name, cube in extended_highres.items()
    }
    harmonic_products["baseline"] = (
        harmonic_products["stochastic_synch"]
        + harmonic_products["freefree"]
        + harmonic_products["extragalactic"]
    )
    harmonic_products["no_sources"] = (
        harmonic_products["baseline"] + harmonic_products["local_shells"]
    )

    # Point sources never enter a HEALPix map. LuSEEpy evaluates their exact
    # band-limited delta coefficients directly at the catalog coordinates.
    source_amplitudes = ateam_integrated_k_sr(
        freqs, np.asarray(em_rate), geometry, tuned.emission_measure_scale, config
    )
    individual_source_alms = exact_ateam_alms(freqs, source_amplitudes, lmax)
    harmonic_products.update(
        {f"ateam_{name.lower()}": alm for name, alm in individual_source_alms.items()}
    )
    harmonic_products["ateam"] = np.sum(list(individual_source_alms.values()), axis=0)
    harmonic_products["total"] = harmonic_products["no_sources"] + harmonic_products["ateam"]

    # NSIDE=32 maps are explicitly diagnostic products. Reconstruct each
    # component only after applying the common display beam.
    diagnostic_components = {
        name: beam_convolved_maps(
            alm, ray_nside, config.nside, lmax, config.output_beam_fwhm_deg
        )
        for name, alm in harmonic_products.items()
        if name in extended_highres
    }
    baseline = (
        diagnostic_components["stochastic_synch"]
        + diagnostic_components["freefree"]
        + diagnostic_components["extragalactic"]
    )
    no_sources = baseline + diagnostic_components["local_shells"]
    individual_sources = {
        name: beam_convolved_maps(
            alm, ray_nside, config.nside, lmax, config.output_beam_fwhm_deg
        )
        for name, alm in individual_source_alms.items()
    }
    ateam = np.sum(list(individual_sources.values()), axis=0)
    total = no_sources + ateam
    tau_coefficient_1mhz = _tau_coefficient(
        jnp.asarray(1.0), config.electron_temperature_k
    )
    tau_1mhz_smooth = tau_coefficient_1mhz * jnp.sum(
        em_rate * tuned.emission_measure_scale * geometry["ds"], axis=1
    )
    partial_transmission_1mhz = 1.0 - inputs.partial_screen_covering_fraction * (
        -jnp.expm1(
            -tau_coefficient_1mhz
            * tuned.emission_measure_scale
            * inputs.partial_screen_emission_measure
        )
    )
    tau_1mhz = tau_1mhz_smooth - jnp.sum(
        jnp.log(jnp.maximum(partial_transmission_1mhz, 1.0e-12)), axis=0
    )
    def degrade_scalar(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=np.float64)
        if geometry["nside"] == config.nside:
            return m
        return hp.ud_grade(m, nside_out=config.nside, order_in="RING", order_out="RING", power=0)

    products = {
        "frequency_mhz": freqs,
        "total": total,
        "no_sources": no_sources,
        "baseline": baseline,
        "smooth_synch": diagnostic_components["smooth_synch"],
        "stochastic_synch": diagnostic_components["stochastic_synch"],
        "local_shells": diagnostic_components["local_shells"],
        "freefree": diagnostic_components["freefree"],
        "extragalactic": diagnostic_components["extragalactic"],
        "ateam": ateam,
        "beta": degrade_scalar(beta),
        "tau_1mhz": degrade_scalar(np.asarray(tau_1mhz)),
        "em_total": degrade_scalar(
            np.sum(np.asarray(em_rate) * tuned.emission_measure_scale * geometry["ds"], axis=1)
        ),
        "anchor_em_total": degrade_scalar(
            np.sum(np.asarray(anchor_em_rate) * tuned.emission_measure_scale * geometry["ds"], axis=1)
            + np.sum(
                np.asarray(inputs.partial_screen_emission_measure)
                * np.asarray(inputs.partial_screen_covering_fraction)
                * tuned.emission_measure_scale,
                axis=0,
            )
        ),
        "gum_covering": degrade_scalar(
            np.asarray(inputs.partial_screen_covering_fraction)[0]
        ),
        "gum_em_clump": degrade_scalar(
            np.asarray(inputs.partial_screen_emission_measure)[0]
        ),
        "orion_covering": degrade_scalar(
            1.0
            - np.prod(
                1.0 - np.asarray(inputs.partial_screen_covering_fraction)[1:], axis=0
            )
        ),
        "orion_em_clump": degrade_scalar(
            np.max(np.asarray(inputs.partial_screen_emission_measure)[1:], axis=0)
        ),
    }
    harmonic_products["frequency_mhz"] = freqs
    harmonic_products["lmax"] = np.asarray(lmax, dtype=np.int32)
    harmonic_products["input_nside"] = np.asarray(ray_nside, dtype=np.int32)
    return products, tuned, individual_sources, local_catalog, harmonic_products


def write_fits(path: Path, products: dict[str, np.ndarray], config: SkyConfig, tuned: TunedParameters) -> None:
    primary = fits.PrimaryHDU(products["total"].astype(np.float32))
    primary.header["MODEL"] = "CF-ULSA"
    primary.header["SKYMODE"] = config.sky_mode
    primary.header["NSIDE"] = config.nside
    primary.header["RAYNSIDE"] = config.nside * config.ray_oversample
    primary.header["ORDERING"] = "RING"
    primary.header["COORDSYS"] = "GALACTIC"
    primary.header["BUNIT"] = "K"
    primary.header["PRODUCT"] = "BEAMED-DIAGNOSTIC"
    primary.header["BEAMFWHM"] = config.output_beam_fwhm_deg
    primary.header["FREQ0"] = 1.0
    primary.header["FREQ1"] = 50.0
    primary.header["DFREQ"] = 1.0
    primary.header["SEED"] = config.seed
    primary.header["EMISSCAL"] = tuned.emissivity_scale
    primary.header["EMSCAL"] = tuned.emission_measure_scale
    primary.header["BETADLT"] = tuned.beta_offset
    hdus = [primary]
    for name in [
        "no_sources",
        "baseline",
        "smooth_synch",
        "stochastic_synch",
        "local_shells",
        "freefree",
        "extragalactic",
        "ateam",
    ]:
        hdu = fits.ImageHDU(products[name].astype(np.float32), name=name.upper())
        hdu.header["BUNIT"] = "K"
        hdus.append(hdu)
    for name, unit in [
        ("beta", ""),
        ("tau_1mhz", ""),
        ("em_total", "pc cm-6"),
        ("anchor_em_total", "pc cm-6"),
        ("gum_covering", ""),
        ("gum_em_clump", "pc cm-6"),
        ("orion_covering", ""),
        ("orion_em_clump", "pc cm-6"),
    ]:
        hdu = fits.ImageHDU(products[name].astype(np.float32), name=name.upper())
        if unit:
            hdu.header["BUNIT"] = unit
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True, checksum=True)


def write_harmonic_product(path: Path, harmonic_products: dict[str, np.ndarray]) -> None:
    """Write the canonical packed-healpy alm product without a sky beam."""
    payload = dict(harmonic_products)
    payload["alm_ordering"] = np.asarray("healpy-packed-m>=0")
    payload["coordinate_frame"] = np.asarray("galactic")
    payload["alm_unit"] = np.asarray("K sr for delta amplitudes; K-map harmonics otherwise")
    payload["point_source_model"] = np.asarray("analytic-delta-via-luseepy")
    np.savez_compressed(path, **payload)


def angular_spectra(maps: np.ndarray, nside: int) -> np.ndarray:
    lmax = 3 * nside - 1
    return np.stack([hp.anafast(m, lmax=lmax, iter=3) for m in maps])


def compare_and_plot(
    output_dir: Path,
    products: dict[str, np.ndarray],
    mounted_maps: np.ndarray,
    config: SkyConfig,
    tuned: TunedParameters,
    individual_sources: dict[str, np.ndarray],
) -> dict[str, object]:
    cf = products["total"]
    cf_diffuse = products["no_sources"]
    cf_baseline = products["baseline"]
    cf_ateam = products["ateam"]
    cf_shells = products["local_shells"]
    cl_cf = angular_spectra(cf, config.nside)
    cl_cf_diffuse = angular_spectra(cf_diffuse, config.nside)
    cl_cf_baseline = angular_spectra(cf_baseline, config.nside)
    cl_cf_ateam = angular_spectra(cf_ateam, config.nside)
    cl_cf_shells = angular_spectra(cf_shells, config.nside)
    cl_ulsa = angular_spectra(mounted_maps, config.nside)
    ell = np.arange(cl_cf.shape[1])
    pixwin = hp.pixwin(config.nside, lmax=len(ell) - 1)
    cl_cf_pw = cl_cf / pixwin[None, :] ** 2
    cl_cf_diffuse_pw = cl_cf_diffuse / pixwin[None, :] ** 2
    cl_cf_baseline_pw = cl_cf_baseline / pixwin[None, :] ** 2
    cl_cf_ateam_pw = cl_cf_ateam / pixwin[None, :] ** 2
    cl_cf_shells_pw = cl_cf_shells / pixwin[None, :] ** 2
    cl_ulsa_pw = cl_ulsa / pixwin[None, :] ** 2
    cl_individual_pw = {
        name: angular_spectra(cube, config.nside) / pixwin[None, :] ** 2
        for name, cube in individual_sources.items()
    }

    metrics: dict[str, object] = {
        "config": asdict(config),
        "tuned": asdict(tuned),
        "conditioned_anchor_em": {
            "mean_pc_cm6": float(np.mean(products["anchor_em_total"])),
            "p95_pc_cm6": float(np.percentile(products["anchor_em_total"], 95.0)),
            "max_pc_cm6": float(np.max(products["anchor_em_total"])),
        },
        "frequency": {},
    }
    for f in [1, 3, 10, 25, 40, 50]:
        i = f - 1
        slopes = {}
        for lo, hi in [(10, 50), (20, 63)]:
            q = (ell >= lo) & (ell <= hi)
            slopes[f"{lo}_{hi}"] = {
                "counterfactual_total": float(np.polyfit(np.log(ell[q]), np.log(cl_cf_pw[i, q]), 1)[0]),
                "counterfactual_diffuse": float(np.polyfit(np.log(ell[q]), np.log(cl_cf_diffuse_pw[i, q]), 1)[0]),
                "ulsa": float(np.polyfit(np.log(ell[q]), np.log(cl_ulsa_pw[i, q]), 1)[0]),
            }
        q_band = (ell >= 20) & (ell <= 63)
        band_variance = lambda spectrum: float(np.sum((2 * ell[q_band] + 1) * spectrum[q_band]))
        def shell_fraction(lo: int, hi: int, denominator: np.ndarray) -> float:
            q = (ell >= lo) & (ell <= hi)
            weight = 2 * ell[q] + 1
            return float(np.sum(weight * cl_cf_shells_pw[i, q]) / np.sum(weight * denominator[i, q]))

        metrics["frequency"][str(f)] = {
            "mean_counterfactual_k": float(np.mean(cf[i])),
            "mean_counterfactual_baseline_k": float(np.mean(cf_baseline[i])),
            "mean_ulsa_k": float(np.mean(mounted_maps[i])),
            "rms_counterfactual_k": float(np.std(cf[i])),
            "rms_ulsa_k": float(np.std(mounted_maps[i])),
            "pixel_correlation": float(np.corrcoef(cf[i], mounted_maps[i])[0, 1]),
            "ateam_bandpower_fraction_ell20_63": band_variance(cl_cf_ateam_pw[i]) / band_variance(cl_cf_pw[i]),
            # This is an isolated-component ratio, not an additive fraction:
            # the total diffuse power also contains shell/baseline cross terms.
            "local_shell_to_diffuse_bandpower_ratio_ell2_10": shell_fraction(2, 10, cl_cf_diffuse_pw),
            "local_shell_to_diffuse_bandpower_ratio_ell10_50": shell_fraction(10, 50, cl_cf_diffuse_pw),
            # These isolated-source fractions need not sum exactly to the
            # combined A-team fraction because angular power has cross terms.
            "individual_source_bandpower_fractions_ell20_63": {
                name: band_variance(spectrum[i]) / band_variance(cl_cf_pw[i])
                for name, spectrum in cl_individual_pw.items()
            },
            "slopes": slopes,
        }

    np.savez_compressed(
        output_dir / "comparison_spectra.npz",
        frequency_mhz=products["frequency_mhz"],
        ell=ell,
        cl_counterfactual_k2=cl_cf,
        cl_counterfactual_diffuse_k2=cl_cf_diffuse,
        cl_counterfactual_baseline_k2=cl_cf_baseline,
        cl_counterfactual_ateam_k2=cl_cf_ateam,
        cl_counterfactual_local_shells_k2=cl_cf_shells,
        cl_ulsa_k2=cl_ulsa,
        cl_counterfactual_pixel_corrected_k2=cl_cf_pw,
        cl_ulsa_pixel_corrected_k2=cl_ulsa_pw,
    )

    fsel = [1, 3, 10, 25, 50]
    fig = plt.figure(figsize=(17, 7.2))
    for col, f in enumerate(fsel):
        i = f - 1
        lo, hi = np.percentile(np.log10(np.maximum(cf[i], 1.0)), [1, 99])
        hp.mollview(
            np.log10(np.maximum(cf[i], 1.0)),
            min=lo,
            max=hi,
            title=f"Fresh {f} MHz",
            sub=(2, len(fsel), col + 1),
            fig=fig.number,
            cmap="inferno",
            cbar=False,
        )
        ulo, uhi = np.percentile(np.log10(np.maximum(mounted_maps[i], 1.0)), [1, 99])
        hp.mollview(
            np.log10(np.maximum(mounted_maps[i], 1.0)),
            min=ulo,
            max=uhi,
            title=f"ULSA {f} MHz",
            sub=(2, len(fsel), len(fsel) + col + 1),
            fig=fig.number,
            cmap="inferno",
            cbar=False,
        )
    fig.suptitle("Fresh realization versus mounted ULSA (log10 K; independently scaled)", fontsize=16)
    fig.savefig(output_dir / "counterfactual_vs_ulsa_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    ax = axes[0, 0]
    for f in fsel:
        i = f - 1
        ax.loglog(ell[2:], cl_cf_diffuse_pw[i, 2:] / cl_cf_diffuse_pw[i, 20], label=f"{f} MHz")
    ax.loglog(ell[2:], (ell[2:] / 20.0) ** -2.7, "k--", label=r"$\ell^{-2.7}$")
    ax.axvspan(2 * config.nside, ell[-1], color="0.9")
    ax.set(title="Fresh diffuse sky (A-team removed)", xlabel=r"$\ell$", ylabel=r"$C_\ell/C_{20}$")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for f in fsel:
        i = f - 1
        ax.loglog(ell[2:], cl_ulsa_pw[i, 2:] / cl_ulsa_pw[i, 20], label=f"{f} MHz")
    ax.loglog(ell[2:], (ell[2:] / 20.0) ** -2.7, "k--")
    ax.axvspan(2 * config.nside, ell[-1], color="0.9")
    ax.set(title="Mounted ULSA", xlabel=r"$\ell$", ylabel=r"$C_\ell/C_{20}$")
    ax = axes[1, 0]
    for f in fsel:
        i = f - 1
        ratio = cl_cf_ateam_pw[i] / np.maximum(cl_cf_pw[i], 1e-300)
        ax.semilogx(ell[2:], ratio[2:], label=f"{f} MHz")
    ax.axvspan(2 * config.nside, ell[-1], color="0.9")
    ax.set(title="Explicit A-team contribution", xlabel=r"$\ell$", ylabel=r"$C_\ell^{A}/C_\ell^{total}$", ylim=(0, 1.1))
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    mean_cf = np.mean(cf, axis=1)
    mean_baseline = np.mean(cf_baseline, axis=1)
    mean_ulsa = np.mean(mounted_maps, axis=1)
    ax.loglog(products["frequency_mhz"], mean_cf, label="fresh + local shells")
    ax.loglog(products["frequency_mhz"], mean_baseline, "--", label="fresh ULSA-like baseline")
    ax.loglog(products["frequency_mhz"], mean_ulsa, label="ULSA")
    ax.set(title="Global spectrum", xlabel="Frequency [MHz]", ylabel="Mean temperature [K]")
    ax.legend()
    for ax in np.ravel(axes):
        ax.grid(alpha=0.25)
    fig.suptitle("Counterfactual ULSA spectra and source separation", fontsize=15)
    fig.savefig(output_dir / "counterfactual_vs_ulsa_spectra.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(15, 4.5))
    hp.mollview(np.log10(np.maximum(products["em_total"], 1e-4)), title="Fresh log10 EM", sub=(1, 3, 1), fig=fig.number, cmap="magma")
    hp.mollview(np.log10(np.maximum(products["tau_1mhz"], 1e-5)), title="Fresh log10 tau(1 MHz)", sub=(1, 3, 2), fig=fig.number, cmap="viridis")
    hp.mollview(products["beta"], title="Fresh spectral index", sub=(1, 3, 3), fig=fig.number, cmap="coolwarm")
    fig.savefig(output_dir / "counterfactual_physical_fields.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(14, 7.5))
    for panel, f in enumerate([1, 10, 50], start=1):
        i = f - 1
        hp.mollview(
            np.log10(np.maximum(cf_shells[i], 1.0)),
            title=f"Local shells {f} MHz (log10 K)",
            sub=(2, 2, panel),
            fig=fig.number,
            cmap="magma",
        )
    ax = fig.add_subplot(2, 2, 4)
    for f in [1, 3, 10, 25, 50]:
        i = f - 1
        ax.semilogx(
            ell[2:],
            cl_cf_shells_pw[i, 2:] / np.maximum(cl_cf_diffuse_pw[i, 2:], 1e-300),
            label=f"{f} MHz",
        )
    ax.axvspan(10, 50, color="0.9", zorder=-1)
    ax.set(
        xlabel=r"$\ell$",
        ylabel=r"$C_\ell^{shell}/C_\ell^{diffuse}$",
        title="Isolated-shell / source-free angular-power ratio",
        ylim=(0, 1.1),
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("Explicit 3D local shell/spur component", fontsize=15)
    fig.savefig(output_dir / "counterfactual_local_shells.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(14, 7.5))
    hp.mollview(
        np.log10(np.maximum(products["anchor_em_total"], 1e-3)),
        title="Conditioned Gum + Orion + Cygnus log10 EM",
        sub=(2, 2, 1),
        fig=fig.number,
        cmap="viridis",
    )
    hp.mollview(
        np.log10(np.maximum(products["tau_1mhz"], 1e-5)),
        title="Total log10 tau at 1 MHz",
        sub=(2, 2, 2),
        fig=fig.number,
        cmap="viridis",
    )
    hp.mollview(
        np.log10(np.maximum(products["local_shells"][9], 1.0)),
        title="Named-loop analogues at 10 MHz",
        sub=(2, 2, 3),
        fig=fig.number,
        cmap="magma",
    )
    hp.mollview(
        np.log10(np.maximum(products["baseline"][9], 1.0)),
        title="Conditioned Galactic baseline at 10 MHz",
        sub=(2, 2, 4),
        fig=fig.number,
        cmap="inferno",
    )
    fig.suptitle("Our-sky-conditioned physical anchors", fontsize=15)
    fig.savefig(output_dir / "counterfactual_our_sky_anchors.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    with (output_dir / "comparison_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mounted-ulsa",
        type=Path,
        default=Path(os.environ["LOWSKY_ULSA_PATH"]) if "LOWSKY_ULSA_PATH" in os.environ else None,
        help="Path to ULSA_32_ddi_smooth.fits (or set LOWSKY_ULSA_PATH).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "lowsky-output")
    parser.add_argument("--seed", type=int, default=SkyConfig.seed)
    parser.add_argument("--n-distance", type=int, default=SkyConfig.n_distance)
    parser.add_argument(
        "--harmonic-lmax",
        type=int,
        default=None,
        help="Canonical harmonic band limit; default is 3*ray-NSIDE-1.",
    )
    parser.add_argument(
        "--sky-mode",
        choices=["random", "ours"],
        default=SkyConfig.sky_mode,
        help="Use fully random local features or literature-conditioned Milky Way anchors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mounted_ulsa is None:
        raise SystemExit("Provide --mounted-ulsa or set LOWSKY_ULSA_PATH.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = SkyConfig(
        seed=args.seed,
        n_distance=args.n_distance,
        sky_mode=args.sky_mode,
        harmonic_lmax=args.harmonic_lmax,
    )
    products, tuned, individual, local_catalog, harmonic_products = generate(
        config, args.mounted_ulsa
    )
    output_fits = args.output_dir / "counterfactual_ulsa_nside32_1_50mhz.fits"
    harmonic_output = args.output_dir / "counterfactual_ulsa_harmonic.npz"
    write_fits(output_fits, products, config, tuned)
    write_harmonic_product(harmonic_output, harmonic_products)
    np.savez_compressed(args.output_dir / "ateam_components.npz", frequency_mhz=products["frequency_mhz"], **individual)
    with (args.output_dir / "local_shell_catalog.json").open("w") as handle:
        json.dump({key: np.asarray(value).tolist() for key, value in asdict(local_catalog).items()}, handle, indent=2)
    mounted = fits.getdata(args.mounted_ulsa).astype(np.float64)
    metrics = compare_and_plot(args.output_dir, products, mounted, config, tuned, individual)
    with (args.output_dir / "run_manifest.json").open("w") as handle:
        json.dump(
            {
                "mounted_ulsa": str(args.mounted_ulsa),
                "output_fits": str(output_fits),
                "harmonic_product": str(harmonic_output),
                "config": asdict(config),
                "tuned": asdict(tuned),
                "metrics_file": str(args.output_dir / "comparison_metrics.json"),
            },
            handle,
            indent=2,
        )
    print(
        json.dumps(
            {
                "output_fits": str(output_fits),
                "harmonic_product": str(harmonic_output),
                "tuned": asdict(tuned),
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
