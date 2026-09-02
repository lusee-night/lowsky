"""Pure, differentiable low-frequency sky model.

This module deliberately contains no file access, catalog generation, HEALPix
operations, NumPy conversions, or optimization.  Build :class:`SkyInputs` in a
pipeline (or directly from JAX arrays), then transform :func:`generate_sky`
with ``jax.jit``, ``jax.grad``, ``jax.jacfwd``, or ``jax.vmap``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


MHZ_TO_GHZ = 1.0e-3


class SkyInputs(NamedTuple):
    """Prepared spatial fields consumed by the differentiable model.

    The first three fields have shape ``(pixel, distance)``.  ``spectral_index``
    has shape ``(pixel,)``.  The shell arrays have shape
    ``(shell, distance_bin, pixel)``, while both shell-index arrays have shape
    ``(shell,)``.
    """

    emissivity_408: jax.Array
    emission_measure_rate: jax.Array
    synchrotron_multiplier: jax.Array
    spectral_index: jax.Array
    shell_emission_408: jax.Array
    shell_foreground_emission_measure: jax.Array
    shell_spectral_index: jax.Array
    shell_low_frequency_spectral_index: jax.Array
    distance_step_kpc: jax.Array


class SkyParameters(NamedTuple):
    """Differentiable physical parameters for a sky realization."""

    emissivity_scale: jax.Array | float = 1.0
    emission_measure_scale: jax.Array | float = 1.0
    spectral_index_offset: jax.Array | float = 0.0
    electron_temperature_k: jax.Array | float = 8_000.0
    shell_spectral_break_mhz: jax.Array | float = 30.0
    shell_spectral_smoothness: jax.Array | float = 2.0


class SkyComponents(NamedTuple):
    """Frequency-by-pixel component maps returned by the transfer model."""

    total: jax.Array
    smooth_synchrotron: jax.Array
    stochastic_synchrotron: jax.Array
    local_shells: jax.Array
    free_free: jax.Array
    extragalactic: jax.Array


def optical_depth_coefficient(
    frequency_mhz: jax.Array | float,
    electron_temperature_k: jax.Array | float,
) -> jax.Array:
    """Free-free optical depth per unit emission measure."""

    frequency_mhz = jnp.asarray(frequency_mhz)
    return (
        3.28e-7
        * (electron_temperature_k / 1.0e4) ** -1.35
        * (frequency_mhz * MHZ_TO_GHZ) ** -2.1
    )


def smooth_broken_power_law(
    frequency_mhz: jax.Array | float,
    high_frequency_index: jax.Array,
    low_frequency_index: jax.Array,
    break_mhz: jax.Array | float,
    smoothness: jax.Array | float,
) -> jax.Array:
    """Scale a 408-MHz component between two asymptotic spectral indices."""

    frequency_mhz = jnp.asarray(frequency_mhz)
    break_ratio = (
        (1.0 + (break_mhz / frequency_mhz) ** smoothness)
        / (1.0 + (break_mhz / 408.0) ** smoothness)
    )
    curvature_exponent = (high_frequency_index - low_frequency_index) / smoothness
    return (
        (frequency_mhz / 408.0) ** high_frequency_index
        * break_ratio**curvature_exponent
    )


def transfer_frequency(
    frequency_mhz: jax.Array | float,
    inputs: SkyInputs,
    parameters: SkyParameters,
) -> SkyComponents:
    """Evaluate radiative transfer at one frequency using only JAX operations."""

    differential_em = (
        inputs.emission_measure_rate
        * parameters.emission_measure_scale
        * inputs.distance_step_kpc
    )
    cumulative_em_edge = jnp.cumsum(differential_em, axis=1)
    cumulative_em_mid = cumulative_em_edge - 0.5 * differential_em
    tau_coefficient = optical_depth_coefficient(
        frequency_mhz, parameters.electron_temperature_k
    )
    tau_mid = tau_coefficient * cumulative_em_mid
    tau_total = tau_coefficient * cumulative_em_edge[:, -1]

    spectral_index = inputs.spectral_index + parameters.spectral_index_offset
    frequency_scale = (frequency_mhz / 408.0) ** spectral_index
    base_emission = (
        inputs.emissivity_408
        * parameters.emissivity_scale
        * frequency_scale[:, None]
    )
    attenuation = jnp.exp(-tau_mid)
    smooth_synchrotron = jnp.sum(
        base_emission * attenuation * inputs.distance_step_kpc, axis=1
    )
    stochastic_synchrotron = jnp.sum(
        base_emission
        * inputs.synchrotron_multiplier
        * attenuation
        * inputs.distance_step_kpc,
        axis=1,
    )
    free_free = parameters.electron_temperature_k * (-jnp.expm1(-tau_total))

    extragalactic_unabsorbed = 1.2 * (frequency_mhz / 1_000.0) ** -2.58
    circumgalactic_optical_depth = 0.95 * (frequency_mhz / 1.0) ** -2.1
    extragalactic = extragalactic_unabsorbed * jnp.exp(
        -tau_total - circumgalactic_optical_depth
    )

    shell_scale = smooth_broken_power_law(
        frequency_mhz,
        inputs.shell_spectral_index[:, None, None],
        inputs.shell_low_frequency_spectral_index[:, None, None],
        parameters.shell_spectral_break_mhz,
        parameters.shell_spectral_smoothness,
    )
    local_shells = jnp.sum(
        inputs.shell_emission_408
        * shell_scale
        * jnp.exp(
            -tau_coefficient
            * parameters.emission_measure_scale
            * inputs.shell_foreground_emission_measure
        ),
        axis=(0, 1),
    )
    total = stochastic_synchrotron + local_shells + free_free + extragalactic
    return SkyComponents(
        total,
        smooth_synchrotron,
        stochastic_synchrotron,
        local_shells,
        free_free,
        extragalactic,
    )


def generate_sky_components(
    frequencies_mhz: jax.Array,
    inputs: SkyInputs,
    parameters: SkyParameters = SkyParameters(),
) -> SkyComponents:
    """Generate component maps with shape ``(frequency, pixel)``.

    All arguments and outputs are JAX pytrees, so the complete operation can be
    differentiated with respect to any floating-point input or parameter.
    """

    frequencies_mhz = jnp.atleast_1d(jnp.asarray(frequencies_mhz))
    return jax.vmap(lambda frequency: transfer_frequency(frequency, inputs, parameters))(
        frequencies_mhz
    )


def generate_sky(
    frequencies_mhz: jax.Array,
    inputs: SkyInputs,
    parameters: SkyParameters = SkyParameters(),
    additive_temperature: jax.Array | float = 0.0,
) -> jax.Array:
    """Generate a differentiable sky temperature cube in kelvin.

    ``additive_temperature`` may be a scalar or a precomputed
    ``(frequency, pixel)`` JAX array.  It lets callers add point sources or
    other externally prepared components without introducing catalog or
    coordinate-system work into this core.
    """

    return (
        generate_sky_components(frequencies_mhz, inputs, parameters).total
        + jnp.asarray(additive_temperature)
    )


__all__ = [
    "SkyComponents",
    "SkyInputs",
    "SkyParameters",
    "generate_sky",
    "generate_sky_components",
    "optical_depth_coefficient",
    "smooth_broken_power_law",
    "transfer_frequency",
]
