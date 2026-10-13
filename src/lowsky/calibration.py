"""Diffuse-synchrotron component algebra at fixed thermal opacity.

Scaling the absorbed volume synchrotron maps corresponds to changing
``SkyParameters.emissivity_scale`` and its spectral-index offset while all
prepared fields stay fixed. An extragalactic ratio is a separate diagnostic
hypothesis, not a current ``SkyParameters`` field. Local shells and free-free
emission remain fixed.
"""
from __future__ import annotations

import numpy as np


def calibrated_no_sources_from_components(
    stochastic_synch: np.ndarray,
    local_shells: np.ndarray,
    freefree: np.ndarray,
    extragalactic: np.ndarray,
    frequency_mhz: float,
    emissivity_ratio: float,
    beta_offset_delta: float,
    extragalactic_ratio: float = 1.0,
) -> np.ndarray:
    """Recombine source-free components without changing their absorption.

    ``stochastic_synch`` is the full absorbed volume-synchrotron map.
    ``SMOOTH_SYNCH`` is an alternate diagnostic and must not be added again.
    """
    if not np.isfinite(frequency_mhz) or not 1 <= frequency_mhz <= 50:
        raise ValueError("frequency_mhz must be within 1--50")
    if not np.isfinite(emissivity_ratio) or emissivity_ratio <= 0:
        raise ValueError("emissivity_ratio must be positive and finite")
    if not np.isfinite(beta_offset_delta):
        raise ValueError("beta_offset_delta must be finite")
    if not np.isfinite(extragalactic_ratio) or extragalactic_ratio <= 0:
        raise ValueError("extragalactic_ratio must be positive and finite")

    components = [
        np.asarray(value, dtype=np.float64)
        for value in (stochastic_synch, local_shells, freefree, extragalactic)
    ]
    if len({value.shape for value in components}) != 1:
        raise ValueError("components must have matching shapes and finite values")
    if not all(np.all(np.isfinite(value)) for value in components):
        raise ValueError("components must have matching shapes and finite values")

    synch_scale = emissivity_ratio * (
        frequency_mhz / 408.0
    ) ** beta_offset_delta
    return (
        synch_scale * components[0]
        + components[1]
        + components[2]
        + extragalactic_ratio * components[3]
    )
