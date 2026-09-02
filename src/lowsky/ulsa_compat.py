"""Compatibility helpers for the legacy LuSEE ULSA sky-map artifact.

The mounted ``ULSA_32_ddi_smooth.fits`` file is a deliberately small FITS
contract: a single primary image containing 50 Galactic, RING-ordered
HEALPix maps.  Spatial metadata is implicit in the legacy consumer
(``lusee.SkyModels.FitsSky``), so this writer preserves that convention and
the original header instead of adding otherwise useful metadata cards.

FITS serialization is necessarily outside JAX transformations.  Generate a
sky with JAX first, then call :func:`write_ulsa_dropin` at the I/O boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

ULSA_NSIDE = 32
ULSA_NPIX = 12 * ULSA_NSIDE**2
ULSA_FREQUENCIES_MHZ = np.arange(1.0, 51.0, 1.0, dtype=np.float64)

# Preserve the original cfitsio card formatting as well as its values. Card
# images are padded to FITS' required 80 bytes by ``Card.fromstring``.
_ULSA_MODEL_CARDS = (
    "HIERARCH FREQ_START =        1",
    "FREQ_END=                   50",
    "HIERARCH FREQ_STEP =         1",
    "NSIDE   =                   32",
    "TYPE    = 'ulsa    '",
    "HIERARCH ULSA_INDX = 'direction_dependent_index'",
    "HIERARCH ULSA_USE_RAW_DIFFUSE = F",
    "HIERARCH ULSA_FIELD = 'smooth_absorb'",
)


def _map_cube(sky: Any) -> np.ndarray:
    """Return a NumPy map cube from an array or a generated product mapping."""
    if isinstance(sky, Mapping):
        if "total" not in sky:
            raise ValueError("a sky product mapping must contain a 'total' map cube")
        sky = sky["total"]
    return np.asarray(sky)


def validate_ulsa_dropin(
    sky: Any,
    frequencies_mhz: Any = ULSA_FREQUENCIES_MHZ,
) -> np.ndarray:
    """Validate and return a cube satisfying the legacy ULSA file contract.

    Parameters
    ----------
    sky
        Array-like map cube, or a generated product mapping whose ``"total"``
        entry is the cube. Its shape must be ``(50, 12288)``. Pixels are
        interpreted as Galactic HEALPix RING ordering, matching LuSEEpy's
        ``FitsSky`` consumer.
    frequencies_mhz
        Frequency assigned to each row. A drop-in artifact requires exactly
        1 through 50 MHz at 1 MHz spacing.

    Returns
    -------
    numpy.ndarray
        A native-endian float64 view/copy suitable for FITS serialization.
    """
    cube = _map_cube(sky)
    expected_shape = (ULSA_FREQUENCIES_MHZ.size, ULSA_NPIX)
    if cube.shape != expected_shape:
        raise ValueError(
            f"ULSA drop-in cube must have shape {expected_shape}; got {cube.shape}"
        )

    frequencies = np.asarray(frequencies_mhz, dtype=np.float64)
    if frequencies.shape != ULSA_FREQUENCIES_MHZ.shape or not np.array_equal(
        frequencies, ULSA_FREQUENCIES_MHZ
    ):
        raise ValueError("ULSA drop-in frequencies must be exactly 1..50 MHz in 1 MHz steps")

    cube = np.asarray(cube, dtype=np.float64)
    if not np.all(np.isfinite(cube)):
        raise ValueError("ULSA drop-in cube contains non-finite temperatures")
    return cube


def write_ulsa_dropin(
    path: str | Path,
    sky: Any,
    frequencies_mhz: Any = ULSA_FREQUENCIES_MHZ,
    *,
    overwrite: bool = False,
) -> Path:
    """Write ``sky`` in the exact layout consumed as a legacy ULSA FITS file.

    The result has one primary HDU, a float64 image of shape ``(50, 12288)``,
    and the same model/frequency cards as ``ULSA_32_ddi_smooth.fits``. FITS
    stores the image as big-endian IEEE float64 (``BITPIX=-64``). No checksum,
    extension, ``BUNIT``, ``ORDERING``, or ``COORDSYS`` cards are added because
    none are present in the mounted artifact.
    """
    output = Path(path)
    cube = validate_ulsa_dropin(sky, frequencies_mhz)

    primary = fits.PrimaryHDU(cube)
    header = primary.header
    # Preserve even the structural comments and two standard COMMENT cards
    # written by the tool that produced the mounted reference.  Downstream
    # readers only need the values, but matching the complete 16-card header
    # makes contract regressions obvious when files are inspected directly.
    header.comments["SIMPLE"] = "file does conform to FITS standard"
    header.comments["BITPIX"] = "number of bits per data pixel"
    header.comments["NAXIS"] = "number of data axes"
    header.comments["NAXIS1"] = "length of data axis 1"
    header.comments["NAXIS2"] = "length of data axis 2"
    header.comments["EXTEND"] = "FITS dataset may contain extensions"
    for card_image in _ULSA_MODEL_CARDS:
        header.append(fits.Card.fromstring(card_image.ljust(80)))
    header.insert(
        "FREQ_START",
        ("COMMENT", "  FITS (Flexible Image Transport System) format is defined in 'Astronomy"),
    )
    header.insert(
        "FREQ_START",
        ("COMMENT", "  and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"),
    )
    primary.writeto(output, overwrite=overwrite, checksum=False)
    return output


__all__ = [
    "ULSA_FREQUENCIES_MHZ",
    "ULSA_NPIX",
    "ULSA_NSIDE",
    "validate_ulsa_dropin",
    "write_ulsa_dropin",
]
