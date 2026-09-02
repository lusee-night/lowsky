from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from lowsky.ulsa_compat import (
    ULSA_FREQUENCIES_MHZ,
    ULSA_NPIX,
    ULSA_NSIDE,
    validate_ulsa_dropin,
    write_ulsa_dropin,
)


def test_write_ulsa_dropin_matches_mounted_file_contract(tmp_path):
    maps = np.arange(50 * ULSA_NPIX, dtype=np.float32).reshape(50, ULSA_NPIX)
    output = write_ulsa_dropin(tmp_path / "ULSA_32_ddi_smooth.fits", maps)

    with fits.open(output, memmap=True, do_not_scale_image_data=True) as hdus:
        assert len(hdus) == 1
        assert isinstance(hdus[0], fits.PrimaryHDU)
        assert hdus[0].data.shape == (50, ULSA_NPIX)
        assert hdus[0].data.dtype == np.dtype(">f8")
        assert hdus[0].header["BITPIX"] == -64
        assert hdus[0].header["NAXIS1"] == ULSA_NPIX
        assert hdus[0].header["NAXIS2"] == 50
        assert hdus[0].header["FREQ_START"] == 1
        assert hdus[0].header["FREQ_END"] == 50
        assert hdus[0].header["FREQ_STEP"] == 1
        assert hdus[0].header["NSIDE"] == ULSA_NSIDE
        assert hdus[0].header["TYPE"] == "ulsa"
        assert hdus[0].header["ULSA_INDX"] == "direction_dependent_index"
        assert hdus[0].header["ULSA_USE_RAW_DIFFUSE"] is False
        assert hdus[0].header["ULSA_FIELD"] == "smooth_absorb"
        assert "CHECKSUM" not in hdus[0].header
        assert "DATASUM" not in hdus[0].header
        assert "ORDERING" not in hdus[0].header
        assert "COORDSYS" not in hdus[0].header
        assert "BUNIT" not in hdus[0].header
        assert list(hdus[0].header) == [
            "SIMPLE",
            "BITPIX",
            "NAXIS",
            "NAXIS1",
            "NAXIS2",
            "EXTEND",
            "COMMENT",
            "COMMENT",
            "FREQ_START",
            "FREQ_END",
            "FREQ_STEP",
            "NSIDE",
            "TYPE",
            "ULSA_INDX",
            "ULSA_USE_RAW_DIFFUSE",
            "ULSA_FIELD",
        ]
        assert hdus[0].header.comments["SIMPLE"] == "file does conform to FITS standard"
        assert [card.image.rstrip() for card in hdus[0].header.cards[8:]] == [
            "HIERARCH FREQ_START =        1",
            "FREQ_END=                   50",
            "HIERARCH FREQ_STEP =         1",
            "NSIDE   =                   32",
            "TYPE    = 'ulsa    '",
            "HIERARCH ULSA_INDX = 'direction_dependent_index'",
            "HIERARCH ULSA_USE_RAW_DIFFUSE = F",
            "HIERARCH ULSA_FIELD = 'smooth_absorb'",
        ]
        np.testing.assert_array_equal(hdus[0].data, maps)


def test_product_mapping_selects_total_cube():
    total = np.ones((50, ULSA_NPIX), dtype=np.float32)
    validated = validate_ulsa_dropin(
        {"total": total, "frequency_mhz": ULSA_FREQUENCIES_MHZ}
    )
    assert validated.dtype == np.float64
    np.testing.assert_array_equal(validated, total)


@pytest.mark.parametrize(
    ("maps", "frequencies", "message"),
    [
        (np.zeros((49, ULSA_NPIX)), ULSA_FREQUENCIES_MHZ, "shape"),
        (np.zeros((50, ULSA_NPIX)), np.arange(50.0), "frequencies"),
        (
            np.full((50, ULSA_NPIX), np.nan),
            ULSA_FREQUENCIES_MHZ,
            "non-finite",
        ),
    ],
)
def test_validate_ulsa_dropin_rejects_contract_violations(maps, frequencies, message):
    with pytest.raises(ValueError, match=message):
        validate_ulsa_dropin(maps, frequencies)


def test_write_ulsa_dropin_refuses_overwrite_by_default(tmp_path):
    output = tmp_path / "existing.fits"
    maps = np.zeros((50, ULSA_NPIX), dtype=np.float64)
    write_ulsa_dropin(output, maps)
    with pytest.raises(OSError):
        write_ulsa_dropin(output, maps)


def test_lusee_fitssky_consumes_dropin_frequency_and_pixel_contract(tmp_path):
    from lusee.SkyModels import FitsSky

    maps = np.ones((50, ULSA_NPIX), dtype=np.float64)
    output = write_ulsa_dropin(tmp_path / "dropin.fits", maps)
    sky = FitsSky(str(output), lmax=0)

    assert sky.Nside == ULSA_NSIDE
    assert sky.Npix == ULSA_NPIX
    assert sky.frame == "galactic"
    assert sky.mapalm.shape == (50, 1)
    np.testing.assert_array_equal(sky.freq, ULSA_FREQUENCIES_MHZ)
