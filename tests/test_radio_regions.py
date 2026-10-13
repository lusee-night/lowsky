"""Focused checks for the matched regional survey operator."""
import importlib.util
from pathlib import Path

import healpy as hp
import numpy as np
import pytest
from astropy.io import fits

SCRIPT = Path(__file__).resolve().parents[1] / "examples/compare_radio_regions.py"
SPEC = importlib.util.spec_from_file_location("compare_radio_regions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mask_weighted_smoothing_preserves_constant_field():
    n = hp.nside2npix(64)
    weight = np.ones(n)
    weight[:n // 5] = 0
    weight[n // 3:n // 3 + 100] = 0.4
    data, smoothed_weight = MODULE.smooth_weighted(np.full(n, 1234.5), weight)
    good = smoothed_weight > MODULE.MIN_VALID_WEIGHT
    assert good.any()
    np.testing.assert_allclose(data[good], 1234.5, rtol=1e-10, atol=1e-7)
    assert np.isnan(data[~good]).all()


@pytest.mark.parametrize("frequency", [0.99, 50.01])
def test_model_rejects_out_of_band(frequency, tmp_path):
    with pytest.raises(ValueError, match="1--50"):
        MODULE.model_map(frequency, tmp_path / "unused.fits")


def test_model_rejects_nonpositive_log_interpolation(tmp_path):
    a = np.ones((50, hp.nside2npix(32)), dtype=np.float32)
    a[9, 0] = 0
    primary = fits.PrimaryHDU(a)
    primary.header["NSIDE"] = 32
    primary.header["ORDERING"] = "RING"
    primary.header["COORDSYS"] = "GALACTIC"
    primary.header["BUNIT"] = "K"
    p = tmp_path / "model.fits"
    fits.HDUList([primary, fits.ImageHDU(a, name="NO_SOURCES")]).writeto(p)
    with pytest.raises(ValueError, match="nonpositive"):
        MODULE.model_map(10.03, p)
