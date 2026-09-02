from pathlib import Path

import numpy as np
from astropy.io import fits

from lowsky.pipeline import SkyConfig, generate


def test_small_end_to_end_pipeline_reaches_diagnostic_products(tmp_path: Path):
    mounted = tmp_path / "ulsa-reference.fits"
    fits.PrimaryHDU(np.full((50, 12 * 32**2), 1.0e6, dtype=np.float64)).writeto(
        mounted
    )
    config = SkyConfig(
        nside=4,
        ray_oversample=1,
        n_distance=12,
        n_shells=2,
        shell_quadrature_steps=16,
        shell_distance_bins=2,
        sky_mode="ours",
        harmonic_lmax=8,
    )

    products, _tuned, _sources, _catalog, harmonics = generate(config, mounted)

    assert products["total"].shape == (50, 12 * config.nside**2)
    assert products["beta"].dtype == np.float64
    assert products["gum_covering"].shape == (12 * config.nside**2,)
    assert products["orion_covering"].shape == (12 * config.nside**2,)
    assert np.all(np.isfinite(products["tau_1mhz"]))
    assert harmonics["total"].shape == (50, 45)
