import numpy as np
import healpy as hp
from scipy.special import sph_harm_y

from plot_harmonic_power import first_sustained_dominance, power_from_alms


def test_power_from_alms_recovers_single_delta_flat_spectrum():
    lmax = 12
    amplitude = 3.5
    theta, phi = 1.1, 0.7
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
    for ell in range(lmax + 1):
        for m in range(ell + 1):
            alm[hp.Alm.getidx(lmax, ell, m)] = amplitude * np.conjugate(
                sph_harm_y(ell, m, theta, phi)
            )
    spectrum = power_from_alms(alm[None, :])[0]
    np.testing.assert_allclose(spectrum, amplitude**2 / (4 * np.pi), rtol=2e-12)


def test_first_sustained_dominance_ignores_short_crossings():
    ratio = np.full(30, 0.5)
    ratio[4:8] = 2.0
    ratio[13:23] = 1.1
    assert first_sustained_dominance(ratio, consecutive=10) == 13
    assert first_sustained_dominance(ratio, consecutive=11) is None
