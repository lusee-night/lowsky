import numpy as np
import pytest
from lowsky.calibration import calibrated_no_sources_from_components


def test_fixed_opacity_linear_component_recombination_and_mean():
    s=np.array([1.,3.,5.]); shell=np.array([2.,4.,6.]); ff=np.array([.1,.2,.3]); eg=np.array([7.,11.,13.])
    out=calibrated_no_sources_from_components(s,shell,ff,eg,22.,.42,0.,.86)
    np.testing.assert_allclose(out,.42*s+shell+ff+.86*eg)
    assert np.mean(out)==pytest.approx(.42*np.mean(s)+np.mean(shell)+np.mean(ff)+.86*np.mean(eg))


@pytest.mark.parametrize('freq,ratio,eg',[(0.9,1,1),(22,0,1),(22,1,0)])
def test_invalid_calibration_rejected(freq,ratio,eg):
    with pytest.raises(ValueError):
        calibrated_no_sources_from_components(np.ones(2),np.ones(2),np.ones(2),np.ones(2),freq,ratio,0,eg)


def test_saved_components_sum_to_saved_source_free_map():
    from astropy.io import fits
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/"outputs_our_sky_gum_gaussian_final/counterfactual_ulsa_nside32_1_50mhz.fits"
    if not path.exists():
        pytest.skip("Saved diagnostic FITS is not present in a clean checkout")
    with fits.open(path,memmap=True) as h:
        i=21  # 22 MHz
        expected=np.asarray(h["NO_SOURCES"].data[i],dtype=float)
        actual=sum(np.asarray(h[n].data[i],dtype=float) for n in ["STOCHASTIC_SYNCH","LOCAL_SHELLS","FREEFREE","EXTRAGALACTIC"])
    np.testing.assert_allclose(actual,expected,rtol=2e-7,atol=.01)
