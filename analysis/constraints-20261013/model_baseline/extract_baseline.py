"""Extract a descriptive spectrum from the existing September artifact; no fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'outputs_our_sky_gum_gaussian_final/counterfactual_ulsa_nside32_1_50mhz.fits'
with fits.open(SOURCE, checksum=True) as hdus:
    header = hdus[0].header
    maps = np.asarray(hdus[0].data, dtype=np.float64)
    assert maps.shape == (50, 12 * header['NSIDE'] ** 2)
    assert header['BUNIT'] == 'K' and np.isfinite(maps).all()
    frequencies = header['FREQ0'] + np.arange(maps.shape[0]) * header['DFREQ']
    records = [dict(frequency_mhz=float(f), all_sky_mean_temperature_k=float(np.mean(m)))
               for f, m in zip(frequencies, maps)]
    result = dict(
        kind='model_prediction_not_observation',
        source=str(SOURCE.relative_to(ROOT)),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        status='Existing September 2 artifact, not regenerated or refit in this gathering pass. Source commit at artifact creation is unverified.',
        observable='Equal-area full-sky arithmetic mean of Galactic RING NSIDE32 total-temperature maps',
        beam_fwhm_deg=header['BEAMFWHM'],
        independence='ULSA-calibrated baseline; neither independent evidence nor an uncertainty ensemble.',
        comparison_requirement='Only directly overlay observations with compatible sky/beam/time/source weighting, or forward-model their measurement operator first.',
        records=records,
    )
(OUT / 'spectrum.json').write_text(json.dumps(result, indent=2) + '\n')
print(f'Wrote {len(records)} model points to {OUT / "spectrum.json"}')
