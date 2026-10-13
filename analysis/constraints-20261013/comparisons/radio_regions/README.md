# Matched regional radio-map check

Run from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/compare_radio_regions.py --output-dir analysis/constraints-20261013/comparisons/radio_regions
```

Optional `--model PATH` and `--survey-dir DIR` override the saved model and input maps; FITS grid metadata and array shapes are checked.

The script compares real [10.03, 22, and 45 MHz survey FITS pixels](../../radio_surveys/manifest.json) with the saved 1–50 MHz, NSIDE=32, 2°-beamed `NO_SOURCES` lowsky diagnostic. It does **not** rerun or refit the model. Each survey and its model prediction receive the **same survey-validity/source mask**, an *additional* 12° Gaussian convolution, and an ell≤64 harmonic cutoff; the surveys' 1–4.6° anisotropic, position-dependent native beams make the effective resolutions only approximate. Survey invalid pixels and 8° circles around Cas A, Cyg A, Tau A, and Vir A are excluded before weighted smoothing. Accepted cells have >98% valid convolution weight. The northern anticenter region uses a common 10.03/22-MHz mask; the southern Orion region uses a common 22/45-MHz mask, with equatorial declination below −2° so its 45-MHz pixels come from the southern Maipu survey rather than the northern 46.5-MHz component. Model temperature at 10.03 MHz is log interpolated from 10 and 11 MHz. All operations stay at NSIDE≥32 and ell≤64; the maps are descriptive and do not assess modes above the validated ell range.

The figure is [PNG](radio_regions.png) and [SVG](radio_regions.svg); [JSON](radio_regions.json) gives exact footprints, per-bin temperatures and residuals, masks, and input SHA-256 hashes. The bands represent 16th–84th **spatial pixel spread**, not errors on the median. No measurement covariance, survey gain, zero-level, or foreground nuisance has been fitted. The 22-MHz declination gain was calibrated to Haslam 408 MHz, 45 MHz has a known EDGES-related absolute correction absent from this hosted map, and 10-MHz zero-level uncertainty was not verified. Some survey information also enters ULSA/model conditioning. Thus this is a direct measured-map confrontation, but neither an independent calibration test nor a statistical rejection.

| Matched region and survey | Cells | Survey median K | Lowsky median K | Median (survey−model)/model |
|---|---:|---:|---:|---:|
| Anticenter north 10.03 MHz | 261 | 225,873 | 324,843 | −31.0% |
| Anticenter north 22 MHz | 261 | 30,437 | 41,957 | −28.2% |
| Orion south 22 MHz | 39 | 19,929 | 43,738 | −54.4% |
| Orion south 45 MHz | 39 | 4,283 | 6,806 | −36.2% |

The large signed discrepancies persist across both patches, with a stronger 22-MHz gap in the small southern patch. Source policy is only approximate: `NO_SOURCES` removes the modeled A-team while survey images may retain other compact sources. The 12° smoothing also washes out localized Gum/Orion/Cygnus absorption, so use a finer-resolution and beam-specific forward operator for object physics. These numbers are diagnostic, not claims of model correctness or a likelihood constraint.
