# Observational constraints for a lowsky paper

Started 2026-10-13 at Zack's request. Three Sol subagents gather complementary observations; this directory is a data/provenance work package, not a fit or a claim of model validation.

Repository: `git@github.com:lusee-night/lowsky.git`, local checkout `/home/zack/Projects/ulsa-counterfactual`, HEAD at discovery `c1eeb4dce1bc6ab43cc476d198a990b0ef9a6d72`. Existing untracked `analysis/ulsa-32-ddi-2026-09-01/` was preserved.

## Figure strategy

| Figure | Observational quantity | Required lowsky comparison | Principal constraint |
| --- | --- | --- | --- |
| Absolute spectrum and turnover | Published brightness or intensity versus frequency, preserving angular response | Apply each experiment's sky/beam/time weighting; use a full-sky mean only when justified | Emissivity, spectral index/curvature, absorption, isotropic background |
| Regional spectra and latitude profiles | Matched-resolution survey brightness in fixed overlapping sky masks | Convolve model to survey response, apply coverage/masks and source treatment, calculate same statistic | Galactic distribution and spectral variation |
| Angular structure | Masked angular power or structure function in survey-covered regions | Same beam, pixel window, mask/mode coupling, source subtraction and noise treatment | Fluctuation amplitudes, shells and spatial correlations |
| HII absorption/emissivity | Foreground/background brightness or inferred emissivity for individual sightlines | Integrate model along the same foreground/background distance interval; reproduce estimator assumptions | 3D synchrotron distribution and free-free transfer |
| Nearby structures | On/off contrast or EM for Gum, Orion–Eridanus and loops | Same apertures/background subtraction, beam, frequency and distance assumptions | Local emission, EM, electron temperature and covering fraction |

Show observational points with their reported errors, instrument labels and shared calibration uncertainties. Show lowsky predictions as a separately identified curve or prediction points. An ensemble band should represent explicit parameter/realization variation, not arbitrary plotting scatter. Other sky models are model comparators, never additional observational points.

## Independence and provenance

`src/lowsky/pipeline.py::tune_to_ulsa` fits emissivity scale, EM scale and spectral-index offset against ULSA sky means, subtracting lowsky's local-shell component from the predicted calibration baseline. Thus agreement with ULSA is a calibration comparison. Surveys underlying ULSA need a lineage check before being called independent holdouts.

The reference notes already condition local shell brightness/spectra and thermal structures on literature, including Gum and Orion–Eridanus. Distinguish (1) an existing calibration/prior input, (2) a potentially shared underlying dataset, (3) a new independent measurement, and (4) a derived estimate with additional astrophysical assumptions. A published number is not automatically an independent datum.

The baseline output nominally covers 1–50 MHz; higher-frequency surveys are useful anchors but require explicitly validated model evaluation outside that range. Diffuse morphology is documented as resolution-validated only through ell=64. Do not infer precision at higher multipoles from the stored harmonic cutoff.

Preserve original units and reported uncertainty definitions. For Rayleigh–Jeans conversion use T=I*c^2/(2*k*nu^2), including the Jy conversion when applicable. Flux density of a compact source requires beam/solid-angle treatment before comparison with diffuse temperature. Use the convention T proportional to nu^(-beta_T); a flux-density spectral index is a different quantity.

## Contents

- `absolute_spectra/`: space/absolute spectrum measurements and retrieval evidence.
- `radio_surveys/`: resolved map availability, numerical constraints and comparison requirements.
- `absorption_structures/`: absorption, emissivity and nearby-structure measurements.
- `model_baseline/`: reproducible extraction of 50 full-sky mean model temperatures from the existing September 2 FITS artifact, with SHA-256. This is an illustrative ULSA-calibrated realization, not a refit or an independent observation; its generating source revision is unverified.

## Next scientific step

After auditing the gathered records, select the first comparison by its measurement operator and independence, not by the number of points available. Implement the exact operator and uncertainty bookkeeping, freeze a calibration/holdout split, and only then overlay or fit lowsky. Large survey downloads and paper-quality inference remain subsequent work. No model parameters or production code were changed by this gathering pass.

## Completed first-pass inventory

All three Sol subtasks completed on 2026-10-13.

| Bundle | Data actually obtained | Source and interpretation |
| --- | --- | --- |
| [Absolute spectra](absolute_spectra/README.md) | 15 tabular entries: 8 Cane points + 7 selected RAE-2 points | 0.25–23 MHz collectively; RAE points are republished observations, not an additional independent Cane experiment; historical beam weights and error semantics retained |
| [Radio surveys](radio_surveys/README.md) | Three FITS maps: 10.03, 22, nominal 45 MHz; 4,155,840 bytes | NASA LAMBDA products; nominal 45 product combines southern 45 and northern 46.5 MHz data; masks, orderings, beams and calibration limitations recorded |
| [Absorption and structures](absorption_structures/README.md) | 152 Su et al. GLEAM HII table rows at 76.2 MHz; four 22-MHz loop brightness entries | HII records include 28 optical-depth lower limits and derived quantities sharing Haslam assumptions; loop priors overlap existing conditioning |
| [Existing model spectrum](model_baseline/spectrum.json) | 50 model predictions, 1–50 MHz | Full-sky means of existing ULSA-calibrated realization; different observable from historical beam-weighted observations |

LWA1 and Haslam data links are verified catalog entries, not downloaded maps. Gum/Orion and fitted-shell summaries are supporting constraints, not extra independent tabular measurements. IMP-6 original source is retrieved but its table is not transcribed; Wind/Voyager gaps are explicit.

Parent verification: all JSON files parse; 15 absolute, 152 HII and 50 baseline records counted; the three local survey byte counts and SHA-256 hashes match their manifest. The baseline extraction executed successfully with shape, finite-value and unit checks. The subagents checked their numerical source extractions and documented provenance. No quantitative model/data agreement has yet been asserted.

Recommended first implementation: matched-mask 10/22/45-MHz regional spectra with explicit beam/source treatment, plus historical north/south spectral points once historical response weighting is implemented. HII tomography is a valuable next track after checking the model at 76.2 MHz and implementing the distance/beam/subgrid measurement operator. These are next actions, not scheduled jobs.

## Modern-data extension

The follow-up search prioritizes modern measurements with different observables:

- **PSP/FIELDS (2023 analysis, 2025 quadrupole publication):** roll-phase spectra below 6 MHz and measured l=2 coefficients from 0.5–6.8 MHz. This is a direct route to testing lowsky's large-scale anisotropy. The published absolute scale has historical calibration ancestry; extracted harmonic means/errors are inferred jointly with antenna parameters. See `absolute_spectra/modern_notes.md` and `modern_sources.json` for acquisition status.
- **OVRO-LWA and LWA1:** resolved maps overlapping 35–50 MHz, useful for morphology and angular structure; preserve interferometric transfer and absolute-level calibration limitations. See `radio_surveys/modern_notes.md`.
- **LoDeSS DR1, released October 6, 2026:** 15–30 MHz source catalogue and images at about 45 arcsec, central frequency 23 MHz. Useful for source populations and spectra, not automatically a diffuse-monopole constraint. The release covers about 15% of the eventual survey area and includes field quality flags. Primary release: https://lofar-surveys.org/lodess.html ; paper: https://arxiv.org/abs/2610.04698 .
- **McKay et al., version 4 June 2026:** absolute 60–350 MHz radiometry with a SKALA4.1 antenna. Valuable higher-frequency anchor and calibration tension, requiring model validation above 50 MHz; not an in-band direct observation. Primary: https://arxiv.org/abs/2509.11846v4 . See `absorption_structures/modern_notes.md` for public numerical data status and EDGES comparisons.

Modern data strengthen three different panels: large-scale harmonic coefficients versus frequency, matched-response map statistics, and source spectral distributions. Do not pool these as measurements of the same sky average.

Modern acquisition outcome: downloaded 56 PSP quadrupole frequency rows (0.514–6.78 MHz), the LoDeSS DR1 catalogue (21,465 sources), and a small published EDGES reduced spectrum (128 rows, 123 valid). The PSP table's row-count discrepancy with the paper is documented, and calibration correlations remain explicit. McKay's primary 7.87-GB collection is linked but not downloaded; reported GSM correction functions are model-specific, not additional measurements. All three modern JSON manifests parse. The modern pass is complete as data discovery/small-product acquisition, not model comparison.
