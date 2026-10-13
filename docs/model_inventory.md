# Lowsky: compact model inventory

Snapshot: 2026-10-13. Scope: the Milky-Way-conditioned (`sky_mode="ours"`)
realization over 1–50 MHz, including the separate survey-calibrated diffuse
product. The organizing principle is **simple components, explicit amplitudes
and spectra, and measurements attached to every fitted parameter**.

## What is in the model now

| Component | Representation | What is fitted or prescribed? |
|---|---|---|
| Bright sources: Cas A, Cyg A, Tau A, Vir A | Analytic point-source spherical harmonics, at catalog positions. Power-law flux spectra with source turnover and foreground free-free absorption. | Cas A normalization and turnover fitted to the 2023.5 peak; other prescriptions are retained. LWA1 source ratios are comparison data, not a joint source fit. |
| Broad Galactic synchrotron | Smooth radial/vertical emissivity distribution with modest spiral-arm and central-bar enhancements; correlated fluctuations add texture. Line-of-sight integration supplies the sky map. | Overall emissivity, spectral-index offset, and emission-measure scale tuned to the ULSA mean spectrum. Geometry and fluctuation statistics are prescribed. |
| Local radio loops | Five broad emitting shell components: Loop I S1/S2 and Loops II–IV. Smooth radial envelopes with partial bright surfaces and seeded texture; broken-power-law spectra. | Literature-conditioned locations, sizes, brightnesses, and spectral priors, with seeded scatter. Not a joint fit to the new evidence index. |
| Galactic absorbing gas | Analytic NE2001-inspired thick/thin disks, five approximate arms, and Galactic center; converted to effective emission measure, with correlated fluctuations. | Global EM scale tuned to ULSA. No full NE2001 clump/void catalog and no new pulsar-data fit. |
| Local Bubble | Smooth irregular cavity reducing the diffuse EM, with polar openings. | Geometry conditioned on literature; not newly fitted here. |
| Gum Nebula | One broad Gaussian EM blob, angular sigma 11.5 degrees, with a smooth partial-covering prescription. | Literature-conditioned geometry and EM/absorption normalization. |
| Orion–Eridanus | Two nearby ionized interfaces with smooth ellipsoidal geometry and partial covering. | Prescribed literature-conditioned geometry and EM. More elaborate than a single Gaussian blob. |
| Cygnus complex | One three-dimensional Gaussian EM enhancement. | Prescribed location, width, and normalization; unresolved internal structure. |
| Isotropic incident radio background | `1.2 (nu/GHz)^(-2.58) K`, attenuated through the Galaxy and an additional prescribed low-frequency optical depth `0.95 (nu/MHz)^(-2.1)`. | Adopted ULSA background prescription, not independently fitted in this pass. |
| Thermal free-free emission | Emission from the same absorbing gas, with default electron temperature 8000 K. | Computed from transmission; not an independently adjustable emission template. |
| Empirical diffuse correction | Positive, smooth angular scale: 48 broad Gaussian basis functions, sigma 20 degrees, at three frequency anchors. Multiplies the saved volume synchrotron and background together. | Fitted to 10.03, 22, and southern 45 MHz surveys. Shells, free-free, and saved point sources are unchanged. |

## How they combine

The physical source-free sky is

`absorbed volume synchrotron + absorbed local loops + free-free emission + absorbed background`.

Point sources are added as analytic `a_lm` coefficients. Extended components
are transformed from their native integration grid to `a_lm`; the normal
NSIDE=32 maps are 2-degree-beam diagnostic products. The smooth and stochastic
synchrotron maps are alternatives: **only the stochastic realization enters
the total**, not both.

Absorption is distance ordered: each emitting layer sees its foreground EM.
An unresolved screen uses transmission `1 - f + f exp(-tau)`, with covering
fraction `f`; gas emission is included consistently. Point-source foreground
absorption currently uses the volume EM column to the source, plus its
source-specific turnover; it does not use the separate local partial-screen
operator. Thus the source and extended-emission absorption operators are not
identical implementations.

The survey-conditioned map instead uses

`g(direction, frequency) * (saved volume synchrotron + saved background) + saved loops + saved free-free + saved sources`.

This is an empirical brightness correction, not a fitted electron-density
model. Its frequency anchors are interpolated in log frequency; the nearest
anchor scale is held fixed outside 10.03–45 MHz. No new un-beamed canonical
harmonic product was generated for that corrected map.

## Keep the fitted model simple

For subsequent development, expose these small parameter groups:

1. **Point sources:** fixed positions; flux normalization, spectral slope,
   and turnover strength. Keep their analytic harmonics.
2. **Diffuse sky:** broad smooth structure plus a regularized fitted angular
   correction and frequency dependence. Keep the empirical correction
   separate from claims about physical electron density or background.
3. **Extended objects:** use a Gaussian blob or broad shell according to the
   measured morphology; fit amplitude, width, spectrum, and optical depth
   only where data constrain them. Keep position/distance fixed or prior-constrained.
4. **Absorption:** retain emission measure, temperature, distance ordering,
   and covering fraction. Do not fit extra geometric texture without evidence.

These are simplicity guidelines, not a claim that all current components
have already been reduced to blobs or jointly fitted. In particular, the
radio loops and Orion interfaces still have prescribed surface structure.

## Calibration status and usable products

- Cas A's 2023.5 peak is a calibration target, not a validation point.
- The diffuse correction uses the survey maps being plotted. Withheld sky
  sectors are correlated spatial checks, not independent-survey validation.
  Many basis coefficients reach bounds; this is not a physical parameter measurement.
- The corrected FITS retains the historical saved A-team. It does not yet
  combine the updated Cas A prescription with a new canonical harmonic sky.
- The complete evidence index is an inventory of constraints, not evidence
  that every entry has entered a fit.

Implementation: [`pipeline.py`](../src/lowsky/pipeline.py),
[`model.py`](../src/lowsky/model.py), and
[`calibrate_diffuse_surveys.py`](../examples/calibrate_diffuse_surveys.py).
Calibration records: [Cas A](../analysis/constraints-20261013/calibration/cas_a/fit.json)
and [diffuse](../analysis/constraints-20261013/calibration/diffuse_survey_conditioned/manifest.json).
