# Object-specific evidence audit

Requested 2026-10-13. Three Sol agents independently gather primary-source observations, numerical products and comparison requirements for all named structures in the current `ours` realization. This is a constraints audit; no fit, source modification or observational agreement is implied.

## Assigned coverage

- [Loops and Local Bubble](../objects_loops_bubble/README.md): Loop I/NPS and S1/S2 interpretations, Loops II–IV, Local Bubble boundaries. Distinguish total intensity from polarization, and gas/dust geometry from ionized emission measure.
- [Thermal structures](../objects_thermal/README.md): Gum, Orion–Eridanus, Cygnus X. Track electron temperature, EM, extinction, distance, morphology and beam-area covering separately.
- [Bright sources](../objects_bright_sources/README.md): Cas A, Cyg A, Tau A/Crab, Vir A/M87. Track integrated flux, epoch/secular evolution, spatially resolved absorption and flux-scale ancestry.

## Common comparison rules

1. A point is observational only if its source and measurement definition are identifiable. Preserve uncertainties, limits, epoch and beam/aperture; null uncertainty means unreported, not zero.
2. Published fitted geometry, EM and spectral parameters are derived constraints. Keep their assumptions and covariance caveats. Numerical sky-model predictions are not observations.
3. Existing conditioning is explicit in `references/our_sky_conditioning.md`, `references/local_features_literature.md`, and the source. Newer literature can reuse the same data; publication recency alone does not establish independence.
4. Current saved maps are a seeded analogue. In particular, shell centres are not bright-arc coordinates, and adopted shell distances need not equal observationally established distances.
5. Compare the same observable: compact-source integrated Jy; loop aperture/background-subtracted K; absorption on/off contrast or transmission; thermal EM/temperature with appropriate extinction assumptions; bubble radial boundary with boundary definition specified.

## Important current-model details

The A-team function in `src/lowsky/pipeline.py` takes a nominal 50-MHz normalization, multiplies by a power law and an internal attenuation normalized to zero at 50 MHz, then applies modeled Galactic foreground absorption. Thus its S50 constant is **before Galactic attenuation**, not automatically the final observed 50-MHz flux. Internal attenuation is clipped to zero above 50 MHz. The source code does not include a time/epoch parameter or secular fading law in this function. A catalogue/calibrator value must not be adopted without checking whether absorption and epoch corrections are already included.

Thermal screens use transmission `1-f+f*exp(-tau)` and affect emission behind their distances. Volume filling factor and projected beam covering are distinct. Global electron temperature and EM scale create shared degeneracies across objects; individual structures are not independently fit by the present high-level parameter interface.

The saved `outputs_our_sky_gum_gaussian_final/` run supplies an illustrative baseline, but its generating source revision has not been established. New evidence should be compared to a pinned regenerated run before a final paper figure.

## Completed evidence pass

All three Sol agents completed the bounded search on 2026-10-13. Read each bundle for exact references and acquisition limits; this is not an exhaustive claim to have found every latest measurement.

- **Loops/Local Bubble:** radio excesses and sightline polarization-distance constraints, with 2024–2025 dust, magnetic-field, X-ray and Eos context. The four 22-MHz loop points duplicate the earlier thermal bundle and must be counted once. No direct measured loop turnover at 30 MHz or new robust Loop II/III distance was secured.
- **Thermal structures:** 50 heterogeneous records (23 Gum, 15 Orion, 12 Cygnus), including measurements, limits and explicitly labeled derived model parameters. Actual products include WHAM H-alpha sightlines and a 2-arcmin 148-MHz Cygnus map. Kumar et al. 2026 offers a different pulsar-derived Gum geometry, not a direct free-free map.
- **Bright sources:** 27 measured source/Cyg ratios plus eight fit-derived peak/ratio or resolved-component constraints. Four calibration-model normalizations are stored separately. Cas A 2023.5 fitted turnover and 2025 Crab shell-only turnover are especially relevant, with epoch and component caveats. Independent Cyg absolute flux remains a gap.

Best immediate comparisons are source/Cyg ratios (with shared-calibrator covariance), epoch-specific Cas A turnover, Orion matched-beam H-alpha/EM profiles, and Cygnus spatial statistics after degrading the observed image appropriately. Gum geometry alternatives merit a controlled sensitivity study before replacing the current screen. Loops need matched spur/background regions; bubble distances need shell intersections and tracer-specific definitions.

Parent validation: all object JSON files parse; locally recorded primary-download hashes were independently checked. No lowsky fit or parameter change was made. Next action is to choose and implement observation operators, starting with the ratio panel and thermal regional profiles; this is a proposed next step, not a scheduled job.
