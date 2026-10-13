# Observational evidence inventory

`index.json` is a portable, machine-readable inventory of the bounded research in `analysis/constraints-20261013/`. Its 486 canonical records reference 59 source entries and four explicit duplicate aliases. It is an **evidence catalog, not a joint fit**. Each record preserves its original table row (`data`) and a repository-relative source path plus pointer (`origin`); `src/lowsky/evidence.py` checks IDs, lists records and resolves pointers. All 486 pointers were checked against the original JSON/CSV files.

The primary collection includes 15 historical absolute-spectrum rows, 56 PSP/FIELDS *quadrupole* rows, 152 H II absorption sightlines, 128 released EDGES channels (123 valid channels, all above lowsky's 1–50 MHz range), 50 thermal-object rows, 39 bright-source rows (including non-observation calibration anchors), four canonical Borka 22 MHz loop measurements, four polarized loop distance constraints, and whole-survey/product records. The LoDeSS DR1 catalogue has 21,465 underlying source rows and appears as **one product record**. Source-only records retain recent NPS, loop and Local Bubble papers where no comparable small table was acquired.

`comparison_status` describes the next required step:

| Status | Meaning |
|---|---|
| `operator_required` | Numeric observation or acquired product exists, but a sky/beam/aperture/calibration/distance operator is required before a model comparison. It does **not** mean independent validation. |
| `out_of_band` | Released EDGES temperature is measured, but its valid channels are above the configured 1–50 MHz model domain. |
| `context_only` | Fitted parameter, reused prior, flagged channel, published prose summary, or source lead. Do not plot as an independent measured sky point. |
| `access_required` | Primary product is documented but a useful small numerical comparison product was not acquired. |

The `evidence_kind`, `operator`, `independence` and verbatim `data` fields refine this status. For example, Page 2025's PSP coefficients are **l=2 harmonic coefficients**, with a historical spacecraft absolute-calibration tie; they are not an all-sky monopole and require the exact Galactic harmonic normalization and antenna treatment. The Borka loop values are background-subtracted spur averages, not absolute sky pixels; the same four values also occur in the `absorption_structures` research bundle. `aliases` maps that duplicate to the four canonical records. O'Neill's Local Bubble dust surface and Wolleben's polarized shell model already inform lowsky's local-feature prior, while X-ray hot-gas geometry and Faraday data are different observables.

Paths in `origin.path` and `sources[].path` are relative to the **repository root**, not the installed Python package. When a source has `sha256`, that hash applies to `sources[].path`; `manifest_path` points separately to the committed research manifest. The default loader locates the checkout through `src/lowsky/evidence.py`, so it is intended for a source checkout; a wheel containing only `lowsky` will not include `evidence/index.json` or the research data. Original PDF/HTML/FITS downloads can be gitignored: source entries carry the public URL and SHA-256 so another checkout can reacquire and verify them. A local `product_path` or original-source `path` is a checkout reference, not a guarantee that the product ships in Git.

From this checkout, use `JAX_PLATFORMS=cpu .venv/bin/python -m lowsky.evidence summary`, `list --group radio_loops`, or `show borka2011_loop_i_t22`. The CPU setting avoids unnecessary JAX accelerator initialization caused by package import; the evidence loader itself uses only the standard library. `list --status operator_required --json` gives structured records for a plot-building script. Recreate the source's measurement operator and inspect calibration/conditioning before drawing or interpreting a lowsky comparison.
