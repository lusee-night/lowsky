# Counterfactual ULSA realization

This project generates a new low-frequency sky realization with the physical
structure of Cong et al. (2021), without using Haslam pixels as a residual
template.

The model includes:

- the fitted cylindrical Galactic synchrotron emissivity from ULSA Eq. 17 and
  Table 1;
- an NE2001-inspired thick disk, thin disk, five logarithmic spiral arms, and
  Galactic-center electron component;
- line-of-sight free-free absorption and thermal free-free emission at
  `Te = 8000 K`;
- correlated Kolmogorov-like emission-measure fluctuations in distance shells,
  replacing catalogued pencil-beam clumps;
- independent synchrotron fluctuations with input `C_ell proportional to
  ell^-2.7`, also distributed across distance shells;
- a fresh, smooth direction-dependent synchrotron-index field using ULSA's
  direction-dependent variant (ULSA treats its frequency-curved index as a
  separate alternative model);
- the ARCADE-2 isotropic extragalactic background and small CGM/IGM opacity;
- explicit Cas A, Cyg A, Tau A, and Vir A components, rather than allowing
  those sources to leak into the diffuse template.
- a seeded population of 3--6 nearby radio-loop/superbubble analogues, including
  one Loop-I-scale object at a random sky position. Continuous Gaussian radial
  emissivity and fine local ray quadrature generate soft limb brightening;
  magnetic shock obliquity, ambient-density gradients, and correlated surface
  modes generate partial turbulent spurs without drawing 2-D arcs.
- a configurable smooth shell-spectrum break near 30 MHz, reflecting the
  observed flattening between the 408- and 22-MHz Loop-I measurements rather
  than extrapolating a steep GHz-band index unchanged to 1 MHz.

JAX evaluates the three-dimensional model, performs radiative transfer, and
gradient-tunes emissivity, emission measure, and spectral-index offsets to the
mounted ULSA global spectrum. Healpy is restricted to HEALPix geometry and
spherical-harmonic synthesis/analysis.

The default calculation traces every extended component at NSIDE 64 and
transforms those native maps directly to packed healpy coefficients through
ell = 191. Cas A, Cyg A, Tau A, and Vir A are never painted into pixels:
LuSEEpy supplies their analytic delta-function coefficients
`a_lm = A_nu Y_lm*(n_source)`. The un-beamed harmonic sum is the canonical sky
product. A common 2-degree beam is applied only when reconstructing the
NSIDE-32 diagnostic FITS maps.

Install from GitHub:

```bash
python -m pip install "lowsky @ git+https://github.com/lusee-night/lowsky.git"
```

Generate a realization using a downloaded or mounted ULSA reference product:

```bash
lowsky-generate \
  --mounted-ulsa /path/to/ULSA_32_ddi_smooth.fits \
  --output-dir lowsky-output
```

The same model is available as a Python API:

```python
from pathlib import Path

import lowsky

config = lowsky.SkyConfig(sky_mode="ours", seed=20260901)
maps, fit, sources, shells, alms = lowsky.generate(
    config,
    Path("/path/to/ULSA_32_ddi_smooth.fits"),
)
```

`maps` contains beam-convolved diagnostic maps. `alms` is the canonical
un-beamed product: every extended component is transformed from the native
high-resolution realization, while each A-team source is inserted analytically
as a band-limited delta function.

For a counterfactual realization conditioned on approximate Milky Way
geography, add `--sky-mode ours`. This anchors named loop directions, an
offset Local Bubble, Gum, Orion--Eridanus, Cygnus X, the solar position, warp,
bar, and modest arm contrast while retaining seeded uncertainty and using no
sky-map pixels.

The literature notes are in `references/`. The canonical
`counterfactual_ulsa_harmonic.npz` contains diffuse, individual-source,
combined-source, and total coefficients. Generated diagnostic FITS maps,
component arrays, spectra, metrics, and comparison plots are also written to
the chosen output directory.

To compare the canonical source-free and analytic point-source angular power
directly, without synthesizing a map or applying a beam, run:

```bash
lowsky-power --input lowsky-output/counterfactual_ulsa_harmonic.npz
```

This writes a log-log comparison figure, all-frequency numerical spectra, and
a CSV of source/diffuse ratios and sustained crossover multipoles beside the
input harmonic product.

The literature and numerical feature audit is reproducible with:

```bash
lowsky-validate lowsky-output
```

It checks Local-Bubble dimensions, named-loop geometry/contrast/spectra,
regional Gum/Orion/Cygnus emission measures, harmonic additivity, and an
NSIDE-64 versus NSIDE-128 shell convergence experiment. The current diffuse
morphology is validated through `ell=64`; coefficients above that band limit
are retained but must not be treated as resolution-converged morphology.

The mounted product identifies itself as ULSA's direction-dependent-index,
smooth-absorber field (`ULSA_FIELD=smooth_absorb`), rather than a final
Haslam-residual-added map.  Comparisons therefore use that field directly.
The output FITS primary HDU is the beam-convolved diagnostic counterfactual
sky. `NO_SOURCES` is
the diffuse sky; `ATEAM` is the sum of the four explicit sources;
`STOCHASTIC_SYNCH`, `LOCAL_SHELLS`, `FREEFREE`, and `EXTRAGALACTIC` add to
`NO_SOURCES`.
`BASELINE` contains the ULSA-tuned source-free sky before `LOCAL_SHELLS` is
added.
`SMOOTH_SYNCH` is the no-fluctuation synchrotron diagnostic and is not an
additional additive component.

The smooth/turbulent baseline is tuned to the mounted ULSA global mean; the
independently normalized local shells are added afterward and are not tuned
away. Its `comparison_metrics.json` separately
records total and diffuse angular slopes, combined A-team bandpower, and each
source's isolated bandpower. The source spectra remain phenomenological
low-frequency extrapolations, but their spatial rendering is an exact
band-limited delta function. Beam convolution belongs to the instrument or a
requested visualization, not to the canonical source model.

This is a controlled counterfactual simulation, not a replacement observational
sky survey. In particular, the statistically represented unresolved HII/WIM
clumping is intentionally not the catalogued NE2001 clump realization.
Ordinary unresolved SNRs remain in the statistical synchrotron component at
the current 2-degree resolution; the explicit shells represent the resolved
local-loop population.
