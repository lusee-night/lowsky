# lowsky

`lowsky` makes physics-based, counterfactual low-frequency skies. It models
the Milky Way in three dimensions instead of copying residual pixels from
Haslam or ULSA.

![One realization at 1, 10, 30, and 50 MHz](docs/sky_frequencies_a07a609.png)

The 1 MHz morphology is especially prior-sensitive because even modest ionized
filaments become optically thick; isolated dark patches are not robust pixels.

The model includes synchrotron emission, free-free transfer, spiral structure,
the Local Bubble, nearby shells and spurs, Gum, Orion–Eridanus, Cygnus X, an
isotropic extragalactic background, and optional analytic A-team sources.
Diffuse WIM and Galactic-center absorption are volume-filling. Thin nearby
ionized filaments use explicit sub-beam partial-covering transfer.

## Install

Python 3.12 is required.

```bash
pip install "lowsky @ git+https://github.com/lusee-night/lowsky.git"
```

For development:

```bash
git clone https://github.com/lusee-night/lowsky.git
cd lowsky
uv sync --extra dev
```

## Differentiable API

`generate_sky` is the main package output. It is written entirely with JAX and
returns brightness temperature in kelvin with shape `(frequency, pixel)`.

```python
import jax
import jax.numpy as jnp

from lowsky import SkyConfig, SkyParameters, generate_sky, prepare_sky_inputs

# Geometry, seeded random fields, and catalogs are prepared outside the JAX graph.
inputs = prepare_sky_inputs(
    SkyConfig(nside=32, ray_oversample=1, sky_mode="ours", seed=20260901)
)
parameters = SkyParameters(
    emissivity_scale=1.0,
    emission_measure_scale=1.0,
    spectral_index_offset=0.0,
    synchrotron_fluctuation_sigma=0.28,
    emission_measure_fluctuation_sigma=0.32,
    spectral_index_fluctuation_sigma=0.055,
    synchrotron_spectral_curvature=0.0,
    local_shell_scale=1.0,
    local_shell_spectral_index_offset=0.0,
)

frequencies_mhz = jnp.array([1.0, 10.0, 30.0, 50.0])
sky_k = jax.jit(generate_sky)(frequencies_mhz, inputs, parameters)
```

`SkyInputs` and `SkyParameters` are JAX pytrees. The output can be used with
`jax.jit`, `jax.grad`, `jax.jacfwd`, and `jax.vmap`. Use
`generate_sky_components` when individual physical components are needed.
The setup step fixes geometry, catalogs, and unit random fields from `seed`;
the amplitudes of those fields and the synchrotron spectral parameters remain
continuous `SkyParameters`, so one realization can be optimized or sampled
without rerunning setup.

## ULSA drop-in file

Download the calibrated 1--50 MHz NSIDE-32 replacement from the
[latest release](https://github.com/lusee-night/lowsky/releases/latest/download/lowsky_32.fits),
or generate a new realization locally without reading the mounted ULSA
artifact:

```bash
uv run python examples/export_ulsa_dropin.py \
  --sky-mode ours \
  --output lowsky_32.fits
```

The writer enforces the LuSEEpy contract: one float64 primary HDU containing
50 Galactic RING-ordered NSIDE-32 maps for 1–50 MHz, with the original ULSA
header layout.

## Full pipeline

The calibrated pipeline compares a realization with an external ULSA reference,
creates diagnostic FITS maps, and saves canonical un-beamed harmonic products:

```bash
lowsky-generate \
  --mounted-ulsa /path/to/ULSA_32_ddi_smooth.fits \
  --sky-mode ours \
  --output-dir lowsky-output
```

Extended components are generated on the higher-resolution ray grid and then
transformed to spherical harmonics. Cas A, Cyg A, Tau A, and Vir A are inserted
as analytic band-limited delta functions. A beam is applied only to diagnostic
maps.

Additional examples:

```bash
uv run python examples/plot_frequency_skies.py lowsky-output/counterfactual_ulsa_nside32_1_50mhz.fits
uv run python examples/plot_harmonic_power.py --help
uv run python examples/validate_feature_fidelity.py --help
```

To test blind A-team spectral recovery over one LuSEE lunar night without the
receiver/electrical forward model, beam-convolve the canonical source-free and
source-only harmonics into separate waterfalls:

```bash
uv run python examples/one_night_source_separation.py \
  lowsky-output/counterfactual_ulsa_harmonic.npz
```

This uses the generated, validated four-port response
``/local/zack/receive_matrix/lusee_bgl_v16_response_v3.fits`` (or the file
passed with ``--response-file``) directly. Its port-specific complex effective
lengths, mutual impedance, layered-regolith/lander response, and JFET receiver
loading produce all 4 auto-power plus 6 complex cross-power products (16 real
channels); no single-port beam rotations are synthesized. The estimator uses
known source positions and an independent low-order temporal Fourier nuisance
model for each channel, but never uses the source-free waterfall in the fit.
Moon and antenna thermal emission are disabled so the saved diffuse-only and
source-only component waterfalls are strictly additive. The default
``lmax=179`` is the native harmonic ceiling of the V16 response and preserves
the modeled lunar-horizon edge at approximately its one-degree grid scale.

Validation notebooks are in [`notebooks/`](notebooks/), including the
[`diffuse versus point-source power-spectrum plot`](notebooks/diffuse_vs_point_sources.ipynb).
The diffuse morphology
has been resolution-validated through `ell = 64`; higher multipoles are retained
but should not be interpreted as converged small-scale structure.

## Scope

This is a controlled simulator, not an observational survey. It uses ULSA’s
radiative-transfer ingredients and global calibration while deliberately
avoiding a Haslam residual template. See [`references/`](references/) for the
physical anchors and literature notes.
