# Physical priors for local radio shells and spurs

The counterfactual treats large radio loops separately from ordinary Galactic
SNRs and from isotropic turbulent fluctuations.

- Mertsch & Sarkar (2013) show that old SNR shells supply intermediate-scale
  power missing from smooth synchrotron models and reproduce the Haslam angular
  spectrum to about 10 percent through ell ~ 200:
  https://arxiv.org/abs/1304.1078
- Urosevic et al. / Borka (2007) measure Loops I--VI with 408-MHz excesses of
  roughly 3--8.5 K and temperature spectral indices about 2.68--3.03:
  https://academic.oup.com/mnras/article/376/2/634/1074743
- Wolleben (2007) models Loop I as two nearby magnetized shells with distances
  78 and 95 pc, radii 63--91 pc, and swept-up tangential magnetic fields:
  https://arxiv.org/abs/0704.0276
- Cong et al. (2025) directly models Loop I at 1--30 MHz. Its local realization
  uses a center distance 0.24 kpc, radius 0.22 kpc, and thickness 0.02 kpc. A
  local hot shell remains visible near 1 MHz because little WIM is foreground:
  https://arxiv.org/abs/2501.00431
- Panopoulou et al. (2021) place high-latitude polarized NPS material at about
  100--140 pc, while emphasizing the distance complexity:
  https://arxiv.org/abs/2106.14267
- La Porta et al. (2008) find high-latitude synchrotron slopes around -2.6 to
  -3.0 with substantial regional variation:
  https://arxiv.org/abs/0801.0547

## Implemented priors

- Number of explicit local loops: clipped Poisson mean 4, range 3--6.
- One conditioned Loop-I-scale analogue: d = 0.24 +/- 0.035 kpc,
  R = 0.22 +/- 0.025 kpc, fractional width about 0.10, but random direction.
- Other loops: lognormal median d = 0.15 kpc and R = 0.08 kpc, with broad
  literature-motivated truncation.
- Background-subtracted excess at 408 MHz: lognormal around 7 K; the large
  anchor uses 8.4 +/- 1.0 K.
- Brightness-temperature spectral index: 2.85 +/- 0.16; the large anchor uses
  2.64 +/- 0.10 to allow the observed low-frequency flattening.
- A smooth spectral break at 30 MHz transitions to a low-frequency temperature
  index 2.55 +/- 0.10 (2.52 +/- 0.08 for the large anchor). This prevents an
  unsupported unbroken 408-to-1-MHz extrapolation and is deliberately exposed
  as a configurable systematic.
- Shell emissivity has a Gaussian radial profile whose FWHM represents a broad
  total-intensity synchrotron envelope, rather than taking a thinner
  geometric/polarized wall fit literally. It is modulated by shock obliquity to a random
  ambient magnetic axis, a smooth density gradient, and three correlated
  low-order surface modes. This yields soft, broken/bilateral spurs without
  hard angular masks.
- The same low-order surface modes corrugate the shell radius by about seven
  percent and vary the local envelope width by up to 35 percent. This encodes
  coherent superbubble deformation and suppresses implausibly perfect circles;
  it does not add independent pixel-scale structure.
- Fine local quadrature is accumulated into eight distance-ordered emitting
  layers. Each is attenuated only by the modeled WIM in front of it. The
  projected 99th-percentile bright rim is normalized to the sampled 408-MHz
  excess, so profile softening does not change the observational anchor.
- A smooth logistic activation restricts synchrotron-bright material to a
  sampled 25--70 percent of the shock surface. The activation coordinate uses
  the same density-gradient and correlated surface modes, so incomplete arcs
  fade continuously rather than ending at a mask boundary.

These priors define a plausible ensemble, not the actual Milky Way realization.

## Orion--Eridanus ionized screens

Orion--Eridanus is treated differently from the radio-emitting shells above.
Its H-alpha structure is an elongated roughly 20 by 45 degree collection of
arcs, while 3-D gas studies favor nested 150--250 pc shells rather than one
filled spherical wall:

- Pon et al. (2014), filament geometry and 5--13 pc widths:
  https://academic.oup.com/mnras/article/441/2/1095/1066551
- Pon et al. (2014), large-scale superbubble geometry:
  https://academic.oup.com/mnras/article/444/4/3657/1028091
- Joubaud et al. (2019), nested H I/CO shells:
  https://arxiv.org/abs/1909.10083
- Haffner et al. (2003), H-alpha intensity to emission-measure calibration:
  https://lorenanderson.faculty.wvu.edu/files/d/a208a023-d139-44c4-abe1-9d50c1bbbf0a/haffner_2003_apjs_149_405.pdf

The 5--25 R bright arcs can be optically thick individually at 1 MHz, but that
does not make a degree-scale pixel an 8000-K black screen. The implementation
ray-traces 5--12 pc walls on a dedicated 2-pc local grid, retains 0.1--0.45
sub-beam filling fractions in selected arc pixels, and averages transfer as
`1 - f + f exp(-tau)`. Its effective coverage is constrained to 2--5 percent
within the broad 35-degree validation aperture; seven percent is retained only
as a hard upper envelope motivated by the 1-degree WHAM map:
https://lambda.gsfc.nasa.gov/product/foreground/fg_wham_h_alpha_map_get.html
