# Literature conditioning for the approximate-Milky-Way realization

This mode conditions uncertain 3D components on measured Galactic geography.
It never reads Haslam, LWA, or ULSA pixels when generating a realization.

## Large-scale Galaxy

- Solar Galactocentric radius and maser spiral-arm constraints:
  Reid et al. (2019), https://arxiv.org/abs/1910.03357
- Solar height above the stellar midplane:
  Bennett & Bovy (2019), https://arxiv.org/abs/1809.03507
- Disk plus extended-halo magnetic geometry:
  Jansson & Farrar (2012), https://arxiv.org/abs/1204.3662
- Radio constraints on the propagated cosmic-ray halo:
  Orlando & Strong (2013),
  https://academic.oup.com/mnras/article/436/3/2127/1246185
- Bar angle and scale:
  Wegg et al. (2015),
  https://academic.oup.com/mnras/article/450/4/4050/989881
- Warp functional form and uncertainty:
  https://academic.oup.com/mnras/article/507/4/5246/6358539

Implementation: R0=8.2 kpc, zSun=15 pc, a smooth outer m=1 warp beginning
near 9 kpc, conservative 25-percent synchrotron arm modulation on the existing
NE2001-like arm ridges, and a 25-percent bar emissivity enhancement. Arm radio
contrast is deliberately weak because low-energy electrons diffuse over kpc
scales and coherent magnetic-arm evidence remains inconclusive; see Unger &
Farrar (2024), https://arxiv.org/abs/2311.12120.

## Named local synchrotron structures

- Loop I two-shell geometry: Wolleben (2007),
  https://arxiv.org/abs/0704.0276
- Loops I--IV directions, 408-MHz contrasts, and GHz spectral indices:
  Borka (2007),
  https://academic.oup.com/mnras/article/376/2/634/1074743
- 22-MHz loop measurements and low-frequency indices:
  https://arxiv.org/abs/1108.3354
- Modern distance constraints and Loop-IV superposition caveat:
  https://arxiv.org/abs/2106.14267

Implementation: five uncertain seeded shells represent Wolleben S1/S2 and
Loops II--IV. Their measured centers are jittered by 1.5 degrees; distances,
radii, thicknesses, brightnesses, spectra, and partial shock coverage are also
sampled. Density-gradient axes are conditioned toward the observed broad arc
sectors, but no angular template or pixel mask is used.

## Conditioned thermal-electron structures

- YMW16 geometry for large-scale electrons, the Local Bubble, Gum, Loop I,
  warp, and Galactic center: https://arxiv.org/abs/1610.09448
- Gum shell geometry, density, and filling factor:
  https://arxiv.org/abs/1502.06296
- Gum low-frequency depression at 2 and 4 MHz:
  https://ntrs.nasa.gov/api/citations/19720004101/downloads/19720004101.pdf
- Orion--Eridanus geometry and distance uncertainty:
  https://arxiv.org/abs/1404.1917 and https://arxiv.org/abs/1909.10083
- Cygnus thermal/nonthermal separation:
  https://arxiv.org/abs/1309.6065
- Modern irregular Local Bubble dimensions:
  https://arxiv.org/abs/2403.04961

Implementation: the Local Bubble is a low-order irregular radial cavity with
a mean wall near 170 pc, ordinary directions spanning roughly 70--220 pc, a
35-pc soft transition, and narrow polar blowouts reaching beyond 600 pc. Gum
and Orion--Eridanus uses broad, ellipsoidal 3D ionized interfaces integrated
on a dedicated 2-pc radial grid. Gum is deliberately simpler: a partial-cover
screen at the Purcell et al. center (l=258 deg, b=-6.6 deg) and 450-pc
distance. The radio-polarization fit's warm-gas filling factor
f=0.3 (+0.3/-0.1) enters its measured EM through EM=n_e^2 f L; it is not a
2D beam-covering fraction. At the 2-degree output scale, multiple clumps along
the shell imply substantial Poisson areal coverage. We infer that coverage
smoothly from effective EM. The parametric morphology is a single circular
Gaussian blob (sigma=11.5 deg) centered on the literature position, with no
closed outer edge, explicit limb, sector, or corrugation. Its
calibrated central EM is about 107 pc cm^-6 and its all-sky EM integral is
constrained to 15--35 pc cm^-6 sr. This keeps the model near the observed
interior EM and the ionized mass implied by the published shell parameters,
without claiming that the confused limb morphology is known. No survey pixels
or hand-selected high-covering sectors enter the model. A smooth Poisson
covering law gives about 0.93 covering at the center while holding the EM
column fixed; the resulting roughly threefold 4-MHz depression is comparable
to the historical low-frequency measurement.
The fitted 22.7-degree circumference is an analytic shell boundary, not a
resolved, uniformly sharp absorption mask. Purcell et al. explicitly restrict
their fit to the clean upper nebula; the Galactic-plane and lower regions are
confused by other HII regions, remnants, and the possibly separate IRAS Vela
Shell. Woermann et al. also find a smaller, asymmetric expanding neutral shell
(10.5-degree mean and 14-degree maximum radius), reinforcing that the phases
and outlines should not be collapsed into one crisp disk.
Orion--Eridanus uses two overlapping, much more sparsely covering shell
pieces across its observed 150--250 pc depth rather than one wall. Cygnus X is
a degree-scale sub-grid surrogate centered at 1.45 kpc and normalized to the
low end of its observed 10^3--10^6 pc cm^-6 distribution; its 4.3-pc filaments
are below the simulator's native resolution. Their
normalizations are intentionally exposed as systematic priors because pulsar
DM constrains integral ne, while absorption depends on integral ne-squared and
therefore on unresolved filling factor.

## Deliberate omissions

- The Fan Region is not a separate total-intensity component; it is primarily
  exceptional in polarization.
- The Cygnus superbubble X-ray outline is not assumed to be a synchrotron shell.
- Fine HII morphology and empirical NE2001 clumps are not imported.
- Ordinary sub-degree SNRs remain statistical at the 2-degree output beam.
