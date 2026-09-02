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
- Orion--Eridanus geometry and distance uncertainty:
  https://arxiv.org/abs/1404.1917 and https://arxiv.org/abs/1909.10083
- Cygnus thermal/nonthermal separation:
  https://arxiv.org/abs/1309.6065
- Modern irregular Local Bubble dimensions:
  https://arxiv.org/abs/2403.04961

Implementation: the Local Bubble is a low-order irregular radial cavity with
a mean wall near 170 pc, ordinary directions spanning roughly 70--220 pc, a
35-pc soft transition, and narrow polar blowouts reaching beyond 600 pc. Gum
and Orion--Eridanus are broad, ellipsoidal 3D ionized
interfaces integrated on a dedicated 2-pc radial grid. Gum uses the
Purcell et al. center (l=258 deg, b=-6.6 deg), 450-pc distance, 160-pc radius,
and 18.5-pc wall. Coherent sectors span roughly 0.2--0.98 projected covering
at the 2-degree output scale, with the high-covering area comparable to the
observed clean upper/central shell, and bright sectors in the measured
220--470 pc cm^-6 range.
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
