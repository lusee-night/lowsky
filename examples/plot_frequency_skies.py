"""Render selected frequencies from a lowsky diagnostic FITS cube."""

from __future__ import annotations

import argparse
from pathlib import Path

import healpy as hp
import matplotlib
import numpy as np
from astropy.io import fits

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits", type=Path)
    parser.add_argument("--output", type=Path, default=Path("sky_frequencies.png"))
    parser.add_argument("--frequencies", type=int, nargs="+", default=[1, 10, 30, 50])
    args = parser.parse_args()

    cube = np.asarray(fits.getdata(args.fits), dtype=float)
    figure = plt.figure(figsize=(15, 8))
    for panel, frequency_mhz in enumerate(args.frequencies, start=1):
        sky_mk = cube[frequency_mhz - 1] / 1.0e6
        hp.mollview(
            np.log10(np.maximum(sky_mk, 1.0e-6)),
            fig=figure.number,
            sub=(2, 2, panel),
            title=f"{frequency_mhz} MHz",
            unit=r"log$_{10}$ brightness [MK]",
            cmap="inferno",
            min=float(np.percentile(np.log10(sky_mk), 1.0)),
            max=float(np.percentile(np.log10(sky_mk), 99.0)),
            xsize=1_600,
        )
        hp.graticule(dpar=30, dmer=45, alpha=0.25)
    figure.suptitle("lowsky counterfactual realization", fontsize=16)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
