"""Generate a differentiable lowsky realization and export a ULSA drop-in.

Example
-------
python examples/export_ulsa_dropin.py --output lowsky_32.fits
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lowsky.model import SkyParameters, generate_sky
from lowsky.pipeline import SkyConfig, prepare_sky_inputs
from lowsky.ulsa_compat import ULSA_FREQUENCIES_MHZ, write_ulsa_dropin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lowsky_32.fits"),
        help="Drop-in FITS path (default: lowsky_32.fits).",
    )
    parser.add_argument("--seed", type=int, default=SkyConfig.seed)
    parser.add_argument("--n-distance", type=int, default=SkyConfig.n_distance)
    parser.add_argument(
        "--sky-mode",
        choices=("random", "ours"),
        default=SkyConfig.sky_mode,
    )
    parser.add_argument("--emissivity-scale", type=float, default=1.0)
    parser.add_argument("--emission-measure-scale", type=float, default=1.0)
    parser.add_argument("--spectral-index-offset", type=float, default=0.0)
    parser.add_argument("--synchrotron-fluctuation-sigma", type=float, default=0.28)
    parser.add_argument("--emission-measure-fluctuation-sigma", type=float, default=0.32)
    parser.add_argument("--spectral-index-fluctuation-sigma", type=float, default=0.055)
    parser.add_argument("--synchrotron-spectral-curvature", type=float, default=0.0)
    parser.add_argument("--local-shell-scale", type=float, default=1.0)
    parser.add_argument("--local-shell-spectral-index-offset", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SkyConfig(
        nside=32,
        ray_oversample=1,
        seed=args.seed,
        n_distance=args.n_distance,
        sky_mode=args.sky_mode,
    )
    inputs = prepare_sky_inputs(config)
    parameters = SkyParameters(
        emissivity_scale=args.emissivity_scale,
        emission_measure_scale=args.emission_measure_scale,
        spectral_index_offset=args.spectral_index_offset,
        synchrotron_fluctuation_sigma=args.synchrotron_fluctuation_sigma,
        emission_measure_fluctuation_sigma=args.emission_measure_fluctuation_sigma,
        spectral_index_fluctuation_sigma=args.spectral_index_fluctuation_sigma,
        synchrotron_spectral_curvature=args.synchrotron_spectral_curvature,
        local_shell_scale=args.local_shell_scale,
        local_shell_spectral_index_offset=args.local_shell_spectral_index_offset,
    )
    sky = generate_sky(ULSA_FREQUENCIES_MHZ, inputs, parameters)
    output = write_ulsa_dropin(
        args.output,
        sky,
        ULSA_FREQUENCIES_MHZ,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
