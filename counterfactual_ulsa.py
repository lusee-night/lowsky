"""Backward-compatible imports for the :mod:`lowsky.pipeline` module.

New code should import from :mod:`lowsky` or :mod:`lowsky.pipeline`.
"""

from lowsky.pipeline import *  # noqa: F401,F403
from lowsky.pipeline import (  # noqa: F401
    _smooth_broken_power_law_scale,
    _tau_coefficient,
    main,
)


if __name__ == "__main__":
    main()
