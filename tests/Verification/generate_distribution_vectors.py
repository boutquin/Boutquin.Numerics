"""SciPy comparison and independent 200-digit Gaussian-integral accuracy vectors."""

from __future__ import annotations

import numpy as np
import math
import mpmath as mp
from scipy import stats

from conftest import save_vector


def generate() -> None:
    cdf_points = np.linspace(-5.0, 5.0, 41).tolist()
    cdf_values = [float(stats.norm.cdf(x)) for x in cdf_points]

    ppf_points = [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
        0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999,
    ]
    ppf_values = [float(stats.norm.ppf(p)) for p in ppf_points]

    # Preserve binary64 inputs exactly in mpmath. 200 decimal digits resolve
    # erfinv(2p-1) down to p=1e-150 without cancellation in the oracle itself.
    xs = sorted(set([-i / 100 for i in range(801)] +
                    [-3.9, math.nextafter(-4., 0.), -4., -1e-10, 0., .1, 1., 3.9, 8.]))
    ps = sorted(set([10.**(-i) for i in range(1, 151)] +
                    [i / 1000 for i in range(1, 1000)] +
                    [5.12e-5, .02425, math.nextafter(.5, 0.),
                     math.nextafter(.5, 1.), math.nextafter(1., 0.)]))
    with mp.workdps(200):
        high_cdf = [float(mp.erfc(-mp.mpf(x) / mp.sqrt(2)) / 2) for x in xs]
        high_ppf = [float(mp.sqrt(2) * mp.erfinv(2 * mp.mpf(p) - 1)) for p in ps]

    save_vector(
        "distributions",
        {
            "cdf": {"x": cdf_points, "y": cdf_values},
            "ppf": {"p": ppf_points, "y": ppf_values},
            "high_precision": {
                "oracle": "mpmath erfc/erfinv at 200 decimal digits; exact binary64 inputs",
                "cdf": {"x": xs, "y": high_cdf},
                "ppf": {"p": ps, "y": high_ppf},
            },
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote distributions.json")
