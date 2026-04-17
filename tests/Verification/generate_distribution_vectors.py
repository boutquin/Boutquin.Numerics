"""Reference Φ(x) and Φ⁻¹(p) vectors from scipy.stats.norm."""

from __future__ import annotations

import numpy as np
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

    save_vector(
        "distributions",
        {
            "cdf": {"x": cdf_points, "y": cdf_values},
            "ppf": {"p": ppf_points, "y": ppf_values},
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote distributions.json")
