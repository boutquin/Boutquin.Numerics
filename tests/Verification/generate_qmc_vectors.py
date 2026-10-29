"""QMC vectors — Sobol and Halton via scipy.stats.qmc."""

from __future__ import annotations

from scipy.stats import qmc

from conftest import save_vector


def generate() -> None:
    dim = 4
    points = 16

    # Both reference arrays include author index zero. Native Sobol emits it;
    # native Halton starts at index one, so its verifier selects rows1 onward.
    sobol = qmc.Sobol(d=dim, scramble=False)
    sobol_pts = sobol.random_base2(m=int.bit_length(points) - 1)

    halton = qmc.Halton(d=dim, scramble=False)
    halton_pts = halton.random(points)

    save_vector(
        "qmc",
        {
            "dimension": dim,
            "num_points": points,
            "sobol": sobol_pts.tolist(),
            "halton": halton_pts.tolist(),
            "sobol_dim_means": sobol_pts.mean(axis=0).tolist(),
            "halton_dim_means": halton_pts.mean(axis=0).tolist(),
        },
    )

    # Pin every embedded Sobol direction row and recurrence bits well beyond
    # the first 16 points. These values come directly from SciPy's independent
    # unscrambled Joe-Kuo D(6) implementation, including its origin point.
    extended_indices = [1, 2, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023]
    sobol32 = qmc.Sobol(d=32, scramble=False, bits=30).random_base2(m=10)
    save_vector(
        "qmc_sobol_extended",
        {
            "dimension": 32,
            "indices": extended_indices,
            "sobol": sobol32[extended_indices].tolist(),
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote qmc.json")
