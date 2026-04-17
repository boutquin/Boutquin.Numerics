"""QMC vectors — Sobol and Halton via scipy.stats.qmc."""

from __future__ import annotations

from scipy.stats import qmc

from conftest import save_vector


def generate() -> None:
    dim = 4
    points = 16

    # Skip-0 Sobol sequence (SciPy emits the [0, 0, ..., 0] point first
    # when scramble=False; the reference C# implementation follows the
    # Joe-Kuo convention which starts at (1/2, ...) for the first non-
    # origin point. We compare the discrepancy and the per-dimension
    # empirical mean — both should converge to 0.5 as points grow.)
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


if __name__ == "__main__":
    generate()
    print("Wrote qmc.json")
