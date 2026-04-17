"""Orchestrator — regenerate every JSON vector used by the C# verification tests."""

from __future__ import annotations

from generate_bootstrap_vectors import generate as gen_bootstrap
from generate_correlation_vectors import generate as gen_correlation
from generate_covariance_vectors import generate as gen_covariance
from generate_distribution_vectors import generate as gen_distributions
from generate_dsr_vectors import generate as gen_dsr
from generate_interpolation_vectors import generate as gen_interpolation
from generate_linalg_vectors import generate as gen_linalg
from generate_lm_vectors import generate as gen_lm
from generate_ols_vectors import generate as gen_ols
from generate_psd_vectors import generate as gen_psd
from generate_qmc_vectors import generate as gen_qmc
from generate_qp_solver_vectors import generate as gen_qp_solver
from generate_sample_moments_vectors import generate as gen_sample_moments
from generate_scalar_stats_vectors import generate as gen_scalar_stats
from generate_solver_vectors import generate as gen_solvers


def main() -> None:
    gen_distributions()
    gen_solvers()
    gen_ols()
    gen_lm()
    gen_interpolation()
    gen_linalg()
    gen_covariance()
    gen_correlation()
    gen_dsr()
    gen_bootstrap()
    gen_scalar_stats()
    gen_qmc()
    gen_psd()
    gen_sample_moments()
    gen_qp_solver()
    print("All vectors regenerated under tests/Verification/vectors/")


if __name__ == "__main__":
    main()
