"""Regenerate computed JSON vectors under vectors/.

jarre_author_20251029.json is a retained upstream-author snapshot, not computed
by this tool. QP corpus files have their separate qp-corpus/generate_corpus.py
entry point. Printed-only reference scripts do not publish JSON vectors.
"""

from __future__ import annotations

from generate_bootstrap_vectors import generate as gen_bootstrap
from generate_correlation_vectors import generate as gen_correlation
from generate_covariance_vectors import generate as gen_covariance
from generate_distribution_vectors import generate as gen_distributions
from generate_dsr_vectors import generate as gen_dsr
from generate_interpolation_vectors import generate as gen_interpolation
from generate_linalg_vectors import generate as gen_linalg
from generate_lm_vectors import generate as gen_lm
from generate_lower_tail_vectors import generate as gen_lower_tail
from generate_nelder_mead_vectors import generate as gen_nelder_mead
from generate_ols_vectors import generate as gen_ols
from generate_psd_vectors import generate as gen_psd
from generate_qmc_vectors import generate as gen_qmc
from generate_qp_solver_vectors import generate as gen_qp_solver
from generate_sample_moments_vectors import generate as gen_sample_moments
from generate_scalar_stats_vectors import generate as gen_scalar_stats
from generate_haircut_fisher_vectors import generate as gen_haircut_fisher
from generate_solver_vectors import generate as gen_solvers
from generate_welford_pca_vectors import generate as gen_welford_pca
from generate_garch_vectors import generate as gen_garch
from generate_graphical_lasso_vectors import generate as gen_graphical_lasso
from generate_jarre_spline_vectors import generate as gen_jarre
from generate_ppw2009_vectors import generate as gen_ppw2009
from generate_rmt_paper_vectors import main as gen_rmt
from generate_garch_independent_qualification import generate as gen_garch_independent
from check_lwsf_spike_precision import generate as gen_lwsf_spike
from generate_covariance_vectors import refresh_selected_contract_vectors


def main() -> None:
    gen_distributions()
    gen_solvers()
    gen_ols()
    gen_lm()
    gen_nelder_mead()
    gen_interpolation()
    gen_linalg()
    gen_covariance()
    refresh_selected_contract_vectors()
    gen_correlation()
    gen_dsr()
    gen_bootstrap()
    gen_scalar_stats()
    gen_qmc()
    gen_psd()
    gen_sample_moments()
    gen_qp_solver()
    gen_haircut_fisher()
    gen_welford_pca()
    gen_garch()
    gen_garch_independent()
    gen_graphical_lasso()
    gen_lower_tail()
    gen_jarre()
    gen_ppw2009()
    gen_rmt()
    gen_lwsf_spike()
    print("Computed vectors regenerated under tests/Verification/vectors/; retained author snapshot unchanged")


if __name__ == "__main__":
    main()
