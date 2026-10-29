"""Discriminators for an independent batch oracle, not streaming recurrences."""
from __future__ import annotations

from generate_welford_pca_vectors import welford_merge, welford_moments, welford_pearson


def test_exact_batch_mean_survives_cancellation():
    result = welford_moments([1e16, 1.0, -1e16])
    assert result['mean'] == 1 / 3, result


def test_merge_oracle_recomputes_combined_sample():
    result = welford_merge([1e16], [1.0, -1e16])
    assert result['merged_mean'] == 1 / 3, result
    assert result['full_mean'] == 1 / 3, result


def test_exact_batch_correlation_preserves_tiny_variance():
    result = welford_pearson([1e-200, 2e-200, 3e-200], [-1e-200, -2e-200, -3e-200])
    assert result['correlation'] == -1, result
