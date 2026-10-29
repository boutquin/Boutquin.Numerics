// Copyright (c) 2026 Pierre G. Boutquin. All rights reserved.
//
//   Licensed under the Apache License, Version 2.0 (the "License").
//   You may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//
//   See the License for the specific language governing permissions and
//   limitations under the License.
//

using Boutquin.Numerics.Statistics;

namespace Boutquin.Numerics.Tests.Verification;

public sealed class CovarianceVerificationTests : CrossLanguageVerificationBase
{
    // ---------------------------------------------------------------------
    //  Legacy top-level tests — unchanged (T=120, N=5 block).
    // ---------------------------------------------------------------------

    [Fact]
    public void SampleCovariance_MatchesNumpyCov()
    {
        using var doc = LoadVector("covariance");
        var returns = GetDecimal2D(doc.RootElement.GetProperty("returns"));
        var expected = GetDouble2D(doc.RootElement.GetProperty("sample_covariance"));

        var actual = new SampleCovarianceEstimator().Estimate(returns);

        AssertMatrixWithin(actual, expected, (decimal)PrecisionNumeric, "sample_cov");
    }

    [Fact]
    public void LedoitWolf_MatchesOwnFormulaReference()
    {
        using var doc = LoadVector("covariance");
        var returns = GetDecimal2D(doc.RootElement.GetProperty("returns"));
        var expected = GetDouble2D(doc.RootElement.GetProperty("ledoit_wolf_scaled_identity"));

        var actual = new LedoitWolfShrinkageEstimator().Estimate(returns);

        // Own-formula match: tight numeric tier.
        AssertMatrixWithin(actual, expected, (decimal)PrecisionNumeric, "lw_scaled_identity");
    }

    [Fact]
    public void Ewma_MatchesRiskMetricsFormula()
    {
        using var doc = LoadVector("covariance");
        var returns = GetDecimal2D(doc.RootElement.GetProperty("returns"));
        var expected = GetDouble2D(doc.RootElement.GetProperty("ewma_lambda_0_94"));

        var actual = new ExponentiallyWeightedCovarianceEstimator(0.94m).Estimate(returns);

        AssertMatrixScaleRelative(
            actual, expected, relativeTolerance: 1e-10m, absoluteFloor: 1e-16m, label: "ewma");
    }

    // ---------------------------------------------------------------------
    //  Phase 2 additions — one test per estimator, parameterised over the
    //  three data regimes from spec §2.2. Regimes:
    //    * small_well_conditioned: T=120, N=8  (concentration ≈ 0.067)
    //    * moderate:               T=60,  N=30 (concentration 0.5)
    //    * overconcentrated:       T=40,  N=50 (concentration 1.25, T < N)
    //
    //  Per-estimator `[Theory]` chosen over a single mega-Theory so that
    //  failure messages name the estimator directly (§6 Decision Rules:
    //  per-regime rows are fine within one Theory — prefer per-estimator
    //  Fact/Theory, not 30-row mega-Theory).
    // ---------------------------------------------------------------------

    public static TheoryData<string> Regimes => new()
    {
        "small_well_conditioned",
        "moderate",
        "overconcentrated",
    };

    public static TheoryData<string> OasRegimes => new()
    {
        "small_well_conditioned",
        "small_scale_oas",
        "moderate",
        "overconcentrated",
    };

    // The full candidate grid includes rank-deficient first subsets in the larger
    // regimes. Their null-space eigenvectors are non-unique, and reconstructed
    // NERCOME matrices can depend on that basis. The well-conditioned panel keeps
    // every candidate subset full rank. This NumPy comparison checks translation
    // and deterministic sampling policy; exact projector unit tests independently
    // check the estimator equations on exhaustive small panels.
    public static TheoryData<string> NercomeRegimes => new()
    {
        "small_well_conditioned",
    };

    /// <summary>
    /// Precision tiers for the covariance verification tests (spec §2.2).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Shrinkage-only estimators (OAS, LW-CC, LW-SF) do NOT call an
    /// eigendecomposition; they agree with the Python reference to the
    /// decimal→double round-trip bar of <c>1e-10</c> relative.
    /// </para>
    /// <para>
    /// Eigendecomposition-based estimators (Denoised, Detoned, TW, QIS,
    /// POET, DSC) cap at roughly <c>1e-5</c> relative agreement. The
    /// algorithmic step is bit-perfect, but Numerics' Jacobi rotation
    /// routine and numpy's LAPACK (<c>dsyevr</c>) are different iterative
    /// algorithms with different convergence bases within near-degenerate
    /// eigenvalue clusters. The reconstruction
    /// <c>V · diag(λ_cleaned) · Vᵀ</c> is sign-and-permutation invariant
    /// but absorbs the <c>~1e-13</c> per-entry eigenvector drift,
    /// amplifying it by the eigenvalue magnitude.
    /// </para>
    /// <para>
    /// POET tolerates slightly looser <c>1e-4</c> relative agreement
    /// because thresholding introduces hard-step nonlinearity at the
    /// residual level; small LAPACK noise can push a near-threshold
    /// residual entry across the <c>τ</c> cut, producing a larger
    /// relative-error amplification near the threshold boundary.
    /// </para>
    /// </remarks>
    private const decimal ShrinkageTolerance = 1e-10m;

    private const decimal EigenDecompTolerance = 1e-5m;

    private const decimal PoetTolerance = 1e-4m;

    // Nercome off-diagonals amplify eigendecomposition drift through the
    // Vᵀ·cov2·V rotation. Noise-level off-diagonals (|entry| ≈ 1e-8) can drift
    // by up to ~10% relative between LAPACK and Jacobi; the algorithm is
    // bit-perfect given the same V. The tolerance is a per-entry relative
    // bound; the verification harness applies an absolute-tolerance floor at
    // 1e-8 so noise-level entries are compared against an absolute threshold
    // rather than a spurious relative-error explosion.
    private const decimal NercomeTolerance = 1e-1m;

    [Theory]
    [MemberData(nameof(OasRegimes))]
    public void Oas_MatchesPaperReference(string regime)
    {
        RunRegime(regime, "oas", new OracleApproximatingShrinkageEstimator(), ShrinkageTolerance);
    }

    [Theory]
    [MemberData(nameof(Regimes))]
    public void LwConstantCorrelation_MatchesPaperReference(string regime)
    {
        RunRegime(regime, "lw_constant_correlation", new LedoitWolfConstantCorrelationEstimator(), ShrinkageTolerance);
    }

    [Theory]
    [MemberData(nameof(Regimes))]
    public void LwSingleFactor_MatchesPaperReference(string regime)
    {
        RunRegime(regime, "lw_single_factor", new LedoitWolfSingleFactorEstimator(), ShrinkageTolerance);
    }

    /// <summary>
    /// Translation-parity check, not paper fidelity: the Denoised golden in
    /// <c>generate_covariance_vectors.py</c> is a numpy port of this library's own
    /// pipeline, so agreement shows the C# and the port compute the same thing — it
    /// cannot show that either matches the cited paper.
    /// </summary>
    [Theory]
    [MemberData(nameof(Regimes))]
    public void Denoised_MatchesPortedPipeline(string regime)
    {
        RunRegime(regime, "denoised", new DenoisedCovarianceEstimator(), EigenDecompTolerance);
    }

    /// <summary>
    /// Translation-parity check, not paper fidelity: the Detoned golden in
    /// <c>generate_covariance_vectors.py</c> is a numpy port of this library's own
    /// pipeline, so agreement shows the C# and the port compute the same thing — it
    /// cannot show that either matches the cited paper.
    /// </summary>
    [Theory]
    [MemberData(nameof(Regimes))]
    public void Detoned_MatchesPortedPipeline(string regime)
    {
        RunRegime(regime, "detoned", new DetonedCovarianceEstimator(detoningAlpha: 1.0m), EigenDecompTolerance);
    }

    /// <summary>
    /// Translation-parity check, not paper fidelity: the Tracy-Widom denoised golden in
    /// <c>generate_covariance_vectors.py</c> is a numpy port of this library's own
    /// pipeline, so agreement shows the C# and the port compute the same thing — it
    /// cannot show that either matches the cited paper.
    /// </summary>
    [Theory]
    [MemberData(nameof(Regimes))]
    public void TracyWidomDenoised_MatchesPortedPipeline(string regime)
    {
        RunRegime(regime, "tracy_widom_denoised", new TracyWidomDenoisedCovarianceEstimator(), EigenDecompTolerance);
    }

    /// <summary>
    /// Translation-parity check, not paper fidelity: the QIS golden in
    /// <c>generate_covariance_vectors.py</c> is a numpy port of this library's own
    /// pipeline, so agreement shows the C# and the port compute the same thing — it
    /// cannot show that either matches the cited paper.
    /// </summary>
    [Theory]
    [MemberData(nameof(Regimes))]
    public void QuadraticInverseShrinkage_MatchesPortedPipeline(string regime)
    {
        RunRegime(regime, "qis", new QuadraticInverseShrinkageEstimator(), EigenDecompTolerance);
    }

    [Theory]
    [MemberData(nameof(NercomeRegimes))]
    public void Nercome_MatchesPaperReference(string regime)
    {
        RunRegime(regime, "nercome", new NercomeCovarianceEstimator(), NercomeTolerance);
    }

    /// <summary>
    /// Paper fidelity: the POET golden is derived directly from Fan, Liao &amp; Mincheva
    /// (2013) — factors by PCA on the T×T matrix and the eq. 3.2 adaptive threshold —
    /// by a different computational route from the C# implementation.
    /// </summary>
    [Theory]
    [MemberData(nameof(Regimes))]
    public void Poet_MatchesPaperReference(string regime)
    {
        RunRegime(regime, "poet", new PoetCovarianceEstimator(numFactors: 1, thresholdMultiplier: 0.5), PoetTolerance);
    }

    /// <summary>
    /// Translation-parity check, not paper fidelity: the doubly sparse golden in
    /// <c>generate_covariance_vectors.py</c> is a numpy port of this library's own
    /// pipeline, so agreement shows the C# and the port compute the same thing — it
    /// cannot show that either matches the cited paper.
    /// </summary>
    [Theory]
    [MemberData(nameof(Regimes))]
    public void DoublySparse_MatchesPortedPipeline(string regime)
    {
        RunRegime(regime, "doubly_sparse", new DoublySparseEstimator(eigenvectorThreshold: 0.1m), EigenDecompTolerance);
    }

    /// <summary>
    /// Loads the named regime and estimator reference, runs the C# estimator,
    /// and asserts per-entry relative agreement at the supplied tolerance tier
    /// (see <c>ShrinkageTolerance</c> / <c>EigenDecompTolerance</c> /
    /// <c>PoetTolerance</c> / <c>NercomeTolerance</c>).
    /// </summary>
    private static void RunRegime(string regime, string estimatorKey, ICovarianceEstimator estimator, decimal tolerance)
    {
        using var doc = LoadVector("covariance");
        var regimeElement = doc.RootElement.GetProperty("regimes").GetProperty(regime);

        var returns = GetDecimal2D(regimeElement.GetProperty("returns"));
        var expected = GetDouble2D(regimeElement.GetProperty(estimatorKey));

        var actual = estimator.Estimate(returns);

        AssertMatrixRelative(actual, expected, tolerance, $"{estimatorKey}/{regime}");
    }

    /// <summary>
    /// Relative matrix comparison with absolute-fallback denominator. The
    /// fallback prevents a noise-level <c>|expected| ≈ 1e-14</c> entry from
    /// dominating the relative-error statistic; those entries are asserted at
    /// absolute tolerance instead.
    /// </summary>
    private static void AssertMatrixRelative(decimal[,] actual, double[,] expected, decimal tolerance, string label)
    {
        Assert.Equal(expected.GetLength(0), actual.GetLength(0));
        Assert.Equal(expected.GetLength(1), actual.GetLength(1));
        const decimal absoluteFloor = 1e-8m;
        for (var i = 0; i < expected.GetLength(0); i++)
        {
            for (var j = 0; j < expected.GetLength(1); j++)
            {
                var expectedEntry = (decimal)expected[i, j];
                var diff = Math.Abs(actual[i, j] - expectedEntry);
                var denominator = Math.Max(Math.Abs(expectedEntry), absoluteFloor);
                var relError = diff / denominator;
                Assert.True(
                    relError <= tolerance,
                    $"{label}[{i},{j}]: expected {expectedEntry}, got {actual[i, j]}, " +
                    $"|Δ|/max(|expected|,{absoluteFloor}) = {relError} > tol = {tolerance}");
            }
        }
    }

    private static void AssertMatrixScaleRelative(
        decimal[,] actual,
        double[,] expected,
        decimal relativeTolerance,
        decimal absoluteFloor,
        string label)
    {
        Assert.Equal(expected.GetLength(0), actual.GetLength(0));
        Assert.Equal(expected.GetLength(1), actual.GetLength(1));
        for (var i = 0; i < expected.GetLength(0); i++)
        {
            for (var j = 0; j < expected.GetLength(1); j++)
            {
                var expectedValue = (decimal)expected[i, j];
                var scale = Math.Max(Math.Abs(actual[i, j]), Math.Abs(expectedValue));
                var tolerance = Math.Max(absoluteFloor, relativeTolerance * scale);
                AssertScalarWithin(actual[i, j], expectedValue, tolerance, $"{label}[{i},{j}]");
            }
        }
    }
}
