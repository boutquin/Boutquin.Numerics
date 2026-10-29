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

using Boutquin.Numerics.LinearAlgebra;
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Verification;

/// <summary>
/// Python cross-checks for <see cref="WelfordMoments{T}"/> and
/// <see cref="PrincipalComponentAnalysis{T}"/>. Reference values come from
/// <c>tests/Verification/generate_welford_pca_vectors.py</c>, which is a
/// batch central-moment oracle using exact rational input values for Welford,
/// and the independent numpy.linalg.eigh solver for PCA.
/// </summary>
public sealed class WelfordPcaVerificationTests : CrossLanguageVerificationBase
{
    // ---------------------------------------------------------------
    //  WelfordMoments — univariate, bivariate, merge
    // ---------------------------------------------------------------

    /// <summary>
    /// Small integer series [1, 2, 3, 4, 5]: mean = 3, sample variance = 2.5,
    /// population variance = 2.0. Both the Python port and C# Welford produce
    /// identical results — the arithmetic is exact in double precision for these
    /// inputs. Tolerance PrecisionExact (1e-10) verifies there is no algorithmic
    /// divergence.
    /// </summary>
    [Fact]
    public void WelfordMoments_SmallIntegerSeries_MatchesReference()
    {
        using var doc = LoadVector("welford_pca");
        var sec = doc.RootElement.GetProperty("welford_moments").GetProperty("small_integer_series");
        var series = GetDoubleArray(sec.GetProperty("series"));

        var w = new WelfordMoments<double>();
        foreach (var x in series)
        {
            w.Add(x);
        }

        AssertScalarWithin(w.Mean, sec.GetProperty("mean").GetDouble(), PrecisionExact, "mean");
        AssertScalarWithin(w.Variance, sec.GetProperty("variance").GetDouble(), PrecisionExact, "variance");
        AssertScalarWithin(w.PopulationVariance, sec.GetProperty("population_variance").GetDouble(), PrecisionExact, "pop variance");
        AssertScalarWithin(w.StdDev, sec.GetProperty("std_dev").GetDouble(), PrecisionExact, "std dev");
        w.Count.Should().Be(sec.GetProperty("count").GetInt64());
    }

    /// <summary>
    /// Fifty-element float series drawn from a normal distribution. The
    /// Python reference is an exact port of the C# Welford loop — both run
    /// in IEEE 754 double precision, so the results are bit-identical to
    /// within a few ULP. Tolerance PrecisionExact (1e-10).
    /// </summary>
    [Fact]
    public void WelfordMoments_FloatSeries_MatchesReference()
    {
        using var doc = LoadVector("welford_pca");
        var sec = doc.RootElement.GetProperty("welford_moments").GetProperty("float_series");
        var series = GetDoubleArray(sec.GetProperty("series"));

        var (mean, variance) = WelfordMoments<double>.Compute(series);

        AssertScalarWithin(mean, sec.GetProperty("mean").GetDouble(), PrecisionExact, "mean");
        AssertScalarWithin(variance, sec.GetProperty("variance").GetDouble(), PrecisionExact, "variance");
    }

    /// <summary>
    /// Bivariate Pearson accumulator on 80 paired observations. Verifies
    /// sample covariance (N-1 divisor) and Pearson correlation coefficient
    /// against the Python Bennett-2009-form bivariate Welford port.
    /// Tolerance PrecisionExact (1e-10) — same algorithm, same precision.
    /// </summary>
    [Fact]
    public void WelfordMoments_BivariatePearson_MatchesReference()
    {
        using var doc = LoadVector("welford_pca");
        var sec = doc.RootElement.GetProperty("welford_moments").GetProperty("bivariate_pearson");
        var xs = GetDoubleArray(sec.GetProperty("x"));
        var ys = GetDoubleArray(sec.GetProperty("y"));

        var pearson = new WelfordMoments<double>.Pearson();
        for (var i = 0; i < xs.Length; i++)
        {
            pearson.Add(xs[i], ys[i]);
        }

        AssertScalarWithin(pearson.Covariance, sec.GetProperty("covariance").GetDouble(), PrecisionExact, "covariance");
        AssertScalarWithin(pearson.Correlation, sec.GetProperty("correlation").GetDouble(), PrecisionExact, "correlation");
        pearson.Count.Should().Be(sec.GetProperty("count").GetInt64());
    }

    [Fact]
    public void WelfordMoments_PearsonLargeDecimalTranslation_PreservesExactCovariance()
    {
        const decimal offset = 10000000000000000000000000000m;
        var centered = new WelfordMoments<decimal>.Pearson();
        var translated = new WelfordMoments<decimal>.Pearson();
        foreach (var residual in new decimal[] { -1m, 0m, 1m })
        {
            centered.Add(residual, residual);
            translated.Add(offset + residual, offset + residual);
        }

        centered.Covariance.Should().Be(1m);
        translated.Covariance.Should().Be(1m);
        translated.Covariance.Should().Be(centered.Covariance);
    }

    /// <summary>
    /// Chan-Golub-LeVeque merge: partitions A (first 30) and B (last 20) of the
    /// float series are accumulated separately then merged. The merged mean and
    /// variance must equal the full-series result to PrecisionExact (1e-10).
    /// The Python reference applies the same merge formula independently, so any
    /// implementation divergence is immediately visible.
    /// </summary>
    [Fact]
    public void WelfordMoments_Merge_MatchesFullSeriesReference()
    {
        using var doc = LoadVector("welford_pca");
        var sec = doc.RootElement.GetProperty("welford_moments").GetProperty("merge_consistency");
        var partA = GetDoubleArray(sec.GetProperty("partition_a"));
        var partB = GetDoubleArray(sec.GetProperty("partition_b"));

        var wA = new WelfordMoments<double>();
        foreach (var x in partA)
        {
            wA.Add(x);
        }

        var wB = new WelfordMoments<double>();
        foreach (var x in partB)
        {
            wB.Add(x);
        }

        var merged = WelfordMoments<double>.Combine(wA, wB);

        AssertScalarWithin(merged.Mean, sec.GetProperty("merged_mean").GetDouble(), PrecisionExact, "merged mean");
        AssertScalarWithin(merged.Variance, sec.GetProperty("merged_variance").GetDouble(), PrecisionExact, "merged variance");

        // Also verify that merged == full-series reference.
        AssertScalarWithin(merged.Mean, sec.GetProperty("full_mean").GetDouble(), PrecisionExact, "merged == full mean");
        AssertScalarWithin(merged.Variance, sec.GetProperty("full_variance").GetDouble(), PrecisionExact, "merged == full variance");
    }

    /// <summary>
    /// Verify the decimal accumulator against the independent reference:
    /// the small-integer series matches exactly (decimal arithmetic is exact here).
    /// </summary>
    [Fact]
    public void WelfordMoments_Decimal_SmallIntegerSeries_MatchesReference()
    {
        using var doc = LoadVector("welford_pca");
        var sec = doc.RootElement.GetProperty("welford_moments").GetProperty("small_integer_series");

        var w = new WelfordMoments<decimal>();
        foreach (var v in sec.GetProperty("series").EnumerateArray())
        {
            w.Add((decimal)v.GetDouble());
        }

        AssertScalarWithin((double)w.Mean, sec.GetProperty("mean").GetDouble(), PrecisionExact, "decimal mean");
        AssertScalarWithin((double)w.Variance, sec.GetProperty("variance").GetDouble(), PrecisionExact, "decimal variance");
        AssertScalarWithin((double)w.PopulationVariance, sec.GetProperty("population_variance").GetDouble(), PrecisionExact, "decimal pop variance");
    }

    // ---------------------------------------------------------------
    //  PrincipalComponentAnalysis — Decompose + FromReturns + Project
    // ---------------------------------------------------------------

    /// <summary>
    /// 3×3 covariance matrix with well-separated eigenvalues. Eigenvalues
    /// (sorted descending), eigenvector columns (sign convention applied),
    /// and explained-variance vectors are all cross-checked against the
    /// numpy.linalg.eigh reference. Tolerance PrecisionStatistical (1e-4) —
    /// Jacobi iteration vs LAPACK dsyevd can diverge by ~1e-5 on these inputs
    /// (matching the established JacobiEigenvalues tolerance).
    /// </summary>
    [Fact]
    public void Pca_Decompose_3x3_EigenvaluesMatchReference()
    {
        using var doc = LoadVector("welford_pca");
        var sec = doc.RootElement.GetProperty("pca").GetProperty("decompose_3x3");
        var cov = GetDecimal2D(sec.GetProperty("covariance_matrix"));
        var expectedEigvals = GetDoubleArray(sec.GetProperty("eigenvalues"));

        var result = PrincipalComponentAnalysis.Decompose(cov);

        for (var k = 0; k < expectedEigvals.Length; k++)
        {
            AssertScalarWithin((double)result.Eigenvalues[k], expectedEigvals[k], PrecisionStatistical, $"eigenvalue[{k}]");
        }
    }

    /// <summary>
    /// Eigenvector columns after the sign convention: for each column, the
    /// component with the largest absolute value is positive. With well-separated
    /// eigenvalues the eigenvectors are unique up to sign, and the convention
    /// resolves that ambiguity the same way on both sides.
    /// </summary>
    [Fact]
    public void Pca_Decompose_3x3_EigenvectorsMatchReference()
    {
        using var doc = LoadVector("welford_pca");
        var sec = doc.RootElement.GetProperty("pca").GetProperty("decompose_3x3");
        var cov = GetDecimal2D(sec.GetProperty("covariance_matrix"));
        var n = cov.GetLength(0);

        // eigenvectors[row][col] — columns are PC loadings.
        var expectedCols = sec.GetProperty("eigenvectors")
            .EnumerateArray()
            .Select(row => row.EnumerateArray().Select(e => e.GetDouble()).ToArray())
            .ToArray();

        var result = PrincipalComponentAnalysis.Decompose(cov);

        for (var row = 0; row < n; row++)
        {
            for (var col = 0; col < n; col++)
            {
                AssertScalarWithin(
                    (double)result.Eigenvectors[row, col],
                    expectedCols[row][col],
                    PrecisionStatistical,
                    $"eigenvector[{row},{col}]");
            }
        }
    }

    /// <summary>
    /// Explained-variance ratio and cumulative explained variance for the 3×3
    /// case. The last cumulative entry must equal 1.0 (up to rounding).
    /// </summary>
    [Fact]
    public void Pca_Decompose_3x3_ExplainedVarianceMatchesReference()
    {
        using var doc = LoadVector("welford_pca");
        var sec = doc.RootElement.GetProperty("pca").GetProperty("decompose_3x3");
        var cov = GetDecimal2D(sec.GetProperty("covariance_matrix"));
        var expectedRatio = GetDoubleArray(sec.GetProperty("explained_variance_ratio"));
        var expectedCumulative = GetDoubleArray(sec.GetProperty("cumulative_explained_variance"));

        var result = PrincipalComponentAnalysis.Decompose(cov);

        for (var k = 0; k < expectedRatio.Length; k++)
        {
            AssertScalarWithin((double)result.ExplainedVarianceRatio[k], expectedRatio[k], PrecisionStatistical, $"ratio[{k}]");
            AssertScalarWithin((double)result.CumulativeExplainedVariance[k], expectedCumulative[k], PrecisionStatistical, $"cumulative[{k}]");
        }

        // Last cumulative entry must be 1 (full variance accounted for).
        var last = result.CumulativeExplainedVariance[^1];
        AssertScalarWithin((double)last, 1.0, PrecisionStatistical, "cumulative[-1] == 1");
    }

    /// <summary>
    /// FromReturns on a 50×4 return matrix: per-column means match numpy
    /// column-mean to PrecisionExact (1e-10) — mean is pure arithmetic with
    /// no approximation involved.
    /// </summary>
    [Fact]
    public void Pca_FromReturns_50x4_MeansMatchReference()
    {
        using var doc = LoadVector("welford_pca");
        var sec = doc.RootElement.GetProperty("pca").GetProperty("from_returns_50x4");
        var returns = GetDecimal2D(sec.GetProperty("returns"));
        var expectedMean = GetDoubleArray(sec.GetProperty("mean"));

        var result = PrincipalComponentAnalysis.FromReturns(returns);

        for (var j = 0; j < expectedMean.Length; j++)
        {
            AssertScalarWithin((double)result.Mean[j], expectedMean[j], PrecisionExact, $"mean[{j}]");
        }
    }

    /// <summary>
    /// FromReturns eigenvalues on the 50×4 matrix cross-checked against numpy.
    /// PC1 carries ~47% of variance; PC4 carries ~8%. PrecisionStatistical (1e-4)
    /// matches the Jacobi eigenvalue tolerance established in LinearAlgebraVerificationTests.
    /// </summary>
    [Fact]
    public void Pca_FromReturns_50x4_EigenvaluesMatchReference()
    {
        using var doc = LoadVector("welford_pca");
        var sec = doc.RootElement.GetProperty("pca").GetProperty("from_returns_50x4");
        var returns = GetDecimal2D(sec.GetProperty("returns"));
        var expectedEigvals = GetDoubleArray(sec.GetProperty("eigenvalues"));

        var result = PrincipalComponentAnalysis.FromReturns(returns);

        for (var k = 0; k < expectedEigvals.Length; k++)
        {
            AssertScalarWithin((double)result.Eigenvalues[k], expectedEigvals[k], PrecisionStatistical, $"eigenvalue[{k}]");
        }
    }

    /// <summary>
    /// FromReturns explained-variance ratios on the 50×4 matrix. Sum of all
    /// ratios must equal 1; each ratio must match the numpy reference to
    /// PrecisionStatistical (1e-4).
    /// </summary>
    [Fact]
    public void Pca_FromReturns_50x4_ExplainedVarianceMatchesReference()
    {
        using var doc = LoadVector("welford_pca");
        var sec = doc.RootElement.GetProperty("pca").GetProperty("from_returns_50x4");
        var returns = GetDecimal2D(sec.GetProperty("returns"));
        var expectedRatio = GetDoubleArray(sec.GetProperty("explained_variance_ratio"));

        var result = PrincipalComponentAnalysis.FromReturns(returns);

        for (var k = 0; k < expectedRatio.Length; k++)
        {
            AssertScalarWithin((double)result.ExplainedVarianceRatio[k], expectedRatio[k], PrecisionStatistical, $"ratio[{k}]");
        }

        // Ratios must sum to 1.
        var ratioSum = result.ExplainedVarianceRatio.Aggregate(0m, (acc, v) => acc + v);
        AssertScalarWithin((double)ratioSum, 1.0, PrecisionStatistical, "ratio sum == 1");
    }

    /// <summary>
    /// Project scores onto top-2 PCs: (X − mean) · V[:, 0:2]. The Python
    /// reference computes the same matrix multiply using the same eigenvectors
    /// and the same mean; scores match to PrecisionStatistical (1e-4) because
    /// the projection inherits the eigenvector tolerance.
    /// </summary>
    [Fact]
    public void Pca_FromReturns_50x4_ProjectTop2_MatchesReference()
    {
        using var doc = LoadVector("welford_pca");
        var sec = doc.RootElement.GetProperty("pca").GetProperty("from_returns_50x4");
        var returns = GetDecimal2D(sec.GetProperty("returns"));

        // expectedScores is a list of 50 rows, each with 2 columns.
        var expectedRows = sec.GetProperty("scores_top2")
            .EnumerateArray()
            .Select(row => row.EnumerateArray().Select(e => e.GetDouble()).ToArray())
            .ToArray();

        var result = PrincipalComponentAnalysis.FromReturns(returns);
        var scores = result.Project(returns, numComponents: 2);

        for (var t = 0; t < expectedRows.Length; t++)
        {
            for (var k = 0; k < 2; k++)
            {
                AssertScalarWithin(
                    (double)scores[t, k],
                    expectedRows[t][k],
                    PrecisionStatistical,
                    $"score[{t},{k}]");
            }
        }
    }
}
