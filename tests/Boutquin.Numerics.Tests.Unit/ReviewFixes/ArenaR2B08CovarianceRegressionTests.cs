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
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Regression tests for the covariance-estimator constructor defaults, the QIS
/// <see cref="Half"/> long-sample overflow, and the legacy interface's PSD contract.
/// </summary>
public sealed class ArenaR2B08CovarianceRegressionTests
{
    [Fact]
    public void Poet_ExplicitDefaultLiteral_UsesDocumentedHalfMultiplier()
    {
        var returns = FactorPanel(observations: 250, assets: 20, strongAssets: 3, seed: 5);
        var documented = new PoetCovarianceEstimator<double>(1, 0.5).Estimate(returns);
        var unthresholded = new PoetCovarianceEstimator<double>(1, 0.0).Estimate(returns);
        MaxAbsDifference(documented, unthresholded).Should().BeGreaterThan(1e-6,
            "the panel must distinguish C = 0.5 from C = 0, or the test below pins nothing");

        var estimator = new PoetCovarianceEstimator<double>(1, default);

        estimator.ThresholdMultiplier.Should().Be(0.5,
            "the parameter is documented to default to 0.5, and the default literal supplies no value");
        MaxAbsDifference(estimator.Estimate(returns), documented).Should().Be(0.0);
    }

    [Fact]
    public void DoublySparse_ExplicitDefaultLiteral_UsesDocumentedTenthThreshold()
    {
        var returns = FactorPanel(observations: 250, assets: 20, strongAssets: 3, seed: 5);
        var documented = new DoublySparseEstimator<double>(0.1).Estimate(returns);
        var unthresholded = new DoublySparseEstimator<double>(0.0).Estimate(returns);
        MaxAbsDifference(documented, unthresholded).Should().BeGreaterThan(1e-9,
            "the panel must have signal-eigenvector entries below 0.1, or the test below pins nothing");

        var viaDefault = new DoublySparseEstimator<double>(default).Estimate(returns);

        MaxAbsDifference(viaDefault, documented).Should().Be(0.0,
            "the eigenvector threshold is documented to default to 0.1, and the default literal supplies no value");
    }

    [Fact]
    public void Nercome_ExplicitDefaultLiteral_UsesDocumentedHalfSplit()
    {
        var construct = () => new NercomeCovarianceEstimator<double>(default);

        construct.Should().NotThrow("the split fraction is documented to default to 0.5");
        construct().SplitFraction.Should().Be(0.5);
    }

    [Fact]
    public void Qis_Half_LongSample_MatchesDoubleInsteadOfOverflowingToNaN()
    {
        // p = 4, T = 60000: the bandwidth h is about 7.3e-4, so the conjugate H-theta is about
        // 342 and its square exceeds Half.MaxValue, while c = p/(T-1) squares below Half's
        // smallest subnormal. The covariance itself is O(0.1).
        const int observations = 60000;
        const int assets = 4;
        var random = new System.Random(3);
        var half = new Half[observations, assets];
        var asDouble = new double[observations, assets];
        for (var i = 0; i < observations; i++)
        {
            var factor = random.NextDouble() - 0.5;
            for (var j = 0; j < assets; j++)
            {
                half[i, j] = (Half)((0.5 * factor) + (random.NextDouble() - 0.5));
                asDouble[i, j] = (double)half[i, j];
            }
        }

        var expected = new QuadraticInverseShrinkageEstimator<double>().Estimate(asDouble);
        var actual = new QuadraticInverseShrinkageEstimator<Half>().Estimate(half);

        var scale = 0.0;
        for (var i = 0; i < assets; i++)
        {
            for (var j = 0; j < assets; j++)
            {
                scale = Math.Max(scale, Math.Abs(expected[i, j]));
            }
        }

        for (var i = 0; i < assets; i++)
        {
            for (var j = 0; j < assets; j++)
            {
                var value = (double)actual[i, j];
                double.IsFinite(value).Should().BeTrue($"entry [{i},{j}] of a representable O(0.1) covariance must be finite");
                Math.Abs(value - expected[i, j]).Should().BeLessThan(0.01 * scale,
                    $"entry [{i},{j}] must agree with the double estimate to Half's precision");
            }
        }
    }

    [Fact]
    public void LegacyInterfaceDocs_StateThePsdPropertiesOfTheCurrentAlgorithms()
    {
        // Denoised, Detoned and Tracy-Widom rescale the cleaned correlation to a unit diagonal
        // (a congruence by a positive diagonal matrix, which preserves PSD); Detoned subtracts
        // the market component, so its output is singular; QIS is positive definite whenever
        // the non-null sample eigenvalues are. The defect is the stale text, so this pins it.
        var source = File.ReadAllText(RepositoryPath(
            "src", "Boutquin.Numerics", "Statistics", "ICovarianceEstimator.cs"));

        source.Should().NotContain("PC1 shrinkage",
            "Detoned removes the market component; it no longer shrinks PC1");
        source.Should().NotContain("final reset of the correlation",
            "the cleaned correlation is rescaled to a unit diagonal, not reset");
        source.Should().NotContain("<see cref=\"QuadraticInverseShrinkageEstimator\"/> — no PSD guarantee",
            "QIS output is positive definite whenever the non-null sample eigenvalues are");
        source.Should().Contain("singular",
            "Detoned output is singular at alpha = 1 and cannot be Cholesky-factored even after projection");
        source.Should().Contain("positive definite",
            "the QIS guarantee must be stated");
    }

    private static double[,] FactorPanel(int observations, int assets, int strongAssets, int seed)
    {
        var random = new System.Random(seed);
        var returns = new double[observations, assets];
        for (var t = 0; t < observations; t++)
        {
            var factor = Normal(random);
            for (var j = 0; j < assets; j++)
            {
                var loading = j < strongAssets ? 1.0 : 0.05;
                returns[t, j] = 0.01 * ((loading * factor) + Normal(random));
            }
        }

        return returns;
    }

    private static double Normal(System.Random random)
    {
        var u1 = 1.0 - random.NextDouble();
        var u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static double MaxAbsDifference(double[,] a, double[,] b)
    {
        var max = 0.0;
        for (var i = 0; i < a.GetLength(0); i++)
        {
            for (var j = 0; j < a.GetLength(1); j++)
            {
                max = Math.Max(max, Math.Abs(a[i, j] - b[i, j]));
            }
        }

        return max;
    }

    private static string RepositoryPath(params string[] parts)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Boutquin.Numerics.slnx")))
        {
            directory = directory.Parent;
        }

        directory.Should().NotBeNull("tests must run beneath the repository checkout");
        return Path.Combine(new[] { directory!.FullName }.Concat(parts).ToArray());
    }
}
