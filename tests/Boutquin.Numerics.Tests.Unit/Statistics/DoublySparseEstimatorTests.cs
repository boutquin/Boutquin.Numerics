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

namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class DoublySparseEstimatorTests
{
    [Fact]
    public void Estimate_PreservesSymmetry()
    {
        var returns = GenerateReturns(100, 5, seed: 42);
        var estimator = new DoublySparseEstimator();

        var result = estimator.Estimate(returns);
        var n = result.GetLength(0);

        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                ((double)Math.Abs(result[i, j] - result[j, i])).Should().BeLessThan(1e-10,
                    $"Symmetry violated at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void Estimate_PreservesTrace()
    {
        var returns = GenerateReturns(100, 5, seed: 42);
        var estimator = new DoublySparseEstimator();

        // Compute sample covariance trace for comparison.
        var n = returns.GetLength(1);
        var t = returns.GetLength(0);
        var means = new decimal[n];
        for (var j = 0; j < n; j++)
        {
            for (var i = 0; i < t; i++)
            {
                means[j] += returns[i, j];
            }

            means[j] /= t;
        }

        var sampleTrace = 0m;
        for (var j = 0; j < n; j++)
        {
            var sum = 0m;
            for (var i = 0; i < t; i++)
            {
                var diff = returns[i, j] - means[j];
                sum += diff * diff;
            }

            sampleTrace += sum / (t - 1);
        }

        var result = estimator.Estimate(returns);
        var resultTrace = 0m;
        for (var i = 0; i < n; i++)
        {
            resultTrace += result[i, i];
        }

        // The DSCE preserves the trace tightly on this well-behaved 5-asset sample:
        // none of the eigenvector entries fall below the default 0.1 threshold here, so
        // the signal eigenvectors stay orthonormal and the reconstructed correlation
        // diagonal stays at 1 to machine precision. (In general, aggressive thresholding
        // can re-normalize the signal eigenvectors and introduce a small bounded trace
        // drift.) Measured relative error is < 1e-15 — pin it to 1e-8, a robust tight
        // bound, instead of the original vacuous ±50% band.
        var relativeError = (double)Math.Abs((resultTrace - sampleTrace) / sampleTrace);
        relativeError.Should().BeLessThan(1e-8,
            "thresholded-eigenvector reconstruction should keep the trace very close to the sample trace");
    }

    [Fact]
    public void Estimate_PositiveDiagonal()
    {
        var returns = GenerateReturns(100, 5, seed: 42);
        var estimator = new DoublySparseEstimator();

        var result = estimator.Estimate(returns);
        var n = result.GetLength(0);

        for (var i = 0; i < n; i++)
        {
            result[i, i].Should().BeGreaterThan(0m, $"Diagonal [{i},{i}] should be positive");
        }
    }

    [Fact]
    public void Estimate_TooFewObservations_Throws()
    {
        var returns = new decimal[1, 3]; // Only 1 observation.
        var estimator = new DoublySparseEstimator();

        var act = () => estimator.Estimate(returns);
        act.Should().Throw<ArgumentException>();
    }

    // -----------------------------------------------------------------------
    //  F-LQ-2: negative eigenvectorThreshold → ArgumentOutOfRangeException
    // -----------------------------------------------------------------------

    /// <summary>
    /// F-LQ-2: A negative eigenvectorThreshold silently disables sparsification
    /// (T.Abs(entry) >= negative ≡ always true). Must throw ArgumentOutOfRangeException
    /// to mirror PoetCovarianceEstimator.
    /// </summary>
    [Fact]
    public void Constructor_ThrowsArgumentOutOfRangeException_ForNegativeEigenvectorThreshold()
    {
        var act = () => new DoublySparseEstimator<double>(eigenvectorThreshold: -0.5);
        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("eigenvectorThreshold");
    }

    /// <summary>
    /// F-LQ-2 regression: zero threshold must be accepted (≥ 0 is the valid range).
    /// </summary>
    [Fact]
    public void Constructor_AcceptsZeroEigenvectorThreshold()
    {
        var act = () => new DoublySparseEstimator<double>(eigenvectorThreshold: 0.0);
        act.Should().NotThrow();
    }

    /// <summary>
    /// The generic parameterless constructor must apply the documented 0.1
    /// eigenvector threshold. With an unconstrained T, `T? = default` is 0 for
    /// value types, which keeps every eigenvector entry and disables sparsification.
    /// </summary>
    [Fact]
    public void Ctor_GenericDefaultAppliesDocumentedPointOneThreshold()
    {
        // One factor loading only the first 8 of 30 assets: the signal eigenvector
        // has ~0.35 entries on those assets and sub-0.1 noise entries elsewhere, so
        // thresholds 0 and 0.1 produce different reconstructions.
        const int t = 400;
        const int n = 30;
        var rng = new System.Random(2024);
        var returns = new double[t, n];
        for (var i = 0; i < t; i++)
        {
            var factor = rng.NextGaussian();
            for (var j = 0; j < n; j++)
            {
                returns[i, j] = (j < 8 ? factor : 0.0) + rng.NextGaussian();
            }
        }

        var byDefault = new DoublySparseEstimator<double>().Estimate(returns);
        var pointOne = new DoublySparseEstimator<double>(0.1).Estimate(returns);
        var zero = new DoublySparseEstimator<double>(0.0).Estimate(returns);

        var maxDiffZeroVsPointOne = 0.0;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                maxDiffZeroVsPointOne = Math.Max(maxDiffZeroVsPointOne, Math.Abs(zero[i, j] - pointOne[i, j]));
            }
        }

        maxDiffZeroVsPointOne.Should().BeGreaterThan(1e-6, "the panel must discriminate threshold 0 from 0.1");
        byDefault.Should().BeEquivalentTo(pointOne, "the documented default eigenvector threshold is 0.1");
    }

    private static decimal[,] GenerateReturns(int t, int n, int seed)
    {
        var rng = new System.Random(seed);
        var returns = new decimal[t, n];
        for (var i = 0; i < t; i++)
        {
            for (var j = 0; j < n; j++)
            {
                returns[i, j] = (decimal)(rng.NextDouble() * 0.02 - 0.01);
            }
        }

        return returns;
    }
}
