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

public sealed class PoetCovarianceEstimatorTests
{
    [Fact]
    public void Ctor_RejectsInvalidArguments()
    {
        FluentActions.Invoking(() => new PoetCovarianceEstimator(0))
            .Should().Throw<ArgumentOutOfRangeException>();
        FluentActions.Invoking(() => new PoetCovarianceEstimator(1, -0.01))
            .Should().Throw<ArgumentOutOfRangeException>();
        FluentActions.Invoking(() => new PoetCovarianceEstimator(1, double.NaN))
            .Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Estimate_ThrowsWhenFactorsNotLessThanAssets()
    {
        var returns = GenerateReturns(50, 3, seed: 1);
        FluentActions.Invoking(() => new PoetCovarianceEstimator(3).Estimate(returns))
            .Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Estimate_ProducesSymmetricMatrix()
    {
        var returns = GenerateReturns(150, 6, seed: 11);
        var cov = new PoetCovarianceEstimator(1, 0.5).Estimate(returns);

        var n = cov.GetLength(0);
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                cov[i, j].Should().Be(cov[j, i]);
            }
        }
    }

    [Fact]
    public void Estimate_PositiveDiagonal()
    {
        var returns = GenerateReturns(200, 5, seed: 41);
        var cov = new PoetCovarianceEstimator(2, 0.5).Estimate(returns);

        for (var i = 0; i < cov.GetLength(0); i++)
        {
            cov[i, i].Should().BeGreaterThan(0m);
        }
    }

    [Fact]
    public void Estimate_WithFactorStructure_RecoversStrongFactorCovariance()
    {
        // Build a return matrix with one strong common factor: r_ij = f_i + noise.
        // A one-factor POET should recover a near-rank-one factor component with
        // all off-diagonal entries of the factor component positive and nearly equal.
        var rng = new System.Random(101);
        var t = 500;
        var n = 5;
        var r = new decimal[t, n];
        for (var i = 0; i < t; i++)
        {
            var factor = (decimal)(rng.NextGaussian() * 0.02);
            for (var j = 0; j < n; j++)
            {
                var noise = (decimal)(rng.NextGaussian() * 0.001);
                r[i, j] = factor + noise;
            }
        }

        var cov = new PoetCovarianceEstimator(1, 0.1).Estimate(r);

        var offDiagonalMin = decimal.MaxValue;
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                if (cov[i, j] < offDiagonalMin)
                {
                    offDiagonalMin = cov[i, j];
                }
            }
        }

        offDiagonalMin.Should().BeGreaterThan(0m, "strong common factor should dominate residual thresholding");
    }

    [Fact]
    public void Estimate_NoThresholdingVsAggressive_DifferOnOffDiagonals()
    {
        var returns = GenerateReturns(120, 5, seed: 77);
        // c = 0 keeps every residual entry; c = 10 zeroes them all.
        var none = new PoetCovarianceEstimator(1, 0.0).Estimate(returns);
        var aggressive = new PoetCovarianceEstimator(1, 10.0).Estimate(returns);

        var differ = false;
        for (var i = 0; i < 5 && !differ; i++)
        {
            for (var j = i + 1; j < 5 && !differ; j++)
            {
                if (none[i, j] != aggressive[i, j])
                {
                    differ = true;
                }
            }
        }

        differ.Should().BeTrue("c=0 retains residual, c=10 zeroes it — off-diagonals must differ");
    }

    [Fact]
    public void Properties_MatchConstructor()
    {
        var estimator = new PoetCovarianceEstimator(2, 1.25);
        estimator.NumFactors.Should().Be(2);
        estimator.ThresholdMultiplier.Should().Be(1.25);
    }

    private static decimal[,] GenerateReturns(int t, int n, int seed)
    {
        var rng = new System.Random(seed);
        var r = new decimal[t, n];
        for (var i = 0; i < t; i++)
        {
            for (var j = 0; j < n; j++)
            {
                r[i, j] = (decimal)(rng.NextDouble() * 0.02 - 0.01);
            }
        }

        return r;
    }
}

internal static class RandomExtensions
{
    public static double NextGaussian(this System.Random rng)
    {
        // Marsaglia polar method — shared helper for tests.
        double u, v, s;
        do
        {
            u = 2.0 * rng.NextDouble() - 1.0;
            v = 2.0 * rng.NextDouble() - 1.0;
            s = u * u + v * v;
        }
        while (s >= 1.0 || s == 0.0);

        var factor = Math.Sqrt(-2.0 * Math.Log(s) / s);
        return u * factor;
    }
}
