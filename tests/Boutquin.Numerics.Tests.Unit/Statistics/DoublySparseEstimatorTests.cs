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

        // Trace should be approximately preserved (may differ slightly due to thresholding).
        ((double)resultTrace).Should().BeApproximately((double)sampleTrace, (double)sampleTrace * 0.5,
            "Trace should be approximately preserved");
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
