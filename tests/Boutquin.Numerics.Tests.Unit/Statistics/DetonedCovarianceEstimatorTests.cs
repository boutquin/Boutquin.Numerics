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

public sealed class DetonedCovarianceEstimatorTests
{
    [Fact]
    public void Constructor_RejectsAlphaOutOfRange()
    {
        Action negative = () => new DetonedCovarianceEstimator(-0.1m);
        Action above = () => new DetonedCovarianceEstimator(1.1m);
        negative.Should().Throw<ArgumentOutOfRangeException>();
        above.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Estimate_ZeroAlphaEqualsDenoised()
    {
        var returns = GenerateFactorStructuredReturns(200, 5, seed: 3);

        var detonedZero = new DetonedCovarianceEstimator(0m).Estimate(returns);
        var denoised = new DenoisedCovarianceEstimator().Estimate(returns);

        for (var i = 0; i < 5; i++)
        {
            for (var j = 0; j < 5; j++)
            {
                ((double)Math.Abs(detonedZero[i, j] - denoised[i, j])).Should().BeLessThan(1e-20);
            }
        }
    }

    [Fact]
    public void Estimate_FullDetoningReducesOffDiagonalsBelowDenoised()
    {
        // Strong common factor → detoning should materially reduce
        // the average magnitude of off-diagonal covariances relative to
        // the denoised baseline.
        var returns = GenerateFactorStructuredReturns(300, 6, seed: 21);

        var denoised = new DenoisedCovarianceEstimator().Estimate(returns);
        var detoned = new DetonedCovarianceEstimator(1m).Estimate(returns);

        decimal denoisedOff = 0m, detonedOff = 0m;
        for (var i = 0; i < 6; i++)
        {
            for (var j = i + 1; j < 6; j++)
            {
                denoisedOff += Math.Abs(denoised[i, j]);
                detonedOff += Math.Abs(detoned[i, j]);
            }
        }

        detonedOff.Should().BeLessThan(denoisedOff);
    }

    [Fact]
    public void Estimate_FallsBackToSampleForFewerThanThreeAssets()
    {
        var returns = GenerateFactorStructuredReturns(80, 2, seed: 12);

        var sample = new SampleCovarianceEstimator().Estimate(returns);
        var detoned = new DetonedCovarianceEstimator(1m).Estimate(returns);

        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                detoned[i, j].Should().Be(sample[i, j]);
            }
        }
    }

    private static decimal[,] GenerateFactorStructuredReturns(int t, int n, int seed)
    {
        var rng = new System.Random(seed);
        var returns = new decimal[t, n];
        for (var i = 0; i < t; i++)
        {
            var factor = rng.NextDouble() * 0.02 - 0.01;
            for (var j = 0; j < n; j++)
            {
                var idio = rng.NextDouble() * 0.002 - 0.001;
                returns[i, j] = (decimal)(factor + idio);
            }
        }

        return returns;
    }
}
