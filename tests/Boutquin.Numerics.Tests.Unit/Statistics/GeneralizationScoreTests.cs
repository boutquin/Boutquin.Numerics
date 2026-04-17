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

public sealed class GeneralizationScoreTests
{
    [Fact]
    public void Compute_PositiveReturns_ProducesPositiveScore()
    {
        // Consistent positive daily returns (like a good strategy).
        var returns = new double[252];
        var rng = new System.Random(42);
        for (var i = 0; i < returns.Length; i++)
        {
            returns[i] = 0.001 + rng.NextDouble() * 0.002; // 0.1% to 0.3% daily
        }

        var result = GeneralizationScore.Compute(returns);

        result.Score.Should().BeGreaterThan(0);
        result.PerformanceComponent.Should().BeGreaterThan(0);
        result.ConsistencyComponent.Should().BeGreaterThan(0);
    }

    [Fact]
    public void Compute_RandomNoise_ProducesLowScore()
    {
        // Pure random noise — no signal.
        var returns = new double[252];
        var rng = new System.Random(42);
        for (var i = 0; i < returns.Length; i++)
        {
            returns[i] = rng.NextDouble() * 0.02 - 0.01; // -1% to +1%
        }

        var result = GeneralizationScore.Compute(returns);

        // Score should be near zero or negative for random noise.
        result.Score.Should().BeLessThan(1.0);
    }

    [Fact]
    public void Compute_HighDrawdown_PenalizesScore()
    {
        // Good returns followed by a crash.
        var returns = new double[252];
        for (var i = 0; i < 200; i++)
        {
            returns[i] = 0.001;
        }

        // Big drawdown.
        for (var i = 200; i < 252; i++)
        {
            returns[i] = -0.02;
        }

        var result = GeneralizationScore.Compute(returns);

        result.DownsideRiskComponent.Should().BeGreaterThan(0);
    }

    [Fact]
    public void Compute_ComponentsDecomposeToScore()
    {
        var returns = new double[252];
        var rng = new System.Random(42);
        for (var i = 0; i < returns.Length; i++)
        {
            returns[i] = 0.0005 + rng.NextDouble() * 0.001;
        }

        var result = GeneralizationScore.Compute(returns);

        var expected = result.PerformanceComponent + result.SignificanceComponent
            + result.ConsistencyComponent - result.DownsideRiskComponent;

        result.Score.Should().BeApproximately(expected, 1e-10);
    }

    [Fact]
    public void Compute_TooFewReturns_Throws()
    {
        var returns = new double[] { 0.01 };
        var act = () => GeneralizationScore.Compute(returns);
        act.Should().Throw<ArgumentException>();
    }
}
