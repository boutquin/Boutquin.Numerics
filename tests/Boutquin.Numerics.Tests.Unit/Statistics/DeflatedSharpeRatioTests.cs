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

public sealed class DeflatedSharpeRatioTests
{
    [Fact]
    public void Compute_SingleTrialProducesZeroExpectedMax()
    {
        var result = DeflatedSharpeRatio.Compute(
            observedSharpe: 1.2m,
            numTrials: 1,
            backTestYears: 5m,
            skewness: 0m,
            kurtosis: 3m);

        result.NumTrials.Should().Be(1);
        result.ExpectedMaxSharpe.Should().Be(0m);
        result.DeflatedSharpe.Should().Be(1.2m);
    }

    [Fact]
    public void Compute_MoreTrialsIncreaseExpectedMax()
    {
        var few = DeflatedSharpeRatio.Compute(1.5m, 10, 5m, 0m, 3m);
        var many = DeflatedSharpeRatio.Compute(1.5m, 1000, 5m, 0m, 3m);

        many.ExpectedMaxSharpe.Should().BeGreaterThan(few.ExpectedMaxSharpe);
        many.DeflatedSharpe.Should().BeLessThan(few.DeflatedSharpe);
    }

    [Fact]
    public void Compute_ClampsPValueToUnitInterval()
    {
        var result = DeflatedSharpeRatio.Compute(-5m, 1000, 1m, 0m, 3m);
        result.PValue.Should().BeGreaterThanOrEqualTo(0m);
        result.PValue.Should().BeLessThanOrEqualTo(1m);
    }

    [Fact]
    public void Compute_HighSharpeLargeSample_ProducesLowPValue()
    {
        // Observed Sharpe well above the expected maximum and a long sample
        // → the null rejection should be decisive.
        var result = DeflatedSharpeRatio.Compute(3m, 10, 10m, 0m, 3m, tradingDaysPerYear: 252);
        result.PValue.Should().BeLessThan(0.01m);
    }

    [Fact]
    public void Compute_ZeroBacktestYearsIsClampedToOne()
    {
        var zero = DeflatedSharpeRatio.Compute(1m, 100, 0m, 0m, 3m);
        var one = DeflatedSharpeRatio.Compute(1m, 100, 1m, 0m, 3m);

        // Both runs should produce the same statistic because zero is clamped up.
        zero.DeflatedSharpe.Should().Be(one.DeflatedSharpe);
        zero.PValue.Should().Be(one.PValue);
    }
}
