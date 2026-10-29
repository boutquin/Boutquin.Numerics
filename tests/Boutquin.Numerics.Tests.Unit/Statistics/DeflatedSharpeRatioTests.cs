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

    // F-L1-01 / F-L5-01: the moments feeding skewness/kurtosis were formed at
    // the caller's raw scale (m3 = E[d^3], m4 = E[d^4]) instead of being
    // normalized first. Skewness and kurtosis are scale-free, but the raw
    // fourth-order products are not — for T = Half at ordinary daily-return
    // magnitudes, d^4 and m2^2 underflow to zero/NaN while sd stays
    // representable, so ComputeFromReturns threw instead of returning a
    // finite p-value. The reference oracle here is DeflatedSharpeRatio<double>
    // run on the SAME Half-rounded/tiny-scaled values — an independent,
    // already-correct sibling instantiation, not a re-derivation of the fix.
    private static double[] BuildDeterministicDailyReturns(int n)
    {
        var values = new double[n];
        for (var i = 0; i < n; i++)
        {
            values[i] = (i % 2 == 0 ? 1.0 : -1.0) * (0.008 + 0.004 * Math.Sin(i * 0.37));
        }

        return values;
    }

    [Fact]
    public void ComputeFromReturns_HalfOrdinaryDailyScale_DoesNotThrowAndAgreesWithDoubleOracle()
    {
        var raw = BuildDeterministicDailyReturns(40);
        var half = new Half[raw.Length];
        var halfAsDouble = new double[raw.Length];
        for (var i = 0; i < raw.Length; i++)
        {
            half[i] = (Half)raw[i];
            halfAsDouble[i] = (double)half[i];
        }

        var expected = DeflatedSharpeRatio<double>.ComputeFromReturns(halfAsDouble, numTrials: 10);

        var act = () => DeflatedSharpeRatio<Half>.ComputeFromReturns(half, numTrials: 10);

        var result = act.Should().NotThrow().Subject;
        Half.IsFinite(result.PValue).Should().BeTrue();
        ((double)result.PValue).Should().BeApproximately(expected.PValue, 1e-2);
    }

    [Fact]
    public void ComputeFromReturns_DecimalTinyScale_DoesNotDivideByZeroAndAgreesWithDoubleOracle()
    {
        var raw = BuildDeterministicDailyReturns(40);
        var decimalTiny = new decimal[raw.Length];
        var decimalTinyAsDouble = new double[raw.Length];
        for (var i = 0; i < raw.Length; i++)
        {
            decimalTiny[i] = (decimal)raw[i] * 1e-6m;
            decimalTinyAsDouble[i] = (double)decimalTiny[i];
        }

        var expected = DeflatedSharpeRatio<double>.ComputeFromReturns(decimalTinyAsDouble, numTrials: 10);

        var act = () => DeflatedSharpeRatio<decimal>.ComputeFromReturns(decimalTiny, numTrials: 10);

        var result = act.Should().NotThrow().Subject;
        ((double)result.PValue).Should().BeApproximately(expected.PValue, 1e-4);
    }

    // F-L1-03 (census site within this file): Half.CreateChecked(n) silently
    // returns +Infinity once n exceeds 65,504, so `mean /= n` (and the
    // backtest-years conversion) collapsed to zero/Infinity for any series
    // longer than that — even though every individual return is finite and
    // ordinary-scale. This asserts a 70,000-point series produces a finite,
    // non-degenerate result instead of the spurious "zero variance" rejection.
    [Fact]
    public void ComputeFromReturns_HalfSeriesLongerThanHalfRange_ProducesFiniteNonDegenerateResult()
    {
        const int n = 70_000;
        var half = new Half[n];
        var halfAsDouble = new double[n];
        for (var i = 0; i < n; i++)
        {
            var noise = Math.Sin(i * 12.9898) * 43758.5453;
            noise -= Math.Floor(noise);
            var v = (i % 2 == 0 ? 1.0 : -1.0) * (0.008 + 0.004 * (noise - 0.5));
            half[i] = (Half)v;
            halfAsDouble[i] = (double)half[i];
        }

        var expected = DeflatedSharpeRatio<double>.ComputeFromReturns(halfAsDouble, numTrials: 10);

        var act = () => DeflatedSharpeRatio<Half>.ComputeFromReturns(half, numTrials: 10);

        var result = act.Should().NotThrow().Subject;
        Half.IsFinite(result.PValue).Should().BeTrue();
        ((double)result.PValue).Should().BeApproximately(expected.PValue, 5e-2);
    }
}
