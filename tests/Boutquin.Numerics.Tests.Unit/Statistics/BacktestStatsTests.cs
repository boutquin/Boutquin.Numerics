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

using Boutquin.Numerics.Random;
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class BacktestStatsTests
{
    [Fact]
    public void NeweyWest_ZeroLagsEqualsSampleVarianceOfMean()
    {
        decimal[] series = [0.01m, 0.02m, -0.005m, 0.015m, -0.01m, 0.02m, 0.005m];
        var hac = NeweyWestVariance.MeanVariance(series, lags: 0);
        // With L = 0, HAC = γ₀ / T (no autocorrelation correction).
        var mean = 0m;
        for (var i = 0; i < series.Length; i++)
        {
            mean += series[i];
        }

        mean /= series.Length;
        var ssq = 0m;
        for (var i = 0; i < series.Length; i++)
        {
            ssq += (series[i] - mean) * (series[i] - mean);
        }

        var expected = ssq / series.Length / series.Length;
        Math.Abs(hac - expected).Should().BeLessThan(1e-20m);
    }

    [Fact]
    public void NeweyWest_AutomaticLagsScaleWithT()
    {
        var l100 = NeweyWestVariance.AutomaticLags(100);
        var l1000 = NeweyWestVariance.AutomaticLags(1000);
        l1000.Should().BeGreaterThan(l100);
    }

    [Fact]
    public void HaircutSharpe_LargeNReducesObservedSharpe()
    {
        // Use a small Sharpe + short backtest where p-values are non-degenerate.
        var fewTrials = HaircutSharpe.Compute(0.1m, numTrials: 1, backTestYears: 1m);
        var manyTrials = HaircutSharpe.Compute(0.1m, numTrials: 1000, backTestYears: 1m);
        manyTrials.HaircutSharpe.Should().BeLessThan(fewTrials.HaircutSharpe);
        manyTrials.HaircutAmount.Should().BeGreaterThan(fewTrials.HaircutAmount);
    }

    [Fact]
    public void Pbo_PerfectlyAlignedStrategiesGivesLowOverfitting()
    {
        // 4 strategies that all share the same (deterministic) Sharpe pattern across blocks
        // → IS winner is OOS winner → PBO ≈ 0.
        var rng = new Pcg64RandomSource(1234UL);
        var n = 4;
        var t = 64; // divisible by 16.
        var returns = new decimal[n, t];
        var gaussian = new GaussianSampler(rng);
        var commonShock = new double[t];
        for (var i = 0; i < t; i++)
        {
            commonShock[i] = gaussian.Next();
        }

        for (var s = 0; s < n; s++)
        {
            for (var i = 0; i < t; i++)
            {
                returns[s, i] = (decimal)((s + 1) * 0.001 + 0.005 * commonShock[i]);
            }
        }

        var pbo = ProbabilityOfBacktestOverfitting.Compute(returns, splitCount: 16);
        pbo.Pbo.Should().BeLessThan(0.5m);
    }

    [Fact]
    public void MinTRL_LongerTrackRequiredForLowerSharpe()
    {
        var lowSharpe = MinimumTrackRecordLength.Compute(observedSharpe: 0.5);
        var highSharpe = MinimumTrackRecordLength.Compute(observedSharpe: 2.0);
        lowSharpe.Should().BeGreaterThan(highSharpe);
    }

    [Fact]
    public void MinTRL_BelowBenchmarkReturnsInfinity()
    {
        var result = MinimumTrackRecordLength.Compute(observedSharpe: 0.0, benchmarkSharpe: 0.5);
        result.Should().Be(double.PositiveInfinity);
    }
}
