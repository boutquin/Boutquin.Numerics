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
    public void NeweyWest_ZeroLagsMatchesExternallyComputedAnswer()
    {
        // Externally computed (not a re-derivation of the source's loop):
        //   series mean = 0.055 / 7 = 0.0078571428571428583
        //   γ₀ = Σ(x - mean)² / 7 = 0.00012040816326530613
        //   HAC(mean) with L = 0 = γ₀ / 7 = 1.7201166180758018e-05
        decimal[] series = [0.01m, 0.02m, -0.005m, 0.015m, -0.01m, 0.02m, 0.005m];
        var hac = NeweyWestVariance.MeanVariance(series, lags: 0);
        Math.Abs(hac - 0.0000172011661807580m).Should().BeLessThan(1e-18m);
    }

    [Fact]
    public void NeweyWest_HalfLongAlternatingSeriesRetainsMeanVariance()
    {
        var series = new Half[65_520];
        for (var i = 0; i < series.Length; i++)
        {
            series[i] = (Half)(i % 2);
        }

        var actual = NeweyWestVariance<Half>.MeanVariance(series, lags: 0);

        ((double)actual).Should().BeApproximately(0.25 / series.Length, 1e-7);
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
    public void HaircutSharpe_ZeroTradingDays_ThrowsArgumentOutOfRangeException()
    {
        // F-L5-2: tradingDaysPerYear must be > 0. At 0 the haircut silently returns
        // -∞ (no exception); the sibling MinimumTrackRecordLength rejects tdy ≤ 0.
        var act = () => HaircutSharpe<double>.Compute(0.1, numTrials: 10, backTestYears: 1.0, tradingDaysPerYear: 0);
        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*Trading days*");
    }

    [Fact]
    public void HaircutSharpe_NegativeTradingDays_ThrowsArgumentOutOfRangeException()
    {
        // F-L5-2: a negative tradingDaysPerYear previously drove Sqrt(negative)=NaN
        // into InverseCdf, which stack-overflowed (process-killing recursion).
        var act = () => HaircutSharpe<double>.Compute(0.1, numTrials: 10, backTestYears: 1.0, tradingDaysPerYear: -1);
        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*Trading days*");
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
    public void Pbo_CountsLogitsAtOrBelowZero()
    {
        // F-L7-02: PBO is the fraction of logits ≤ 0 (Bailey et al. 2014, and the
        // repo's own Python reference: pbo = mean(logits <= 0)). The code previously
        // counted strictly < 0, dropping the ω = 0.5 boundary (logit == 0), which is
        // reachable for any ODD strategy count N (rank = (N+1)/2 → ω = 0.5).
        //
        // Deterministic N=3, T=8, splitCount=4 panel (6 splits). Of the 6 CSCV
        // logits, two are exactly 0 (ω = 0.5). The correct ≤ 0 count is 6/6 = 1.0;
        // the old < 0 count was 4/6 ≈ 0.6667.
        var returns = new double[,]
        {
            { 0.345584192064786, 0.8216181435011584, 0.33043707618338714, -1.303157231604361, 0.9053558666731177, 0.4463745723640113, -0.5369532353602852, 0.5811181041963531 },
            { 0.36457239618607573, 0.294132496655526, 0.02842224131579679, 0.5467129866124469, -0.7364540870016669, -0.16290994799305278, -0.48211931267997826, 0.5988462126346276 },
            { 0.03972210748165899, -0.2924567509650886, -0.7819084623568421, -0.2571922406188707, 0.008142180518343508, -0.2756029052993704, 1.2940638143982073, 1.0067243153057943 },
        };

        var result = ProbabilityOfBacktestOverfitting<double>.Compute(returns, splitCount: 4);
        result.Pbo.Should().BeApproximately(1.0, 1e-12);
    }

    [Fact]
    public void Pbo_HalfTopRankHasFiniteLogit()
    {
        var returns = new Half[2048, 4];
        for (var strategy = 0; strategy < 2048; strategy++)
        {
            var baseline = strategy == 2047 ? (Half)1 : (Half)0;
            returns[strategy, 0] = baseline;
            returns[strategy, 1] = baseline + (Half)0.25f;
            returns[strategy, 2] = baseline;
            returns[strategy, 3] = baseline + (Half)0.25f;
        }

        var result = ProbabilityOfBacktestOverfitting<Half>.Compute(returns, 2);

        Half.IsFinite(result.LogitMedian).Should().BeTrue();
        ((double)result.LogitMedian).Should().BeApproximately(Math.Log(2048), 0.01);
    }

    [Fact]
    public void Pbo_HalfTwentyBlocksKeepsFiniteFoldFraction()
    {
        var returns = new Half[2, 40];
        for (var t = 0; t < 40; t++)
        {
            returns[0, t] = (Half)(t % 2 == 0 ? 0.25f : 0.5f);
            returns[1, t] = (Half)(t % 2 == 0 ? 0.5f : 0.25f);
        }

        var result = ProbabilityOfBacktestOverfitting<Half>.Compute(returns, 20);

        result.LogitValues.Should().HaveCount(184_756);
        Half.IsFinite(result.Pbo).Should().BeTrue();
        result.Pbo.Should().Be((Half)1);
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
