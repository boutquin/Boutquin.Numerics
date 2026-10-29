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

/// <summary>
/// Regression pins for intermediate scalars narrowed to the working type before a later
/// step would have made the result representable. Expected values come from independent
/// oracles (scipy.stats.t for the Harvey-Liu panel, mpmath first-principles moments for the
/// deflated Sharpe, exact rational arithmetic for the GT-inspired z), rounded once to the
/// working type, never from the production code.
/// </summary>
public sealed class SharpeNarrowingTests
{
    // ---- HaircutSharpe.ComputePanel: single-test p-values must not be narrowed before × M ----

    private static T[] StrategyAmongNulls<T>(T observed)
        where T : System.Numerics.IFloatingPoint<T>
    {
        var panel = new T[1000];
        panel[0] = observed;
        for (var i = 1; i < panel.Length; i++)
        {
            panel[i] = T.Zero;
        }

        return panel;
    }

    [Fact]
    public void ComputePanel_HalfStrongStrategyAmongNulls_AppliesTheBonferroniHaircut()
    {
        // scipy: t = 1.8125·√10, dof 2519, p = 1.1132482223955526e-8 (below Half's least
        // subnormal half-step); Bonferroni 1000·p = 1.1132482223955526e-5, BHY 1000·H(1000)·p =
        // 8.333187129301379e-5, haircut Sharpe t⁻¹(Bonferroni)/√10 = 1.392284728387857.
        var results = HaircutSharpe<Half>.ComputePanel(StrategyAmongNulls((Half)1.8125), (Half)10);

        var top = results[0];
        ((double)top.HaircutSharpe).Should().Be(1.392578125);
        ((double)top.HaircutAmount).Should().Be((double)((Half)1.8125 - (Half)1.392578125));
        ((double)top.BonferroniPValue).Should().Be(1.1146068572998047e-05);
        ((double)top.HolmPValue).Should().Be(1.1146068572998047e-05);
        ((double)top.BhyPValue).Should().Be(8.33272933959961e-05);
    }

    [Fact]
    public void ComputePanel_HalfSubnormalSingleTestPValue_IsNotRoundedBeforeTheMultiplication()
    {
        // scipy: SR 1.5 gives p = 2.218741727746019e-6 (a Half subnormal 2.2054e-6 when narrowed
        // first); Bonferroni 2.218741727746019e-3 and haircut Sharpe 0.9684229136000545.
        var results = HaircutSharpe<Half>.ComputePanel(StrategyAmongNulls((Half)1.5), (Half)10);

        ((double)results[0].HaircutSharpe).Should().Be(0.96826171875);
        ((double)results[0].BonferroniPValue).Should().Be(0.0022182464599609375);
    }

    [Fact]
    public void ComputePanel_DecimalSingleTestPValueBelowDecimalResolution_AppliesTheBonferroniHaircut()
    {
        // scipy: SR 3.6 gives p = 2.576056295428448e-29 (decimal 0 when narrowed first);
        // Bonferroni 2.5760562954284482e-26, BHY 1.928299433456692e-25, haircut 3.394535932601844.
        var results = HaircutSharpe<decimal>.ComputePanel(StrategyAmongNulls(3.6m), 10m);

        var top = results[0];
        ((double)top.HaircutSharpe).Should().BeApproximately(3.394535932601844, 1e-12);
        top.HaircutAmount.Should().Be(3.6m - top.HaircutSharpe);
        ((double)top.BonferroniPValue).Should().BeApproximately(2.5760562954284482e-26, 1e-28);
        ((double)top.HolmPValue).Should().BeApproximately(2.5760562954284482e-26, 1e-28);
        ((double)top.BhyPValue).Should().BeApproximately(1.928299433456692e-25, 1e-28);
    }

    [Fact]
    public void ComputePanel_FloatSingleTestPValueBelowFloatRange_AppliesTheBonferroniHaircut()
    {
        // scipy: SR 4.6f (4.599999904632568) gives p = 4.3096944282523454e-46 (float 0 when narrowed
        // first); Bonferroni 4.309694428252345e-43 and haircut Sharpe 4.435502583060435.
        var results = HaircutSharpe<float>.ComputePanel(StrategyAmongNulls(4.6f), 10f);

        var top = results[0];
        ((double)top.HaircutSharpe).Should().Be(4.435502529144287);
        ((double)top.BonferroniPValue).Should().BeApproximately(4.309694428252345e-43, 3e-45);
        ((double)top.BhyPValue).Should().BeApproximately(3.226009206055911e-42, 3e-45);
    }

    // ---- GtInspiredScore<Half>: the double mean must feed z before the reported Mu is narrowed ----

    private static Half[] AlternatingHalfWithTinyDrift(int count)
    {
        // ±1/2 alternating with the first value raised by 2⁻¹¹: the exact mean is 2⁻¹¹/count,
        // below Half's least subnormal half-step (2.98e-8) for count ≥ 20,000.
        var values = new Half[count];
        for (var i = 0; i < count; i++)
        {
            values[i] = i % 2 == 0 ? (Half)0.5 : (Half)(-0.5);
        }

        values[0] = (Half)(0.5 + 1.0 / 2048.0);
        return values;
    }

    [Theory]
    [InlineData(20_000, 6.9051666870444476e-06)]
    [InlineData(40_000, 4.8827513451962746e-06)]
    public void GtCompute_HalfNearZeroPositiveMean_TakesTheMarginalBranch(int count, double oracleZ)
    {
        // Exact rational oracle: z = mean/(s/√n) is positive and far below one, so the
        // marginal branch gives 100(1 − e^−|z−1|) = 63.2118…, binary16 63.21875.
        var result = GtInspiredScore<Half>.Compute(AlternatingHalfWithTinyDrift(count));

        ((double)result.Score).Should().Be(63.21875);
        ((double)result.Z).Should().BeApproximately(oracleZ, 2.5e-7);
    }

    [Fact]
    public void GtCompute_HalfNearZeroPositiveMeanBelowSafeguard_ReportsThePositiveZ()
    {
        // n ≤ minimumObservations takes the safeguard score 999 − 899 = 100; the reported
        // diagnostic Z must still be the oracle's 6.9051666870444476e-6, not the zero obtained
        // from a mean already rounded to Half.
        var result = GtInspiredScore<Half>.Compute(AlternatingHalfWithTinyDrift(20_000), Half.Zero, (Half)0.001, 20_000);

        ((double)result.Score).Should().Be(100.0);
        ((double)result.Z).Should().BeApproximately(6.9051666870444476e-06, 2.5e-7);
    }

    // ---- DeflatedSharpeRatio<Half>.ComputeFromReturns: kurtosis above Half.MaxValue ----

    [Theory]
    [InlineData(100_000, 10.0, 0.02065647865574191)]
    [InlineData(200_000, 10.0, 0.021687224116210273)]
    [InlineData(100_000, 2.0, 0.022321947800472117)]
    public void DsrFromReturns_HalfKurtosisAboveHalfRange_MatchesFirstPrinciplesPValue(int count, double outlier, double oraclePValue)
    {
        // mpmath, 50 digits, on the Half-rounded values: population moments give a Pearson
        // kurtosis near n (above Half.MaxValue 65,504), and eq. 2 with one trial gives the p-value.
        var returns = new Half[count];
        for (var i = 0; i < count - 1; i++)
        {
            returns[i] = (Half)0.01;
        }

        returns[^1] = (Half)outlier;

        var result = DeflatedSharpeRatio<Half>.ComputeFromReturns(returns, 1, 252);

        ((double)result.PValue).Should().BeApproximately(oraclePValue, 1e-4);
    }
}
