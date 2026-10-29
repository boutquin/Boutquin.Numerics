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

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Unit-consistency and paper-fidelity pins for the Deflated Sharpe Ratio
/// (Bailey &amp; López de Prado 2014) and the Minimum Track Record Length
/// (Bailey &amp; López de Prado 2012). Every literal expected value below was
/// computed with SciPy directly from the papers' equations (not from the C#
/// code), or is a number printed in the papers themselves.
/// </summary>
public sealed class ArenaR1B09DsrMinTrlTests
{
    // Per-period Sharpe 0.1 over 1000 observations, N = 10 trials, Gaussian moments,
    // trials' SR variance at its null value 1/(T−1). SciPy, BLdP (2014) eqs. 1–2.
    private const double PerPeriodDsrPValueN10T1000 = 0.05680796385938458;

    [Theory]
    [InlineData(52)]
    [InlineData(252)]
    [InlineData(365)]
    public void Dsr_Compute_PValueDependsOnlyOnPerPeriodData_NotOnAnnualizationLabel(int periodsPerYear)
    {
        // The same per-period data (SR 0.1, T = 1000) described with different
        // annualization labels must give the same test: SR̂ and T enter eq. 2 at
        // the same, non-annualized frequency.
        var sqrtP = Math.Sqrt(periodsPerYear);
        var result = DeflatedSharpeRatio<double>.Compute(
            observedSharpe: 0.1 * sqrtP,
            numTrials: 10,
            backTestYears: 1000.0 / periodsPerYear,
            skewness: 0.0,
            kurtosis: 3.0,
            tradingDaysPerYear: periodsPerYear);

        result.PValue.Should().BeApproximately(PerPeriodDsrPValueN10T1000, 1e-7);
    }

    [Theory]
    [InlineData(12)]
    [InlineData(52)]
    [InlineData(252)]
    public void Dsr_ComputeFromReturns_PValueIsInvariantToTradingDaysPerYear(int periodsPerYear)
    {
        // ComputeFromReturns annualizes the Sharpe with √p and converts n into
        // n/p years. The statistic of a fixed return series cannot depend on p.
        var returns = new double[500];
        var state = 12345u;
        for (var i = 0; i < returns.Length; i++)
        {
            state = (state * 1664525u) + 1013904223u;
            returns[i] = 0.0006 + (((state >> 8) / 16777216.0) - 0.5) * 0.02;
        }

        var reference = DeflatedSharpeRatio<double>.ComputeFromReturns(returns, numTrials: 20, tradingDaysPerYear: 1);
        var labelled = DeflatedSharpeRatio<double>.ComputeFromReturns(returns, numTrials: 20, tradingDaysPerYear: periodsPerYear);

        labelled.PValue.Should().BeApproximately(reference.PValue, 1e-12);
    }

    [Fact]
    public void Dsr_Compute_ThresholdIsPaperEquation1AtTheNullTrialVariance()
    {
        // BLdP (2014) eq. 1: SR̂₀ = √V·((1−γ)Z⁻¹[1−1/N] + γZ⁻¹[1−e⁻¹/N]).
        // With no trial variance supplied, V is the null sampling variance of a
        // per-period SR estimate, 1/(T−1). N = 100, T = 5·250 = 1250:
        // maxZ = 2.5306028932016846 (SciPy), SR̂₀ = maxZ/√1249 per period,
        // ×√250 annualized = 1.1321729784527885.
        var result = DeflatedSharpeRatio<double>.Compute(
            observedSharpe: 2.5,
            numTrials: 100,
            backTestYears: 5.0,
            skewness: 0.0,
            kurtosis: 3.0,
            tradingDaysPerYear: 250);

        result.ExpectedMaxSharpe.Should().BeApproximately(1.1321729784527885, 1e-7);
        result.DeflatedSharpe.Should().BeApproximately(2.5 - 1.1321729784527885, 1e-7);
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void Dsr_ComputeWithTrialVariance_RejectsInvalidTrialVariance(double trialVariance)
    {
        var act = () => DeflatedSharpeRatio<double>.ComputeWithTrialVariance(
            observedSharpe: 2.5,
            numTrials: 100,
            trialSharpeVariance: trialVariance,
            backTestYears: 5.0,
            skewness: 0.0,
            kurtosis: 3.0,
            tradingDaysPerYear: 250);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("trialSharpeVariance");
    }

    [Fact]
    public void Dsr_ComputeWithTrialVariance_ZeroVarianceIsThePsrAgainstZero()
    {
        // V = 0 collapses SR̂₀ to 0: DSR is the PSR against a zero benchmark.
        // SciPy: 1 − Φ((2.5/√250)·√1249 / √(1 + 0.5·(2.5/√250)²)).
        var result = DeflatedSharpeRatio<double>.ComputeWithTrialVariance(
            observedSharpe: 2.5,
            numTrials: 100,
            trialSharpeVariance: 0.0,
            backTestYears: 5.0,
            skewness: 0.0,
            kurtosis: 3.0,
            tradingDaysPerYear: 250);

        result.ExpectedMaxSharpe.Should().Be(0.0);
        result.PValue.Should().BeApproximately(1.4013680051404235e-08, 1e-11);
    }

    [Theory]
    [InlineData(12, 272.90711713658885)]
    [InlineData(252, 272.90711713658885)]
    public void MinTrl_ReturnsPerPeriodObservationCount_IndependentOfAnnualizationLabel(int periodsPerYear, double expected)
    {
        // Per-period SR 0.1 vs benchmark 0, Gaussian, α = 0.05:
        // BLdP (2012) eq. 13 in SciPy gives 272.907 observations whatever label
        // is used to annualize the Sharpe.
        var result = MinimumTrackRecordLength<double>.Compute(
            observedSharpe: 0.1 * Math.Sqrt(periodsPerYear),
            benchmarkSharpe: 0.0,
            skewness: 0.0,
            kurtosis: 3.0,
            significanceLevel: 0.05,
            tradingDaysPerYear: periodsPerYear);

        result.Should().BeApproximately(expected, 1e-6);
    }

    [Fact]
    public void MinTrl_PaperWorkedExample_AnnualizedTwoVersusOne_Is688Observations()
    {
        // BLdP (2012): an annualized Sharpe of 2 needs 2.73 years of daily data
        // (688.2 observations) to exceed a benchmark of 1 at 95% confidence.
        // SciPy value of eq. 13 at SR = 2/√252, SR* = 1/√252: 688.2080373402348.
        var result = MinimumTrackRecordLength<double>.Compute(
            observedSharpe: 2.0,
            benchmarkSharpe: 1.0,
            skewness: 0.0,
            kurtosis: 3.0,
            significanceLevel: 0.05,
            tradingDaysPerYear: 252);

        result.Should().BeApproximately(688.2080373402348, 1e-5);
        (result / 252.0).Should().BeApproximately(2.73, 0.005);
    }
}
