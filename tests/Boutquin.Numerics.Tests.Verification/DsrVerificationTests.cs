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

namespace Boutquin.Numerics.Tests.Verification;

public sealed class DsrVerificationTests : CrossLanguageVerificationBase
{
    // Goldens come from tests/Verification/generate_dsr_vectors.py, which transcribes
    // Bailey & López de Prado (2014) eqs. 1–2 and (2012) eq. 13 directly (SciPy
    // norm.ppf / norm.cdf), with Sharpe ratios and T at the same per-period frequency.
    // The papers' own worked examples are pinned as independent oracles as well.

    [Fact]
    public void DeflatedSharpe_Scalar_MatchesReferenceImplementation()
    {
        using var doc = LoadVector("dsr");
        var returns = GetDecimalArray(doc.RootElement.GetProperty("returns"));
        var numTrials = doc.RootElement.GetProperty("num_trials").GetInt32();
        var tradingDays = doc.RootElement.GetProperty("trading_days_per_year").GetInt32();
        var expectedScalar = doc.RootElement.GetProperty("deflated_sharpe_scalar_100_trials").GetDouble();
        var expectedMaxSharpe = doc.RootElement.GetProperty("expected_max_sharpe_100_trials").GetDouble();
        var expectedPValue = doc.RootElement.GetProperty("deflated_sharpe_statistic_100_trials").GetDouble();

        var result = DeflatedSharpeRatio.ComputeFromReturns(returns, numTrials, tradingDays);

        // SR̂₀ at the null trials' variance 1/(T−1), annualized.
        AssertScalarWithin((double)result.ExpectedMaxSharpe, expectedMaxSharpe, PrecisionNumeric, "SR̂₀ (annualized)");
        AssertScalarWithin((double)result.DeflatedSharpe, expectedScalar, PrecisionStatistical, "SR − SR̂₀");
        AssertScalarWithin((double)result.PValue, expectedPValue, PrecisionNumeric, "1 − DSR");
    }

    [Fact]
    public void DeflatedSharpe_DisclosedTrialVariance_MatchesReferenceImplementation()
    {
        using var doc = LoadVector("dsr");
        var root = doc.RootElement;
        var tradingDays = root.GetProperty("trading_days_per_year").GetInt32();
        var observations = root.GetProperty("returns").GetArrayLength();

        var result = DeflatedSharpeRatio<double>.ComputeWithTrialVariance(
            observedSharpe: root.GetProperty("sharpe").GetDouble(),
            numTrials: root.GetProperty("num_trials").GetInt32(),
            trialSharpeVariance: root.GetProperty("trial_sharpe_variance_annualized").GetDouble(),
            backTestYears: (double)observations / tradingDays,
            skewness: root.GetProperty("skew").GetDouble(),
            kurtosis: root.GetProperty("kurtosis_pearson").GetDouble(),
            tradingDaysPerYear: tradingDays);

        AssertScalarWithin(result.ExpectedMaxSharpe, root.GetProperty("expected_max_sharpe_100_trials_disclosed_variance").GetDouble(), PrecisionNumeric, "SR̂₀ (annualized)");
        AssertScalarWithin(result.PValue, root.GetProperty("deflated_sharpe_statistic_100_trials_disclosed_variance").GetDouble(), PrecisionNumeric, "1 − DSR");
    }

    [Fact]
    public void DeflatedSharpe_PaperWorkedExample_Is0Point9004()
    {
        // Bailey & López de Prado (2014), p. 10: annualized SR 2.5 over five years of
        // daily data (250 per year), N = 100 trials, V[{SR̂ₙ}] = 1/2, skewness −3,
        // kurtosis 10 → SR̂₀ = 0.1132 (non-annualized) and DSR = 0.9004.
        using var doc = LoadVector("dsr");
        var root = doc.RootElement;

        var result = DeflatedSharpeRatio<double>.ComputeWithTrialVariance(
            observedSharpe: 2.5,
            numTrials: 100,
            trialSharpeVariance: 0.5,
            backTestYears: 5.0,
            skewness: -3.0,
            kurtosis: 10.0,
            tradingDaysPerYear: 250);

        var dsr = 1.0 - result.PValue;
        var thresholdPerPeriod = result.ExpectedMaxSharpe / Math.Sqrt(250.0);

        AssertScalarWithin(dsr, root.GetProperty("paper_dsr").GetDouble(), PrecisionNumeric, "DSR vs SciPy eq. 2");
        AssertScalarWithin(thresholdPerPeriod, root.GetProperty("paper_dsr_threshold_per_period").GetDouble(), PrecisionNumeric, "SR̂₀ vs SciPy eq. 1");
        dsr.Should().BeApproximately(0.9004, 5e-5, "the paper prints DSR = 0.9004");
        thresholdPerPeriod.Should().BeApproximately(0.1132, 5e-5, "the paper prints SR̂₀ = 0.1132");
    }

    [Theory]
    [InlineData(46, -3.0, 10.0, 0.9505017068755787, 1.5868684126287262)]
    [InlineData(88, 0.0, 3.0, 0.9504908166760138, 1.7573687975531247)]
    public void DeflatedSharpe_PaperAlternativeTrialCounts_Match0Point9505(
        int numTrials,
        double skewness,
        double kurtosis,
        double expectedDsr,
        double expectedAnnualThreshold)
    {
        // Bailey & López de Prado (2014), p. 10, worked example following Exhibit 2:
        // N = 46 with skewness -3 / raw kurtosis 10, or N = 88 with Normal returns,
        // both print DSR = 0.9505. Annualized SR, trial variance and frequency are
        // unchanged from the N = 100 example above.
        // Unrounded goldens independently evaluate eqs. 1-2 with Python's
        // statistics.NormalDist.inv_cdf/cdf, not the library's Acklam/erfc routines:
        // SR = 2.5/sqrt(250), V = 0.5/250, T = 1250, gamma = 0.5772156649015329.
        var result = DeflatedSharpeRatio<double>.ComputeWithTrialVariance(
            observedSharpe: 2.5,
            numTrials: numTrials,
            trialSharpeVariance: 0.5,
            backTestYears: 5.0,
            skewness: skewness,
            kurtosis: kurtosis,
            tradingDaysPerYear: 250);

        var dsr = 1.0 - result.PValue;
        dsr.Should().BeApproximately(expectedDsr, 1e-10, "eq. 2 uses the frequency-matched Sharpe and raw kurtosis");
        result.ExpectedMaxSharpe.Should().BeApproximately(expectedAnnualThreshold, 1e-10, "eq. 1 uses the disclosed annualized trial variance");
        dsr.Should().BeApproximately(0.9505, 5e-5, "the paper prints four decimal places");
        dsr.Should().BeGreaterThan(0.95, "both worked examples meet the investor's 95% confidence requirement");
    }

    [Fact]
    public void MinimumTrackRecordLength_MatchesReferenceImplementation()
    {
        // Inputs come from dsr.json rather than being recomputed from `returns` so the
        // moment-convention difference between SciPy and the C# moment path is factored
        // out of this test — it compares the MinTRL arithmetic only. The generator
        // converts the annualized Sharpe to per-period before applying eq. 13.
        using var doc = LoadVector("dsr");
        var observedSharpe = doc.RootElement.GetProperty("sharpe").GetDouble();
        var skew = doc.RootElement.GetProperty("skew").GetDouble();
        var kurtosis = doc.RootElement.GetProperty("kurtosis_pearson").GetDouble();
        var tradingDays = doc.RootElement.GetProperty("trading_days_per_year").GetInt32();

        // T* is a number of observations at the per-period frequency.
        var expectedObservations = doc.RootElement.GetProperty("min_trl_sr0_conf95").GetDouble();

        var resultObservations = MinimumTrackRecordLength.Compute(
            observedSharpe: observedSharpe,
            benchmarkSharpe: 0.0,
            skewness: skew,
            kurtosis: kurtosis,
            significanceLevel: 0.05,
            tradingDaysPerYear: tradingDays);

        AssertScalarWithin(resultObservations, expectedObservations, PrecisionExact, "MinTRL (observations) vs reference");

        // Boundary: MinTRL is +infinity when the observed Sharpe does not exceed the benchmark.
        var unreachable = MinimumTrackRecordLength.Compute(
            observedSharpe: 0.5,
            benchmarkSharpe: 1.0,
            skewness: skew,
            kurtosis: kurtosis,
            significanceLevel: 0.05,
            tradingDaysPerYear: tradingDays);
        unreachable.Should().Be(double.PositiveInfinity);
    }

    [Fact]
    public void MinimumTrackRecordLength_PaperWorkedExample_Is688Observations()
    {
        // Bailey & López de Prado (2012): an annualized Sharpe of 2 needs 2.73 years of
        // daily data (688.2 observations) to exceed a benchmark of 1 at 95% confidence.
        using var doc = LoadVector("dsr");

        var observations = MinimumTrackRecordLength.Compute(
            observedSharpe: 2.0,
            benchmarkSharpe: 1.0,
            skewness: 0.0,
            kurtosis: 3.0,
            significanceLevel: 0.05,
            tradingDaysPerYear: 252);

        AssertScalarWithin(observations, doc.RootElement.GetProperty("paper_min_trl_observations").GetDouble(), PrecisionNumeric, "MinTRL vs SciPy eq. 13");
        observations.Should().BeApproximately(688.2, 0.05, "the paper prints 688.2 observations");
        (observations / 252.0).Should().BeApproximately(2.73, 0.005, "the paper prints 2.73 years");
    }

    [Theory]
    [InlineData(52, 147.0993465211522, 2.83)]
    [InlineData(12, 38.87760835733575, 3.24)]
    public void MinimumTrackRecordLength_PaperWeeklyAndMonthlyFrequencies_MatchPrintedYears(
        int observationsPerYear,
        double expectedObservations,
        double printedYears)
    {
        // Bailey & López de Prado (2012), p. 11, Figures 9-10: annualized SR 2
        // vs benchmark 1, IID Normal returns and 95% confidence require 2.83 years
        // of weekly returns or 3.24 years of monthly returns.
        // Independently evaluating eq. 13 with Python statistics.NormalDist gives
        // z_0.95 = 1.6448536269514715. For these inputs its observation-count
        // expression simplifies to 1 + (frequency + 2) * z_0.95^2.
        var observations = MinimumTrackRecordLength<double>.Compute(
            observedSharpe: 2.0,
            benchmarkSharpe: 1.0,
            skewness: 0.0,
            kurtosis: 3.0,
            significanceLevel: 0.05,
            tradingDaysPerYear: observationsPerYear);

        observations.Should().BeApproximately(expectedObservations, 1e-9, "eq. 13 returns a number of observations at the supplied frequency");
        (observations / observationsPerYear).Should().BeApproximately(printedYears, 0.005, "the paper prints track record years to two decimal places");
    }
}
