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
/// Regression pins for the Harvey-Liu (2015) haircut Sharpe: the annualized
/// Sharpe and the observation count must be put on the same sampling
/// frequency before the t-ratio is formed.
/// </summary>
public sealed class ArenaR1B10HaircutSharpeTests
{
    /// <summary>
    /// With a single trial there is no multiple-testing adjustment
    /// (p_M = p_S), so the haircut Sharpe equals the observed Sharpe. The
    /// annualized-SR-times-sqrt(daily-T) statistic drove p_S to 0 and the
    /// inverse to its clamp, fabricating a 0.84 "haircut" for N = 1.
    /// </summary>
    [Fact]
    public void SingleTrial_AnnualSharpeOnDailyData_HasNoHaircut()
    {
        var result = HaircutSharpe<double>.Compute(observedSharpe: 1.0, numTrials: 1, backTestYears: 10.0, tradingDaysPerYear: 252);

        result.HaircutAmount.Should().BeApproximately(0.0, 1e-9, "one trial means p_M = p_S and therefore HSR = SR");
        result.HaircutSharpe.Should().BeApproximately(1.0, 1e-9);
    }

    /// <summary>
    /// The t-ratio of an annualized Sharpe is SR · √years, not SR · √(years · 252).
    /// A ten-year daily backtest whose annual Sharpe gives SR · √10 = 1.959964
    /// (the normal 97.5% quantile) is significant at exactly the two-sided 5%
    /// level up to the t(2519)-versus-normal gap (≈ 1e-4); the per-period count
    /// inflates the statistic by √252 and returns p ≈ 0.
    /// </summary>
    [Fact]
    public void AnnualSharpeOnDailyData_TwoSidedFivePercentQuantile_GivesFivePercent()
    {
        var result = HaircutSharpe<double>.Compute(
            observedSharpe: 1.959964 / Math.Sqrt(10.0), numTrials: 1, backTestYears: 10.0, tradingDaysPerYear: 252);

        result.BonferroniPValue.Should().BeApproximately(0.05, 5e-4, "SR · √years is the t-ratio; z(0.975) = 1.959964");
    }

    /// <summary>
    /// The same per-period return series (per-period Sharpe 0.05 over 600
    /// observations) must give the same p-values whether its calendar is
    /// labelled monthly, weekly or daily: SR_annual · sqrt(years) is
    /// invariant to the label, SR_annual · sqrt(years · periods) is not.
    /// </summary>
    [Theory]
    [InlineData(52)]
    [InlineData(252)]
    public void SamePerPeriodData_DifferentCalendarLabel_GivesSamePValues(int periodsPerYear)
    {
        const double PerPeriodSharpe = 0.05;
        const double Observations = 600.0;

        var monthly = HaircutSharpe<double>.Compute(
            PerPeriodSharpe * Math.Sqrt(12), numTrials: 50, backTestYears: Observations / 12, tradingDaysPerYear: 12);
        var relabelled = HaircutSharpe<double>.Compute(
            PerPeriodSharpe * Math.Sqrt(periodsPerYear), numTrials: 50, backTestYears: Observations / periodsPerYear, tradingDaysPerYear: periodsPerYear);

        relabelled.BonferroniPValue.Should().BeApproximately(monthly.BonferroniPValue, 1e-12);
        relabelled.BhyPValue.Should().BeApproximately(monthly.BhyPValue, 1e-12);
    }

    /// <summary>
    /// Harvey-Liu eq. 3 is two-sided on a Student t with T − 1 degrees of
    /// freedom. With T = 10 annual observations, the textbook critical values
    /// t(0.995; 9) = 3.249836 and t(0.975; 9) = 2.262157 pin both directions:
    /// a t-ratio of 3.249836 has p_S = 0.01, five trials make p_M = 0.05, and
    /// eq. 5 maps p_M = 0.05 back to a t-ratio of 2.262157.
    /// </summary>
    [Fact]
    public void TwoSidedStudentT_TextbookCriticalValues_PinPValueAndHaircut()
    {
        var years = 10.0;
        var result = HaircutSharpe<double>.Compute(
            observedSharpe: 3.249836 / Math.Sqrt(years), numTrials: 5, backTestYears: years, tradingDaysPerYear: 1);

        result.BonferroniPValue.Should().BeApproximately(0.05, 1e-6, "5 × two-sided t(9) tail at 3.249836 = 5 × 0.01");
        result.HaircutSharpe.Should().BeApproximately(2.262157 / Math.Sqrt(years), 1e-6, "eq. 5 inverts p_M = 0.05 to t(0.975; 9)");
    }

    /// <summary>
    /// A two-sided test is symmetric in the sign of the Sharpe: a short
    /// position in the same strategy is equally significant, and its haircut
    /// shrinks the magnitude toward zero.
    /// </summary>
    [Fact]
    public void TwoSidedTest_NegatedSharpe_IsSymmetric()
    {
        var positive = HaircutSharpe<double>.Compute(observedSharpe: 0.9, numTrials: 20, backTestYears: 8.0, tradingDaysPerYear: 12);
        var negative = HaircutSharpe<double>.Compute(observedSharpe: -0.9, numTrials: 20, backTestYears: 8.0, tradingDaysPerYear: 12);

        negative.BonferroniPValue.Should().BeApproximately(positive.BonferroniPValue, 1e-15);
        negative.HaircutSharpe.Should().BeApproximately(-positive.HaircutSharpe, 1e-15);
        positive.HaircutSharpe.Should().BeInRange(0.0, 0.9);
    }

    /// <summary>
    /// BHY is a step-up recursion over the ordered p-values with factor
    /// M·c(M)/i, c(M) = Σ 1/j (Harvey-Liu 2015 p. 16). The tested strategy is
    /// the best of M trials (rank 1) and the other trials carry no evidence
    /// (p = 1), so its adjusted value is min(1, M·c(M)·p_S): with p_S = 0.01
    /// and M = 5 that is 5 × (1 + 1/2 + 1/3 + 1/4 + 1/5) × 0.01 = 0.1141667.
    /// </summary>
    [Fact]
    public void Bhy_TopRankedStrategy_UsesStepUpFactorMTimesHarmonic()
    {
        var years = 10.0;
        var result = HaircutSharpe<double>.Compute(
            observedSharpe: 3.249836 / Math.Sqrt(years), numTrials: 5, backTestYears: years, tradingDaysPerYear: 1);

        result.BhyPValue.Should().BeApproximately(5.0 * (137.0 / 60.0) * 0.01, 1e-6);
    }

    /// <summary>
    /// The Student-t test needs T − 1 &gt; 0 degrees of freedom. One annual
    /// observation has none, so the call is refused rather than answered
    /// with a degenerate p-value of 0.
    /// </summary>
    [Fact]
    public void SingleObservation_NoDegreesOfFreedom_IsRejected()
    {
        var act = () => HaircutSharpe<double>.Compute(observedSharpe: 1.0, numTrials: 1, backTestYears: 1.0, tradingDaysPerYear: 1);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("backTestYears");
    }
}
