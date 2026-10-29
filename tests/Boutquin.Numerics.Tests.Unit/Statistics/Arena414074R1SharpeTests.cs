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

using System.Numerics;
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

/// <summary>
/// Regression pins for the deflated Sharpe ratio, probability of backtest overfitting,
/// GT-inspired score and minimum track record length. Expected values come from exact
/// closed-form algebra on the stated inputs (population/sample moments of small integer
/// offsets, exact rational cumulative-return regression), never from the production code.
/// </summary>
public sealed class Arena414074R1SharpeTests
{
    // ---- DSR: a common location must not erase the representable centered deviations ----

    [Fact]
    public void Dsr_DecimalTranslatedReturns_KeepPopulationMomentsOfTheOffsets()
    {
        // returns = a + {0,2,4,6}: mean a+3, population variance 5, skewness 0, kurtosis 1.64
        // (the offsets' own moments), so Sharpe = (a+3)/sqrt(5) with one trial and one period/year.
        var a = 70000000000000000000000000000m;
        decimal[] returns = [a, a + 2m, a + 4m, a + 6m];

        var result = DeflatedSharpeRatio<decimal>.ComputeFromReturns(returns, 1, 1);

        var expectedSharpe = ((double)(a + 3m)) / Math.Sqrt(5.0);
        ((double)result.DeflatedSharpe).Should().BeApproximately(expectedSharpe, expectedSharpe * 1e-13);
        // z = Sharpe*sqrt(3)/sqrt(1 + (1.64-1)/4*Sharpe^2); p = 1 - Phi(z) = 7.4511678962138e-6.
        ((double)result.PValue).Should().BeApproximately(7.4511678962138465e-06, 1e-11);
    }

    [Fact]
    public void Dsr_DoubleTranslatedReturns_KeepPopulationMomentsOfTheOffsets()
    {
        // Same offsets at a location of 1e16 (spacing 2 in double): all four values are exact.
        double[] returns = [1e16, 1e16 + 2, 1e16 + 4, 1e16 + 6];

        var result = DeflatedSharpeRatio<double>.ComputeFromReturns(returns, 1, 1);

        // Mean of the stored values is 1e16+3 (population variance 5).
        var expectedSharpe = 10000000000000003.0 / Math.Sqrt(5.0);
        result.DeflatedSharpe.Should().BeApproximately(expectedSharpe, expectedSharpe * 1e-13);
    }

    // ---- DSR: skew*SR and the quadratic term both overflow ----

    [Theory]
    [InlineData(1e308, 2.0, 0.0023388674905236322)]
    [InlineData(-1e308, -2.0, 0.9976611325094764)]
    public void Dsr_OverflowingSkewAndQuadraticTerms_KeepNormalLimit(double sharpe, double skew, double expectedP)
    {
        // Five observations, one trial, kurtosis 3: z -> 2*sqrt(2)*sign as |SR| -> infinity.
        DeflatedSharpeRatio<double>.Compute(sharpe, 1, 5, skew, 3, 1).PValue
            .Should().BeApproximately(expectedP, 1e-13);
        DeflatedSharpeRatio<double>.ComputeWithTrialVariance(sharpe, 1, 0, 5, skew, 3, 1).PValue
            .Should().BeApproximately(expectedP, 1e-13);
        DeflatedSharpeRatio<double>.ComputeWithTrialMoments(sharpe, 1, 0, 0, 5, skew, 3, 1).PValue
            .Should().BeApproximately(expectedP, 1e-13);
    }

    [Theory]
    [InlineData(1e308, 2.0, 0.0)]
    [InlineData(-1e308, -2.0, 1.0)]
    public void Dsr_OverflowingSkewTermWithNonPositiveVariance_TakesTheGaussianVariance(double sharpe, double skew, double expectedP)
    {
        // Unit kurtosis with |skew*SR| beyond range gives a negative variance estimate (-Infinity before
        // scaling, -2e-308 after). As for any negative estimate the Gaussian variance applies, so the test
        // statistic is +/-Infinity and the p-value saturates at 0 or 1 instead of becoming NaN or 0.5.
        DeflatedSharpeRatio<double>.Compute(sharpe, 1, 5, skew, 1, 1).PValue.Should().Be(expectedP);
    }

    [Fact]
    public void DsrHalf_AnnualSharpeBeyondHalfRange_IsStillRefused()
    {
        // Daily Sharpe 1025 at 2e9 periods per year is 4.6e7 annualized, beyond binary16: the finite-Sharpe
        // guard of the shared evaluation must refuse it rather than pass +Infinity on.
        Half[] returns = [(Half)0.5, (Half)0.5009765625, (Half)0.5, (Half)0.5009765625];

        var act = () => DeflatedSharpeRatio<Half>.ComputeFromReturns(returns, 1, 2_000_000_000);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("observedSharpe");
    }

    // ---- PBO: translation must not create a false tie ----

    [Fact]
    public void Pbo_DecimalTranslatedPanel_KeepsStrictSharpeOrder()
    {
        // A = a+{0,2,4,6}, B = a+{0,4,8,12}, each repeated: sample SDs sqrt(20/3) and twice that,
        // means a+3 and a+6, so A has the larger Sharpe in every half. A wins in-sample and is the
        // out-of-sample best of two: omega = 2/3, logit = ln 2, PBO = 0.
        var a = 7e28m;
        var panel = new decimal[2, 8];
        decimal[] first = [0, 2, 4, 6, 0, 2, 4, 6];
        decimal[] second = [0, 4, 8, 12, 0, 4, 8, 12];
        for (var t = 0; t < 8; t++)
        {
            panel[0, t] = a + first[t];
            panel[1, t] = a + second[t];
        }

        var result = ProbabilityOfBacktestOverfitting<decimal>.Compute(panel, 2);

        result.Pbo.Should().Be(0m);
        ((double)result.LogitMedian).Should().BeApproximately(Math.Log(2.0), 1e-12);
    }

    [Fact]
    public void Pbo_DecimalLargeOffsetTinyRelativeSpread_RanksFoldsByExactSharpe()
    {
        // m = 1e15; strategy 0 = m +/- 1 and strategy 1 = m +/- 2 in every two-observation fold:
        // sample SDs sqrt(2) and 2*sqrt(2) around the same mean m, so strategy 0 is the strict
        // winner in both halves (omega = 2/3, logit = ln 2, PBO = 0).
        var m = 1000000000000000m;
        decimal[,] panel =
        {
            { m - 1, m + 1, m - 1, m + 1 },
            { m - 2, m + 2, m - 2, m + 2 },
        };

        var result = ProbabilityOfBacktestOverfitting<decimal>.Compute(panel, 2);

        result.Pbo.Should().Be(0m);
        ((double)result.LogitMedian).Should().BeApproximately(Math.Log(2.0), 1e-12);
    }

    // ---- GT-inspired: mean/SD, equity-curve R^2 and an SD beyond the type range ----

    [Fact]
    public void Gt_DoubleTranslatedReturns_KeepSampleStandardDeviationOfTheOffsets()
    {
        // Values 1e16+{0,2,4,6}: mean 1e16+3, sample SD sqrt(20/3); z = mean*sqrt(4)/SD.
        double[] returns = [1e16, 1e16 + 2, 1e16 + 4, 1e16 + 6];
        var expectedZ = 10000000000000003.0 * 2.0 / Math.Sqrt(20.0 / 3.0);

        var result = GtInspiredScore<double>.Compute(returns);

        result.Z.Should().BeApproximately(expectedZ, expectedZ * 1e-13);
    }

    [Fact]
    public void Gt_FloatTranslatedReturns_KeepSampleStandardDeviationOfTheOffsets()
    {
        // 1.6e7 + {0,2,4,6} is exact in float (spacing 1); z = (1.6e7+3)*2/sqrt(20/3).
        float[] returns = [1.6e7f, 1.6e7f + 2, 1.6e7f + 4, 1.6e7f + 6];
        var expectedZ = 16000003.0 * 2.0 / Math.Sqrt(20.0 / 3.0);

        var result = GtInspiredScore<float>.Compute(returns);

        ((double)result.Z).Should().BeApproximately(expectedZ, expectedZ * 1e-5);
    }

    [Fact]
    public void Gt_LargeFirstReturn_DoesNotChangeTheEquityCurveRSquared()
    {
        // The first return only shifts the whole equity curve by a constant, which cannot
        // change a correlation. Exact rational R^2 of the stored returns (squared correlation of
        // the running sum with 1..N) is 1/21 for every first return.
        long[] returns = [10_000_000_000_000_000, 2, -2, 2, -2, 2, -2, 2];
        var expected = ExactCumulativeRSquared(returns);
        expected.Should().BeApproximately(1.0 / 21.0, 1e-15);

        var result = GtInspiredScore<double>.Compute(returns.Select(r => (double)r).ToArray());

        result.RSquared.Should().BeApproximately(expected, 1e-12);
    }

    [Fact]
    public void Gt_LargeFirstReturnFloat_DoesNotChangeTheEquityCurveRSquared()
    {
        float[] returns = [1.6e7f, 2, -2, 2, -2, 2, -2, 2];

        var result = GtInspiredScore<float>.Compute(returns);

        ((double)result.RSquared).Should().BeApproximately(1.0 / 21.0, 1e-6);
    }

    [Fact]
    public void Gt_DoubleSampleStandardDeviationBeyondRange_KeepsFiniteZ()
    {
        // {-s, s, s}: z = (s/3)*sqrt(3)/(s*sqrt(4/3)) = 1/2 for every scale s, even when the
        // sample SD (1.1547*s) exceeds the type range.
        double[] returns = [-1.7e308, 1.7e308, 1.7e308];

        var result = GtInspiredScore<double>.Compute(returns, 0.0, 1e-6, 2);

        result.Z.Should().BeApproximately(0.5, 1e-12);
        // Marginal branch 100*(1 - exp(-|z - 1|)) at z = 1/2.
        result.Score.Should().BeApproximately(100.0 * (1.0 - Math.Exp(-0.5)), 1e-9);
    }

    [Fact]
    public void Gt_FloatSampleStandardDeviationBeyondRange_KeepsFiniteZ()
    {
        float[] returns = [-3e38f, 3e38f, 3e38f];

        var result = GtInspiredScore<float>.Compute(returns, 0f, 1e-6f, 2);

        ((double)result.Z).Should().BeApproximately(0.5, 1e-5);
    }

    [Fact]
    public void Gt_HalfSampleStandardDeviationBeyondRange_KeepsFiniteZ()
    {
        Half[] returns = [(Half)(-60000), (Half)60000, (Half)60000];

        var result = GtInspiredScore<Half>.Compute(returns, Half.Zero, (Half)1e-6, 2);

        ((double)result.Z).Should().BeApproximately(0.5, 2e-3);
    }

    [Fact]
    public void Gt_HalfSampleStandardDeviationBeyondRange_DefaultOverloadKeepsFiniteZ()
    {
        // The default minimum-observation threshold (50) takes the small-sample branch, which
        // reports the Half-typed z rather than the widened one.
        Half[] returns = [(Half)(-60000), (Half)60000, (Half)60000];

        var result = GtInspiredScore<Half>.Compute(returns);

        ((double)result.Z).Should().BeApproximately(0.5, 2e-3);
    }

    [Fact]
    public void Gt_DoubleDefaultOverload_SampleStandardDeviationBeyondRange_KeepsFiniteZ()
    {
        double[] returns = [-1.7e308, 1.7e308, 1.7e308];

        var result = GtInspiredScore<double>.Compute(returns);

        result.Z.Should().BeApproximately(0.5, 1e-12);
    }

    // ---- MinTRL: confidence at or below one half needs only the first observation ----

    [Theory]
    [InlineData(0.5)]
    [InlineData(0.5000001)]
    [InlineData(0.6)]
    [InlineData(0.9)]
    [InlineData(0.95)]
    [InlineData(0.999999)]
    public void MinTrl_ConfidenceAtOrBelowOneHalf_ReturnsOneObservation(double alpha)
    {
        // BLdP 2012 eq. 12: the minimum n with PSR(n) >= 1 - alpha. PSR(1) = Phi(0) = 1/2 already
        // meets any target 1 - alpha <= 1/2, so the minimum is exactly one observation.
        MinimumTrackRecordLength<double>.Compute(2, 1, 0, 3, alpha, 252).Should().Be(1.0);
    }

    [Theory]
    [InlineData(0.5000001)]
    [InlineData(0.9)]
    [InlineData(0.999999)]
    public void MinTrl_ConfidenceAtOrBelowOneHalf_DefaultedOverloadReturnsOneObservation(double alpha)
    {
        MinimumTrackRecordLength<double>.Compute(2, alpha, 252).Should().Be(1.0);
    }

    [Theory]
    [InlineData(0.5000001)]
    [InlineData(0.9)]
    [InlineData(0.999999)]
    public void MinTrl_ConfidenceAtOrBelowOneHalf_FloatReturnsOneObservation(double alpha)
    {
        MinimumTrackRecordLength<float>.Compute(2f, 1f, 0f, 3f, alpha, 252).Should().Be(1f);
    }

    [Theory]
    [InlineData(0.5000001)]
    [InlineData(0.9)]
    [InlineData(0.999999)]
    public void MinTrl_ConfidenceAtOrBelowOneHalf_HalfReturnsOneObservation(double alpha)
    {
        MinimumTrackRecordLength<Half>.Compute((Half)2, (Half)1, Half.Zero, (Half)3, alpha, 252).Should().Be(Half.One);
    }

    [Fact]
    public void MinTrl_AlphaGrid_IsNonIncreasingAndKeepsEq13BelowOneHalf()
    {
        var previous = double.PositiveInfinity;
        for (var step = 1; step <= 99; step++)
        {
            var alpha = step / 100.0;
            var length = MinimumTrackRecordLength<double>.Compute(2, 1, 0, 3, alpha, 252);
            length.Should().BeLessThanOrEqualTo(previous, $"a smaller alpha (greater confidence) never needs fewer observations; alpha = {alpha}");
            length.Should().BeGreaterThanOrEqualTo(1.0);
            previous = length;
        }

        // In-domain branch is eq. 13: 1 + (1 + (3 - 1)/4*SR^2)*(z/(SR - SR*))^2 with the per-period
        // SR = 2/sqrt(252), SR* = 1/sqrt(252) and z the standard-normal quantile at confidence 0.6
        // (0.2533471031357997).
        var srPerPeriod = 2.0 / Math.Sqrt(252.0);
        var ratio = 0.2533471031357997 / (1.0 / Math.Sqrt(252.0));
        var expected = 1.0 + (1.0 + 0.5 * srPerPeriod * srPerPeriod) * ratio * ratio;
        MinimumTrackRecordLength<double>.Compute(2, 1, 0, 3, 0.4, 252).Should().BeApproximately(expected, 1e-9);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(1.5)]
    [InlineData(-0.1)]
    public void MinTrl_AlphaOutsideUnitInterval_IsStillRejected(double alpha)
    {
        var act = () => MinimumTrackRecordLength<double>.Compute(2, 1, 0, 3, alpha, 252);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // ---- DSR Half: the observation count must not pass through a Half-narrowed years fraction ----

    [Fact]
    public void DsrHalf_ShortSeriesAtVeryHighFrequency_UsesTheObservationCount()
    {
        // Six binary16 returns at 2e9 periods per year: backtest years = 3e-9 is below the least
        // Half subnormal, but the observation count (6) and every output are representable.
        Half[] returns = [Half.Zero, (Half)0.01, (Half)0.02, (Half)0.01, Half.Zero, (Half)0.02];
        var wide = returns.Select(r => (double)r).ToArray();

        var expected = DeflatedSharpeRatio<double>.ComputeFromReturns(wide, 10, 2_000_000_000);
        var actual = DeflatedSharpeRatio<Half>.ComputeFromReturns(returns, 10, 2_000_000_000);

        // Independent of the sibling: eq. 1 with T = 6 gives SR0 = 31491.966026914997.
        expected.ExpectedMaxSharpe.Should().BeApproximately(31491.966026914997, 1e-6);
        ((double)actual.ExpectedMaxSharpe).Should().BeApproximately(expected.ExpectedMaxSharpe, 32.0);
        ((double)actual.DeflatedSharpe).Should().BeApproximately(expected.DeflatedSharpe, 64.0);
        ((double)actual.PValue).Should().BeApproximately(expected.PValue, 0.02);
    }

    private static double ExactCumulativeRSquared(long[] returns)
    {
        var n = returns.Length;
        BigInteger sumX = 0, sumY = 0, sumXX = 0, sumYY = 0, sumXY = 0, equity = 0;
        for (var i = 0; i < n; i++)
        {
            BigInteger x = i + 1;
            equity += returns[i];
            sumX += x;
            sumY += equity;
            sumXX += x * x;
            sumYY += equity * equity;
            sumXY += x * equity;
        }

        var covariance = (n * sumXY) - (sumX * sumY);
        var varianceX = (n * sumXX) - (sumX * sumX);
        var varianceY = (n * sumYY) - (sumY * sumY);
        return (double)(covariance * covariance) / ((double)varianceX * (double)varianceY);
    }
}
