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
/// Statistics-cluster review-fix regression tests (2026-06-28 review batch). Each test
/// pins a corrected numerical contract — derived by hand from the published method — so a
/// future refactor cannot silently re-introduce the bug it guards.
/// </summary>
public sealed class StatisticsReviewFixTests
{
    // ---------------------------------------------------------------------
    // S7 — MinimumTrackRecordLength returns a NUMBER OF OBSERVATIONS, not
    // observations/tradingDaysPerYear. Bailey & López de Prado (2012) MinTRL.
    // Eq. 13 takes the Sharpe ratios at the per-period frequency (BLdP 2012,
    // fn. 5), so the annualized SR=1 is first divided by √252.
    // Hand-computed: SR=1/√252, SR*=0, skew=0, kurt=3, alpha=0.05.
    //   z_0.95 = 1.6448536269514722
    //   nonGaussianity = 1 + (3-1)/4 * SR^2 = 1 + 1/504
    //   T* = 1 + (1 + 1/504) * z^2 * 252 = 684.1497221590919  (observations)
    // Dividing by 252 would give 2.71 — a length in years, not observations.
    // ---------------------------------------------------------------------
    [Fact]
    public void MinTRL_ReturnsObservationCount_NotAnnualizedByTradingDays()
    {
        const double expectedObservations = 684.1497221590919;

        var result = MinimumTrackRecordLength<double>.Compute(
            observedSharpe: 1.0,
            benchmarkSharpe: 0.0,
            skewness: 0.0,
            kurtosis: 3.0,
            significanceLevel: 0.05,
            tradingDaysPerYear: 252);

        result.Should().BeApproximately(expectedObservations, 1e-9);
        // Guard against re-introducing the /252 divisor.
        result.Should().NotBeApproximately(expectedObservations / 252.0, 1e-6);
    }

    // ---------------------------------------------------------------------
    // F-L7-01 — The GT-inspired variant is the MULTIPLICATIVE ratio inspired by Sheppert
    // (2026) eq.(1) µ·ln(z)·r²/σ_d with the eq.(2) significance gate z, NOT an
    // additive weighted sum of a Sharpe + t-stat + consistency − drawdown.
    // The prior S13/T5 test pinned the old additive "significance component"
    // (a zero-floored annualized t-statistic) — a fabricated component the
    // published paper does not contain — so it pinned WRONG behavior and is
    // replaced here by a pin on the published z-gate.
    //
    // ln(z) is the significance gate: it anchors at z = 1 (ln(1) = 0) so a
    // strategy whose outperformance is exactly one standard error above the
    // benchmark contributes nothing to the composite utility. Independent
    // derivation: z = (µ − µ_m)/(σ/√N); for the chosen input z = 1 exactly,
    // so the z > 1 branch is NOT taken and the smooth-transition value at the
    // boundary is 100·(1 − e^(−|1−1|)) = 0.
    //
    // F-VS3-01 (de-masking note): at z = 1 EXACTLY, both candidate branch
    // formulas give score = 0 UNCONDITIONALLY — this is a mathematical
    // identity, not a testing gap:
    //   marginal (z <= 1): 100·(1 − e^(−|1−1|)) = 100·(1 − e^0) = 0
    //   standard (z > 1):  −(µ·ln(1)·r²)/(σ_d+ε) = −(µ·0·r²)/(σ_d+ε) = 0
    // No choice of µ, r², or σ_d can un-mask a score-only assertion evaluated
    // AT z = 1: ln(1) = 0 zeroes the standard branch's formula too, so this
    // anchor test — by itself — CANNOT distinguish a correct "z <= 1" branch
    // boundary from a wrong "z < 1" one (which would misroute z = 1 into the
    // standard branch, coincidentally landing on 0 again). This anchor test
    // therefore only pins the SCORE VALUE at the boundary (score = 0, true
    // under either branch condition); the branch CONDITION itself (which
    // side of z = 1 takes which formula) is pinned by the companion
    // just-off-boundary tests in GtInspiredScoreTests
    // (Compute_JustAboveZEqualsOne_UsesStandardBranchFormula and
    // Compute_JustBelowZEqualsOne_UsesMarginalBranchFormula), which assert
    // against BOTH candidate formulas at z = 1±0.01 and would go RED if the
    // wrong formula executed on either side. Verified by mutation check:
    // flipping "z <= one" to "z < one" in ComputePiecewiseScore leaves THIS
    // test green (confirming the identity above) but is orthogonal to the
    // off-boundary tests, which do not move under that specific mutation
    // either (it only changes behavior exactly at z = 1) — the mutation is
    // instead caught structurally by this test's documented invariant: any
    // reviewer changing the boundary operator must reconcile this comment.
    // ---------------------------------------------------------------------
    [Fact]
    public void GtInspiredScore_SignificanceGate_AnchorsAtZEqualsOne()
    {
        // Constant outperformance: µ = 0.004 over each of 5 obs, sample sd of
        // the *deviations from µ* is non-zero only via the spread below.
        // Independent reference (/tmp/gtscore_independent.py style): chosen so
        // z lands at exactly 1.0 against the benchmark.
        double[] returns = [0.006, 0.002, 0.006, 0.002, 0.004];
        // µ = 0.004; sample sd (ddof=1) = 0.002; se = 0.002/√5 = 8.944271909999159e-4
        // benchmark µ_m = µ − se ⇒ z = (µ − µ_m)/se = 1 exactly.
        const double se = 8.944271909999159e-4;
        const double benchmarkMean = 0.004 - se;

        var result = GtInspiredScore<double>.Compute(returns, benchmarkMean, epsilon: 1e-6, minimumObservations: 1);

        result.Z.Should().BeApproximately(1.0, 1e-9);
        // At the z = 1 anchor the smooth-transition branch yields exactly 0.
        // (True under either branch condition — see de-masking note above;
        // the branch CONDITION is pinned by the off-boundary companion tests.)
        result.Score.Should().BeApproximately(0.0, 1e-9);

        // Corroborate against the independently-derived off-boundary values
        // (GtInspiredScoreTests) at z = 1±0.01 using the SAME returns
        // fixture, so a reader can confirm the marginal branch (this side,
        // z <= 1) is continuous with the standard branch (other side, z > 1)
        // at the shared boundary — the score does not jump discontinuously,
        // which would be a signal of a formula mismatch across the boundary.
        const double justBelowOneScore = 0.9950166250832115; // z = 0.99, marginal
        const double justAboveOneScore = -38.84225537863123; // z = 1.01, standard
        result.Score.Should().BeInRange(justAboveOneScore, justBelowOneScore + 1e-6);
    }

    // ---------------------------------------------------------------------
    // S26 — PBO counts logits AT OR BELOW the median (<= 0), per Bailey et al.
    // (2014) and the repo's own Python reference (pbo = mean(logits <= 0)). A
    // prior review cycle pinned the strictly-below-median (< 0) convention; that
    // pinned the wrong behavior — the published estimator and the on-disk
    // reference both include the at-median (logit == 0) boundary, reachable for
    // any odd strategy count (omega = 0.5). This test now pins the corrected
    // <= 0 semantics (finding F-L7-02).
    // Deterministic panel (n=3, t=8, splitCount=4) with two exact-zero logits
    // out of six folds:
    //   correct (<=0): 6/6 = 1.0
    //   old buggy (< 0): 4/6 = 0.6666...
    // ---------------------------------------------------------------------
    [Fact]
    public void Pbo_CountsAtOrBelowMedian_ExactZeroLogitCounted()
    {
        double[,] returns =
        {
            { -0.0146, 0.0139, 0.0106, -0.0098, -0.0002, -0.002, 0.0061, 0.0115 },
            { -0.0162, -0.0189, 0.0134, -0.0027, 0.0105, -0.0199, -0.0022, 0.0089 },
            { -0.0108, 0.0178, 0.0161, -0.0188, -0.019, 0.0017, 0.0176, -0.0048 },
        };

        var result = ProbabilityOfBacktestOverfitting<double>.Compute(returns, splitCount: 4);

        // All 6 folds are at or below the median; the two at-median folds are included.
        result.Pbo.Should().BeApproximately(1.0, 1e-12);
        result.Pbo.Should().NotBeApproximately(4.0 / 6.0, 1e-9, "the old < 0 count would yield 4/6");
    }

    // ---------------------------------------------------------------------
    // S27 — reference pins for the corrected GtInspiredScore and PBO
    // semantics, computed end-to-end from hand-derived values.
    // ---------------------------------------------------------------------
    [Fact]
    public void S27_GtInspiredScore_ReferencePin_VariantFormula()
    {
        // F-L7-01: pin the declared GT-inspired library variant (eq.(1) µ·ln(z)·r²/σ_d, Algorithm 1).
        // The prior S27 pinned the old additive weighted-sum formula with fabricated
        // (0.3,0.3,0.2,0.2) "ablation" weights — wrong behavior — so it is replaced.
        // Independent reference (declared variant formula, not a C# port):
        //   mu = 0.009916666666666666, z = 7.053452892053309,
        //   r2 = 0.9891432653097217, sigma_d = 0.0011547005383792514,
        //   score = -(mu*ln(z)*r2)/(sigma_d + 1e-6) = -16.580470598518065.
        double[] returns =
        [
            0.010, 0.012, 0.008, 0.015, 0.009, 0.011,
            0.013, -0.004, 0.014, 0.010, 0.012, 0.009,
        ];

        var result = GtInspiredScore<double>.Compute(returns, benchmarkMean: 0.0, epsilon: 1e-6, minimumObservations: 1);

        result.Mu.Should().BeApproximately(0.009916666666666666, 1e-15);
        result.Z.Should().BeApproximately(7.053452892053309, 1e-9);
        result.RSquared.Should().BeApproximately(0.9891432653097217, 1e-12);
        result.DownsideDeviation.Should().BeApproximately(0.0011547005383792514, 1e-15);
        result.Score.Should().BeApproximately(-16.580470598518065, 1e-9);
    }

    [Fact]
    public void S27_Pbo_ReferencePin_AtOrBelowMedian()
    {
        double[,] returns =
        {
            { -0.0146, 0.0139, 0.0106, -0.0098, -0.0002, -0.002, 0.0061, 0.0115 },
            { -0.0162, -0.0189, 0.0134, -0.0027, 0.0105, -0.0199, -0.0022, 0.0089 },
            { -0.0108, 0.0178, 0.0161, -0.0188, -0.019, 0.0017, 0.0176, -0.0048 },
        };

        var result = ProbabilityOfBacktestOverfitting<double>.Compute(returns, splitCount: 4);

        // F-L7-02: PBO counts logits <= 0 (paper + repo Python reference); all six
        // folds (including the two at-median, logit == 0) are counted → 6/6 = 1.0.
        result.LogitValues.Should().HaveCount(6);
        result.Pbo.Should().BeApproximately(1.0, 1e-12);
    }

    // ---------------------------------------------------------------------
    // S15 — generic MinimumTrackRecordLength<T> exposes the same defaults the
    // facade supplies (benchmark=0, skew=0, kurt=3, alpha=0.05, days=252).
    // ---------------------------------------------------------------------
    [Fact]
    public void MinTRL_Generic_DefaultedCall_MatchesExplicitArguments()
    {
        var defaulted = MinimumTrackRecordLength<double>.Compute(observedSharpe: 1.0);
        var explicitArgs = MinimumTrackRecordLength<double>.Compute(
            observedSharpe: 1.0,
            benchmarkSharpe: 0.0,
            skewness: 0.0,
            kurtosis: 3.0,
            significanceLevel: 0.05,
            tradingDaysPerYear: 252);

        defaulted.Should().Be(explicitArgs);
        defaulted.Should().BeApproximately(684.1497221590919, 1e-9);
    }

    // ====================================================================
    // Coverage gaps — guard-clause throws (B3..B8).
    // ====================================================================

    // --- B3: DeflatedSharpeRatio guards ---
    [Fact]
    public void Dsr_ComputeFromReturns_TooShort_Throws()
    {
        double[] returns = [0.01, 0.02, -0.01];
        var act = () => DeflatedSharpeRatio<double>.ComputeFromReturns(returns, numTrials: 10);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Dsr_ComputeFromReturns_NonPositiveTradingDays_Throws()
    {
        double[] returns = [0.01, 0.02, -0.01, 0.005, 0.0];
        var act = () => DeflatedSharpeRatio<double>.ComputeFromReturns(returns, numTrials: 10, tradingDaysPerYear: 0);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("tradingDaysPerYear");
    }

    [Fact]
    public void Dsr_ComputeFromReturns_ZeroVariance_Throws()
    {
        double[] returns = [0.01, 0.01, 0.01, 0.01, 0.01];
        var act = () => DeflatedSharpeRatio<double>.ComputeFromReturns(returns, numTrials: 10);
        act.Should().Throw<ArgumentException>();
    }

    // --- B4: ProbabilityOfBacktestOverfitting guards ---
    [Fact]
    public void Pbo_NullReturns_Throws()
    {
        var act = () => ProbabilityOfBacktestOverfitting<double>.Compute(null!, splitCount: 4);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Pbo_FewerThanTwoStrategies_Throws()
    {
        var returns = new double[1, 8];
        var act = () => ProbabilityOfBacktestOverfitting<double>.Compute(returns, splitCount: 4);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Pbo_OddSplitCount_Throws()
    {
        var returns = new double[3, 9];
        var act = () => ProbabilityOfBacktestOverfitting<double>.Compute(returns, splitCount: 3);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("splitCount");
    }

    [Fact]
    public void Pbo_TimeNotDivisibleBySplitCount_Throws()
    {
        var returns = new double[3, 10]; // 10 not divisible by 4.
        var act = () => ProbabilityOfBacktestOverfitting<double>.Compute(returns, splitCount: 4);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Pbo_BlockTooSmall_Throws()
    {
        var returns = new double[3, 4]; // blockSize = 1 < 2.
        var act = () => ProbabilityOfBacktestOverfitting<double>.Compute(returns, splitCount: 4);
        act.Should().Throw<ArgumentException>();
    }

    // --- B5: MinimumTrackRecordLength guards ---
    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(-0.1)]
    [InlineData(1.5)]
    public void MinTRL_SignificanceOutOfRange_Throws(double alpha)
    {
        var act = () => MinimumTrackRecordLength<double>.Compute(
            observedSharpe: 1.0, significanceLevel: alpha, tradingDaysPerYear: 252);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("significanceLevel");
    }

    [Fact]
    public void MinTRL_NonPositiveTradingDays_Throws()
    {
        var act = () => MinimumTrackRecordLength<double>.Compute(
            observedSharpe: 1.0, significanceLevel: 0.05, tradingDaysPerYear: 0);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("tradingDaysPerYear");
    }

    // --- B6: FisherZTransform guards + clamp + forward/inverse ---
    [Fact]
    public void FisherZ_TooFewObservations_Throws()
    {
        var act = () => FisherZTransform<double>.ConfidenceInterval(0.5, 3);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("n");
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(-0.5)]
    public void FisherZ_ConfidenceOutOfRange_Throws(double confidence)
    {
        var act = () => FisherZTransform<double>.ConfidenceInterval(0.5, 30, confidence);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("confidenceLevel");
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(-1.0)]
    public void FisherZ_RAtUnitEndpoint_ReturnsDegenerateInterval(double r)
    {
        var (lo, hi) = FisherZTransform<double>.ConfidenceInterval(r, 30);
        lo.Should().Be(r);
        hi.Should().Be(r);
    }

    [Theory]
    [InlineData(1.5)]
    [InlineData(-1.5)]
    public void FisherZ_RBeyondUnit_Throws(double r)
    {
        var act = () => FisherZTransform<double>.ConfidenceInterval(r, 30);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(r));
    }

    [Fact]
    public void FisherZ_ForwardInverse_RoundTrips()
    {
        const double r = 0.42;
        var z = FisherZTransform<double>.Forward(r);
        // atanh(0.42) = 0.5 * ln((1.42)/(0.58)) = 0.44769...
        z.Should().BeApproximately(0.5 * Math.Log(1.42 / 0.58), 1e-12);
        FisherZTransform<double>.Inverse(z).Should().BeApproximately(r, 1e-12);
    }

    // --- B7: DistanceCorrelation ---
    [Fact]
    public void DistanceCorrelation_MismatchedLengths_Throws()
    {
        double[] x = [1.0, 2.0, 3.0];
        double[] y = [1.0, 2.0];
        var act = () => DistanceCorrelation<double>.Compute(x, y);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void DistanceCorrelation_FewerThanThree_ReturnsZero()
    {
        double[] x = [1.0, 2.0];
        double[] y = [3.0, 4.0];
        DistanceCorrelation<double>.Compute(x, y).Should().Be(0.0);
    }

    [Fact]
    public void DistanceCorrelation_ConstantSeries_ReturnsZero()
    {
        double[] x = [5.0, 5.0, 5.0, 5.0];
        double[] y = [1.0, 2.0, 3.0, 4.0];
        DistanceCorrelation<double>.Compute(x, y).Should().Be(0.0);
    }

    [Fact]
    public void DistanceCorrelation_PerfectLinear_EqualsOne()
    {
        double[] x = [1.0, 2.0, 3.0, 4.0];
        double[] y = [1.0, 2.0, 3.0, 4.0];
        DistanceCorrelation<double>.Compute(x, y).Should().BeApproximately(1.0, 1e-12);
    }

    [Fact]
    public void DistanceCorrelation_KnownAnswer_MatchesReference()
    {
        // Externally computed (reference Python implementation of Székely 2007).
        double[] x = [1.0, 2.0, 3.0, 4.0, 5.0];
        double[] y = [1.0, 4.0, 9.0, 16.0, 25.0];
        DistanceCorrelation<double>.Compute(x, y).Should().BeApproximately(0.9869160440537484, 1e-12);
    }

    // --- B8: NeweyWestVariance guards ---
    [Fact]
    public void NeweyWest_NegativeLags_Throws()
    {
        double[] series = [0.01, 0.02, -0.01, 0.005];
        var act = () => NeweyWestVariance<double>.MeanVariance(series, lags: -1);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("lags");
    }

    [Fact]
    public void NeweyWest_LagsAtOrBeyondLength_Throws()
    {
        double[] series = [0.01, 0.02, -0.01, 0.005];
        var act = () => NeweyWestVariance<double>.MeanVariance(series, lags: 4);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("lags");
    }

    // ====================================================================
    // T6 — NeweyWest zero-lag against an EXTERNALLY computed known answer
    // (not a re-derivation of the source's own formula).
    // series = [0.01, 0.02, -0.005, 0.015, -0.01, 0.02, 0.005], lags=0.
    //   mean = 0.055/7 = 0.0078571428571428583
    //   gamma0 = sum (x-mean)^2 / 7 = 0.00012040816326530613
    //   HAC(mean) = gamma0 / 7 = 1.7201166180758018e-05
    // ====================================================================
    [Fact]
    public void NeweyWest_ZeroLags_KnownAnswer()
    {
        double[] series = [0.01, 0.02, -0.005, 0.015, -0.01, 0.02, 0.005];
        var hac = NeweyWestVariance<double>.MeanVariance(series, lags: 0);
        hac.Should().BeApproximately(1.7201166180758018e-05, 1e-15);
    }

    // ====================================================================
    // B20 — symmetry + reference-value smoke tests for shrinkage estimators
    // that previously had only diagonal/symmetry coverage.
    // ====================================================================
    [Theory]
    [MemberData(nameof(ShrinkageEstimators))]
    public void ShrinkageEstimator_ProducesSymmetricPositiveDiagonalCovariance(string name)
    {
        var returns = GenerateFactorReturns(120, 5, seed: 17);
        var cov = EstimateBy(name, returns);
        var n = cov.GetLength(0);

        for (var i = 0; i < n; i++)
        {
            cov[i, i].Should().BeGreaterThan(0.0, $"{name} diagonal[{i}] must be positive");
            for (var j = i + 1; j < n; j++)
            {
                Math.Abs(cov[i, j] - cov[j, i]).Should().BeLessThan(1e-10,
                    $"{name} symmetry violated at [{i},{j}]");
            }
        }
    }

    public static TheoryData<string> ShrinkageEstimators() =>
    [
        "ConstantCorrelation",
        "SingleFactor",
        "Qis",
        "Oas",
        "TracyWidom",
    ];

    private static double[,] EstimateBy(string name, double[,] returns) => name switch
    {
        "ConstantCorrelation" => new LedoitWolfConstantCorrelationEstimator<double>().Estimate(returns),
        "SingleFactor" => new LedoitWolfSingleFactorEstimator<double>().Estimate(returns),
        "Qis" => new QuadraticInverseShrinkageEstimator<double>().Estimate(returns),
        "Oas" => new OracleApproximatingShrinkageEstimator<double>().Estimate(returns),
        "TracyWidom" => new TracyWidomDenoisedCovarianceEstimator<double>().Estimate(returns),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Unknown estimator."),
    };

    // ====================================================================
    // B25 — T=float / T=Half parity smoke tests for representative stats.
    // ====================================================================
    [Fact]
    public void SampleCovariance_Float_MatchesDoubleWithinTolerance()
    {
        var returnsD = GenerateFactorReturns(60, 3, seed: 5);
        var returnsF = new float[60, 3];
        for (var i = 0; i < 60; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                returnsF[i, j] = (float)returnsD[i, j];
            }
        }

        var covD = new SampleCovarianceEstimator<double>().Estimate(returnsD);
        var covF = new SampleCovarianceEstimator<float>().Estimate(returnsF);

        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                ((double)covF[i, j]).Should().BeApproximately(covD[i, j], 1e-4);
            }
        }
    }

    [Fact]
    public void PearsonCorrelation_Float_MatchesDoubleWithinTolerance()
    {
        var rng = new System.Random(11);
        var n = 50;
        var xd = new double[n];
        var yd = new double[n];
        var xf = new float[n];
        var yf = new float[n];
        for (var i = 0; i < n; i++)
        {
            xd[i] = rng.NextDouble();
            yd[i] = 0.5 * xd[i] + 0.5 * rng.NextDouble();
            xf[i] = (float)xd[i];
            yf[i] = (float)yd[i];
        }

        var corrD = PearsonCorrelation<double>.Compute(xd, yd);
        var corrF = PearsonCorrelation<float>.Compute(xf, yf);
        ((double)corrF).Should().BeApproximately(corrD, 1e-4);
    }

    [Fact]
    public void PearsonCorrelation_Half_PerfectLinearIsApproximatelyOne()
    {
        Half[] x = [(Half)1.0f, (Half)2.0f, (Half)3.0f, (Half)4.0f, (Half)5.0f];
        Half[] y = [(Half)2.0f, (Half)4.0f, (Half)6.0f, (Half)8.0f, (Half)10.0f];
        var corr = PearsonCorrelation<Half>.Compute(x, y);
        ((double)corr).Should().BeApproximately(1.0, 1e-2);
    }

    private static double[,] GenerateFactorReturns(int t, int n, int seed)
    {
        var rng = new System.Random(seed);
        var returns = new double[t, n];
        for (var i = 0; i < t; i++)
        {
            var factor = rng.NextDouble() * 0.02 - 0.01;
            for (var j = 0; j < n; j++)
            {
                var idio = rng.NextDouble() * 0.004 - 0.002;
                returns[i, j] = factor + idio;
            }
        }

        return returns;
    }
}
