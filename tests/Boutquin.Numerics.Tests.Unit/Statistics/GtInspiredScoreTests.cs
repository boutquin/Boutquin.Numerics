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
/// Pins <see cref="GtInspiredScore{T}"/> to the declared GT-inspired library variant,
/// inspired by Sheppert (2026), arXiv:2602.00080 — the utility
/// <c>GTScore = µ·ln(z)·r²/σ_d</c>, equation (2) <c>z = (µ−µ_m)/(σ/√N)</c>, and
/// Algorithm 1 (the piecewise definition).
/// </summary>
/// <remarks>
/// <para>
/// <b>Independent golden derivation.</b> Every expected value below was computed
/// from the declared library formulas (not from a port of the C# code) by a standalone Python
/// reference. The retained oracle is <c>tests/Verification/generate_scalar_stats_vectors.py</c>.
/// The Python independently implements the declared variant utility and branches,
/// with the following explicit component definitions:
/// </para>
/// <list type="bullet">
///   <item><c>r²</c> (Kestner 1996): R-squared of the cumulative-equity curve
///   (running sum of returns) regressed on the linear time trend <c>1..N</c>.</item>
///   <item><c>σ_d</c> (Sortino &amp; van der Meer 1991): downside deviation
///   <c>√(mean(min(0, rᵢ)²))</c> with a zero minimum acceptable return.</item>
/// </list>
/// <para>
/// Because the goldens are derived from the declared variant independently of the C#
/// implementation, a regression that re-introduced the old additive
/// weighted-sum formula (or any other structural drift) would flip these
/// assertions RED.
/// </para>
/// </remarks>
// Component-only short fixtures explicitly use minimumObservations = 1.
// The default inclusive safeguard is pinned separately in C4ContractRegressionTests.
public sealed class GtInspiredScoreTests
{
    [Fact]
    public void Compute_LargeStandardBranchNumerator_PreservesFiniteScore()
    {
        var returns = Enumerable.Repeat(1e308, 100).ToArray();
        returns[^1] = -1e308;
        var normalized = returns.Select(value => value / 1e308).ToArray();
        var expected = GtInspiredScore<double>.Compute(normalized, 0, 1e-314).Score;

        GtInspiredScore<double>.Compute(returns).Score.Should().BeApproximately(expected, 1e-10);

        var narrow = Enumerable.Repeat((Half)60000, 100).ToArray();
        narrow[^1] = (Half)(-60000);
        var widened = GtInspiredScore<double>.Compute(narrow.Select(value => (double)value).ToArray()).Score;
        ((double)GtInspiredScore<Half>.Compute(narrow).Score).Should().BeApproximately(widened, 0.2);
    }

    [Fact]
    public void Compute_HalfSignificanceBeyondRange_PreservesFiniteSmoothedScore()
    {
        var returns = Enumerable.Repeat((Half)1, 1000).ToArray();
        returns[^1] = Half.BitIncrement((Half)1);
        var expected = GtInspiredScore<double>.Compute(returns.Select(value => (double)value).ToArray(), 0, 1000);

        var result = GtInspiredScore<Half>.Compute(returns, (Half)0, (Half)1000);

        ((double)result.Score).Should().BeApproximately(expected.Score, 0.00002);
        Half.IsPositiveInfinity(result.Z).Should().BeTrue("the exposed Half Z cannot represent the significance");
    }

    // ------------------------------------------------------------------
    // CASE z > 1 — the "standard GT-Score" branch of Algorithm 1, with a
    // strictly-positive downside deviation (one negative return present).
    //   independent library-variant reference:
    //     mu      = 0.009916666666666666
    //     z       = 7.053452892053309
    //     r2      = 0.9891432653097217
    //     sigma_d = 0.0011547005383792514
    //     score   = -(mu*ln(z)*r2)/(sigma_d + 1e-6) = -16.580470598518065
    // ------------------------------------------------------------------
    [Fact]
    public void Compute_OutperformingStrategy_MatchesVariantStandardBranch()
    {
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
        // The composite is the negated equation-(1) utility (loss form).
        result.Score.Should().BeApproximately(-16.580470598518065, 1e-9);
    }

    // ------------------------------------------------------------------
    // CASE z <= 0 — underperforms the benchmark: large-penalty branch.
    //   score = 100 + 100*(1 - exp(-|z-1|)).
    //   independent reference:
    //     mu = -0.00013333333333333334, z = -1.1541678045384953
    //     score = 188.40003151324478
    // ------------------------------------------------------------------
    [Fact]
    public void Compute_Underperformer_UsesLargePenaltyBranch()
    {
        double[] returns = [0.001, -0.002, 0.0015, -0.001, 0.0005, -0.0008];

        var result = GtInspiredScore<double>.Compute(returns, benchmarkMean: 0.0005, epsilon: 1e-6, minimumObservations: 1);

        result.Z.Should().BeApproximately(-1.1541678045384953, 1e-9);
        result.Score.Should().BeApproximately(188.40003151324478, 1e-9);
        // Penalty branch always lands in [100, 200].
        result.Score.Should().BeGreaterThanOrEqualTo(100.0).And.BeLessThanOrEqualTo(200.0);
    }

    // ------------------------------------------------------------------
    // CASE 0 < z <= 1 — marginal outperformance: smooth-transition branch.
    //   score = 100*(1 - exp(-|z-1|)).
    //   independent reference:
    //     z = 0.40824829046386324, score = 44.664288567984045
    // ------------------------------------------------------------------
    [Fact]
    public void Compute_MarginalOutperformer_UsesSmoothTransitionBranch()
    {
        double[] returns = [0.002, 0.001, 0.003, 0.0015, 0.0025, 0.001, 0.002, 0.0018];

        var result = GtInspiredScore<double>.Compute(returns, benchmarkMean: 0.00175, epsilon: 1e-6, minimumObservations: 1);

        result.Z.Should().BeApproximately(0.40824829046386324, 1e-9);
        result.Score.Should().BeApproximately(44.664288567984045, 1e-9);
        result.Score.Should().BeGreaterThanOrEqualTo(0.0).And.BeLessThanOrEqualTo(100.0);
    }

    // ------------------------------------------------------------------
    // CASE σ_d == 0 edge (all non-negative returns, z > 1): the epsilon guard
    // (1e-6) prevents division by zero under the library variant contract.
    //   independent reference:
    //     mu = 0.014374999999999999, z = 11.608296486010875
    //     r2 = 0.995979581243123, sigma_d = 0.0
    //     score = -(mu*ln(z)*r2)/(0 + 1e-6) = -35101.78228497962
    // ------------------------------------------------------------------
    [Fact]
    public void Compute_ZeroDownsideDeviation_EpsilonGuardsDenominator()
    {
        double[] returns = [0.01, 0.02, 0.015, 0.012, 0.018, 0.011, 0.013, 0.016];

        var result = GtInspiredScore<double>.Compute(returns, benchmarkMean: 0.0, epsilon: 1e-6, minimumObservations: 1);

        result.DownsideDeviation.Should().Be(0.0);
        double.IsFinite(result.Score).Should().BeTrue("the epsilon guard keeps the denominator non-zero");
        result.Score.Should().BeApproximately(-35101.78228497962, 1e-6);
    }

    [Fact]
    public void Compute_HalfHundredAlternatingReturns_HasFiniteEquityCurveRSquared()
    {
        var returns = new Half[100];
        for (var i = 0; i < returns.Length; i++)
        {
            returns[i] = (Half)(i % 2 == 0 ? 0.01f : 0.02f);
        }

        var result = GtInspiredScore<Half>.Compute(returns);

        Half.IsFinite(result.RSquared).Should().BeTrue();
        ((double)result.RSquared).Should().BeApproximately(0.9999666811101855, 0.002);
        Half.IsFinite(result.Score).Should().BeTrue();
    }

    [Fact]
    public void Compute_HalfBalancedLargeReturnsKeepsZeroMeanAndFiniteScore()
    {
        var returns = new Half[2000];
        for (var i = 0; i < returns.Length; i++)
        {
            returns[i] = i < 1000 ? (Half)100 : (Half)(-100);
        }

        var result = GtInspiredScore<Half>.Compute(returns);

        result.Mu.Should().Be((Half)0);
        Half.IsFinite(result.Score).Should().BeTrue();
        Half.IsFinite(result.DownsideDeviation).Should().BeTrue();
    }

    [Fact]
    public void Compute_HalfLongBinaryReturnsPreservesMean()
    {
        var returns = new Half[10_000];
        for (var i = 0; i < returns.Length; i++)
        {
            returns[i] = (Half)(i % 2);
        }

        var result = GtInspiredScore<Half>.Compute(returns);

        result.Mu.Should().Be((Half)0.5f);
    }

    [Theory]
    [InlineData(0.001f, 0.002f)]
    [InlineData(0.1f, 0.2f)]
    [InlineData(1.0f, 2.0f)]
    [InlineData(-0.01f, 0.02f)]
    public void Compute_HalfEquityCurveRSquared_AgreesWithWidenedInputOracle(
        float firstReturn, float secondReturn)
    {
        var returns = new Half[100];
        for (var i = 0; i < returns.Length; i++)
        {
            returns[i] = (Half)(i % 2 == 0 ? firstReturn : secondReturn);
        }

        // This oracle uses the exact Half-quantized observations but computes
        // Pearson's squared correlation independently in double precision.
        var equity = new double[returns.Length];
        var running = 0.0;
        for (var i = 0; i < returns.Length; i++)
        {
            running += (double)returns[i];
            equity[i] = running;
        }

        var meanEquity = equity.Average();
        var meanTime = (returns.Length + 1.0) / 2.0;
        var centeredTimeSquared = 0.0;
        var centeredEquitySquared = 0.0;
        var crossProduct = 0.0;
        for (var i = 0; i < returns.Length; i++)
        {
            var centeredTime = i + 1.0 - meanTime;
            var centeredEquity = equity[i] - meanEquity;
            centeredTimeSquared += centeredTime * centeredTime;
            centeredEquitySquared += centeredEquity * centeredEquity;
            crossProduct += centeredTime * centeredEquity;
        }

        var expected = crossProduct * crossProduct / (centeredTimeSquared * centeredEquitySquared);
        var result = GtInspiredScore<Half>.Compute(returns);

        Half.IsFinite(result.RSquared).Should().BeTrue();
        ((double)result.RSquared).Should().BeInRange(0.0, 1.0);
        ((double)result.RSquared).Should().BeApproximately(expected, 0.01);
    }

    // F-L1-02: ComputeEquityCurveRSquared and ComputeDownsideDeviation have no
    // typeof(T) == typeof(Half) widening branch (unlike ComputeMeanAndStdDev,
    // just above), so their per-term Half accumulation drifts for series of a
    // few thousand points even though every input is finite and ordinary-scale.
    // The oracle is GtInspiredScore<double> run on the SAME Half-rounded
    // values — an independent, already-correct sibling instantiation.
    private static Half[] BuildDeterministicNoisyHalfReturns(int n)
    {
        var values = new Half[n];
        for (var i = 0; i < n; i++)
        {
            var noise = Math.Sin(i * 12.9898) * 43758.5453;
            noise -= Math.Floor(noise);
            var v = 0.0005 + 0.01 * (noise - 0.5) * 2.0;
            values[i] = (Half)v;
        }

        return values;
    }

    [Fact]
    public void Compute_HalfFiveThousandPoints_RSquaredAndDownsideDeviationAgreeWithDoubleOracle()
    {
        var half = BuildDeterministicNoisyHalfReturns(5000);
        var asDouble = half.Select(h => (double)h).ToArray();

        var expected = GtInspiredScore<double>.Compute(asDouble);
        var result = GtInspiredScore<Half>.Compute(half);

        ((double)result.RSquared).Should().BeApproximately(expected.RSquared, 0.02);
        ((double)result.DownsideDeviation).Should().BeApproximately(expected.DownsideDeviation, 0.001);
    }

    // F-L1-03 (census site within this file): Half.CreateChecked(n) silently
    // returns +Infinity once n exceeds 65,504, which zeroed the standard-error
    // denominator (line ~196) and the R²/downside-deviation reductions
    // (lines ~345/353/364), collapsing z, R² and the score to 0 for any series
    // longer than that.
    [Fact]
    public void Compute_HalfSeriesLongerThanHalfRange_ZAndRSquaredAgreeWithDoubleOracle()
    {
        var half = BuildDeterministicNoisyHalfReturns(70_000);
        var asDouble = half.Select(h => (double)h).ToArray();

        var expected = GtInspiredScore<double>.Compute(asDouble);
        var result = GtInspiredScore<Half>.Compute(half);

        Half.IsFinite(result.Z).Should().BeTrue();
        ((double)result.Z).Should().NotBe(0.0);
        ((double)result.RSquared).Should().BeApproximately(expected.RSquared, 0.05);
    }

    // ------------------------------------------------------------------
    // Multiplicative structure, NOT an additive weighted sum: the old code
    // computed w1*Sharpe + w2*tstat + w3*consistency - w4*drawdown, which for
    // this outperforming series would be a small positive number. The paper
    // formula returns a large negative loss. This assertion would go RED if the
    // additive formula were re-introduced.
    // ------------------------------------------------------------------
    [Fact]
    public void Compute_IsMultiplicativeRatio_NotAdditiveWeightedSum()
    {
        double[] returns =
        [
            0.010, 0.012, 0.008, 0.015, 0.009, 0.011,
            0.013, -0.004, 0.014, 0.010, 0.012, 0.009,
        ];

        var result = GtInspiredScore<double>.Compute(returns, benchmarkMean: 0.0, epsilon: 1e-6, minimumObservations: 1);

        // The old additive composite (0.3*Sharpe + 0.3*tstat + 0.2*consist - 0.2*dd)
        // is a small positive number (~ +9). The library variant loss is strongly negative.
        result.Score.Should().BeNegative();
        result.Score.Should().BeLessThan(-1.0);
    }

    // ------------------------------------------------------------------
    // De-mask F-VS3-01: the z = 1 boundary test previously only checked
    // score == 0 at z = 1, which cannot distinguish a correct branch
    // condition (z <= 1 takes the marginal branch, giving
    // 100*(1-exp(0)) = 0) from an incorrect one (if z <= 1 were wrongly
    // z < 1, z = 1 would fall into the z > 1 standard branch, where
    // ln(1) = 0 ALSO gives a score of 0 — a coincidental match that masks
    // the bug). Pin the branch CONDITION instead by asserting values just
    // above and below z = 1, each independently derived from the Sheppert
    // (2026) formulas (not ported from the C# under test):
    //   z <= 1  (marginal, smooth-transition): score = 100*(1-exp(-|z-1|))
    //   z > 1   (standard GT-score):           score = -(mu*ln(z)*r2)/(sigma_d+eps)
    //
    // Fixture: reuse the returns array from
    // StatisticsReviewFixTests.GtInspiredScore_SignificanceGate_AnchorsAtZEqualsOne
    // (returns = [0.006, 0.002, 0.006, 0.002, 0.004]; mu = 0.004, sample sd
    // (ddof=1) = 0.002, se = 0.002/sqrt(5) = 8.944271909999158e-4). Target z
    // is hit algebraically via benchmarkMean = mu - z*se (z = (mu-benchmarkMean)/se).
    // delta = 0.01.
    //
    // Independent reference (/tmp/gtscore_zboundary.py, implementing eq.(1)/(2)
    // and Algorithm 1 directly from the paper — NOT from this C# code):
    //   z = 1.01 (z > 1, STANDARD branch):
    //     benchmarkMean = 0.0030966285370900852
    //     computed z    = 1.0099999999999998
    //     r2            = 0.9759036144578316
    //     sigma_d       = 0.0
    //     score         = -(mu*ln(z)*r2)/(sigma_d+1e-6) = -38.84225537863123
    //     (the MARGINAL formula at this same z would give +0.9950166250831782 —
    //      a value the test would wrongly accept if the branch boundary leaked)
    //   z = 0.99 (z <= 1, MARGINAL branch):
    //     benchmarkMean = 0.0031145170809100836
    //     computed z    = 0.9899999999999999
    //     r2            = 0.9759036144578316
    //     sigma_d       = 0.0
    //     score         = 100*(1-exp(-|z-1|)) = 0.9950166250832115
    //     (the STANDARD formula at this same z would give +39.23263634378924 —
    //      a value the test would wrongly accept if the branch boundary leaked)
    // ------------------------------------------------------------------
    [Fact]
    public void Compute_JustAboveZEqualsOne_UsesStandardBranchFormula()
    {
        double[] returns = [0.006, 0.002, 0.006, 0.002, 0.004];
        const double benchmarkMean = 0.0030966285370900852;

        var result = GtInspiredScore<double>.Compute(returns, benchmarkMean, epsilon: 1e-6, minimumObservations: 1);

        result.Z.Should().BeApproximately(1.01, 1e-9);
        // Pins the STANDARD branch (z > 1) formula, independently derived.
        result.Score.Should().BeApproximately(-38.84225537863123, 1e-6);
        // Would go RED if the marginal formula's value at this z leaked in.
        result.Score.Should().NotBeApproximately(0.9950166250831782, 1e-6);
    }

    [Fact]
    public void Compute_JustBelowZEqualsOne_UsesMarginalBranchFormula()
    {
        double[] returns = [0.006, 0.002, 0.006, 0.002, 0.004];
        const double benchmarkMean = 0.0031145170809100836;

        var result = GtInspiredScore<double>.Compute(returns, benchmarkMean, epsilon: 1e-6, minimumObservations: 1);

        result.Z.Should().BeApproximately(0.99, 1e-9);
        // Pins the MARGINAL branch (z <= 1) formula, independently derived.
        result.Score.Should().BeApproximately(0.9950166250832115, 1e-6);
        // Would go RED if the standard formula's value at this z leaked in.
        result.Score.Should().NotBeApproximately(39.23263634378924, 1e-6);
    }

    [Fact]
    public void Compute_TooFewReturns_Throws()
    {
        double[] returns = [0.01];
        var act = () => GtInspiredScore<double>.Compute(returns);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Compute_NonFiniteReturnElement_ThrowsArgumentException()
    {
        // M-3 guard: a NaN element must be rejected explicitly, naming the
        // parameter and index. Pre-guard, the z<=0 penalty branch's formula
        // never touches rSquared/downsideDeviation, so a NaN silently produces
        // a plausible-looking FINITE score (~163.2) instead of throwing or
        // propagating NaN — a silent-corruption defect, not a crash.
        double[] returns = [0.01, 0.02, double.NaN, 0.015, 0.01];

        var act = () => GtInspiredScore<double>.Compute(returns);

        act.Should().Throw<ArgumentException>().WithMessage("*index 2*");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Compute_NonFiniteBenchmarkMean_ThrowsArgumentOutOfRangeException(double benchmarkMean)
    {
        var act = () => GtInspiredScore<double>.Compute([0.01, 0.02, -0.01], benchmarkMean);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName(nameof(benchmarkMean));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1e-6)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Compute_NonPositiveOrNonFiniteEpsilon_ThrowsArgumentOutOfRangeException(double epsilon)
    {
        var act = () => GtInspiredScore<double>.Compute(
            [0.01, 0.02, 0.015], benchmarkMean: 0.0, epsilon);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName(nameof(epsilon));
    }
}
