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

using Boutquin.Numerics.Interpolation;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Interpolation;

/// <summary>
/// Tests for <see cref="MonotoneConvexInterpolator"/>.
///
/// Node convention: xs[0]=0, ys[0]=0 is the virtual origin (P(0)=1 → NCR=0).
/// xs[1..n] are year fractions; ys[i] = −ln(P(xs[i])) are NCR values.
/// Discount factor at x: P(x) = exp(−result).
/// </summary>
public sealed class MonotoneConvexInterpolatorTests
{
    // Standard yield-curve nodes expressed as (time, NCR = -ln(DF)):
    //   DF:  0.9956, 0.9868, 0.9737, 0.9479, 0.8979, 0.7788, 0.6065, 0.2231
    //   NCR: -ln(DF) — monotonically increasing for an upward-sloping curve
    private static readonly double[] s_xs =
    [
        0.0,                            // virtual origin
        1.0 / 12.0,                     // 1 month
        3.0 / 12.0,                     // 3 months
        6.0 / 12.0,                     // 6 months
        1.0,                            // 1 year
        2.0,                            // 2 years
        5.0,                            // 5 years
        10.0,                           // 10 years
        30.0                            // 30 years
    ];

    private static readonly double[] s_ys =
    [
        0.0,                            // virtual origin NCR = 0
        -Math.Log(0.9956),              // 1M
        -Math.Log(0.9868),              // 3M
        -Math.Log(0.9737),              // 6M
        -Math.Log(0.9479),              // 1Y
        -Math.Log(0.8979),              // 2Y
        -Math.Log(0.7788),              // 5Y
        -Math.Log(0.6065),              // 10Y
        -Math.Log(0.2231)               // 30Y
    ];

    private readonly MonotoneConvexInterpolator _sut = MonotoneConvexInterpolator.Instance;

    [Fact]
    public void Name_Returns_MonotoneConvex()
    {
        _sut.Name.Should().Be("MonotoneConvex");
    }

    [Fact]
    public void Instance_IsSingleton()
    {
        var first = MonotoneConvexInterpolator.Instance;
        var second = MonotoneConvexInterpolator.Instance;
        ReferenceEquals(first, second).Should().BeTrue();
    }

    [Fact]
    public void Interpolate_AtOrigin_ReturnsZero()
    {
        var result = _sut.Interpolate(0.0, s_xs, s_ys);
        result.Should().BeApproximately(0.0, 1e-10);
    }

    [Theory]
    [InlineData(-0.5, -0.375)]
    [InlineData(3.0, 5.25)]
    public void Interpolate_Exterior_ContinuesEndpointForward(double x, double expected)
    {
        // F1=1, F2=2; interior forward=1.5, endpoints=.75 and2.25.
        _sut.Interpolate(x, [0, 1, 2], [0, 1, 3]).Should().BeApproximately(expected, 1e-12);
    }

    [Fact]
    public void Interpolate_AtActualNode_ReturnsNodeNcr()
    {
        // At xs[4] = 1.0 (1Y), should return ys[4] = -ln(0.9479)
        var result = _sut.Interpolate(1.0, s_xs, s_ys);
        result.Should().BeApproximately(s_ys[4], 1e-10);
    }

    [Fact]
    public void Interpolate_BetweenNodes_NcrIsMonotonicallyIncreasing()
    {
        // NCR = -ln(DF) must be monotonically non-decreasing (DFs are decreasing)
        double prev = 0.0;
        for (int months = 1; months <= 360; months++)
        {
            double x = months / 12.0;
            double ncr = _sut.Interpolate(x, s_xs, s_ys);
            ncr.Should().BeGreaterThanOrEqualTo(prev - 1e-10,
                $"NCR at month {months} ({ncr:F8}) must not decrease (prev={prev:F8})");
            prev = ncr;
        }
    }

    [Fact]
    public void Interpolate_ImpliedDiscountFactors_ArePositive()
    {
        for (int months = 1; months <= 360; months++)
        {
            double x = months / 12.0;
            double ncr = _sut.Interpolate(x, s_xs, s_ys);
            double df = Math.Exp(-ncr);
            df.Should().BeGreaterThan(0d,
                $"DF at month {months} must be strictly positive");
        }
    }

    [Fact]
    public void Interpolate_ImpliedForwardRates_NonNegative()
    {
        // f(t) ≈ [NCR(t+ε) - NCR(t)] / ε must be non-negative for no-arbitrage
        double epsilon = 1.0 / 365.0;
        for (int months = 1; months <= 359; months++)
        {
            double x = months / 12.0;
            double ncr = _sut.Interpolate(x, s_xs, s_ys);
            double ncrNext = _sut.Interpolate(x + epsilon, s_xs, s_ys);
            double forwardRate = (ncrNext - ncr) / epsilon;
            forwardRate.Should().BeGreaterThanOrEqualTo(-1e-8,
                $"Forward rate at month {months} should be non-negative, got {forwardRate:F8}");
        }
    }

    // Discrete forwards F = [1.0%, 1.1%, 3.0%] on unit intervals. Hagan & West (2006) eq. 30 gives
    // f₁ = 1.05%, f₂ = 2.05%, so on [1, 2] g₁ = f₁ − F₂ = −0.05% and g₂ = f₂ − F₂ = 0.95%. Since
    // g₂ > −2g₁ this is sector (ii): eqs. 49–50 hold g at g₁ up to η = (g₂ + 2g₁)/(g₂ − g₁) = 0.85,
    // so at u = 0.315 the NCR is 0.01 + 0.011·0.315 − 0.0005·0.315 = 0.0133075 (eq. 37).
    private static readonly double[] s_increasingForwardXs = [0.0, 1.0, 2.0, 3.0];
    private static readonly double[] s_increasingForwardYs = [0.0, 0.01, 0.021, 0.051];

    [Fact]
    public void Interpolate_IncreasingForwards_UsesSectorTwoFlatSegment()
    {
        MonotoneConvexInterpolator<double>.Instance
            .Interpolate(1.315, s_increasingForwardXs, s_increasingForwardYs)
            .Should().BeApproximately(0.0133075, 1e-14);
    }

    // Hand-derived from Hagan & West (2006) on unit intervals, evaluated at t = 1.5 (u = 0.5).
    // Sector (iii): F = [3%, 1%, 0.8%] gives f₁ = 2%, f₂ = 0.9%, so g₁ = 1%, g₂ = −0.1% and
    // η = 3g₂/(g₂ − g₁) = 3/11 < u (eq. 52): NCR = 0.03 + 0.005 − 0.0005 + 0.011·η/3 = 0.0355.
    // Sector (iv): F = [1%, 3%, 2%] gives f₁ = 2%, f₂ = 2.5%, so g₁ = −1%, g₂ = −0.5%, η = 1/3 and
    // A = 1/300 (eqs. 55-56): NCR = 0.025 + 1/600 − 1/675 − 1/34560 = 0.025 + 1/6400.
    [Theory]
    [InlineData(0.03, 0.04, 0.048, 0.0355)]
    [InlineData(0.01, 0.04, 0.06, 0.02515625)]
    public void Interpolate_SectorThreeAndFour_MatchPaperFormulas(double y1, double y2, double y3, double expected)
    {
        MonotoneConvexInterpolator<double>.Instance
            .Interpolate(1.5, [0.0, 1.0, 2.0, 3.0], [0.0, y1, y2, y3])
            .Should().BeApproximately(expected, 1e-14);
    }

    [Fact]
    public void Interpolate_IncreasingForwards_ForwardCurveIsNonDecreasing()
    {
        // Hagan & West: with strictly increasing inputs every interval lies in sector (i) or (ii),
        // where the interpolated forward is non-decreasing. The basic quadratic alone dips inside
        // [1, 2] (to about 0.78%, below every input), which this sweep would expose.
        const double epsilon = 1e-6;
        var previous = double.NegativeInfinity;
        for (var step = 1; step < 300; step++)
        {
            var t = step / 100.0;
            var forward = (MonotoneConvexInterpolator<double>.Instance.Interpolate(t + epsilon, s_increasingForwardXs, s_increasingForwardYs)
                - MonotoneConvexInterpolator<double>.Instance.Interpolate(t - epsilon, s_increasingForwardXs, s_increasingForwardYs))
                / (2 * epsilon);
            forward.Should().BeGreaterThanOrEqualTo(previous - 1e-8, $"forward at t = {t}");
            previous = forward;
        }
    }

    [Fact]
    public void Interpolate_TwoActualNodes_ProducesReasonableResult()
    {
        // Minimal case: virtual node + 2 actual nodes
        double[] xs = [0.0, 0.5, 1.0];
        double[] ys = [0.0, -Math.Log(0.97), -Math.Log(0.94)];

        var result = _sut.Interpolate(0.75, xs, ys);

        // Must be between ys[1] and ys[2]
        result.Should().BeGreaterThanOrEqualTo(ys[1]);
        result.Should().BeLessThanOrEqualTo(ys[2]);
    }

    [Fact]
    public void Interpolate_SingleActualNode_HandledGracefully()
    {
        // Virtual node + 1 actual node
        double[] xs = [0.0, 1.0];
        double[] ys = [0.0, -Math.Log(0.95)];

        var result = _sut.Interpolate(0.5, xs, ys);

        // With only one segment the boundary formula reduces to F[1];
        // result must be a valid NCR value
        result.Should().BeGreaterThan(0.0);
        result.Should().BeLessThan(ys[1]);
    }

    [Fact]
    public void Interpolate_VirtualOriginWithoutActualNode_IsRejected()
    {
        Action act = () => MonotoneConvexInterpolator<double>.Instance.Interpolate(0.5, [0], [0]);

        act.Should().Throw<ArgumentException>().WithParameterName("xs");
    }

    [Fact]
    public void Interpolate_WideFiniteNcr_StaysWithinNodes()
    {
        var result = MonotoneConvexInterpolator<double>.Instance.Interpolate(
            0.5, [0, 1, 2], [0, 1e308, 1e308]);

        double.IsFinite(result).Should().BeTrue();
        result.Should().BeInRange(0, 1e308);

        // Normalized: F = [1, 0], f₀ = 1.25, f₁ → 0 (eq. 61), so g₀ = 0.25, g₁ = −1 < −2g₀:
        // sector (ii) with η = 0.4 (eq. 50). At u = 0.5 the NCR is
        // 0.5 + 0.25·0.5 − 1.25·0.6·(1/6)³/3 = 539/864 (eq. 37).
        (result / 1e308).Should().BeApproximately(539.0 / 864.0, 1e-12);
    }

    [Fact]
    public void Interpolate_WideNcrAndNarrowIntervals_StaysFinite()
    {
        var result = MonotoneConvexInterpolator<double>.Instance.Interpolate(
            0.5e-10, [0, 1e-10, 2e-10], [0, 1e300, 1e300]);

        double.IsFinite(result).Should().BeTrue();
        result.Should().BeInRange(0, 1e300);
    }

    [Fact]
    public void Interpolate_ThrowsOnMismatchedLengths()
    {
        double[] xs = [0.0, 1.0];
        double[] ys = [0.0];
        var act = () => _sut.Interpolate(0.5, xs, ys);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Interpolate_ThrowsOnEmptyArrays()
    {
        var act = () => _sut.Interpolate(0.5, [], []);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Interpolate_DecreasingNcr_RejectsCurveBeforeEvaluation()
    {
        double[] xs = [0.0, 1.0, 2.0];
        double[] ys = [0.0, 0.1, 0.05];

        Action interior = () => _sut.Interpolate(1.5, xs, ys);
        Action extrapolated = () => MonotoneConvexInterpolator<double>.Instance.Interpolate(3.0, xs, ys);

        interior.Should().Throw<ArgumentException>().WithParameterName("ys");
        extrapolated.Should().Throw<ArgumentException>().WithParameterName("ys");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Interpolate_NonFiniteNcr_RejectsCurve(double ncr)
    {
        double[] xs = [0.0, 1.0, 2.0];
        double[] ys = [0.0, 0.1, ncr];

        Action act = () => _sut.Interpolate(1.5, xs, ys);

        act.Should().Throw<ArgumentException>().WithParameterName("ys");
    }

    [Fact]
    public void Interpolate_NonFiniteVirtualOriginNcr_RejectsCurve()
    {
        double[] xs = [0.0, 1.0];
        double[] ys = [double.NaN, 0.1];

        Action act = () => _sut.Interpolate(0.5, xs, ys);

        act.Should().Throw<ArgumentException>().WithParameterName("ys");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Interpolate_NonFiniteTarget_IsRejected(double target)
    {
        double[] xs = [0, 1, 2];
        double[] ys = [0, 0.1, 0.2];

        Action act = () => _sut.Interpolate(target, xs, ys);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("x");
    }
}
