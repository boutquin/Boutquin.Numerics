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

    [Fact]
    public void Interpolate_BelowFirstNode_ExtrapolatesFlat()
    {
        // x < xs[0]: flat extrapolation returns ys[0]
        var result = _sut.Interpolate(-0.5, s_xs, s_ys);
        result.Should().BeApproximately(s_ys[0], 1e-10);
    }

    [Fact]
    public void Interpolate_AboveLastNode_ExtrapolatesFlat()
    {
        // x > xs[^1]: flat extrapolation returns ys[^1]
        var result = _sut.Interpolate(50.0, s_xs, s_ys);
        result.Should().BeApproximately(s_ys[^1], 1e-10);
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
}
