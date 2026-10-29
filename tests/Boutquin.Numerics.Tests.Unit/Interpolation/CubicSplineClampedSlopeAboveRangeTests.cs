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

#pragma warning disable BNU0001

namespace Boutquin.Numerics.Tests.Unit.Interpolation;

/// <summary>
/// Pins for <see cref="CubicSplineInterpolator{T}"/> built from the non-generic double
/// <see cref="SplineBoundary.CreateClamped"/> when the caller's slope lies ABOVE the working type's range in the
/// caller's own units but the slope scaled to the solver's normalized coordinates, and every value it drives, is
/// representable. Every expected value is the clamped cubic spline solved in exact rational arithmetic (Gaussian
/// elimination over <c>fractions.Fraction</c>, no rounding) from the exact double slope and the exact working-type
/// knots and ordinates, then reported as a double; none of it comes from the implementation under test.
/// </summary>
public sealed class CubicSplineClampedSlopeAboveRangeTests
{
    // Half-exact knots: 16 / width exceeds Half.MaxValue, so the coordinate scale (~0.0102) engages and the scaled
    // slope (slope * 0.0102) is about 1018 for a caller slope of 1e5, far above Half.MaxValue (65504) unscaled.
    private static readonly Half[] s_halfKnots = [(Half)0.01, (Half)0.01006, (Half)0.010124, (Half)0.010185];

    // Zero ordinates, so the whole spline is driven by the clamped slopes. Half arithmetic on knots this narrow
    // carries about 3 % relative error on a bump of height ~1 (the natural-boundary Half spline on the same grid
    // errs by the same amount), so the bound is 0.05 absolute: a lost or mis-scaled slope moves the bump by >= 0.4.
    public static IEnumerable<object[]> HalfCases() =>
    [
        [1e5, 0.0, 0.01003265380859375, 0.9663899739583334],
        [1e5, 0.0, 0.01009368896484375, -0.2543131510416667],
        [1e5, 0.0, 0.0101470947265625, 0.059604644775390625],
        [0.0, -1e5, 0.01003265380859375, 0.050862630208333336],
        [0.0, -1e5, 0.01009368896484375, -0.2543131510416667],
        [0.0, -1e5, 0.0101470947265625, 0.7748603820800781],
        [7e4, 0.0, 0.01003265380859375, 0.6764729817708334],
        [7e4, 0.0, 0.01009368896484375, -0.17801920572916666],
    ];

    [Theory]
    [MemberData(nameof(HalfCases))]
    public void ClampedDoubleSlopeAboveHalfRange_ScaledSlopeRepresentable_MatchesExactRationalSpline(
        double left, double right, double query, double expected)
    {
        var spline = new CubicSplineInterpolator<Half>(s_halfKnots, new Half[4], SplineBoundary.CreateClamped(left, right));

        var actual = (double)spline.Interpolate((Half)query);

        Math.Abs(actual - expected).Should().BeLessThanOrEqualTo(
            0.05, $"the exact-rational clamped spline value is {expected:R} but the Half spline returned {actual:R}");
    }

    [Fact]
    public void ClampedDoubleSlopeAboveFloatRange_ScaledSlopeRepresentable_MatchesExactRationalSpline()
    {
        float[] knots = [9.99999991097579e-38f, 1.1000000126281123e-37f, 1.2500000449239123e-37f, 1.3999999651158351e-37f];
        float[] ys = [0f, 10.000001907348633f, 25.0000057220459f, 39.999996185302734f];
        var spline = new CubicSplineInterpolator<float>(knots, ys, SplineBoundary.CreateClamped(1e39, 1e39));

        // Exact-rational values at the queries 1.0199999954036856e-37, 1.150000023393379e-37, 1.300000055689179e-37.
        (double Query, double Expected)[] cases =
        [
            (1.0199999954036856e-37, 2.000000401352293),
            (1.150000023393379e-37, 15.00000321845279),
            (1.300000055689179e-37, 30.000006280880122),
        ];
        foreach (var (query, expected) in cases)
        {
            var actual = spline.Interpolate((float)query);
            Math.Abs(actual - expected).Should().BeLessThanOrEqualTo(
                64 * (double)(float.BitIncrement((float)expected) - (float)expected),
                $"exact-rational value at {query:R} is {expected:R} but the float spline returned {actual:R}");
        }
    }

    [Theory]
    [InlineData(1e300, 1e300)]
    [InlineData(double.PositiveInfinity, 0.0)]
    [InlineData(0.0, double.NegativeInfinity)]
    [InlineData(double.NaN, 0.0)]
    [InlineData(0.0, double.NaN)]
    public void ClampedDoubleSlope_GenuinelyUnrepresentableAfterScaling_IsRefused(double left, double right)
    {
        var act = () => new CubicSplineInterpolator<Half>(
            s_halfKnots, new Half[] { (Half)0, (Half)6, (Half)12, (Half)18 }, SplineBoundary.CreateClamped(left, right));

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void ClampedDoubleSlopeAboveRange_NoScalingEngaged_IsRefused()
    {
        // Knots one unit wide: neither coordinate nor abscissa scaling changes the slope, so 1e5 stays above Half range.
        Half[] knots = [(Half)0, (Half)1, (Half)2, (Half)3];
        var act = () => new CubicSplineInterpolator<Half>(
            knots, new Half[] { (Half)0, (Half)1, (Half)2, (Half)3 }, SplineBoundary.CreateClamped(1e5, 1e5));

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Theory]
    [InlineData(1e30, 0.0)]
    [InlineData(0.0, -1e30)]
    public void ClampedDoubleSlopeAboveDecimalRange_NoScalingEngaged_IsRefusedAsArgumentOutOfRange(double left, double right)
    {
        // decimal has no infinity: a slope beyond +-7.9e28 that scaling cannot bring back into range is refused
        // with the documented ArgumentOutOfRangeException, the same type the IEEE working types raise.
        decimal[] knots = [0m, 1m, 2m, 3m];
        var act = () => new CubicSplineInterpolator<decimal>(
            knots, new decimal[4], SplineBoundary.CreateClamped(left, right));

        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
