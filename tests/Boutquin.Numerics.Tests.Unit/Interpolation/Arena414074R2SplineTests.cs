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

using System.Globalization;
using Boutquin.Numerics.Interpolation;
using FluentAssertions;

#pragma warning disable BNU0001

namespace Boutquin.Numerics.Tests.Unit.Interpolation;

/// <summary>
/// Round-2 pins for <see cref="CubicSplineInterpolator{T}"/> built from the non-generic double
/// <see cref="SplineBoundary.CreateClamped"/>: the caller's slopes are scaled to the solver's normalized units while
/// still in double and narrowed to the working type once, so a slope below the working type's range in the caller's
/// own units no longer erases a spline whose values are representable. Every expected value below is the clamped
/// cubic spline solved in exact rational arithmetic (Thomas elimination over <c>fractions.Fraction</c>, no rounding)
/// from the exact double slope and the exact working-type knots, then rounded once to the working type; none of it
/// comes from the implementation under test.
/// </summary>
public sealed class Arena414074R2SplineTests
{
    // One working-type ulp at the expected value, measured on the working type's own grid.
    private static double HalfUlp(Half at) => (double)Half.BitIncrement(at) - (double)at;

    private static double FloatUlp(float at) => (double)float.BitIncrement(at) - at;

    private static void AssertHalf(Half actual, double expected, int ulps)
    {
        var rounded = (Half)expected;
        var tolerance = ulps * HalfUlp(rounded);
        Math.Abs((double)actual - (double)rounded).Should().BeLessThanOrEqualTo(
            tolerance,
            $"the spline value must stay within {ulps} Half ulps of the exact-rational value {expected:R}, but was {(double)actual:R}");
    }

    private static void AssertFloat(float actual, double expected, int ulps)
    {
        var rounded = (float)expected;
        var tolerance = ulps * FloatUlp(rounded);
        Math.Abs((double)actual - rounded).Should().BeLessThanOrEqualTo(
            tolerance,
            $"the spline value must stay within {ulps} float ulps of the exact-rational value {expected:R}, but was {(double)actual:R}");
    }

    // ------------------------------------------------------------------------------------ F-L8-R2-01, Half

    public static IEnumerable<object[]> HalfGrids() =>
    [
        // Coordinate and abscissa scaling both engaged (60000 * 16 overflows binary16). Slope 2e-8 is below half of
        // the least Half subnormal (5.96e-8), so narrowing it first returns 0 although every value is a normal Half.
        [new double[] { 0, 30000, 60000 }, 2e-8, 0.0, 7500.0, 9.143352508544922e-05],
        [new double[] { 0, 30000, 60000 }, 2e-8, 0.0, 15000.0, 9.375810623168945e-05],
        [new double[] { 0, 30000, 60000 }, 2e-8, 0.0, 22500.0, 4.9233436584472656e-05],
        // The right slope behaves the same way (mirror image of the first row).
        [new double[] { 0, 30000, 60000 }, 0.0, -2e-8, 52500.0, 9.143352508544922e-05],
        [new double[] { 0, 30000, 60000 }, 0.0, -2e-8, 45000.0, 9.375810623168945e-05],
        // A slope that narrows to a subnormal keeps only a few bits: 1e-6 -> 1.0133e-6, a 1.3 % error (16 ulps).
        [new double[] { 0, 30000, 60000 }, 1e-6, 0.0, 7500.0, 0.00457000732421875],
        // Abscissa scaling alone (2000 * 16 does not overflow, so the coordinate scale stays 1).
        [new double[] { 0, 1000, 2000 }, 1e-7, 0.0, 250.0, 1.52587890625e-05],
        [new double[] { 0, 1000, 2000 }, 1e-7, 0.0, 500.0, 1.5616416931152344e-05],
        [new double[] { 0, 1000, 2000 }, -3e-8, 2e-8, 250.0, -4.351139068603516e-06],
        [new double[] { 0, 1000, 2000 }, -3e-8, 2e-8, 1500.0, -2.205371856689453e-06],
    ];

    [Theory]
    [MemberData(nameof(HalfGrids))]
    public void ClampedDoubleSlopes_Half_MatchTheExactRationalSpline(double[] xs, double left, double right, double query, double expected)
    {
        var knots = xs.Select(x => (Half)x).ToArray();
        var ordinates = new Half[knots.Length];
        var spline = new CubicSplineInterpolator<Half>(knots, ordinates, SplineBoundary.CreateClamped(left, right));

        AssertHalf(spline.Interpolate((Half)query), expected, ulps: 8);
    }

    // ----------------------------------------------------------------------------------- F-L8-R2-01, float

    public static IEnumerable<object[]> FloatGrids() =>
    [
        // 1e-46 is below half of the least float subnormal (1.4e-45); the scaled slope 1e-46 * 2^99 is a normal float.
        [1e-46, 0.0, 5.000000075237331e+29, 1.562500005667959e-17],
        [1e-46, 0.0, 1.5000000225711993e+30, -3.1250000526949485e-18],
        [0.0, -1e-46, 5.000000075237331e+29, -3.1250000526949485e-18],
        [0.0, -1e-46, 1.5000000225711993e+30, 1.562500005667959e-17],
    ];

    [Theory]
    [MemberData(nameof(FloatGrids))]
    public void ClampedDoubleSlopes_Float_MatchTheExactRationalSpline(double left, double right, double query, double expected)
    {
        var edge = (float)1e30;
        float[] knots = [0f, edge, 2 * edge];
        var spline = new CubicSplineInterpolator<float>(knots, new float[3], SplineBoundary.CreateClamped(left, right));

        AssertFloat(spline.Interpolate((float)query), expected, ulps: 16);
    }

    // --------------------------------------------------------------------------------- F-L8-R2-01, decimal

    public static IEnumerable<object[]> DecimalGrids() =>
    [
        // 1e-30 is below decimal's 1e-28 grid and becomes 0 when narrowed first.
        ["1E-30", "0", "50000000000000000000", "1.562500000000000130213157199E-11"],
        ["1E-30", "0", "150000000000000000000", "-3.125000000000000260426314399E-12"],
        // A slope of 1.2345678901234567e-20 keeps only 9 significant digits on the 1e-28 grid: 1.0e-10 relative error.
        ["1.2345678901234567E-20", "0", "50000000000000000000", "0.1929012328317901111446384632"],
        ["0", "-1.2345678901234567E-20", "50000000000000000000", "-0.03858024656635802222892769264"],
        ["0", "-1.2345678901234567E-20", "150000000000000000000", "0.1929012328317901111446384632"],
    ];

    [Theory]
    [MemberData(nameof(DecimalGrids))]
    public void ClampedDoubleSlopes_Decimal_MatchTheExactRationalSpline(string left, string right, string query, string expected)
    {
        var inv = CultureInfo.InvariantCulture;
        decimal[] knots = [0m, 100000000000000000000m, 200000000000000000000m];
        var spline = new CubicSplineInterpolator<decimal>(
            knots,
            new decimal[3],
            SplineBoundary.CreateClamped(double.Parse(left, inv), double.Parse(right, inv)));

        var want = decimal.Parse(expected, NumberStyles.Float, inv);
        var got = spline.Interpolate(decimal.Parse(query, NumberStyles.Float, inv));

        // A double slope carries 53 bits (about 16 digits) and decimal.CreateChecked keeps 15, so 1e-13 relative is
        // the honest ceiling; the 9-digit narrowing this guards against misses it by three orders of magnitude.
        (Math.Abs(got - want) / Math.Abs(want)).Should().BeLessThanOrEqualTo(1e-13m,
            $"expected {want} but was {got}");
    }

    // --------------------- a slope outside the working type's range in the caller's own units is still refused

    [Theory]
    [InlineData(1e5, 0.0, "leftSlope")]
    [InlineData(0.0, -1e5, "rightSlope")]
    [InlineData(-1e300, 1.0, "leftSlope")]
    public void ClampedDoubleSlopes_Half_BeyondTheFiniteRange_AreStillRefused(double left, double right, string parameter)
    {
        Half[] knots = [(Half)0, (Half)1, (Half)2];

        var construct = () => new CubicSplineInterpolator<Half>(knots, new Half[3], SplineBoundary.CreateClamped(left, right));

        construct.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(parameter);
    }

    [Theory]
    [InlineData(1e39, 0.0, "leftSlope")]
    [InlineData(0.0, -1e300, "rightSlope")]
    public void ClampedDoubleSlopes_Float_BeyondTheFiniteRange_AreStillRefused(double left, double right, string parameter)
    {
        var construct = () => new CubicSplineInterpolator<float>([0f, 1f, 2f], new float[3], SplineBoundary.CreateClamped(left, right));

        construct.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(parameter);
    }

    [Theory]
    [InlineData(1e30, 0.0)]
    [InlineData(0.0, -1e300)]
    public void ClampedDoubleSlopes_Decimal_BeyondTheFiniteRange_AreStillRefused(double left, double right)
    {
        var construct = () => new CubicSplineInterpolator<decimal>([0m, 1m, 2m], new decimal[3], SplineBoundary.CreateClamped(left, right));

        construct.Should().Throw<ArgumentOutOfRangeException>();
    }

}
