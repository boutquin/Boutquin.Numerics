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

namespace Boutquin.Numerics.Tests.Unit.Interpolation;

/// <summary>
/// Regression pins for the interpolation coordinate / weight normalisation and the
/// derivative-scale loop. Every expected value is the closed form of a line through the
/// origin (the interpolant of linear data is the line), so none of them comes from the
/// implementation under test.
/// </summary>
public sealed class Arena414074R1InterpolationTests
{
    private const double Relative = 1e-14;

    // Runs on a worker so that a non-terminating loop fails the assertion instead of hanging the runner.
    private static T RunBounded<T>(Func<T> evaluate)
    {
        var task = Task.Run(evaluate);
        if (Task.WaitAny([task], TimeSpan.FromSeconds(10)) < 0)
        {
            throw new TimeoutException("The evaluation did not terminate within 10 seconds.");
        }

        return task.GetAwaiter().GetResult();
    }

    private static void ShouldMatch(double actual, double expected, double relative = Relative) =>
        actual.Should().BeApproximately(expected, Math.Abs(expected) * relative);

    private static void ShouldMatch(decimal actual, decimal expected) =>
        actual.Should().BeApproximately(expected, expected * 1e-20m);

    // ------------------------------------------------------------------ F-L23-001 / F-L23-002

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void MonotoneCubicInterpolator_NonFiniteOrdinate_IsRefusedInsteadOfLooping(double bad)
    {
        var act = () => RunBounded(() =>
            MonotoneCubicInterpolator<double>.Instance.Interpolate(0.5, [0, 1, 2], [0, bad, 2]));
        act.Should().Throw<ArgumentException>().WithParameterName("ys");
    }

    [Fact]
    public void MonotoneCubicInterpolator_NonFiniteOrdinate_Float_IsRefusedInsteadOfLooping()
    {
        var act = () => RunBounded(() =>
            MonotoneCubicInterpolator<float>.Instance.Interpolate(0.5f, [0f, 1f, 2f], [0f, float.NegativeInfinity, 2f]));
        act.Should().Throw<ArgumentException>().WithParameterName("ys");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.NegativeInfinity)]
    public void MonotoneCubicSpline_NonFiniteOrdinate_IsRefusedInsteadOfLooping(double bad)
    {
        var act = () => RunBounded(() => new MonotoneCubicSpline<double>([0, 1, 2], [0, bad, 2]));
        act.Should().Throw<ArgumentException>().WithParameterName("ys");
    }

    [Fact]
    public void MonotoneCubicSpline_NonFiniteOrdinate_Half_IsRefusedInsteadOfLooping()
    {
        var act = () => RunBounded(() =>
            new MonotoneCubicSpline<Half>([(Half)0, (Half)1, (Half)2], [(Half)0, Half.NaN, (Half)2]));
        act.Should().Throw<ArgumentException>().WithParameterName("ys");
    }

    // ------------------------------------------------------------------ F-L1-08

    [Theory]
    [InlineData("linear", 3e-24)]
    [InlineData("linear", 7e-24)]
    [InlineData("linear", 1.4e-23)]
    [InlineData("linear", 1e-22)]
    [InlineData("linear", 1e-20)]
    [InlineData("twopoint", 3e-24)]
    [InlineData("twopoint", 1e-20)]
    [InlineData("monotone", 3e-24)]
    [InlineData("monotone", 1e-20)]
    [InlineData("spline", 3e-24)]
    [InlineData("spline", 1e-20)]
    public void LinearWeight_SubnormalFraction_PreservesRepresentableContribution(string route, double q)
    {
        double[] xs = [0, 1e300];
        double[] ys = [0, 1e300];
        var actual = route switch
        {
            "linear" => LinearInterpolator<double>.Instance.Interpolate(q, xs, ys),
            "twopoint" => TwoPointLinearInterpolator<double>.Interpolate(0, 0, 1e300, 1e300, q),
            "monotone" => MonotoneCubicInterpolator<double>.Instance.Interpolate(q, xs, ys),
            _ => new MonotoneCubicSpline<double>(xs, ys).Interpolate(q),
        };
        ShouldMatch(actual, q);
    }

    [Theory]
    [InlineData("linear", "0.000000035")]
    [InlineData("linear", "0.000000015")]
    [InlineData("linear", "0.123456789012345678")]
    [InlineData("twopoint", "0.000000035")]
    [InlineData("monotone", "0.000000015")]
    public void LinearWeight_DecimalGridQuantisation_PreservesRepresentableContribution(string route, string text)
    {
        var q = decimal.Parse(text, CultureInfo.InvariantCulture);
        decimal[] xs = [0m, 1e20m];
        decimal[] ys = [0m, 1e20m];
        var actual = route switch
        {
            "linear" => LinearInterpolator<decimal>.Instance.Interpolate(q, xs, ys),
            "twopoint" => TwoPointLinearInterpolator<decimal>.Interpolate(0m, 0m, 1e20m, 1e20m, q),
            _ => MonotoneCubicInterpolator<decimal>.Instance.Interpolate(q, xs, ys),
        };
        ShouldMatch(actual, q);
    }

    // ------------------------------------------------------------------ F-L1-09 (stateless monotone cubic)

    [Theory]
    [InlineData(1e-100)]
    [InlineData(3e-24)]
    [InlineData(7e-24)]
    [InlineData(1e-20)]
    public void MonotoneCubicInterpolator_WideFirstInterval_PreservesLocalContribution(double q)
    {
        double[] xs = [0, 1e300, 2e300];
        ShouldMatch(MonotoneCubicInterpolator<double>.Instance.Interpolate(q, xs, xs), q);
    }

    [Fact]
    public void MonotoneCubicInterpolator_ModerateSpan_PreservesNormalQueryWithSubnormalFraction()
    {
        double[] xs = [0, 1e20, 2e20];
        ShouldMatch(MonotoneCubicInterpolator<double>.Instance.Interpolate(3e-304, xs, xs), 3e-304);
    }

    [Theory]
    [InlineData("0.000000001")]
    [InlineData("0.000000035")]
    public void MonotoneCubicInterpolator_DecimalWideInterval_PreservesLocalContribution(string text)
    {
        var q = decimal.Parse(text, CultureInfo.InvariantCulture);
        decimal[] xs = [0m, 1e20m, 2e20m];
        ShouldMatch(MonotoneCubicInterpolator<decimal>.Instance.Interpolate(q, xs, xs), q);
    }

    // ------------------------------------------------------------------ F-L1-10 (cached monotone cubic)

    [Theory]
    [InlineData(1e-100)]
    [InlineData(3e-24)]
    [InlineData(7e-24)]
    [InlineData(1e-20)]
    public void MonotoneCubicSpline_WideFirstInterval_PreservesLocalContribution(double q)
    {
        double[] xs = [0, 1e300, 2e300];
        ShouldMatch(new MonotoneCubicSpline<double>(xs, xs).Interpolate(q), q);
    }

    [Fact]
    public void MonotoneCubicSpline_ModerateSpan_PreservesNormalQueryWithSubnormalFraction()
    {
        double[] xs = [0, 1e20, 2e20];
        ShouldMatch(new MonotoneCubicSpline<double>(xs, xs).Interpolate(3e-304), 3e-304);
    }

    [Theory]
    [InlineData("0.000000001")]
    [InlineData("0.000000035")]
    public void MonotoneCubicSpline_DecimalWideInterval_PreservesLocalContribution(string text)
    {
        var q = decimal.Parse(text, CultureInfo.InvariantCulture);
        decimal[] xs = [0m, 1e20m, 2e20m];
        ShouldMatch(new MonotoneCubicSpline<decimal>(xs, xs).Interpolate(q), q);
    }

    // ------------------------------------------------------------------ F-L1-11 (monotone convex)

    [Theory]
    [InlineData(2, 1e-100)]
    [InlineData(2, 3e-24)]
    [InlineData(3, 1e-100)]
    [InlineData(3, 3e-24)]
    [InlineData(3, 7e-24)]
    public void MonotoneConvexInterpolator_WideFirstInterval_PreservesConstantForwardContribution(int count, double q)
    {
        double[] xs = count == 2 ? [0, 1e300] : [0, 1e300, 2e300];
        ShouldMatch(MonotoneConvexInterpolator<double>.Instance.Interpolate(q, xs, xs), q);
    }

    [Fact]
    public void MonotoneConvexInterpolator_ModerateSpan_PreservesNormalQueryWithSubnormalFraction()
    {
        double[] xs = [0, 1e20, 2e20];
        ShouldMatch(MonotoneConvexInterpolator<double>.Instance.Interpolate(3e-304, xs, xs), 3e-304);
    }

    [Theory]
    [InlineData("0.000000001")]
    [InlineData("0.000000035")]
    public void MonotoneConvexInterpolator_DecimalWideInterval_PreservesConstantForwardContribution(string text)
    {
        var q = decimal.Parse(text, CultureInfo.InvariantCulture);
        decimal[] xs = [0m, 1e20m, 2e20m];
        ShouldMatch(MonotoneConvexInterpolator<decimal>.Instance.Interpolate(q, xs, xs), q);
    }

    // ------------------------------------------------------------------ F-L1-12 (cubic spline)

    [Theory]
    [InlineData(1e-100)]
    [InlineData(3e-24)]
    [InlineData(7e-24)]
    [InlineData(1e-20)]
    public void CubicSplineInterpolator_WideFirstInterval_PreservesLocalContribution(double q)
    {
        double[] xs = [0, 1e300, 2e300];
        double[] four = [0, 1e300, 2e300, 3e300];
        ShouldMatch(new CubicSplineInterpolator<double>(xs, xs).Interpolate(q), q);
        ShouldMatch(new CubicSplineInterpolator<double>(four, four, SplineBoundary.NotAKnot).Interpolate(q), q);
    }

    [Fact]
    public void CubicSplineInterpolator_ModerateSpan_PreservesNormalQueryWithSubnormalFraction()
    {
        double[] xs = [0, 1e20, 2e20];
        ShouldMatch(new CubicSplineInterpolator<double>(xs, xs).Interpolate(3e-304), 3e-304);
    }

    [Theory]
    [InlineData("0.000000001")]
    [InlineData("0.000000035")]
    public void CubicSplineInterpolator_DecimalWideInterval_PreservesLocalContribution(string text)
    {
        var q = decimal.Parse(text, CultureInfo.InvariantCulture);
        decimal[] xs = [0m, 1e20m, 2e20m];
        ShouldMatch(new CubicSplineInterpolator<decimal>(xs, xs).Interpolate(q), q);
    }

    // ------------------------------------------------------------------ F-L1-13 (coordinate normalisation)

    [Theory]
    [InlineData("monotone", 1e-200)]
    [InlineData("monotone", 1e-10)]
    [InlineData("cached", 1e-200)]
    [InlineData("cached", 1e-10)]
    [InlineData("cubic", 1e-200)]
    [InlineData("cubic", 1e-10)]
    public void CoordinateNormalisation_SmallQuery_IsNotErasedByTheGridMagnitude(string route, double q)
    {
        double[] xs = [-1e308, 0, 1e308];
        double[] ys = [-1e300, 0, 1e300];
        var actual = route switch
        {
            "monotone" => MonotoneCubicInterpolator<double>.Instance.Interpolate(q, xs, ys),
            "cached" => new MonotoneCubicSpline<double>(xs, ys).Interpolate(q),
            _ => new CubicSplineInterpolator<double>(xs, ys).Interpolate(q),
        };
        ShouldMatch(actual, q * (1e300 / 1e308), 1e-13);
    }

    [Theory]
    [InlineData(1e-200)]
    [InlineData(1e-10)]
    public void CoordinateNormalisation_MonotoneConvex_SmallQueryIsNotErasedByTheGridMagnitude(double q)
    {
        double[] xs = [0, 1e308, 1.5e308];
        double[] ys = [0, 1e300, 1.5e300];
        ShouldMatch(MonotoneConvexInterpolator<double>.Instance.Interpolate(q, xs, ys), q * (1e300 / 1e308), 1e-13);
    }

    // ------------------------------------------------------------------ F-L1-14 (secant and forward headroom scaling)

    // y = 1e-600 x on a grid whose secant underflows although every ordinate is normal.
    private static readonly double[] s_wideXs = [0, 1e300, 2e300, 3e300];
    private static readonly double[] s_tinyYs = [0, 1e-300, 2e-300, 3e-300];

    [Theory]
    [InlineData("monotone", 0.5e300)]
    [InlineData("monotone", 1.25e300)]
    [InlineData("monotone", 2.5e300)]
    [InlineData("cached", 1.25e300)]
    [InlineData("cached", 2.5e300)]
    [InlineData("convex", 0.5e300)]
    [InlineData("convex", 1.25e300)]
    [InlineData("convex", 2.5e300)]
    public void DerivativeScale_SecantUnderflowOnWideGrid_ReproducesLine(string route, double q)
    {
        var actual = route switch
        {
            "monotone" => MonotoneCubicInterpolator<double>.Instance.Interpolate(q, s_wideXs, s_tinyYs),
            "cached" => new MonotoneCubicSpline<double>(s_wideXs, s_tinyYs).Interpolate(q),
            _ => MonotoneConvexInterpolator<double>.Instance.Interpolate(q, s_wideXs, s_tinyYs),
        };
        ShouldMatch(actual, q / 1e300 * 1e-300, 1e-12);
    }

    // f(2^a x; 2^a xs, 2^b ys) = 2^b f(x; xs, ys): power-of-two scaling is exact, so the unit-scale
    // evaluation of the same route (whose scale is one) is the oracle for every scaled evaluation.
    [Theory]
    [InlineData("monotone", 1000, -1000)]
    [InlineData("monotone", 600, -600)]
    [InlineData("cached", 1000, -1000)]
    [InlineData("cached", 600, -600)]
    [InlineData("convex", 1000, -1000)]
    [InlineData("convex", 600, -600)]
    public void DerivativeScale_PowerOfTwoScaling_IsEquivariant(string route, int a, int b)
    {
        double[] ux = [0, 1, 3, 4];
        double[] uy = [0, 1, 1.5, 4];
        var sx = ux.Select(v => Math.ScaleB(v, a)).ToArray();
        var sy = uy.Select(v => Math.ScaleB(v, b)).ToArray();
        foreach (var q in new[] { 0.5, 2.0, 3.5 })
        {
            var expected = Math.ScaleB(Evaluate(route, q, ux, uy), b);
            ShouldMatch(Evaluate(route, Math.ScaleB(q, a), sx, sy), expected, 1e-12);
        }

        static double Evaluate(string route, double q, double[] xs, double[] ys) => route switch
        {
            "monotone" => MonotoneCubicInterpolator<double>.Instance.Interpolate(q, xs, ys),
            "cached" => new MonotoneCubicSpline<double>(xs, ys).Interpolate(q),
            _ => MonotoneConvexInterpolator<double>.Instance.Interpolate(q, xs, ys),
        };
    }

    // One steep narrow interval beside a wide one: the wide interval's secant 1e-292 must survive the
    // headroom scaling the steep secant 1e308 forces. Fritsch-Carlson gives m1 = 3 delta1 (the circle
    // limiter clips the steep neighbour), so at s = 1/4 of the wide interval:
    // y = 0.84375e8 + 0.140625 * 3e8 + 0.15625 * 2e8 = 1.578125e8.
    [Theory]
    [InlineData("monotone")]
    [InlineData("cached")]
    public void DerivativeScale_SteepNarrowNextToWideInterval_KeepsTheWideSecant(string route)
    {
        double[] xs = [0, 1e-300, 1e300];
        double[] ys = [0, 1e8, 2e8];
        var actual = route == "monotone"
            ? MonotoneCubicInterpolator<double>.Instance.Interpolate(2.5e299, xs, ys)
            : new MonotoneCubicSpline<double>(xs, ys).Interpolate(2.5e299);
        ShouldMatch(actual, 1.578125e8, 1e-12);
    }

    // Hagan-West on the same grid, with F1 = 1e308 and F2 = 1e-292 the segment forwards. The node forward
    // g1 is limited to 2 F2 (eq. 62), and the boundary extrapolation g2 = (3 F2 - g1_raw) / 2 uses the
    // unlimited g1_raw ~ F1, so it is negative and is floored to 0 (eqs. 60-62). On [x1, x2] the forward
    // deviation is sector (i) with left = g1 - F2 = F2 and right = g2 - F2 = -F2; at u = 1/4 the integral is
    // (0.25 + 0.140625 + 0.046875) F2 h = 0.4375e8, hence y = 1e8 + 4.375e7.
    [Fact]
    public void DerivativeScale_SteepNarrowNextToWideInterval_MonotoneConvex_KeepsTheWideForward()
    {
        double[] xs = [0, 1e-300, 1e300];
        double[] ys = [0, 1e8, 2e8];
        ShouldMatch(MonotoneConvexInterpolator<double>.Instance.Interpolate(2.5e299, xs, ys), 1.4375e8, 1e-12);
    }

    // float: the shared scale for a steep interval (1e-26 of the span) used to leave scale / h at 1e-54,
    // below float's subnormal range, so the wide interval's secant vanished. The double instance on the
    // same float data has no such overflow and is the oracle.
    [Theory]
    [InlineData("monotone", 2.523996251421622E+26f)]
    [InlineData("monotone", 6.226833323660642E+26f)]
    [InlineData("cached", 6.226833323660642E+26f)]
    public void DerivativeScale_FloatNearMaximumOrdinatesOnMixedWidths_AgreesWithDouble(string route, float q)
    {
        float[] xs = [0f, 11.632639275227042f, 1.01725149365014E+27f];
        float[] ys = [-2.815821649857568E+38f, -3.2398659350298477E+38f, -1.1242885214671933E+38f];
        var oracle = MonotoneCubicInterpolator<double>.Instance.Interpolate(
            q, xs.Select(v => (double)v).ToArray(), ys.Select(v => (double)v).ToArray());
        var actual = route == "monotone"
            ? MonotoneCubicInterpolator<float>.Instance.Interpolate(q, xs, ys)
            : new MonotoneCubicSpline<float>(xs, ys).Interpolate(q);
        ((double)actual).Should().BeApproximately(oracle, Math.Abs(oracle) * 1e-5);
    }

    // decimal: ordinates of 1e-8 over a 1e20 grid give secants (1e-28) at decimal's grid; the integrated
    // forward F * u = 5e-29 rounded to zero at the midpoint of the first interval.
    [Theory]
    [InlineData("convex", "50000000000000000000", "0.000000005")]
    [InlineData("convex", "125000000000000000000", "0.0000000125")]
    [InlineData("monotone", "50000000000000000000", "0.000000005")]
    [InlineData("cached", "125000000000000000000", "0.0000000125")]
    public void DerivativeScale_DecimalSecantAtTheGridOnWideGrid_ReproducesLine(string route, string query, string expectedText)
    {
        decimal[] xs = [0m, 1e20m, 2e20m, 3e20m];
        decimal[] ys = [0m, 0.00000001m, 0.00000002m, 0.00000003m];
        var q = decimal.Parse(query, CultureInfo.InvariantCulture);
        var actual = route switch
        {
            "convex" => MonotoneConvexInterpolator<decimal>.Instance.Interpolate(q, xs, ys),
            "monotone" => MonotoneCubicInterpolator<decimal>.Instance.Interpolate(q, xs, ys),
            _ => new MonotoneCubicSpline<decimal>(xs, ys).Interpolate(q),
        };
        ShouldMatch(actual, decimal.Parse(expectedText, CultureInfo.InvariantCulture));
    }

    // The fraction u = q / h = 2e-309 sits just above the bound below which a subnormal is too coarse to
    // multiply back up (1.4e-309), so it is accurate to 1e-15; but the integrand F * u with F = 1/8 is
    // eight times smaller, falls below that bound and loses three more bits before the segment width
    // scales it back up. y = x / 8 exactly, so the answer is q / 8.
    [Fact]
    public void MonotoneConvexInterpolator_IntegrandBelowTheCoarseBoundWhileTheFractionIsNot_KeepsItsDigits()
    {
        double[] xs = [0, 5.362842736380558E+245, 1.1E+246];
        double[] ys = xs.Select(v => v * 0.125).ToArray();
        const double q = 1.0931302804307119E-63;
        ShouldMatch(MonotoneConvexInterpolator<double>.Instance.Interpolate(q, xs, ys), q * 0.125, 5e-15);
    }

    // A steep interval 1e-30 wide overflows float's secant by a factor of 1e30, so the common scale shrinks to
    // about 2^-100; the wide interval's secant then needs rise * scale / run, whereas scale / run (8e-58) is
    // below float's subnormal range and erased it. The double instance on the same float data is the oracle.
    [Theory]
    [InlineData("monotone", 2.5E+26f)]
    [InlineData("monotone", 7.5E+26f)]
    [InlineData("cached", 5E+26f)]
    public void DerivativeScale_FloatScaleShrunkByAThousandBillionBillion_KeepsTheWideSecant(string route, float q)
    {
        float[] xs = [0f, 1e-30f, 1e27f];
        float[] ys = [0f, 1.5e38f, 3e38f];
        var oracle = MonotoneCubicInterpolator<double>.Instance.Interpolate(
            q, xs.Select(v => (double)v).ToArray(), ys.Select(v => (double)v).ToArray());
        var actual = route == "monotone"
            ? MonotoneCubicInterpolator<float>.Instance.Interpolate(q, xs, ys)
            : new MonotoneCubicSpline<float>(xs, ys).Interpolate(q);
        ((double)actual).Should().BeApproximately(oracle, Math.Abs(oracle) * 1e-5);
    }

    // Hagan-West in float with near-maximum ordinates on a 1e-2 .. 1e24 grid: the first interval is a hundred
    // billion billion times narrower than the second, so its forward is far above the headroom the second
    // interval's forward needs. The double instance on the same float data has no overflow and is the oracle.
    [Fact]
    public void DerivativeScale_FloatNearMaximumOrdinatesOnMixedWidths_MonotoneConvex_AgreesWithDouble()
    {
        float[] xs = [0f, 0.008553998f, 1.04746695E+24f];
        float[] ys = [0f, 8.686109E+36f, 1.0889035E+38f];
        const float q = 7.97961E+23f;
        var oracle = MonotoneConvexInterpolator<double>.Instance.Interpolate(
            q, xs.Select(v => (double)v).ToArray(), ys.Select(v => (double)v).ToArray());
        ((double)MonotoneConvexInterpolator<float>.Instance.Interpolate(q, xs, ys))
            .Should().BeApproximately(oracle, Math.Abs(oracle) * 1e-5);
    }
}
