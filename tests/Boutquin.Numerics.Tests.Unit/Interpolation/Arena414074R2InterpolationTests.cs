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

using System.Reflection;
using Boutquin.Numerics.Interpolation;
using FluentAssertions;

#pragma warning disable BNU0001

namespace Boutquin.Numerics.Tests.Unit.Interpolation;

/// <summary>
/// Round-2 pins for the monotone interpolators: a non-finite ordinate is refused on every public route that
/// reaches the derivative-scale search (and by the search itself), and the Hagan-West deviation integral keeps
/// its first-order term for a small local fraction in the sectors that used to subtract two nearly equal numbers.
/// Every expected value is a closed form worked out from the definition of the piecewise forward, not from the
/// implementation under test.
/// </summary>
public sealed class Arena414074R2InterpolationTests
{
    private static readonly double[] s_nonFinite = [double.NaN, double.PositiveInfinity, double.NegativeInfinity];

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

    // ------------------------------------------------------------------------------------ F-L5-010

    private static readonly Dictionary<string, Func<double[], double[], double, double>> s_cubicRoutes = new()
    {
        ["generic stateless"] = (xs, ys, x) => MonotoneCubicInterpolator<double>.Instance.Interpolate(x, xs, ys),
        ["generic stateless, spans"] = (xs, ys, x) => MonotoneCubicInterpolator<double>.Instance.Interpolate(x, xs.AsSpan(), ys.AsSpan()),
        ["generic cached, arrays"] = (xs, ys, x) => new MonotoneCubicSpline<double>(xs, ys).Interpolate(x),
        ["generic cached, lists"] = (xs, ys, x) => new MonotoneCubicSpline<double>(new List<double>(xs), new List<double>(ys)).Interpolate(x),
        ["legacy stateless"] = (xs, ys, x) => MonotoneCubicInterpolator.Instance.Interpolate(x, xs, ys),
        ["legacy cached"] = (xs, ys, x) => new MonotoneCubicSpline(xs, ys).Interpolate(x),
        ["generic factory"] = (xs, ys, x) => InterpolatorFactory<double>.Create(InterpolatorKind.MonotoneCubic).Interpolate(x, xs, ys),
        ["legacy factory"] = (xs, ys, x) => InterpolatorFactory.Create(InterpolatorKind.MonotoneCubic).Interpolate(x, xs, ys),
    };

    public static IEnumerable<object[]> CubicRouteNames() => s_cubicRoutes.Keys.Select(k => new object[] { k });

    // The query kinds cover the paths that return before the derivative-scale search as well as the one that reaches it.
    private static IEnumerable<double> Queries(double[] xs) =>
        [xs[0] - 1, xs[0], (xs[0] + xs[1]) / 2, (xs[^2] + xs[^1]) / 2, xs[^1], xs[^1] + 1];

    private static IEnumerable<(double[] Xs, double[] Ys)> NonFiniteInstances()
    {
        foreach (var n in new[] { 2, 3, 5 })
        {
            foreach (var at in new[] { 0, n / 2, n - 1 })
            {
                foreach (var bad in s_nonFinite)
                {
                    var xs = Enumerable.Range(0, n).Select(i => (double)i).ToArray();
                    var ys = Enumerable.Range(0, n).Select(i => (double)(i * i)).ToArray();
                    ys[at] = bad;
                    yield return (xs, ys);
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(CubicRouteNames))]
    public void MonotoneCubic_NonFiniteOrdinate_IsRefusedOnEveryPublicRoute(string route)
    {
        var run = s_cubicRoutes[route];
        foreach (var (xs, ys) in NonFiniteInstances())
        {
            foreach (var x in Queries(xs))
            {
                var act = () => RunBounded(() => run(xs, ys, x));
                act.Should().Throw<ArgumentException>($"route '{route}', n={xs.Length}, ys=[{string.Join(", ", ys)}], x={x}")
                    .Which.ParamName.Should().Be("ys");
            }
        }
    }

    [Theory]
    [MemberData(nameof(CubicRouteNames))]
    public void MonotoneCubic_FiniteOrdinates_AreStillAcceptedOnEveryPublicRoute(string route)
    {
        var run = s_cubicRoutes[route];
        double[] xs = [0, 1, 2, 4];
        double[] ys = [0, 1e300, -1e300, 3];
        foreach (var x in Queries(xs))
        {
            double.IsFinite(RunBounded(() => run(xs, ys, x))).Should().BeTrue($"route '{route}', x={x}");
        }
    }

    [Fact]
    public void MonotoneCubic_NonFiniteOrdinate_IsRefusedOnFloatAndHalfRoutes()
    {
        Check<float>();
        Check<Half>();

        static void Check<T>()
            where T : unmanaged, System.Numerics.IFloatingPoint<T>
        {
            foreach (var (xsD, ysD) in NonFiniteInstances())
            {
                var xs = xsD.Select(v => T.CreateChecked(v)).ToArray();
                var ys = ysD.Select(v => T.CreateChecked(v)).ToArray();
                foreach (var xD in Queries(xsD))
                {
                    var x = T.CreateChecked(xD);
                    var routes = new Func<T>[]
                    {
                        () => MonotoneCubicInterpolator<T>.Instance.Interpolate(x, xs, ys),
                        () => new MonotoneCubicSpline<T>(xs, ys).Interpolate(x),
                        () => InterpolatorFactory<T>.Create(InterpolatorKind.MonotoneCubic).Interpolate(x, xs, ys),
                    };
                    foreach (var route in routes)
                    {
                        var act = () => RunBounded(route);
                        act.Should().Throw<ArgumentException>($"{typeof(T).Name}, n={xs.Length}, x={xD}").Which.ParamName.Should().Be("ys");
                    }
                }
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void MonotoneConvex_NonFiniteOrdinate_IsRefusedOnEveryPublicRoute(int at)
    {
        Func<double[], double[], double, double>[] routes =
        [
            (xs, ys, x) => MonotoneConvexInterpolator<double>.Instance.Interpolate(x, xs, ys),
            (xs, ys, x) => MonotoneConvexInterpolator.Instance.Interpolate(x, xs, ys),
            (xs, ys, x) => InterpolatorFactory<double>.Create(InterpolatorKind.MonotoneConvex).Interpolate(x, xs, ys),
            (xs, ys, x) => InterpolatorFactory.Create(InterpolatorKind.MonotoneConvex).Interpolate(x, xs, ys),
        ];
        foreach (var bad in s_nonFinite)
        {
            double[] xs = [0, 1, 2, 3];
            double[] ys = [0, 1, 2, 3];
            ys[at] = bad;
            foreach (var x in new[] { 0.0, 0.5, 1.5, 2.5, 3.0, 4.0 })
            {
                foreach (var route in routes)
                {
                    var act = () => RunBounded(() => route(xs, ys, x));
                    act.Should().Throw<ArgumentException>($"at={at}, bad={bad}, x={x}").Which.ParamName.Should().Be("ys");
                }
            }
        }
    }

    // The derivative-scale search itself is the one place every route reaches, so it refuses a non-finite
    // ordinate on its own instead of relying on each caller to have validated first.
    [Fact]
    public void DerivativeScale_NonFiniteOrdinate_IsRefusedByTheSearchItself()
    {
        foreach (var (xs, ys) in NonFiniteInstances().Where(i => i.Xs.Length >= 3))
        {
            var act = () => RunBounded(() => InterpolationHelper.DerivativeScale<double>(xs, ys, out _));
            act.Should().Throw<ArgumentException>($"n={xs.Length}, ys=[{string.Join(", ", ys)}]").Which.ParamName.Should().Be("ys");

            var floatXs = xs.Select(v => (float)v).ToArray();
            var floatYs = ys.Select(v => (float)v).ToArray();
            var floatAct = () => RunBounded(() => InterpolationHelper.DerivativeScale<float>(floatXs, floatYs, out _));
            floatAct.Should().Throw<ArgumentException>().Which.ParamName.Should().Be("ys");

            var halfXs = xs.Select(v => (Half)v).ToArray();
            var halfYs = ys.Select(v => (Half)v).ToArray();
            var halfAct = () => RunBounded(() => InterpolationHelper.DerivativeScale<Half>(halfXs, halfYs, out _));
            halfAct.Should().Throw<ArgumentException>().Which.ParamName.Should().Be("ys");
        }
    }

    [Fact]
    public void DerivativeScale_FiniteExtremeOrdinates_StillTerminatesWithAScale()
    {
        double[] xs = [0, 1e-300, 2e-300];
        double[] ys = [0, double.MaxValue, -double.MaxValue];
        var scale = RunBounded(() => InterpolationHelper.DerivativeScale<double>(xs, ys, out _));
        double.IsFinite(scale).Should().BeTrue();
        scale.Should().BeGreaterThan(0);
    }

    // ------------------------------------------------------------------------------------ F-SURF-S-1

    private static Func<double, double, double, double> Deviation()
    {
        var method = typeof(MonotoneConvexInterpolator<double>).GetMethod("IntegrateDeviation", BindingFlags.NonPublic | BindingFlags.Static);
        method.Should().NotBeNull();
        return method!.CreateDelegate<Func<double, double, double, double>>();
    }

    private static void ShouldMatch(double actual, decimal expected, double relative)
    {
        var e = (double)expected;
        actual.Should().BeApproximately(e, Math.Abs(e) * relative);
    }

    // Sector (iii), g(0) = 1 falling to g = -1/4 at eta = 3/5 and flat after: the quadratic is
    // g(x) = 1 - (5/4)(2x/eta - x^2/eta^2) = 1 - (25/6) x + (125/36) x^2 ... integrated from the definition,
    //   int_0^u g = u - (25/12) u^2 + (125/108) u^3        (u <= 3/5)
    // and, because the deviation integrates to zero over the whole interval, int_0^u g = R (u - 1) afterwards.
    [Theory]
    [InlineData(1e-3)]
    [InlineData(1e-6)]
    [InlineData(1e-9)]
    [InlineData(1e-12)]
    [InlineData(1e-15)]
    [InlineData(1e-18)]
    [InlineData(0.25)]
    [InlineData(0.59)]
    public void DeviationIntegral_SectorIII_SmallFraction_KeepsTheFirstOrderTerm(double u)
    {
        var du = (decimal)u;
        var expected = du - (25m / 12m) * du * du + (125m / 108m) * du * du * du;
        var deviation = Deviation();
        ShouldMatch(deviation(1.0, -0.25, u), expected, 1e-14);
        ShouldMatch(deviation(-1.0, 0.25, u), -expected, 1e-14);
    }

    [Theory]
    [InlineData(0.6, "0.1")]
    [InlineData(0.8, "0.05")]
    [InlineData(0.99, "0.0025")]
    public void DeviationIntegral_SectorIII_AfterTheKink_IsTheFlatTail(double u, string expectedText)
    {
        var expected = decimal.Parse(expectedText, System.Globalization.CultureInfo.InvariantCulture);
        var deviation = Deviation();
        ShouldMatch(deviation(1.0, -0.25, u), expected, 1e-13);
        ShouldMatch(deviation(-1.0, 0.25, u), -expected, 1e-13);
    }

    // Close to the end of the interval the integral is minus the small remaining flat area, R (u - 1); the
    // sum of the two near-equal terms it used to be built from keeps only the absolute rounding error.
    [Theory]
    [InlineData(0.999999)]
    [InlineData(0.999999999)]
    [InlineData(0.999999999999)]
    public void DeviationIntegral_SectorIII_NearTheEndOfTheInterval_KeepsTheRelativeAccuracy(double u)
    {
        // L = 0.7, R = -0.1 (sector iii, eta = 3/8). 1 - u is exact in double, and R * (u - 1) = 0.1 * (1 - u).
        var expected = 0.1m * (decimal)(1.0 - u);
        var deviation = Deviation();
        ShouldMatch(deviation(0.7, -0.1, u), expected, 1e-10);
        ShouldMatch(deviation(-0.7, 0.1, u), -expected, 1e-10);
    }

    // Sector (iv), g(0) = 1, g(1) = 1/2: the two quadratics meet at eta = 1/3 with extremum A = -1/3,
    //   g(x) = -1/3 + (4/3)(1 - 3x)^2 on [0, 1/3],  so  int_0^u g = u - 4u^2 + 4u^3.
    [Theory]
    [InlineData(1e-3)]
    [InlineData(1e-6)]
    [InlineData(1e-9)]
    [InlineData(1e-12)]
    [InlineData(1e-15)]
    [InlineData(1e-18)]
    [InlineData(0.2)]
    [InlineData(0.33)]
    public void DeviationIntegral_SectorIV_SmallFraction_KeepsTheFirstOrderTerm(double u)
    {
        var du = (decimal)u;
        var expected = du - 4m * du * du + 4m * du * du * du;
        var deviation = Deviation();
        ShouldMatch(deviation(1.0, 0.5, u), expected, 1e-14);
        ShouldMatch(deviation(-1.0, -0.5, u), -expected, 1e-14);
    }

    // Past eta the same integral is minus the remaining area: int_0^u g = -(1-u)(A + (R-A) k), k = 1 - v + v^2/3,
    // v = (1-u)/omega, omega = 2/3 (u = 1/2 gives v = 3/4, k = 7/16 and -1/64).
    [Theory]
    [InlineData(0.5, "-0.015625")]
    [InlineData(1.0, "0")]
    public void DeviationIntegral_SectorIV_AfterTheExtremum_IsMinusTheRemainingArea(double u, string expectedText)
    {
        var expected = decimal.Parse(expectedText, System.Globalization.CultureInfo.InvariantCulture);
        var deviation = Deviation();
        deviation(1.0, 0.5, u).Should().BeApproximately((double)expected, 1e-15);
        deviation(-1.0, -0.5, u).Should().BeApproximately(-(double)expected, 1e-15);
    }

    // Sector (ii), g = -1 on [0, 1/4] and then a quadratic to g(1) = 3, so past the kink g = -1 + 4 ((x - 1/4)/(3/4))^2;
    // integrated from the definition, int_0^u g = -(u - ((u - 1/4)/(3/4))^3) for u > 1/4.
    [Theory]
    [InlineData(0.3, "0.3")]
    [InlineData(0.6, "0.6")]
    [InlineData(0.9, "0.9")]
    public void DeviationIntegral_SectorII_PastTheKink_IsTheIntegralOfTheQuadratic(double u, string text)
    {
        var du = decimal.Parse(text, System.Globalization.CultureInfo.InvariantCulture);
        var tail = (du - 0.25m) / 0.75m;
        var expected = -(du - (tail * tail * tail));
        var deviation = Deviation();
        ShouldMatch(deviation(-1.0, 3.0, u), expected, 1e-14);
        ShouldMatch(deviation(1.0, -3.0, u), -expected, 1e-14);
    }

    // A deviation whose plateau (sector ii) or whose first node value (sector iv) is a million times smaller than the
    // other node value: the quadratic part is then about 1e-6 wide, and close to the end of the interval the integral
    // must not be rebuilt from one minus the rounded kink position. The same method instantiated at decimal, whose
    // 28 digits resolve that subtraction, is the reference. Every input is a power of two or one minus one, so both
    // instances see exactly the same values.
    [Theory]
    [InlineData(1, -1, 26)]
    [InlineData(-1, 1, 26)]
    [InlineData(1, -1, 22)]
    [InlineData(-1, 1, 22)]
    [InlineData(1, -1, 18)]
    [InlineData(-1, 1, 18)]
    public void DeviationIntegral_SectorII_NearTheEnd_AgreesWithDecimalInstance(int leftSign, int rightSign, int uExponent) =>
        CompareWithDecimal(leftSign, rightSign, uExponent);

    [Theory]
    [InlineData(1, 1, 26)]
    [InlineData(-1, -1, 26)]
    [InlineData(1, 1, 22)]
    [InlineData(-1, -1, 22)]
    [InlineData(1, 1, 18)]
    [InlineData(-1, -1, 18)]
    public void DeviationIntegral_SectorIV_NearTheEnd_AgreesWithDecimalInstance(int leftSign, int rightSign, int uExponent) =>
        CompareWithDecimal(leftSign, rightSign, uExponent);

    // left = leftSign * 2^-20, right = rightSign * 1, u = 1 - 2^-uExponent.
    private static void CompareWithDecimal(int leftSign, int rightSign, int uExponent)
    {
        var left = leftSign * Math.Pow(2, -20);
        var right = (double)rightSign;
        var u = 1.0 - Math.Pow(2, -uExponent);
        var leftDecimal = leftSign * (1m / 1048576m);
        var uDecimal = 1m - (1m / (1 << uExponent));
        ((double)leftDecimal).Should().Be(left);
        ((double)uDecimal).Should().Be(u);

        var method = typeof(MonotoneConvexInterpolator<decimal>).GetMethod("IntegrateDeviation", BindingFlags.NonPublic | BindingFlags.Static);
        method.Should().NotBeNull();
        var oracle = (decimal)method!.Invoke(null, [leftDecimal, (decimal)rightSign, uDecimal])!;
        oracle.Should().NotBe(0m);
        Deviation()(left, right, u).Should().BeApproximately((double)oracle, Math.Abs((double)oracle) * 1e-12, $"left={left}, right={right}, u={u}");
    }

    // The public entry in double against the same entry instantiated at decimal, on nodes whose second segment
    // (width 1, after a first segment 1e-9 wide) has deviation sector (iii) / (iv). The result is the NCR at the
    // first node (5e-10) plus the integral over the tiny fraction, so the loss of the first-order term shows up in
    // the leading digits.
    [Theory]
    [InlineData("iii", 0.5, 0.2, 0.1, 1e-3)]
    [InlineData("iii", 0.5, 0.2, 0.1, 1e-6)]
    [InlineData("iii", 0.5, 0.2, 0.1, 1e-9)]
    [InlineData("iii", 0.5, 0.2, 0.1, 1e-12)]
    [InlineData("iv", 0.5, 0.1, 0.2, 1e-3)]
    [InlineData("iv", 0.5, 0.1, 0.2, 1e-6)]
    [InlineData("iv", 0.5, 0.1, 0.2, 1e-9)]
    [InlineData("iv", 0.5, 0.1, 0.2, 1e-12)]
    public void MonotoneConvex_TinyFractionInLaterSegment_AgreesWithDecimalInstance(string sector, double f1, double f2, double f3, double u)
    {
        _ = sector;
        const decimal firstWidth = 0.000000001m;
        decimal[] xsD = [0m, firstWidth, 1m, 2m];
        var ysD = new decimal[4];
        decimal[] forwards = [0m, (decimal)f1, (decimal)f2, (decimal)f3];
        for (var j = 1; j < 4; j++)
        {
            ysD[j] = ysD[j - 1] + (forwards[j] * (xsD[j] - xsD[j - 1]));
        }

        var xD = firstWidth + ((decimal)u * (1m - firstWidth));
        double[] xs = xsD.Select(v => (double)v).ToArray();
        double[] ys = ysD.Select(v => (double)v).ToArray();
        var x = (double)xD;

        var actual = MonotoneConvexInterpolator<double>.Instance.Interpolate(x, xs, ys);
        var oracle = MonotoneConvexInterpolator<decimal>.Instance.Interpolate((decimal)x, xsD, ysD);
        actual.Should().BeApproximately((double)oracle, (double)oracle * 1e-12);
    }
}
