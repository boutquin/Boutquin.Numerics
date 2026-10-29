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

namespace Boutquin.Numerics.Tests.Unit.Interpolation;

public sealed class CubicSplineInterpolatorTests
{
    [Fact]
    public void Interpolate_NaNTarget_IsRejected()
    {
        var spline = new CubicSplineInterpolator<double>([0, 1, 2], [0, 1, 4]);

        Action act = () => spline.Interpolate(double.NaN);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("x");
    }

    // Natural spline through (0,0), (1,b), (2,−b), (3,b), (4,0): by symmetry σ₁ = σ₃, and the
    // interior equations 4σ₁ + σ₂ = −18b, 2σ₁ + 4σ₂ = 24b give σ₁ = −48b/7, σ₂ = 66b/7. At the
    // midpoint of [1, 2], S = (y₁ + y₂)/2 − (σ₁ + σ₂)/16 = −9b/56. The spline is linear in the
    // ordinates, so this holds for every representable b.
    [Fact]
    public void Natural_LargeFiniteOrdinates_DoubleMatchesClosedForm()
    {
        const double b = 1e307;
        var spline = new CubicSplineInterpolator<double>([0, 1, 2, 3, 4], [0, b, -b, b, 0]);

        spline.Interpolate(1.5).Should().BeApproximately(-9.0 / 56.0 * b, 1e-12 * b);
    }

    [Fact]
    public void Natural_LargeFiniteOrdinates_DecimalMatchesClosedForm()
    {
        const decimal b = 1e28m;
        var build = () => new CubicSplineInterpolator<decimal>([0m, 1m, 2m, 3m, 4m], [0m, b, -b, b, 0m]);

        build.Should().NotThrow<OverflowException>();
        build().Interpolate(1.5m).Should().BeApproximately(-9m / 56m * b, 1e6m);
    }

    [Fact]
    public void Natural_Interpolate_AtNodes_ReturnsExactValues()
    {
        double[] xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        double[] ys = [0.0, 1.0, 0.0, 1.0, 0.0];
        var spline = new CubicSplineInterpolator(xs, ys);

        for (int i = 0; i < xs.Length; i++)
        {
            spline.Interpolate(xs[i]).Should().BeApproximately(ys[i], 1e-10);
        }
    }

    [Fact]
    public void Clamped_Interpolate_BetweenNodes_ReproducesCubicExactly()
    {
        // A clamped cubic spline reproduces ANY cubic polynomial exactly (interior and along the
        // end segments) when the end slopes are supplied. f(x) = x³ on [0,3] has f'(0)=0, f'(3)=27.
        double[] xs = [0.0, 1.0, 2.0, 3.0];
        double[] ys = [0.0, 1.0, 8.0, 27.0];
        var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.CreateClamped(0.0, 27.0));

        // Closed-form expected values: f(x) = x³ at each between-node sample.
        spline.Interpolate(0.5).Should().BeApproximately(0.125, 1e-9);
        spline.Interpolate(1.5).Should().BeApproximately(3.375, 1e-9);
        spline.Interpolate(2.5).Should().BeApproximately(15.625, 1e-9);
    }

    [Fact]
    public void Natural_Interpolate_AtNodes_ReproducesExactly()
    {
        // Natural spline passes through every node exactly (closed-form: ys[i]).
        double[] xs = [0.0, 1.0, 2.0, 3.0];
        double[] ys = [0.0, 1.0, 4.0, 9.0];
        var spline = new CubicSplineInterpolator(xs, ys);

        for (var i = 0; i < xs.Length; i++)
        {
            spline.Interpolate(xs[i]).Should().BeApproximately(ys[i], 1e-12);
        }

        // Between-node value stays within the bracketing node values for this monotone data.
        var mid = spline.Interpolate(1.5);
        mid.Should().BeInRange(1.0, 4.0);
    }

    [Fact]
    public void Constructor_LessThanThreePoints_Throws()
    {
        double[] xs = [1.0, 2.0];
        double[] ys = [0.0, 1.0];

        var act = () => new CubicSplineInterpolator(xs, ys);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Constructor_NonIncreasingX_Throws()
    {
        double[] xs = [1.0, 3.0, 2.0];
        double[] ys = [0.0, 1.0, 2.0];

        var act = () => new CubicSplineInterpolator(xs, ys);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Constructor_MismatchedLengths_Throws()
    {
        double[] xs = [1.0, 2.0, 3.0];
        double[] ys = [0.0, 1.0];

        var act = () => new CubicSplineInterpolator(xs, ys);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Clamped_Interpolate_BelowRange_ExtrapolatesAlongCubicExactly()
    {
        // A clamped cubic spline of f(x) = x³ reproduces the cubic exactly, including the
        // extrapolation region: it continues along the first segment's cubic, which equals x³.
        double[] xs = [0.0, 1.0, 2.0, 3.0];
        double[] ys = [0.0, 1.0, 8.0, 27.0];
        var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.CreateClamped(0.0, 27.0));

        // Closed-form: f(-1) = (-1)³ = -1.
        spline.Interpolate(-1.0).Should().BeApproximately(-1.0, 1e-9);
    }

    [Fact]
    public void Clamped_Interpolate_AboveRange_ExtrapolatesAlongCubicExactly()
    {
        double[] xs = [0.0, 1.0, 2.0, 3.0];
        double[] ys = [0.0, 1.0, 8.0, 27.0];
        var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.CreateClamped(0.0, 27.0));

        // Closed-form: f(4) = 4³ = 64.
        var result = spline.Interpolate(4.0);
        result.Should().BeApproximately(64.0, 1e-9);
    }

    [Fact]
    public void Clamped_AtNodes_ReturnsExactValues()
    {
        // f(x) = x^2 on [1,3], f'(1) = 2, f'(3) = 6
        double[] xs = [1.0, 2.0, 3.0];
        double[] ys = [1.0, 4.0, 9.0];
        var boundary = SplineBoundary.CreateClamped(2.0, 6.0);
        var spline = new CubicSplineInterpolator(xs, ys, boundary);

        for (int i = 0; i < xs.Length; i++)
        {
            spline.Interpolate(xs[i]).Should().BeApproximately(ys[i], 1e-10);
        }
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Clamped_NonFiniteSlope_IsRejectedAtBoundaryCreation(double slope)
    {
        Action genericLeft = () => SplineBoundary<double>.CreateClamped(slope, 0.0);
        Action genericRight = () => SplineBoundary<double>.CreateClamped(0.0, slope);
        Action legacyLeft = () => SplineBoundary.CreateClamped(slope, 0.0);
        Action legacyRight = () => SplineBoundary.CreateClamped(0.0, slope);

        genericLeft.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("leftSlope");
        genericRight.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("rightSlope");
        legacyLeft.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("leftSlope");
        legacyRight.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("rightSlope");
    }

    [Fact]
    public void Clamped_MoreAccurateThanNatural_ForKnownFunction()
    {
        // f(x) = x^2, f'(1) = 2, f'(3) = 6
        // Clamped spline should approximate x^2 better at midpoints than natural.
        double[] xs = [1.0, 2.0, 3.0];
        double[] ys = [1.0, 4.0, 9.0];

        var natural = new CubicSplineInterpolator(xs, ys, SplineBoundary.Natural);
        var clamped = new CubicSplineInterpolator(xs, ys, SplineBoundary.CreateClamped(2.0, 6.0));

        var x = 1.5;
        var exact = x * x; // 2.25
        var naturalError = Math.Abs(natural.Interpolate(x) - exact);
        var clampedError = Math.Abs(clamped.Interpolate(x) - exact);

        clampedError.Should().BeLessThan(naturalError,
            "clamped spline with known derivatives should be more accurate");
    }

    [Fact]
    public void NotAKnot_AtNodes_ReturnsExactValues()
    {
        double[] xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        double[] ys = [1.0, 4.0, 9.0, 16.0, 25.0];
        var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.NotAKnot);

        for (int i = 0; i < xs.Length; i++)
        {
            spline.Interpolate(xs[i]).Should().BeApproximately(ys[i], 1e-10);
        }
    }

    [Fact]
    public void NotAKnot_RequiresFourPoints()
    {
        double[] xs = [1.0, 2.0, 3.0];
        double[] ys = [1.0, 4.0, 9.0];

        var act = () => new CubicSplineInterpolator(xs, ys, SplineBoundary.NotAKnot);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void NaturalBoundary_IsDefaultConstructor()
    {
        double[] xs = [1.0, 2.0, 3.0, 4.0];
        double[] ys = [0.0, 1.0, 0.0, 1.0];

        var withDefault = new CubicSplineInterpolator(xs, ys);
        var withExplicit = new CubicSplineInterpolator(xs, ys, SplineBoundary.Natural);

        var x = 1.5;
        withDefault.Interpolate(x).Should().BeApproximately(withExplicit.Interpolate(x), 1e-14);
    }

    // ── Modified Not-a-Knot (Jarre 2025) ───────────────────────

    [Fact]
    public void ModifiedNotAKnot_AtNodes_ReturnsExactValues()
    {
        double[] xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        double[] ys = [1.0, 4.0, 9.0, 16.0, 25.0];
        var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.ModifiedNotAKnot);

        for (int i = 0; i < xs.Length; i++)
        {
            spline.Interpolate(xs[i]).Should().BeApproximately(ys[i], 1e-10);
        }
    }

    [Fact]
    public void ModifiedNotAKnot_MatchesNotAKnot_OnUniformGrid()
    {
        // On uniform grids, Modified Not-a-Knot should be very close to standard Not-a-Knot
        // (ratio = 1 → alpha = 1 → full not-a-knot behavior).
        double[] xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        double[] ys = [0.0, 1.0, 0.0, 1.0, 0.0];

        var standard = new CubicSplineInterpolator(xs, ys, SplineBoundary.NotAKnot);
        var modified = new CubicSplineInterpolator(xs, ys, SplineBoundary.ModifiedNotAKnot);

        for (double x = 0.0; x <= 4.0; x += 0.25)
        {
            modified.Interpolate(x).Should().BeApproximately(
                standard.Interpolate(x), 1e-10,
                $"Should match standard Not-a-Knot on uniform grid at x={x}");
        }
    }

    [Fact]
    public void ModifiedNotAKnot_RequiresFourPoints()
    {
        double[] xs = [1.0, 2.0, 3.0];
        double[] ys = [1.0, 4.0, 9.0];

        var act = () => new CubicSplineInterpolator(xs, ys, SplineBoundary.ModifiedNotAKnot);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ModifiedNotAKnot_NonUniformGrid_ProducesFiniteResults()
    {
        // Highly non-uniform grid where standard Not-a-Knot may have poor conditioning.
        double[] xs = [0.0, 0.01, 0.02, 1.0, 10.0];
        double[] ys = [0.0, 0.1, 0.2, 1.0, 2.0];

        var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.ModifiedNotAKnot);

        for (double x = 0.0; x <= 10.0; x += 0.5)
        {
            double.IsFinite(spline.Interpolate(x)).Should().BeTrue($"at x={x}");
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ModifiedNotAKnot_TinyPositiveBoundaryRatio_PreservesCurvatureCorrection(bool tinyOnLeft)
    {
        // The reference recurrence applies for every positive spacing ratio, even below 1e-15.
        double[] xs = tinyOnLeft
            ? [0.0, 1e-16, 1.0, 2.0]
            : [-2.0, -1.0, -1e-16, 0.0];
        double[] ys = xs.Select(x => x * x).ToArray();
        double[] h = [xs[1] - xs[0], xs[2] - xs[1], xs[3] - xs[2]];

        var solver = typeof(CubicSplineInterpolator<double>).GetMethod(
            "SolveModifiedNotAKnot", BindingFlags.NonPublic | BindingFlags.Static)!;
        var sigma = (double[])solver.Invoke(null, [xs.Length, h, ys])!;

        double ratio = tinyOnLeft
            ? Math.Min(h[0], h[1]) / Math.Max(h[0], h[1])
            : Math.Min(h[1], h[2]) / Math.Max(h[1], h[2]);
        double expected = tinyOnLeft
            ? ratio * ((h[0] + h[1]) * sigma[1] - h[0] * sigma[2]) / h[1]
            : ratio * ((h[1] + h[2]) * sigma[2] - h[2] * sigma[1]) / h[1];
        double actual = tinyOnLeft ? sigma[0] : sigma[3];

        expected.Should().BeGreaterThan(1e-17);
        actual.Should().BeApproximately(expected, 1e-20);
    }

    // ── Q-Spline (Jarre 2025) ──────────────────────────────────

    [Fact]
    public void QSpline_AtNodes_ReturnsExactValues()
    {
        double[] xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        double[] ys = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0];
        var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.QSpline);

        for (int i = 0; i < xs.Length; i++)
        {
            spline.Interpolate(xs[i]).Should().BeApproximately(ys[i], 1e-10);
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void QSpline_BetweenNodes_ReproducesQuadraticAndCubicExactly(bool irregular, bool cubic)
    {
        double[] xs = irregular
            ? [0.0, 0.2, 1.1, 2.3, 4.5, 5.7]
            : [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        double[] ys = new double[xs.Length];
        for (int i = 0; i < xs.Length; i++)
        {
            ys[i] = Polynomial(xs[i], cubic);
        }

        var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.QSpline);
        double[] fractions = [0.25, 0.5, 0.75];
        for (int i = 0; i < xs.Length - 1; i++)
        {
            foreach (double fraction in fractions)
            {
                double x = xs[i] + fraction * (xs[i + 1] - xs[i]);
                spline.Interpolate(x).Should().BeApproximately(Polynomial(x, cubic), 1e-10);
            }
        }

        static double Polynomial(double x, bool useCubic) => useCubic
            ? x * x * x - 2 * x * x + 3 * x - 1
            : 2 * x * x - 3 * x + 1;
    }

    [Theory]
    [InlineData(false, 0.5, -7.0 / 76.0)]
    [InlineData(false, 3.5, 5701.0 / 38.0)]
    [InlineData(true, 0.1, 6844729.0 / 1819220000.0)]
    [InlineData(true, 3.4, 1194051751.0 / 9096100.0)]
    public void QSpline_Quartic_UndampedOnZeroFifthDifference(bool irregular, double x, double expected)
    {
        // For f(x) = x^4, the five-point endpoint estimates equal f''(x) = 12x^2.
        // The fifth difference vanishes. Independent rational moment equations on six
        // knots give these values with exact second-derivative boundary data.
        double[] xs = irregular
            ? [0.0, 0.2, 1.1, 2.3, 4.5, 5.7]
            : [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        double[] ys = new double[xs.Length];
        for (int i = 0; i < xs.Length; i++)
        {
            ys[i] = xs[i] * xs[i] * xs[i] * xs[i];
        }

        var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.QSpline);

        spline.Interpolate(x).Should().BeApproximately(expected, 1e-10);
    }

    [Fact]
    public void QSpline_ApproximatesSinWell()
    {
        // Test on sin(x) with 11 points on [0, π].
        int n = 11;
        double[] xs = new double[n];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++)
        {
            xs[i] = i * Math.PI / (n - 1);
            ys[i] = Math.Sin(xs[i]);
        }

        var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.QSpline);

        // Check at midpoints between nodes.
        double maxError = 0;
        for (int i = 0; i < n - 1; i++)
        {
            double x = (xs[i] + xs[i + 1]) / 2;
            double error = Math.Abs(spline.Interpolate(x) - Math.Sin(x));
            maxError = Math.Max(maxError, error);
        }

        maxError.Should().BeLessThan(0.01, "Q-spline should approximate sin(x) well with 11 points");
    }

    [Fact]
    public void QSpline_RequiresSixPoints()
    {
        double[] xs = [0.0, 1.0, 2.0, 3.0];
        double[] ys = [0.0, 1.0, 4.0, 9.0];

        var act = () => new CubicSplineInterpolator(xs, ys, SplineBoundary.QSpline);
        act.Should().Throw<ArgumentException>();
    }
}
