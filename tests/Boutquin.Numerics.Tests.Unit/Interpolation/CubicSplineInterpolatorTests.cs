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

public sealed class CubicSplineInterpolatorTests
{
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
    public void Natural_Interpolate_BetweenNodes_ReturnsSmooth()
    {
        double[] xs = [0.0, 1.0, 2.0, 3.0];
        double[] ys = [0.0, 1.0, 4.0, 9.0];
        var spline = new CubicSplineInterpolator(xs, ys);

        var result = spline.Interpolate(1.5);
        result.Should().BeGreaterThan(1.0);
        result.Should().BeLessThan(4.0);
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
    public void Natural_Interpolate_BelowRange_ExtrapolatesUsingFirstSegment()
    {
        double[] xs = [1.0, 2.0, 3.0];
        double[] ys = [1.0, 4.0, 9.0];
        var spline = new CubicSplineInterpolator(xs, ys);

        var result = spline.Interpolate(0.0);
        double.IsFinite(result).Should().BeTrue();
    }

    [Fact]
    public void Natural_Interpolate_AboveRange_ExtrapolatesUsingLastSegment()
    {
        double[] xs = [1.0, 2.0, 3.0];
        double[] ys = [1.0, 4.0, 9.0];
        var spline = new CubicSplineInterpolator(xs, ys);

        var result = spline.Interpolate(5.0);
        double.IsFinite(result).Should().BeTrue();
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

    // ── Q-Spline (Jarre 2025) ──────────────────────────────────

    [Fact]
    public void QSpline_AtNodes_ReturnsExactValues()
    {
        double[] xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        double[] ys = [0.0, 1.0, 4.0, 9.0, 16.0];
        var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.QSpline);

        for (int i = 0; i < xs.Length; i++)
        {
            spline.Interpolate(xs[i]).Should().BeApproximately(ys[i], 1e-10);
        }
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
    public void QSpline_RequiresFourPoints()
    {
        double[] xs = [1.0, 2.0, 3.0];
        double[] ys = [1.0, 4.0, 9.0];

        var act = () => new CubicSplineInterpolator(xs, ys, SplineBoundary.QSpline);
        act.Should().Throw<ArgumentException>();
    }
}
