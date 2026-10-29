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

public sealed class RevisedNotAKnotTests
{
    [Fact]
    public void Constructor_FewerThanSixPoints_RejectsBothBoundaryFacades()
    {
        double[] xs = [0, 1, 2, 3, 4];
        double[] ys = [0, 1, 16, 81, 256];
        Action generic = () => _ = new CubicSplineInterpolator<double>(xs, ys, SplineBoundary<double>.RevisedNotAKnot);
        Action facade = () => _ = new CubicSplineInterpolator<double>(xs, ys, SplineBoundary.RevisedNotAKnot);

        generic.Should().Throw<ArgumentException>().WithParameterName("xs");
        facade.Should().Throw<ArgumentException>().WithParameterName("xs");
    }

    [Fact]
    public void CubicOnIrregularKnots_IsReproducedInDecimalAndDouble()
    {
        decimal[] xs = [0m, 0.25m, 1m, 2m, 4m, 7m];
        static decimal Polynomial(decimal x) => x * x * x - 2m * x + 1m;
        var ys = xs.Select(Polynomial).ToArray();
        var decimalSpline = new CubicSplineInterpolator<decimal>(xs, ys, SplineBoundary<decimal>.RevisedNotAKnot);
        var doubleSpline = new CubicSplineInterpolator<double>(xs.Select(x => (double)x).ToArray(),
            ys.Select(y => (double)y).ToArray(), SplineBoundary.RevisedNotAKnot);

        // Independent exact cubic oracle. Fourth and fifth divided differences
        // vanish, so the safeguarded RNAK rule must reduce to classical NAK.
        for (var i = 0; i < xs.Length - 1; i++)
        {
            var x = xs[i] + (xs[i + 1] - xs[i]) / 3m;
            decimalSpline.Interpolate(x).Should().BeApproximately(Polynomial(x), 1e-23m);
            doubleSpline.Interpolate((double)x).Should().BeApproximately((double)Polynomial(x), 2e-12);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(10000000000)]
    public void QuarticOnUniformKnots_MatchesAnalyticSplineIncludingAbscissaScaling(long scale)
    {
        decimal[] xs = [0m, scale, 2m * scale, 3m * scale, 4m * scale, 5m * scale];
        decimal[] ys = [0m, 1m, 16m, 81m, 256m, 625m];
        var decimalSpline = new CubicSplineInterpolator<decimal>(xs, ys, SplineBoundary<decimal>.RevisedNotAKnot);
        var doubleSpline = new CubicSplineInterpolator<double>(xs.Select(x => (double)x).ToArray(),
            ys.Select(y => (double)y).ToArray(), SplineBoundary<double>.RevisedNotAKnot);

        // For f(t)=t^4 at integer knots the exact RNAK knot curvatures are
        // 12*i^2-2. Thus s(i+u)=(i+u)^4-u^2*(1-u)^2. f[0..5]=0,
        // so this exercises the undamped fourth-difference jump at both ends.
        for (var i = 0; i < 5; i++)
        {
            foreach (var u in new[] { 0.25m, 0.5m, 0.75m })
            {
                var t = i + u;
                var expected = t * t * t * t - u * u * (1m - u) * (1m - u);
                decimalSpline.Interpolate(t * scale).Should().BeApproximately(expected, 1e-22m);
                doubleSpline.Interpolate((double)(t * scale)).Should().BeApproximately((double)expected, 2e-12);
            }
        }
    }

    [Fact]
    public void SafeguardedIrregularSpline_IsInvariantUnderReflectionAndOrdinateSign()
    {
        decimal[] xs = [0m, 0.25m, 1m, 2m, 4m, 7m];
        decimal[] ys = [2m, 1m, -1m, 3m, 2m, -4m];
        var reflectedXs = xs.Reverse().Select(x => -x).ToArray();
        var reflectedYs = ys.Reverse().Select(y => -y).ToArray();
        var spline = new CubicSplineInterpolator<decimal>(xs, ys, SplineBoundary<decimal>.RevisedNotAKnot);
        var reflected = new CubicSplineInterpolator<decimal>(reflectedXs, reflectedYs, SplineBoundary<decimal>.RevisedNotAKnot);
        var ieee = new CubicSplineInterpolator<double>(xs.Select(x => (double)x).ToArray(),
            ys.Select(y => (double)y).ToArray(), SplineBoundary<double>.RevisedNotAKnot);

        for (var i = 0; i < xs.Length - 1; i++)
        {
            var x = (xs[i] + xs[i + 1]) / 2m;
            spline.Interpolate(x).Should().BeApproximately(-reflected.Interpolate(-x), 1e-22m);
            ieee.Interpolate((double)x).Should().BeApproximately((double)spline.Interpolate(x), 2e-12);
        }
    }
}
