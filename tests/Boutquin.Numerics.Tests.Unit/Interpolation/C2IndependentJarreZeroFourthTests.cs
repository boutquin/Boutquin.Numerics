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

public sealed class C2IndependentJarreZeroFourthTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void VanishingFourthButNonzeroFifth_PreservesUncorrectedEndpoint(bool reflected, bool revised)
    {
        decimal[] xs = [0, 1, 2, 3, 4, 5];
        // The fourth divided difference of x^5 - 10x^4 at 0..4 is
        // sum(0..4)-10=0, while its fifth difference is exactly1.
        var ys = xs.Select(x => { var z = reflected ? 5m - x : x; return z * z * z * z * (z - 10m); }).ToArray();
        CubicSplineInterpolator<decimal>? spline = null;
        var boundary = revised ? SplineBoundary<decimal>.RevisedNotAKnot : SplineBoundary<decimal>.QSpline;
        FluentActions.Invoking(() => spline = new CubicSplineInterpolator<decimal>(xs, ys, boundary)).Should().NotThrow();
        if (revised)
        {
            var jump = reflected ? Third(spline!, 4m) - Third(spline!, 3m) : Third(spline!, 1m) - Third(spline!, 0m);
            jump.Should().BeApproximately(0m, 1e-15m);
        }
        else
        {
            var origin = reflected ? 5m : 0m;
            var direction = reflected ? -1m : 1m;
            var f0 = spline!.Interpolate(origin);
            var f1 = spline.Interpolate(origin + direction * .25m);
            var f2 = spline.Interpolate(origin + direction * .5m);
            var f3 = spline.Interpolate(origin + direction * .75m);
            // Cubic through (0,0),(1,-9),(2,-128),(3,-567) is
            // -35x^3+50x^2-24x, hence endpoint curvature100.
            ((2m * f0 - 5m * f1 + 4m * f2 - f3) * 16m).Should().BeApproximately(100m, 1e-15m);
        }
    }

    private static decimal Third(CubicSplineInterpolator<decimal> spline, decimal origin)
    {
        return (spline.Interpolate(origin + .75m) - 3m * spline.Interpolate(origin + .5m)
            + 3m * spline.Interpolate(origin + .25m) - spline.Interpolate(origin)) * 64m;
    }
}
