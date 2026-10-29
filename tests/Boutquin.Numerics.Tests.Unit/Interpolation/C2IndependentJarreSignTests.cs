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

public sealed class C2IndependentJarreSignTests
{
    [Fact]
    public void NegatedQuintic_PreservesSignedCurvaturesAndJumps()
    {
        decimal[] xs = [0, 1, 2, 3, 4, 5];
        // Negative of x^4 + .02x^5. Divided differences at the left are
        // fourth=-1.2, fifth=-.02. Damped fourth is -1.15 for Q and
        // -1.1 for RNAK. Right values follow reflection, not a sign guess.
        var ys = xs.Select(x => -x * x * x * x * (1m + .02m * x)).ToArray();
        var q = new CubicSplineInterpolator<decimal>(xs, ys, SplineBoundary<decimal>.QSpline);
        var r = new CubicSplineInterpolator<decimal>(xs, ys, SplineBoundary<decimal>.RevisedNotAKnot);
        Second(q, 0, 1).Should().BeApproximately(-.9m, 1e-18m);
        Second(q, 5, -1).Should().BeApproximately(-348m, 1e-18m);
        (Third(r, 1) - Third(r, 0)).Should().BeApproximately(-26.4m, 1e-17m);
        (Third(r, 4) - Third(r, 3)).Should().BeApproximately(-31.2m, 1e-17m);
    }

    [Fact]
    public void Cubic_ZeroFourthAndFifthDifferences_PreserveCubicBoundary()
    {
        decimal[] xs = [0, 1, 2, 3, 4, 5];
        var ys = xs.Select(x => x * x * x + 2m * x * x + 3m).ToArray();
        foreach (var boundary in new[] { SplineBoundary<decimal>.QSpline, SplineBoundary<decimal>.RevisedNotAKnot })
        {
            var spline = new CubicSplineInterpolator<decimal>(xs, ys, boundary);
            // f''(0)=4 is deliberately nonzero: zero fourth difference means
            // zero correction, not a natural boundary. A cubic has no jump
            // in its constant third derivative and is its own interpolant.
            Second(spline, 0, 1).Should().BeApproximately(4m, 1e-18m);
            Second(spline, 5, -1).Should().BeApproximately(34m, 1e-18m);
            foreach (var x in new decimal[] { .25m, 1.5m, 2.75m, 4.5m })
            {
                spline.Interpolate(x).Should().BeApproximately(x * x * x + 2m * x * x + 3m, 1e-18m);
            }
        }
    }

    private static decimal Second(CubicSplineInterpolator<decimal> spline, decimal origin, decimal direction)
    {
        var f0 = spline.Interpolate(origin);
        var f1 = spline.Interpolate(origin + direction * .25m);
        var f2 = spline.Interpolate(origin + direction * .5m);
        var f3 = spline.Interpolate(origin + direction * .75m);
        return (2m * f0 - 5m * f1 + 4m * f2 - f3) * 16m;
    }

    private static decimal Third(CubicSplineInterpolator<decimal> spline, decimal origin)
    {
        var f0 = spline.Interpolate(origin);
        var f1 = spline.Interpolate(origin + .25m);
        var f2 = spline.Interpolate(origin + .5m);
        var f3 = spline.Interpolate(origin + .75m);
        return (f3 - 3m * f2 + 3m * f1 - f0) * 64m;
    }
}
