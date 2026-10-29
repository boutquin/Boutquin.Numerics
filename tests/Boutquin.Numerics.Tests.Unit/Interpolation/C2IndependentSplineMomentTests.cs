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

public sealed class C2IndependentSplineMomentTests
{
    [Fact]
    public void NaturalQuadraticData_HasExactPiecewiseCubicAndEndpointCurvatures()
    {
        // For x² at0,1,2 the only interior equation is4M1=12, hence M=(0,3,0).
        var spline = new CubicSplineInterpolator<decimal>([0, 1, 2], [0, 1, 4], SplineBoundary<decimal>.Natural);
        for (var k = 0; k <= 8; k++)
        {
            var u = k / 8m;
            spline.Interpolate(u).Should().BeApproximately((u + u * u * u) / 2m, 1e-25m);
            spline.Interpolate(2m - u).Should().BeApproximately(4m - 3.5m * u + .5m * u * u * u, 1e-25m);
        }
        Second(spline, 0, 1).Should().BeApproximately(0, 1e-24m);
        Second(spline, 2, -1).Should().BeApproximately(0, 1e-24m);
        Second(spline, 1, -1).Should().BeApproximately(3, 1e-24m);
        Second(spline, 1, 1).Should().BeApproximately(3, 1e-24m);
    }

    [Fact]
    public void NotAKnot_NonuniformCubic_HasExactCurvaturesAndThirdDerivatives()
    {
        decimal[] xs = [0, 1, 3, 4];
        var ys = xs.Select(Polynomial).ToArray();
        var spline = new CubicSplineInterpolator<decimal>(xs, ys, SplineBoundary<decimal>.NotAKnot);
        for (var k = -2; k <= 18; k++)
        {
            var x = k / 4m;
            spline.Interpolate(x).Should().BeApproximately(Polynomial(x), 1e-22m);
        }
        Second(spline, 0, 1).Should().BeApproximately(4, 1e-22m);
        Second(spline, 4, -1).Should().BeApproximately(28, 1e-22m);
        Third(spline, 0).Should().BeApproximately(6, 1e-22m);
        Third(spline, 1).Should().BeApproximately(6, 1e-22m);
        Third(spline, 3).Should().BeApproximately(6, 1e-22m);
    }

    [Fact]
    public void Clamped_NonuniformCubic_UsesBothSuppliedSlopes()
    {
        decimal[] xs = [0, 1, 3, 4];
        var spline = new CubicSplineInterpolator<decimal>(xs, xs.Select(Polynomial).ToArray(), SplineBoundary<decimal>.CreateClamped(3, 67));
        for (var k = 1; k <= 15; k++)
        {
            var x = k / 4m;
            spline.Interpolate(x).Should().BeApproximately(Polynomial(x), 1e-22m);
        }
    }

    [Fact]
    public void Clamped_TwoNodes_AreRejectedBySelectedApi()
    {
        FluentActions.Invoking(() => new CubicSplineInterpolator<decimal>([0, 1], [0, 1], SplineBoundary<decimal>.CreateClamped(1, 1)))
            .Should().Throw<ArgumentException>();
    }

    [Fact]
    public void CoincidentInteriorKnots_AreRejectedBeforeDivision()
    {
        FluentActions.Invoking(() => new CubicSplineInterpolator<decimal>([0, 1, 1, 2], [0, 1, 1, 4]))
            .Should().Throw<ArgumentException>();
    }

    private static decimal Polynomial(decimal x) => x * x * x + 2 * x * x + 3 * x + 5;

    private static decimal Second(CubicSplineInterpolator<decimal> spline, decimal x, decimal direction) =>
        16m * (2m * spline.Interpolate(x) - 5m * spline.Interpolate(x + direction * .25m)
            + 4m * spline.Interpolate(x + direction * .5m) - spline.Interpolate(x + direction * .75m));

    private static decimal Third(CubicSplineInterpolator<decimal> spline, decimal x) =>
        64m * (spline.Interpolate(x + .75m) - 3m * spline.Interpolate(x + .5m)
            + 3m * spline.Interpolate(x + .25m) - spline.Interpolate(x));
}
