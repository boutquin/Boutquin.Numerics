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

public sealed class ReleaseQualificationInterpolationTests
{
    [Theory]
    [InlineData("0.000000000000001", 1)]
    [InlineData("0.000000000000001", 2)]
    [InlineData("0.00000000000000000001", 1)]
    [InlineData("0.00000000000000000001", 2)]
    public void DecimalScaledTangents_ReproduceLinearFunction(string slopeText, int width)
    {
        var slope = decimal.Parse(slopeText, CultureInfo.InvariantCulture);
        var xs = Enumerable.Range(0, 5).Select(i => (decimal)(i * width)).ToArray();
        var ys = xs.Select(x => slope * x).ToArray();
        foreach (var sign in new[] { 1m, -1m })
        {
            var signed = ys.Select(y => sign * y).ToArray();
            var spline = new MonotoneCubicSpline<decimal>(xs, signed);
            foreach (var x in new[] { .25m, 1.25m, 2.5m, 3.75m }.Select(x => x * width))
            {
                var expected = sign * slope * x;
                MonotoneCubicInterpolator<decimal>.Instance.Interpolate(x, xs, signed)
                    .Should().BeApproximately(expected, 2e-27m);
                spline.Interpolate(x).Should().BeApproximately(expected, 2e-27m);
                if (sign > 0)
                {
                    MonotoneConvexInterpolator<decimal>.Instance.Interpolate(x, xs, ys)
                        .Should().BeApproximately(expected, 2e-27m);
                }
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void CubicFallback_PreservesOriginalKnotValues(int boundaryIndex)
    {
        decimal[] xs = [0, 1e-9m, 1, 2, 3, 4];
        decimal[] ys = [1e-20m, 1e14m, 1e-20m, 1, 1e-20m, 2];
        SplineBoundary<decimal>[] boundaries = [
            SplineBoundary<decimal>.Natural, SplineBoundary<decimal>.CreateClamped(0m, 0m),
            SplineBoundary<decimal>.NotAKnot, SplineBoundary<decimal>.ModifiedNotAKnot,
            SplineBoundary<decimal>.RevisedNotAKnot, SplineBoundary<decimal>.QSpline];
        var spline = new CubicSplineInterpolator<decimal>(xs, ys, boundaries[boundaryIndex]);
        for (var i = 0; i < xs.Length; i++)
        {
            spline.Interpolate(xs[i]).Should().Be(ys[i], "interpolation reproduces each original knot even when coefficients require scaling");
        }
    }
}
