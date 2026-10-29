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

public sealed class C2IndependentCubicPolynomialTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnequalKnots_ReproduceQuadraticAndItsNodalDerivatives(bool cached)
    {
        decimal[] xs = [0, 1, 3, 6];
        decimal[] ys = [0, 1, 9, 36];
        var spline = new MonotoneCubicSpline<decimal>(xs, ys);
        decimal Evaluate(decimal x) => cached ? spline.Interpolate(x) : MonotoneCubicInterpolator<decimal>.Instance.Interpolate(x, xs, ys);
        // Three-point differentiation is exact for x squared. The resulting
        // endpoint tangent ratios lie inside the radius-three circle, so no
        // limiter changes them. Uniqueness of Hermite interpolation gives x².
        foreach (var x in new decimal[] { 0, .25m, .5m, .75m, 1, 1.5m, 2, 2.5m, 3, 3.75m, 4.5m, 5.25m, 6 })
        {
            Evaluate(x).Should().BeApproximately(x * x, 1e-24m);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void WrongSignEndpointAndCircleLimiter_HaveExactCubicValues(bool cached)
    {
        decimal[] xs = [0, 1, 2];
        decimal[] ys = [0, 1, 10];
        var spline = new MonotoneCubicSpline<decimal>(xs, ys);
        decimal Evaluate(decimal x) => cached ? spline.Interpolate(x) : MonotoneCubicInterpolator<decimal>.Instance.Interpolate(x, xs, ys);
        // Raw tangents (-3,5,13): left sign repair gives zero; the first
        // circle correction scales (0,5) to (0,3). The second pair (3,13)
        // is inside its circle. First polynomial is x³; second midpoint
        // is (1+10)/2 + (3-13)/8 = 17/4.
        Evaluate(.25m).Should().BeApproximately(1m / 64m, 1e-24m);
        Evaluate(.5m).Should().BeApproximately(1m / 8m, 1e-24m);
        Evaluate(1.5m).Should().BeApproximately(17m / 4m, 1e-24m);
    }
}
