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

public sealed class C2IndependentJarreBoundaryTests
{
    [Theory]
    [InlineData(0, 0, 300, 24, 24)]
    [InlineData(2, 0.9, 348, 26.4, 31.2)]
    [InlineData(-2, -2, 250.9, 19.2, 14.4)]
    [InlineData(-10, -10, 60, 0, -12)]
    [InlineData(-11, -8.8, 36, 0, -15.6)]
    public void QuinticBoundaryData_HasIndependentCurvatureAndJump(
        int hundredths, double qLeft, double qRight, double jumpLeft, double jumpRight)
    {
        // f=x^4+c*x^5 on 0,...,5: fourth divided differences are
        // 1+10c and 1+15c; both fifth differences equal c.
        // These polynomial identities give the endpoint targets directly,
        // without running a divided-difference or spline implementation as oracle.
        // c=-.10 gives exact zero fourth difference; c=-.11 clips to zero.
        decimal c = hundredths / 100m;
        decimal[] xs = [0, 1, 2, 3, 4, 5];
        var ys = xs.Select(x => x * x * x * x * (1m + c * x)).ToArray();
        foreach (var reflected in new[] { false, true })
        {
            var ordinates = reflected ? ys.Reverse().ToArray() : ys;
            var q = new CubicSplineInterpolator<decimal>(xs, ordinates, SplineBoundary<decimal>.QSpline);
            var rnak = new CubicSplineInterpolator<decimal>(xs, ordinates, SplineBoundary<decimal>.RevisedNotAKnot);

            // Four samples determine each cubic's derivatives exactly.
            // Dyadic offsets avoid a rounded finite-difference spacing.
            // Reflection exchanges endpoint curvatures and preserves oriented jumps.
            SecondAtStart(q, 0m, 1m).Should().BeApproximately((decimal)(reflected ? qRight : qLeft), 1e-18m);
            SecondAtStart(q, 5m, -1m).Should().BeApproximately((decimal)(reflected ? qLeft : qRight), 1e-18m);
            (ThirdOnSegment(rnak, 1m) - ThirdOnSegment(rnak, 0m))
                .Should().BeApproximately((decimal)(reflected ? jumpRight : jumpLeft), 1e-17m);
            (ThirdOnSegment(rnak, 4m) - ThirdOnSegment(rnak, 3m))
                .Should().BeApproximately((decimal)(reflected ? jumpLeft : jumpRight), 1e-17m);
        }
    }

    private static decimal SecondAtStart(CubicSplineInterpolator<decimal> spline, decimal origin, decimal direction)
    {
        var f0 = spline.Interpolate(origin);
        var f1 = spline.Interpolate(origin + direction * .25m);
        var f2 = spline.Interpolate(origin + direction * .5m);
        var f3 = spline.Interpolate(origin + direction * .75m);
        return (2m * f0 - 5m * f1 + 4m * f2 - f3) * 16m;
    }

    private static decimal ThirdOnSegment(CubicSplineInterpolator<decimal> spline, decimal origin)
    {
        var f0 = spline.Interpolate(origin);
        var f1 = spline.Interpolate(origin + .25m);
        var f2 = spline.Interpolate(origin + .5m);
        var f3 = spline.Interpolate(origin + .75m);
        return (f3 - 3m * f2 + 3m * f1 - f0) * 64m;
    }
}
