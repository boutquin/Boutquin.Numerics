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

public sealed class C2IndependentMonotoneRemainingTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(-1)]
    public void Limiter_NonunitSecant_HasExactInsideBoundaryAndOutsideImages(int sign)
    {
        var delta = 2m * sign;
        var left = 6m * sign;
        var right = 8m * sign;
        MonotoneTangentLimiter<decimal>.Limit(delta, ref left, ref right);
        left.Should().BeApproximately(3.6m * sign, 1e-26m);
        right.Should().BeApproximately(4.8m * sign, 1e-26m);
        left = 4m * sign; right = 2m * sign;
        MonotoneTangentLimiter<decimal>.Limit(delta, ref left, ref right);
        left.Should().Be(4m * sign); right.Should().Be(2m * sign);
        left = 6m * sign; right = 0m;
        MonotoneTangentLimiter<decimal>.Limit(delta, ref left, ref right);
        left.Should().Be(6m * sign); right.Should().Be(0m);
    }

    [Theory]
    [InlineData(false, 1)]
    [InlineData(true, 1)]
    [InlineData(false, -1)]
    [InlineData(true, -1)]
    public void OpposingSecants_ForceSharedTurningDerivativeToZero(bool cached, int sign)
    {
        decimal[] xs = [0, 1, 3];
        decimal[] ys = [0, 2m * sign, 1m * sign];
        var spline = new MonotoneCubicSpline<decimal>(xs, ys);
        decimal Evaluate(decimal x) => cached ? spline.Interpolate(x) : MonotoneCubicInterpolator<decimal>.Instance.Interpolate(x, xs, ys);
        // One-sided cubic derivative recovery is exact for each segment.
        // Four samples spaced1/4 recover the endpoint derivative without
        // using the implementation's tangent array or Hermite coefficients.
        foreach (var direction in new[] { -1m, 1m })
        {
            var f0 = Evaluate(1m);
            var f1 = Evaluate(1m + direction * .25m);
            var f2 = Evaluate(1m + direction * .5m);
            var f3 = Evaluate(1m + direction * .75m);
            ((-11m * f0 + 18m * f1 - 9m * f2 + 2m * f3) / (1.5m * direction)).Should().BeApproximately(0m, 1e-24m);
        }
        Evaluate(.5m).Should().BeApproximately(sign * 65m / 48m, 1e-25m);
        Evaluate(2m).Should().BeApproximately(sign * 15m / 8m, 1e-25m);
    }
}
