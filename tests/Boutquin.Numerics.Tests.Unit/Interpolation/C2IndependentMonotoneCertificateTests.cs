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

public sealed class C2IndependentMonotoneCertificateTests
{
    [Fact]
    public void Limiter_ExactPythagoreanPairs_RespectCircleAndFlatBranch()
    {
        var left = 3m;
        var right = 4m;
        MonotoneTangentLimiter<decimal>.Limit(1, ref left, ref right);
        left.Should().BeApproximately(1.8m, 1e-26m);
        right.Should().BeApproximately(2.4m, 1e-26m);
        left = 4;
        right = 3;
        MonotoneTangentLimiter<decimal>.Limit(1, ref left, ref right);
        left.Should().BeApproximately(2.4m, 1e-26m);
        right.Should().BeApproximately(1.8m, 1e-26m);
        left = 3;
        right = 0;
        MonotoneTangentLimiter<decimal>.Limit(1, ref left, ref right);
        left.Should().Be(3);
        right.Should().Be(0);
        left = 2;
        right = 3;
        MonotoneTangentLimiter<decimal>.Limit(0, ref left, ref right);
        left.Should().Be(0);
        right.Should().Be(0);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PublicCubics_HaveContinuousSlopesAndNonnegativeDerivativeEverywhere(bool cached)
    {
        double[] xs = [0, 1, 2, 3, 5, 6];
        double[] ys = [-10, 0, 1, 11, 11, 13];
        var spline = new MonotoneCubicSpline<double>(xs, ys);
        double Evaluate(double x) => cached ? spline.Interpolate(x) : MonotoneCubicInterpolator<double>.Instance.Interpolate(x, xs, ys);
        var previousSlope = double.NaN;
        for (var i = 0; i < xs.Length - 1; i++)
        {
            var width = xs[i + 1] - xs[i];
            var f0 = Evaluate(xs[i]);
            var f1 = Evaluate(xs[i] + width / 4);
            var f2 = Evaluate(xs[i] + width / 2);
            var f3 = Evaluate(xs[i] + 3 * width / 4);
            // Newton forward differences reconstruct the unique cubic from four
            // public samples, independently of the production Hermite formulas.
            var d1 = f1 - f0;
            var d2 = f2 - 2 * f1 + f0;
            var d3 = f3 - 3 * f2 + 3 * f1 - f0;
            var a = 32d * d3 / 3;
            var b = 8 * (d2 - d3);
            var c = 4 * d1 - 2 * d2 + 4d * d3 / 3;
            (f0 + a + b + c).Should().BeApproximately(ys[i + 1], 2e-12);
            var minimum = Math.Min(c, 3 * a + 2 * b + c);
            if (a > 0)
            {
                var vertex = -b / (3 * a);
                if (vertex > 0 && vertex < 1)
                {
                    minimum = Math.Min(minimum, (3 * a * vertex + 2 * b) * vertex + c);
                }
            }
            minimum.Should().BeGreaterThanOrEqualTo(-2e-12, "the derivative quadratic minimum certifies the whole segment");
            if (i > 0) { (c / width).Should().BeApproximately(previousSlope, 2e-12); }
            previousSlope = (3 * a + 2 * b + c) / width;
            if (ys[i] == ys[i + 1])
            {
                a.Should().BeApproximately(0, 2e-12);
                b.Should().BeApproximately(0, 2e-12);
                c.Should().BeApproximately(0, 2e-12);
            }
        }
    }

    [Fact]
    public void CachedSpline_CopiesInputBeforePrecomputation()
    {
        double[] xs = [0, 1, 2];
        double[] ys = [0, 2, 4];
        var spline = new MonotoneCubicSpline<double>(xs, ys);
        xs[1] = 100;
        ys[1] = -100;
        spline.Interpolate(.5).Should().Be(1);
        spline.Interpolate(1.5).Should().Be(3);
    }
}
