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

public sealed class Arena20261001PublicationTests
{
    [Fact]
    public void SuccessorQSpline_RequiresSixKnotsForFifthDifferenceSafeguard()
    {
        double[] xs = [0, 1, 2, 3, 4];
        double[] ys = [0, 1, 4, 9, 16];
        var act = () => new CubicSplineInterpolator<double>(xs, ys, SplineBoundary<double>.QSpline);
        act.Should().Throw<ArgumentException>().WithParameterName("xs");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FritschCarlson_S2ActiveCircle_HasAnalyticInteriorValue(bool cached)
    {
        double[] xs = [0, 1, 2, 3];
        double[] ys = [-10, 0, 1, 11];
        // FC (1980), S2 circle: on the middle interval both initial ratios are
        // 5.5, so the limiter gives 3/sqrt(2). Hermite at t=1/4 yields this value.
        var expected = 5d / 32 + 9d / (32 * Math.Sqrt(2));
        var actual = cached ? new MonotoneCubicSpline<double>(xs, ys).Interpolate(1.25)
            : MonotoneCubicInterpolator<double>.Instance.Interpolate(1.25, xs, ys);
        actual.Should().BeApproximately(expected, 2e-15);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FritschCarlson_WrongSignEndpoint_IsClippedToZero(bool cached)
    {
        double[] xs = [0, 1, 2];
        double[] ys = [0, 1, 10];
        // Three-point left derivative -3 has the wrong sign and becomes zero.
        // The first interval's S2-clipped right derivative is 3: p(t)=t^3.
        var actual = cached ? new MonotoneCubicSpline<double>(xs, ys).Interpolate(.25)
            : MonotoneCubicInterpolator<double>.Instance.Interpolate(.25, xs, ys);
        actual.Should().BeApproximately(1d / 64, 2e-15);
    }

    [Theory]
    [InlineData(8, false)]
    [InlineData(17, false)]
    [InlineData(8, true)]
    [InlineData(17, true)]
    public void JarreQ_TheoremTwo_DerivativeDependentBetweenKnotBound(int n, bool irregular)
    {
        // Jarre arXiv2507.05083v1 Th.2: f=sin on [pi/4,5pi/4], ||f4||=||f5||=1.
        // This pins an applicable analytic bound, not an unknown historical mesh.
        var xs = Enumerable.Range(0, n).Select(i => Math.PI / 4 + Math.PI * (irregular ? Math.Pow((double)i / (n - 1), 1.3) : (double)i / (n - 1))).ToArray();
        var ys = xs.Select(Math.Sin).ToArray();
        var h = xs.Zip(xs.Skip(1), (left, right) => right - left).Max();
        var r = Math.Min(11d / 6, 5 * h / 12);
        var bound = (5d / 384 + r / 8) * Math.Pow(h, 4);
        var spline = new CubicSplineInterpolator<double>(xs, ys, SplineBoundary<double>.QSpline);
        for (var i = 0; i < n - 1; i++)
        {
            for (var j = 1; j < 50; j++)
            {
                var x = xs[i] + (xs[i + 1] - xs[i]) * j / 50;
                Math.Abs(spline.Interpolate(x) - Math.Sin(x)).Should().BeLessThanOrEqualTo(bound + 2e-15);
            }
        }
    }
}
