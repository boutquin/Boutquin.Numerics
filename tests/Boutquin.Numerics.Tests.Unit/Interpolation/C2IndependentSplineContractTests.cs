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

public sealed class C2IndependentSplineContractTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void DuplicateKnots_AreRejectedBeforeInterpolation(int kind)
    {
        double[] xs = [0, 0, 1];
        double[] ys = [1, 2, 3];
        Action act = kind switch
        {
            0 => () => LinearInterpolator<double>.Instance.Interpolate(0, xs, ys),
            1 => () => MonotoneCubicInterpolator<double>.Instance.Interpolate(0, xs, ys),
            2 => () => new MonotoneCubicSpline<double>(xs, ys).Interpolate(0),
            _ => () => new CubicSplineInterpolator<double>(xs, ys).Interpolate(0),
        };
        act.Should().Throw<ArgumentException>();
    }

    [Theory]
    [InlineData(0, 3)]
    [InlineData(1, 4)]
    [InlineData(2, 6)]
    [InlineData(3, 6)]
    public void BoundaryMinimum_RejectsOneTooFewAndAcceptsExactMinimum(int kind, int minimum)
    {
        var boundary = kind switch
        {
            0 => SplineBoundary<double>.Natural,
            1 => SplineBoundary<double>.NotAKnot,
            2 => SplineBoundary<double>.QSpline,
            _ => SplineBoundary<double>.RevisedNotAKnot,
        };
        var xs = Enumerable.Range(0, minimum).Select(i => (double)i).ToArray();
        var ys = xs.Select(x => 1 + x).ToArray();
        FluentActions.Invoking(() => new CubicSplineInterpolator<double>(xs[..^1], ys[..^1], boundary)).Should().Throw<ArgumentException>();
        CubicSplineInterpolator<double>? valid = null;
        FluentActions.Invoking(() => valid = new CubicSplineInterpolator<double>(xs, ys, boundary)).Should().NotThrow();
        valid!.Interpolate(.5).Should().BeApproximately(1.5, 1e-14);
    }

    [Fact]
    public void ModifiedSelector_UniformGridReproducesQuadratic()
    {
        decimal[] xs = [0, 1, 2, 3];
        decimal[] ys = [0, 1, 4, 9];
        var generic = new CubicSplineInterpolator<decimal>(xs, ys, SplineBoundary<decimal>.ModifiedNotAKnot);
        var facade = new CubicSplineInterpolator<decimal>(xs, ys, SplineBoundary.ModifiedNotAKnot);
        foreach (var x in new decimal[] { .25m, .5m, 1.5m, 2.5m, 2.75m })
        {
            generic.Interpolate(x).Should().BeApproximately(x * x, 1e-24m);
            facade.Interpolate(x).Should().BeApproximately(x * x, 1e-24m);
        }
    }
}
