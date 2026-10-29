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

using System.Reflection;
using Boutquin.Numerics.Interpolation;
using Boutquin.Numerics.LinearAlgebra;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Interpolation;

public sealed class Arena20261001PoolTests
{
    [Theory]
    [InlineData(257, false)]
    [InlineData(300, false)]
    [InlineData(257, true)]
    [InlineData(300, true)]
    public void StatelessScratch_IsReusedAndPreservesOffKnotLine(int n, bool convex)
    {
        var xs = Enumerable.Range(0, n).Select(i => (double)i).ToArray();
        var ys = xs.Select(x => 2 * x).ToArray();
        IInterpolator<double> interpolator = convex ? MonotoneConvexInterpolator<double>.Instance : MonotoneCubicInterpolator<double>.Instance;
        {
            for (var i = 0; i < 20; i++)
            {
                interpolator.Interpolate(150.5, xs, ys).Should().BeApproximately(301, 1e-10);
            }

            var before = GC.GetAllocatedBytesForCurrentThread();
            var actual = interpolator.Interpolate(150.5, xs, ys);
            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            actual.Should().BeApproximately(301, 1e-10);
            allocated.Should().Be(0, "large transient buffers must be reused from the pool");
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void EveryCubicBoundary_ReusesLargeThomasScratch(int boundaryIndex)
    {
        const int N = 300;
        var xs = Enumerable.Range(0, N).Select(i => (double)i).ToArray();
        var ys = xs.Select(x => 2 * x + 1).ToArray();
        SplineBoundary<double>[] boundaries =
        [
            SplineBoundary<double>.Natural, SplineBoundary<double>.CreateClamped(2, 2),
            SplineBoundary<double>.NotAKnot, SplineBoundary<double>.ModifiedNotAKnot,
            SplineBoundary<double>.RevisedNotAKnot, SplineBoundary<double>.QSpline,
        ];
        for (var i = 0; i < 20; i++)
        {
            _ = new CubicSplineInterpolator<double>(xs, ys, boundaries[boundaryIndex]);
        }

        var before = GC.GetAllocatedBytesForCurrentThread();
        var spline = new CubicSplineInterpolator<double>(xs, ys, boundaries[boundaryIndex]);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        spline.Interpolate(150.5).Should().BeApproximately(302, 1e-10);
        // Eight owned/work-output arrays plus object headers: < 22 KB. Three or
        // four extra 300-double Thomas arrays would exceed this independent budget.
        allocated.Should().BeLessThan(22000);
    }

    [Fact]
    public void GaussianPivotSearch_ReusesCandidatesAboveThreshold()
    {
        const int N = 1025;
        var augmented = new double[N, 1];
        augmented[0, 0] = 2;
        augmented[N - 1, 0] = 1;
        var method = typeof(GaussianElimination<double>).GetMethod("SelectPivotRow", BindingFlags.NonPublic | BindingFlags.Static)!;
        object?[] arguments = [augmented, N, 0, new System.Random(47), 0d, 0d];
        for (var i = 0; i < 20; i++)
        {
            _ = method.Invoke(null, arguments);
        }

        var before = GC.GetAllocatedBytesForCurrentThread();
        var selected = (int)method.Invoke(null, arguments)!;
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        selected.Should().BeOneOf(0, N - 1);
        allocated.Should().BeLessThan(1000, "reflection boxing is allowed, a 1025-int candidate array is not");
    }
}
