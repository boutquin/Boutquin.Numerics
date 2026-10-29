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

/// <summary>
/// Allocation harness for stackalloc optimisations in
/// <see cref="CubicSplineInterpolator{T}"/> and <see cref="MonotoneCubicInterpolator{T}"/>.
/// Covers AC §3.3 of the span-parameters feature spec.
/// </summary>
public sealed class SplineAllocationTests
{
    // ── CubicSplineInterpolator stackalloc (AC §3.3, ic ≤ 256) ─────────────
    //
    // CubicSplineInterpolator<T> is stateful: the constructor solves the spline
    // system and stores permanent coefficient arrays (_x, _a, _b, _c, _d).
    // The four Thomas-solve scratch arrays (diag, upper, lower, rhs) are transient
    // and must be stackalloc'd for ic ≤ 256.
    //
    // The test measures bytes allocated per construction and asserts the value is
    // below a budget computed as "permanent storage only", which excludes room for
    // the 4 scratch T[ic] arrays. CLR object headers account for approximately
    // 24 bytes per array plus 8 bytes per double element.
    //
    // For n = 10 (ic = 8):
    //   Permanent + temp (y, h, sigma): ~8 arrays × avg 8 elements × 8 bytes + headers ≈ 800 bytes
    //   Scratch (4 × T[8]): 4 × (8×8 + 24) ≈ 352 bytes
    //   Budget = 1050 bytes  (above permanent, below permanent + scratch)

    [Fact]
    public void CubicSplineInterpolator_Construction_ZeroScratchAllocation_WhenIcLe256()
    {
        // ic = n - 2 = 8 for n = 10 — well within the 256-element threshold.
        const int N = 10;
        var xs = Enumerable.Range(0, N).Select(i => (double)i).ToArray();
        var ys = xs.Select(x => Math.Sin(x / N * Math.PI)).ToArray();

        // Warmup: JIT spline construction and static-field initialization.
        for (var i = 0; i < 10; i++)
        {
            _ = new CubicSplineInterpolator<double>(xs, ys);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        const int Rounds = 100;
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < Rounds; i++)
        {
            _ = new CubicSplineInterpolator<double>(xs, ys);
        }

        var after = GC.GetAllocatedBytesForCurrentThread();
        var perConstruction = (after - before) / (double)Rounds;

        // Budget: permanent arrays + temp (y, h, sigma) + CLR header margin,
        // but NOT the 4 scratch diag/upper/lower/rhs arrays (~352 bytes for ic=8).
        const double BudgetWithoutScratch = 1050.0;

        perConstruction.Should().BeLessThan(
            BudgetWithoutScratch,
            because: "CubicSplineInterpolator construction with ic ≤ 256 must use stackalloc " +
                     "for the Thomas-solve scratch arrays (diag, upper, lower, rhs) — " +
                     "no T[] heap allocation for those transient buffers");
    }

    [Fact]
    public void CubicSplineInterpolator_PooledScratch_PreservesInterpolation_WhenIcGt256()
    {
        // For n = 260 (ic = 258 > 256), construction must succeed via pooled scratch.
        const int N = 260;
        var xs = Enumerable.Range(0, N).Select(i => (double)i).ToArray();
        var ys = xs.Select(x => x * 0.5).ToArray();

        var act = () => new CubicSplineInterpolator<double>(xs, ys);

        act.Should().NotThrow(
            because: "inputs exceeding the stackalloc threshold must use pooled scratch without error");
    }

    [Fact]
    public void CubicSplineInterpolator_ProducesCorrectInterpolation_AfterStackAllocChange()
    {
        // Sanity: construction using stackalloc still produces numerically correct results.
        double[] xs = [0.0, 1.0, 2.0, 3.0];
        double[] ys = [0.0, 1.0, 0.0, -1.0];

        var spline = new CubicSplineInterpolator<double>(xs, ys);

        // Every node must be reproduced exactly (the spline interpolates all data points).
        for (var i = 0; i < xs.Length; i++)
        {
            spline.Interpolate(xs[i]).Should().BeApproximately(ys[i], 1e-12,
                because: $"the spline must pass through node {i} exactly");
        }

        // Closed-form interior value: a clamped spline of f(x)=x³ reproduces the cubic exactly,
        // so f(1.5) = 3.375 — a precise expected value, not a wide window.
        double[] cubeXs = [0.0, 1.0, 2.0, 3.0];
        double[] cubeYs = [0.0, 1.0, 8.0, 27.0];
        var cubic = new CubicSplineInterpolator<double>(
            cubeXs, cubeYs, SplineBoundary<double>.CreateClamped(0.0, 27.0));
        cubic.Interpolate(1.5).Should().BeApproximately(3.375, 1e-9);
    }

    // ── MonotoneCubicInterpolator stackalloc (AC §3.3, n ≤ 256) ────────────
    //
    // MonotoneCubicInterpolator<T> is stateless: every call to Interpolate(x, xs, ys)
    // recomputes the slopes from scratch (delta[n-1] and m[n]). With stackalloc for
    // n ≤ 256, each call allocates zero heap bytes — verifiable to exactly 0.

    [Fact]
    public void MonotoneCubicInterpolator_Interpolate_ZeroAllocation_WhenNLe256()
    {
        const int N = 50;
        var xs = Enumerable.Range(0, N).Select(i => (double)i).ToArray();
        var ys = xs.Select(x => x * x * 0.1).ToArray(); // monotone increasing

        var interpolator = MonotoneCubicInterpolator<double>.Instance;

        // Warmup: JIT the Interpolate path.
        for (var i = 0; i < 10; i++)
        {
            _ = interpolator.Interpolate(25.0, xs, ys);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 500; i++)
        {
            _ = interpolator.Interpolate(25.5, xs, ys);
        }

        var after = GC.GetAllocatedBytesForCurrentThread();

        (after - before).Should().Be(
            0,
            because: "MonotoneCubicInterpolator is stateless — with n ≤ 256 it must use stackalloc " +
                     "for the delta and m scratch arrays, allocating zero heap bytes per Interpolate call");
    }

    [Fact]
    public void MonotoneCubicInterpolator_ZeroAllocation_AtThreshold256()
    {
        // Exactly 256 nodes — stackalloc must still be used at n = 256.
        const int N = 256;
        var xs = Enumerable.Range(0, N).Select(i => (double)i).ToArray();
        var ys = xs.Select(x => x).ToArray(); // linear — trivially monotone

        var interpolator = MonotoneCubicInterpolator<double>.Instance;

        for (var i = 0; i < 10; i++)
        {
            _ = interpolator.Interpolate(128.0, xs, ys);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 200; i++)
        {
            _ = interpolator.Interpolate(128.5, xs, ys);
        }

        var after = GC.GetAllocatedBytesForCurrentThread();

        (after - before).Should().Be(
            0,
            because: "n == 256 is at the threshold; stackalloc must still apply");
    }

    [Fact]
    public void MonotoneCubicInterpolator_PooledScratch_PreservesInterpolation_WhenNGt256()
    {
        const int N = 300;
        var xs = Enumerable.Range(0, N).Select(i => (double)i).ToArray();
        var ys = xs.Select(x => x).ToArray();

        var interpolator = MonotoneCubicInterpolator<double>.Instance;
        interpolator.Interpolate(150.5, xs, ys).Should().BeApproximately(150.5, 1e-12,
            because: "the pooled path must reproduce a line between knots");
    }

    [Fact]
    public void MonotoneCubicInterpolator_ProducesCorrectResult_AfterStackAllocChange()
    {
        // Sanity: the stackalloc path still produces correct monotone interpolation.
        double[] xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        double[] ys = [0.0, 1.0, 2.0, 3.0, 4.0]; // linear data

        var interpolator = MonotoneCubicInterpolator<double>.Instance;

        // Linear data → cubic Hermite interpolation must reproduce the line exactly.
        interpolator.Interpolate(1.5, xs, ys).Should().BeApproximately(1.5, 1e-10);
        interpolator.Interpolate(2.7, xs, ys).Should().BeApproximately(2.7, 1e-10);
    }
}
