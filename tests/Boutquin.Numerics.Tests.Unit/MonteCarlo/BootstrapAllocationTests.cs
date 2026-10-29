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

using Boutquin.Numerics.MonteCarlo;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

/// <summary>
/// Allocation harness for <see cref="HarrellDavisPercentile"/> stackalloc optimisation.
/// Covers AC §3.3 of the span-parameters feature spec.
/// </summary>
public sealed class BootstrapAllocationTests
{
    // ── HarrellDavisPercentile stackalloc (AC §3.3) ─────────────────────────

    [Fact]
    public void HarrellDavisPercentile_ZeroHeapAllocation_WhenNLe512()
    {
        // With n ≤ 512, HarrellDavisPercentile.Compute must allocate no double[] on the
        // heap for the Beta-weights vector — the stackalloc path must be used.
        const int N = 200; // well within the 512-element threshold
        const double P = 0.5;

        var sorted = Enumerable.Range(1, N).Select(i => (double)i).ToArray();

        // Warmup: JIT and static initialization.
        for (var i = 0; i < 10; i++)
        {
            _ = HarrellDavisPercentile.Compute<double>(sorted, P);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 100; i++)
        {
            _ = HarrellDavisPercentile.Compute<double>(sorted, P);
        }

        var after = GC.GetAllocatedBytesForCurrentThread();

        (after - before).Should().Be(
            0,
            because: "HarrellDavisPercentile with n ≤ 512 must use stackalloc for the " +
                     "Beta-weights vector and allocate zero heap bytes");
    }

    [Fact]
    public void HarrellDavisPercentile_ZeroHeapAllocation_AtThreshold512()
    {
        // Exactly at the 512-element threshold the stackalloc path still applies.
        const int N = 512;
        const double P = 0.25;

        var sorted = Enumerable.Range(1, N).Select(i => (double)i).ToArray();

        for (var i = 0; i < 10; i++)
        {
            _ = HarrellDavisPercentile.Compute<double>(sorted, P);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 100; i++)
        {
            _ = HarrellDavisPercentile.Compute<double>(sorted, P);
        }

        var after = GC.GetAllocatedBytesForCurrentThread();

        (after - before).Should().Be(
            0,
            because: "n == 512 is still within the threshold; stackalloc path should be used");
    }

    [Fact]
    public void HarrellDavisPercentile_HeapFallback_DoesNotThrow_WhenNGt512()
    {
        // For n > 512, the method must fall back to heap allocation without throwing.
        const int N = 1000;
        const double P = 0.75;

        var sorted = Enumerable.Range(1, N).Select(i => (double)i).ToArray();

        var act = () => HarrellDavisPercentile.Compute<double>(sorted, P);

        act.Should().NotThrow(
            because: "inputs exceeding the stackalloc threshold must fall back to new double[] without error");
    }

    [Fact]
    public void HarrellDavisPercentile_WithSpan_ProducesCorrectMedian()
    {
        // Sanity: after the ReadOnlySpan<T> migration, the method produces correct results.
        // Median of 1..99 should be approximately 50.
        var sorted = Enumerable.Range(1, 99).Select(i => (double)i).ToArray();

        var result = HarrellDavisPercentile.Compute<double>(sorted, 0.5);

        result.Should().BeApproximately(50.0, 2.0,
            because: "HD median of integers 1..99 should be near 50");
    }

    [Fact]
    public void HarrellDavisPercentile_AcceptsReadOnlySpan_NotOnlyArray()
    {
        // Verifies the public signature accepts ReadOnlySpan<T> (the span-migration AC).
        ReadOnlySpan<double> span = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // This must compile. If it fails to compile, the signature was not migrated.
        var result = HarrellDavisPercentile.Compute<double>(span, 0.5);

        result.Should().BeApproximately(3.0, 0.5);
    }

    // ── Percentile span migration (AC §3.1) ─────────────────────────────────

    [Fact]
    public void Percentile_AcceptsReadOnlySpan()
    {
        ReadOnlySpan<double> span = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        var result = Percentile.Compute<double>(span, 0.5);

        result.Should().BeApproximately(3.0, 0.01,
            because: "median of [1, 2, 3, 4, 5] is 3");
    }

    // ── PolitisWhiteBlockLength span migration (AC §3.1) ─────────────────────

    [Fact]
    public void PolitisWhiteBlockLength_AcceptsReadOnlySpan()
    {
        // Builds a deterministic AR(1) series.
        const int N = 200;
        var series = new double[N];
        series[0] = 0.0;
        for (var i = 1; i < N; i++)
        {
            series[i] = 0.5 * series[i - 1] + (i % 7 - 3) * 0.01;
        }

        ReadOnlySpan<double> span = series;

        // Must compile — verifies ReadOnlySpan<double> signature.
        var blockLength = PolitisWhiteBlockLength.EstimateDouble(span);

        blockLength.Should().BeGreaterThan(0,
            because: "estimated block length must be positive");
    }
}
