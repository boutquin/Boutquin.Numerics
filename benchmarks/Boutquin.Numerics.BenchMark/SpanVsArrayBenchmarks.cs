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

using BenchmarkDotNet.Attributes;
using Boutquin.Numerics.MonteCarlo;
using Boutquin.Numerics.Statistics;

namespace Boutquin.Numerics.BenchMark;

/// <summary>
/// Before/after allocation comparison for the <c>T[]</c> → <c>ReadOnlySpan&lt;T&gt;</c>
/// parameter migration.
/// </summary>
/// <remarks>
/// <para>
/// Run this class explicitly: <c>--filter '*SpanVsArrayBenchmarks*'</c>
/// </para>
/// <para>
/// Acceptance criteria:
/// <list type="bullet">
/// <item><description><c>BootstrapSpanCallback.Allocated</c> allocates less than <c>BootstrapArrayCallback.Allocated</c> —
/// the span callback avoids the per-sample <c>ToArray</c> copy the array callback performs.</description></item>
/// <item><description><c>NeweyWestSpan.Allocated</c> is identical to <c>NeweyWestArray.Allocated</c> —
/// implicit <c>T[]</c> → <c>ReadOnlySpan&lt;T&gt;</c> conversion adds no extra copy.</description></item>
/// </list>
/// </para>
/// </remarks>
[MemoryDiagnoser]
[BenchmarkCategory("Span")]
public class SpanVsArrayBenchmarks
{
    private const int N = 1000;
    private double[] _data = null!;
    private BootstrapMonteCarloEngine<double> _engine = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new System.Random(5765);
        _data = new double[N];
        for (var i = 0; i < N; i++)
        {
            _data[i] = (rng.NextDouble() - 0.5) * 0.04;
        }

        _engine = BootstrapMonteCarloEngine<double>.FromSeed(simulationCount: 500, seed: 5765);
    }

    /// <summary>
    /// Baseline bootstrap: callback copies the resampled span to an array before computing,
    /// simulating pre-span code that expected a <c>T[]</c> argument.
    /// </summary>
    [Benchmark]
    public double BootstrapArrayCallback()
        => _engine.Run(_data.AsSpan(), static span =>
        {
            var arr = span.ToArray();
            var sum = 0.0;
            foreach (var v in arr)
            {
                sum += v;
            }

            return sum / arr.Length;
        }).Mean;

    /// <summary>
    /// Span bootstrap: callback works directly on the resampled <see cref="ReadOnlySpan{T}"/> —
    /// no per-sample array allocation.
    /// </summary>
    [Benchmark]
    public double BootstrapSpanCallback()
        => _engine.Run(_data.AsSpan(), static span =>
        {
            var sum = 0.0;
            for (var i = 0; i < span.Length; i++)
            {
                sum += span[i];
            }

            return sum / span.Length;
        }).Mean;

    [Benchmark(Baseline = true)]
    public double NeweyWestArray() => NeweyWestVariance<double>.MeanVariance(_data, lags: 5);

    /// <summary>
    /// Span overload: passes <c>_data.AsSpan()</c> explicitly. Expected to be allocation-identical
    /// to <see cref="NeweyWestArray"/> — the implicit <c>T[]</c> → <c>ReadOnlySpan&lt;T&gt;</c>
    /// conversion at the call site adds no copy.
    /// </summary>
    [Benchmark]
    public double NeweyWestSpan() => NeweyWestVariance<double>.MeanVariance(_data.AsSpan(), lags: 5);
}
