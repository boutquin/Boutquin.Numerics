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

using Boutquin.Numerics.Statistics;
using FluentAssertions;
using Xunit.Abstractions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

/// <summary>
/// Characterization harness for <see cref="ProbabilityOfBacktestOverfitting{T}"/>. Pins the
/// CSCV output (PBO, logit median, full logit vector) on a panel with deliberate
/// cross-strategy ties, and bounds the per-fold allocation of the OOS ranking. Expected
/// values come from a Python reference written from Bailey et al. (2014) §2: Sharpe on the
/// joined blocks, and an out-of-sample tie group sharing its average rank.
/// </summary>
public sealed class PboSortBufferCharacterizationTests
{
    private readonly ITestOutputHelper _output;

    public PboSortBufferCharacterizationTests(ITestOutputHelper output) => _output = output;

    // Logits from the paper-derived Python reference (ties take the average rank; strategies
    // 1 and 3 are identical, so every fold has an OOS tie).
    private static readonly double[] s_expectedLogits =
    [
        -1.6094379124341005, -0.6931471805599454, -0.33647223662121273,
        0.0, 0.336472236621213, 0.336472236621213,
    ];

    // Deterministic panel: 5 strategies × 8 observations, split into 4 blocks (blockSize 2),
    // C(4,2) = 6 folds. Two strategy pairs are constructed to produce equal joined Sharpes,
    // exercising the tie path in the OOS ranking.
    private static double[,] BuildPanel()
    {
        var data = new double[5, 8];
        // Base wave per strategy + small strategy-specific offsets; strategies 1 and 3
        // share a pattern (ties), strategies 0/2/4 differ.
        for (var t = 0; t < 8; t++)
        {
            data[0, t] = ((t % 3) - 1) * 0.02 + 0.01;
            data[1, t] = ((t % 2) == 0 ? 0.03 : -0.01);
            data[2, t] = ((t % 4) - 1.5) * 0.015;
            data[3, t] = ((t % 2) == 0 ? 0.03 : -0.01); // same as strategy 1 → ties
            data[4, t] = ((t % 3) - 1) * 0.02 - 0.005;
        }

        return data;
    }

    [Fact]
    public void Compute_Decimal_PinnedResult()
    {
        var d = BuildPanel();
        var panel = new decimal[5, 8];
        for (var i = 0; i < 5; i++)
        {
            for (var t = 0; t < 8; t++)
            {
                panel[i, t] = (decimal)d[i, t];
            }
        }

        var r = ProbabilityOfBacktestOverfitting<decimal>.Compute(panel, splitCount: 4);
        _output.WriteLine($"dec pbo={r.Pbo} med={r.LogitMedian} logits=[{string.Join(",", r.LogitValues)}]");

        r.LogitValues.Should().HaveCount(6);
        ((double)r.Pbo).Should().BeApproximately(2.0 / 3.0, 1e-15);
        ((double)r.LogitMedian).Should().BeApproximately(-0.16823611831060636, 1e-12);
        r.LogitValues.Select(x => (double)x).Should().Equal(
            s_expectedLogits, (a, e) => Math.Abs(a - e) <= 1e-12);
    }

    [Fact]
    public void Compute_Double_PerFoldSortDoesNotAllocate()
    {
        // splitCount 8 → C(8,4) = 70 folds. The former per-fold LINQ
        // `Enumerable.Range(...).OrderBy(...).ToArray()` allocated three objects on every
        // one of those folds; the allocation-free rank count allocates none. This
        // budget is set comfortably below the ~70×(Range+OrderBy+ToArray) the old path added,
        // so a regression that reintroduces per-fold LINQ blows through it.
        const int Strategies = 6;
        const int Obs = 24; // 8 blocks × 3 obs
        var panel = new double[Strategies, Obs];
        var state = 12345u;
        for (var i = 0; i < Strategies; i++)
        {
            for (var t = 0; t < Obs; t++)
            {
                state = (1664525u * state) + 1013904223u;
                panel[i, t] = (((state & 0xFFFFFFu) / (double)0x1000000) - 0.5) * 0.04;
            }
        }

        for (var w = 0; w < 5; w++)
        {
            _ = ProbabilityOfBacktestOverfitting<double>.Compute(panel, splitCount: 8);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var w = 0; w < 20; w++)
        {
            _ = ProbabilityOfBacktestOverfitting<double>.Compute(panel, splitCount: 8);
        }

        var after = GC.GetAllocatedBytesForCurrentThread();
        var perCompute = (after - before) / 20.0;
        _output.WriteLine($"per-Compute bytes (splitCount=8, 70 folds): {perCompute}");

        // Budget: the surviving allocations (combos list + its 70 int[4] entries, 70 Complement
        // int[4] arrays, logits, oosSharpes, sortedLogits clone) sit near ~12 KB.
        // The old LINQ path added ~70×3 objects on top; a 20 KB ceiling passes the reused-buffer
        // form and fails a LINQ reintroduction.
        perCompute.Should().BeLessThan(20_000.0,
            because: "the per-fold OOS ranking must not allocate, e.g. through a fresh " +
                     "per-fold LINQ OrderBy");
    }

    [Fact]
    public void Compute_Double_PinnedResult()
    {
        var panel = BuildPanel();

        var r = ProbabilityOfBacktestOverfitting<double>.Compute(panel, splitCount: 4);
        _output.WriteLine($"dbl pbo={r.Pbo:R} med={r.LogitMedian:R} logits=[{string.Join(",", r.LogitValues.Select(x => x.ToString("R", System.Globalization.CultureInfo.InvariantCulture)))}]");

        r.LogitValues.Should().HaveCount(6);
        r.Pbo.Should().BeApproximately(2.0 / 3.0, 1e-15);
        r.LogitMedian.Should().BeApproximately(-0.16823611831060636, 1e-12);
        r.LogitValues.Should().Equal(
            s_expectedLogits, (actual, expected) => Math.Abs(actual - expected) <= 1e-12);
    }
}
