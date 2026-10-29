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
using Boutquin.Numerics.Random;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

/// <summary>
/// End-to-end oracle for <see cref="FastDoubleBootstrap.PValue"/> built from the
/// Davidson-MacKinnon fast-double-bootstrap definition rather than from the C#
/// implementation. The statistic delegate ignores the resampled data and replays a
/// scripted sequence, so the outer distribution τ* and the inner distribution τ**
/// are known exactly and the expected p-value is computed by hand:
/// <list type="bullet">
/// <item><description>right tail: <c>p̂₁ = #{τ* > τ̂}/B</c>, <c>Q = (1 − p̂₁)</c>-quantile of τ**, <c>p = #{τ* > Q}/B</c>;</description></item>
/// <item><description>left tail: <c>p̂₁ = #{τ* &lt; τ̂}/B</c>, <c>Q = p̂₁</c>-quantile of τ**, <c>p = #{τ* &lt; Q}/B</c>;</description></item>
/// <item><description>two-sided: <c>min(2·min(p_right, p_left), 1)</c> of the two one-sided FDB p-values.</description></item>
/// </list>
/// Quantiles use the linear-interpolation convention of <see cref="Percentile"/>
/// (index <c>p·(B − 1)</c>). With B = 101, τ*_b = b and τ**_b = b + shift
/// (b = 0..100), every expected value below is a small rational worked out in
/// the comment beside it.
/// </summary>
public sealed class FastDoubleBootstrapPaperOracleTests
{
    private const int OuterCount = 101;

    [Theory]
    // Right tail, inner shifted down by 20. τ̂ = 89.5 → p̂₁ = #{90..100}/101 = 11/101.
    // Q = (90/101)-quantile of τ** = 90/101·100 − 20 = 69.109 → p = #{70..100}/101 = 31/101.
    [InlineData(PValueTail.RightTail, 89.5, -20, 31)]
    // Right tail, τ̂ beyond every outer draw → p̂₁ = 0. Q = 1-quantile of τ** = max = 80
    // → p = #{81..100}/101 = 20/101. (The strongest rejection must not read as p ≈ 1.)
    [InlineData(PValueTail.RightTail, 1000.0, -20, 20)]
    // Left tail, inner shifted up by 20. τ̂ = 10.5 → p̂₁ = #{0..10}/101 = 11/101.
    // Q = (11/101)-quantile of τ** = 11/101·100 + 20 = 30.891 → p = #{0..30}/101 = 31/101.
    [InlineData(PValueTail.LeftTail, 10.5, 20, 31)]
    // Two-sided, inner shifted down by 20, τ̂ = 89.5.
    // Right FDB = 31/101 (row 1). Left: p̂₁ = #{0..89}/101 = 90/101, Q = 69.109,
    // p = #{0..69}/101 = 70/101. Two-sided = min(2·min(31, 70)/101, 1) = 62/101.
    [InlineData(PValueTail.TwoSided, 89.5, -20, 62)]
    public void PValue_MatchesHandComputedFastDoubleBootstrap(
        PValueTail tail, double observed, int shift, int expectedNumerator)
    {
        var engine = new FastDoubleBootstrap(OuterCount, new Pcg64RandomSource(7UL));
        var statistic = ScriptedStatistic(shift);

        var p = engine.PValue(new[] { 1m, 2m, 3m }, (decimal)observed, statistic, tail);

        // 1e-20 absorbs only the last-digit decimal rounding of 2·(31/101) vs 62/101;
        // the p-value grid spacing is 1/101.
        p.Should().BeApproximately(
            expectedNumerator / (decimal)OuterCount,
            1e-20m,
            "the FDB p-value must follow the Davidson-MacKinnon definition for the {0} tail",
            tail);
    }

    [Fact]
    public void PValue_TwoSided_EqualsFoldedOneSidedFastDoubleBootstrapValues()
    {
        // The two-sided value is the bilateral fold of the two one-sided FDB p-values,
        // each of which takes its own tail's quantile of τ**.
        var right = new FastDoubleBootstrap(OuterCount, new Pcg64RandomSource(3UL))
            .PValue(new[] { 1m }, 60.25m, ScriptedStatistic(-7), PValueTail.RightTail);
        var left = new FastDoubleBootstrap(OuterCount, new Pcg64RandomSource(3UL))
            .PValue(new[] { 1m }, 60.25m, ScriptedStatistic(-7), PValueTail.LeftTail);
        var two = new FastDoubleBootstrap(OuterCount, new Pcg64RandomSource(3UL))
            .PValue(new[] { 1m }, 60.25m, ScriptedStatistic(-7), PValueTail.TwoSided);

        two.Should().Be(Math.Min(2m * Math.Min(right, left), 1m));
    }

    [Theory]
    // Outer = [1,2,2,2,3,4], observed = 2. With inner = [0,1,2,3,4,5],
    // strict p* is right 2/6 or left 1/6. Repeat each pattern ten times (B=60). Type-7 Q is 3 or 5/6,
    // giving final right 1/6 and left 0. Inclusive p* picks different Q.
    [InlineData(PValueTail.RightTail, false, 1)]
    [InlineData(PValueTail.LeftTail, false, 0)]
    [InlineData(PValueTail.TwoSided, false, 0)]
    // With inner = [2,2,2,2,2,2], Q=2 for both tails, so the final
    // strict rank must also exclude the three tied outer statistics.
    [InlineData(PValueTail.RightTail, true, 2)]
    [InlineData(PValueTail.LeftTail, true, 1)]
    [InlineData(PValueTail.TwoSided, true, 2)]
    public void PValue_TiedStatistics_ExcludesEqualityAtBothRanks(
        PValueTail tail, bool tiedInner, int numerator)
    {
        decimal[] outer = [1m, 2m, 2m, 2m, 3m, 4m];
        decimal[] inner = tiedInner ? [2m, 2m, 2m, 2m, 2m, 2m] : [0m, 1m, 2m, 3m, 4m, 5m];
        foreach (var imposeNull in new[] { false, true })
        {
            var call = 0;
            decimal Statistic(ReadOnlySpan<decimal> sample)
            {
                var b = call / 2 % 6;
                return call++ % 2 == 0 ? outer[b] : inner[b];
            }

            var engine = new FastDoubleBootstrap(60, new Pcg64RandomSource(17UL));
            var actual = imposeNull
                ? engine.PValueImposingNull([1m], 2m, Statistic, static _ => { }, tail)
                : engine.PValue([1m], 2m, Statistic, tail);

            actual.Should().BeApproximately(numerator / 6m, 1e-25m);
            call.Should().Be(120);
        }
    }

    // Calls alternate outer, inner, outer, inner, ... (one inner draw per outer draw).
    // Outer call b returns b; inner call b returns b + shift.
    private static Func<ReadOnlySpan<decimal>, decimal> ScriptedStatistic(int shift)
    {
        var call = 0;
        return _ =>
        {
            var b = call / 2;
            var isInner = call % 2 == 1;
            call++;
            return isInner ? b + shift : b;
        };
    }
}
