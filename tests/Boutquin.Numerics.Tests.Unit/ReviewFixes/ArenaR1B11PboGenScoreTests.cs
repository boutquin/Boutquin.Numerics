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

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Regression pins for the CSCV Probability of Backtest Overfitting (Bailey,
/// Borwein, López de Prado &amp; Zhu 2014) and the GT-inspired library variant.
/// </summary>
public sealed class ArenaR1B11PboGenScoreTests
{
    /// <summary>
    /// Deterministic 32-bit LCG panel, identical to the generator used by the
    /// independent Python reference that produced the expected values below.
    /// </summary>
    private static double[,] LcgPanel(int strategies, int observations, uint seed)
    {
        var panel = new double[strategies, observations];
        var state = seed;
        for (var i = 0; i < strategies; i++)
        {
            for (var t = 0; t < observations; t++)
            {
                state = (1664525u * state) + 1013904223u;
                panel[i, t] = (((state & 0xFFFFFFu) / (double)0x1000000) - 0.5) * 0.04;
            }
        }

        return panel;
    }

    /// <summary>
    /// An exact duplicate of the in-sample winner ties it out-of-sample. Ranking
    /// the winner at the bottom of its tie group turned a panel whose IS winner is
    /// the OOS best in every fold into PBO = 1. With the tied pair sharing the
    /// average rank (2 + 3) / 2 = 2.5 of N = 3, ω = 2.5 / 4 and every fold's logit
    /// is ln(0.625 / 0.375) = ln(5/3) &gt; 0, so PBO is 0 — exactly as without the copy.
    /// </summary>
    [Fact]
    public void Pbo_WinnerWithExactOutOfSampleTwin_TakesAverageRank()
    {
        var noise = LcgPanel(2, 32, seed: 7);
        var panel = new double[3, 32];
        for (var t = 0; t < 32; t++)
        {
            panel[0, t] = noise[0, t] + 0.01;
            panel[1, t] = noise[0, t] + 0.01;
            panel[2, t] = noise[1, t] - 0.01;
        }

        var result = ProbabilityOfBacktestOverfitting<double>.Compute(panel, splitCount: 4);

        result.Pbo.Should().Be(0.0, "the IS winner and its twin are the OOS best in every fold");
        result.LogitValues.Should().AllSatisfy(l => l.Should().BeApproximately(Math.Log(5.0 / 3.0), 1e-12));
    }

    /// <summary>
    /// Bailey et al. (2014) §2 step (a) joins the S/2 training blocks into one
    /// T/2 × N matrix and computes the Sharpe ratio on each joined column; step (d)
    /// does the same for the complementary test set. The mean of per-block Sharpe
    /// ratios is a different statistic. On this panel the joined method gives
    /// PBO = 0.5 (three of six logits ≤ 0) while the per-block mean gives 1.0.
    /// Expected values come from a Python reference written from the paper
    /// (numpy concatenation of the selected blocks, sample sd, average-rank ties).
    /// </summary>
    [Fact]
    public void Pbo_SharpeIsComputedOnJoinedBlocks()
    {
        var panel = LcgPanel(6, 24, seed: 20);

        var result = ProbabilityOfBacktestOverfitting<double>.Compute(panel, splitCount: 4);

        result.Pbo.Should().Be(0.5);
        result.LogitMedian.Should().BeApproximately(-0.31430432971118716, 1e-12);
        result.LogitValues.Should().Equal(
            new[]
            {
                -1.791759469228055, -0.9162907318741551, -0.9162907318741551,
                0.28768207245178085, 0.28768207245178085, 0.28768207245178085,
            },
            (a, e) => Math.Abs(a - e) <= 1e-12);
    }

    /// <summary>
    /// A constant strategy has zero standard deviation and, by the estimator's
    /// convention, a Sharpe ratio of zero. Twelve joined copies of 0.01 do not sum
    /// to exactly 0.12, so a mean formed by summation leaves a 1e-18 residue that
    /// became a 1e16 Sharpe and made the constant series the winner of every fold.
    /// Expected values come from the paper-derived Python reference, where an
    /// all-equal column has Sharpe 0.
    /// </summary>
    [Fact]
    public void Pbo_ConstantStrategy_HasZeroSharpeRegardlessOfRounding()
    {
        var noise = LcgPanel(2, 24, seed: 3);
        var panel = new double[3, 24];
        for (var t = 0; t < 24; t++)
        {
            panel[0, t] = 0.01;
            panel[1, t] = noise[0, t] + 0.002;
            panel[2, t] = noise[1, t] + 0.001;
        }

        var result = ProbabilityOfBacktestOverfitting<double>.Compute(panel, splitCount: 4);

        result.Pbo.Should().BeApproximately(2.0 / 3.0, 1e-15);
        result.LogitMedian.Should().BeApproximately(-Math.Log(3.0), 1e-12);
    }

    /// <summary>
    /// A constant series has zero standard error, so Sheppert (2026) eq. (2)
    /// gives no z; the estimator takes z = 0, which selects Algorithm 1's
    /// penalty branch 100 + 100·(1 − e^(−1)). The result must not depend on
    /// whether 1/N is exactly representable: summing N inexact 1/N terms left a
    /// rounding residue that produced z ≈ 1e16 and a score near −4e5.
    /// </summary>
    [Theory]
    [InlineData(0.01, 3)]
    [InlineData(0.01, 4)]
    [InlineData(0.01, 10)]
    [InlineData(0.01, 100)]
    [InlineData(0.003, 7)]
    [InlineData(-0.02, 10)]
    public void GtScore_ConstantSeries_TakesZeroZ(double value, int count)
    {
        var returns = Enumerable.Repeat(value, count).ToArray();

        var result = GtInspiredScore<double>.Compute(returns, 0, 1e-6, minimumObservations: 1);

        result.Z.Should().Be(0.0);
        result.Mu.Should().Be(value, "the mean of a constant series is that constant");
        result.Score.Should().BeApproximately(100.0 + (100.0 * (1.0 - Math.Exp(-1.0))), 1e-12);
    }
}
