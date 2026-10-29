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

namespace Boutquin.Numerics.Tests.Unit.Statistics;

/// <summary>
/// Pins for the GT-inspired Half score keeping its denominator terms (the documented default
/// epsilon 10⁻⁶ and the completed downside RMS) in double until the score itself is narrowed
/// once. The expected scores come from a 60-digit mpmath evaluation of the documented library
/// variant (sample SD, equity-curve R², zero-target downside RMS over all N, additive ε) on the
/// exact Half-rounded inputs; they are never taken from the production code.
/// </summary>
public sealed class GtInspiredScoreHalfNarrowingTests
{
    // Relative allowance: two binary16 units in the last place at the score's magnitude. The
    // defects under test shift the score by 1.3% to 15%, an order of magnitude beyond it.
    private const double RelativeTolerance = 2e-3;

    private static Half[] AlternatingSeries(int count, double? firstReturn = null)
    {
        var values = new Half[count];
        for (var i = 0; i < count; i++)
        {
            values[i] = i % 2 == 0 ? (Half)0.001 : (Half)0.0015;
        }

        if (firstReturn is { } first)
        {
            values[0] = (Half)first;
        }

        return values;
    }

    private static void ScoreShouldMatch(Half score, double oracle)
        => ((double)score).Should().BeApproximately(oracle, Math.Abs(oracle) * RelativeTolerance);

    // ---- F-L8-R2-LIT-01: the documented default epsilon 1e-6 must not be rounded to a Half subnormal ----

    [Fact]
    public void Compute_HalfDefaultEpsilon_NonNegativeSeries_UsesTheExactDocumentedEpsilon()
    {
        // Zero downside, so the denominator is epsilon alone: score ∝ 1/ε. Oracle (ε = 1e-6
        // exactly): -4562.0348065…; Half rounds the default to 1.0132789611816406e-6 (+1.33%).
        var result = GtInspiredScore<Half>.Compute(AlternatingSeries(60));

        ScoreShouldMatch(result.Score, -4562.034806507173);
    }

    [Fact]
    public void Compute_HalfDefaultEpsilonWithBenchmark_NonNegativeSeries_UsesTheExactDocumentedEpsilon()
    {
        var result = GtInspiredScore<Half>.Compute(AlternatingSeries(60), Half.Zero);

        ScoreShouldMatch(result.Score, -4562.034806507173);
    }

    [Fact]
    public void Compute_HalfExplicitEpsilon_IsHonouredAsGiven()
    {
        // A caller who passes (Half)1e-6 chose the Half value 1.0132789611816406e-6; the score
        // must follow that argument and not be replaced by the documented double default.
        // Oracle with ε = 1.0132789611816406e-6: -4502.2496087…
        var result = GtInspiredScore<Half>.Compute(AlternatingSeries(60), Half.Zero, (Half)1e-6);

        ScoreShouldMatch(result.Score, -4502.249608722885);
    }

    // ---- F-L8-R2-LIT-02: the completed downside RMS must feed the score quotient unrounded ----

    [Fact]
    public void Compute_HalfSubnormalDownsideRms_ScoreUsesTheUnroundedRms()
    {
        // One return of -12·2⁻²⁴ among positives: exact σ_d = 9.2339e-8, which Half rounds to
        // 1.1921e-7 (+29%). ε = Half.Epsilon is exactly representable, so only σ_d can differ.
        // Oracle: -28158.6357539…
        var result = GtInspiredScore<Half>.Compute(
            AlternatingSeries(60, -7.152557373046875e-7), Half.Zero, Half.Epsilon);

        ScoreShouldMatch(result.Score, -28158.635753949236);
    }

    [Fact]
    public void Compute_HalfSubnormalDownsideRms_ReportedDeviationIsStillRoundedOnce()
    {
        // The returned field is the completed output: exact 9.2339e-8 rounded once to Half.
        var result = GtInspiredScore<Half>.Compute(
            AlternatingSeries(60, -7.152557373046875e-7), Half.Zero, Half.Epsilon);

        ((double)result.DownsideDeviation).Should().Be(1.1920928955078125e-07);
    }

    [Fact]
    public void Compute_HalfSecondSubnormalDownsideRms_ScoreUsesTheUnroundedRms()
    {
        // σ_d = 1.29275e-6 rounds to 1.31130e-6 (+1.44%). Oracle: -3161.2523223…
        var result = GtInspiredScore<Half>.Compute(
            AlternatingSeries(60, -1.0013580322265625e-5), Half.Zero, Half.Epsilon);

        ScoreShouldMatch(result.Score, -3161.2523222756537);
    }

    // ---- Both denominator terms together on the default overload ----

    [Fact]
    public void Compute_HalfDefaultOverload_SubnormalDownsideRms_UsesBothUnroundedTerms()
    {
        // Default ε and σ_d = 9.2339e-8: oracle -3916.8505602…
        var result = GtInspiredScore<Half>.Compute(AlternatingSeries(60, -7.152557373046875e-7));

        ScoreShouldMatch(result.Score, -3916.850560246063);
    }

    [Fact]
    public void Compute_HalfDefaultOverload_SecondSubnormalDownsideRms_UsesBothUnroundedTerms()
    {
        // Default ε and σ_d = 1.29275e-6: oracle -1864.6303470…
        var result = GtInspiredScore<Half>.Compute(AlternatingSeries(60, -1.0013580322265625e-5), Half.Zero);

        ScoreShouldMatch(result.Score, -1864.6303470855061);
    }
}
