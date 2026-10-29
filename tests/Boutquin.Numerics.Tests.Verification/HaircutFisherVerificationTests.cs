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

namespace Boutquin.Numerics.Tests.Verification;

/// <summary>
/// Python cross-checks for <see cref="HaircutSharpe{T}"/> and
/// <see cref="FisherZTransform{T}"/>. Reference values come from
/// <c>tests/Verification/generate_haircut_fisher_vectors.py</c>. The haircut
/// vectors are derived from Harvey &amp; Liu (2015) "Backtesting" equations 1–5
/// with <c>scipy.stats.t</c> (two-sided Student t on T − 1 degrees of freedom,
/// Sharpe and T at the same sampling frequency, Holm/BHY evaluated as the
/// paper's step-down/step-up recursions) — not from the C# arithmetic — and
/// include the paper's own worked example as an independent oracle.
/// </summary>
public sealed class HaircutFisherVerificationTests : CrossLanguageVerificationBase
{
    // ---------------------------------------------------------------
    //  HaircutSharpe — seven cases: single trial, moderate, many trials,
    //  saturated p-value, negative Sharpe, tiny p-value, and the
    //  Harvey-Liu (2015) worked example.
    // ---------------------------------------------------------------

    /// <summary>
    /// Relative tolerance on the adjusted p-values: the C# Student-t tail
    /// (incomplete beta by continued fraction) against scipy.stats.t.sf.
    /// </summary>
    private const double PValueRelativeTolerance = 1e-9;

    /// <summary>
    /// Absolute tolerance on the haircut Sharpe: the C# inverse Student-t
    /// (bisection) against scipy.stats.t.isf, scaled by sqrt(periods / T).
    /// </summary>
    private const double HaircutSharpeTolerance = 1e-9;

    /// <summary>Single trial: p_M = p_S, so there is no haircut.</summary>
    [Fact]
    public void HaircutSharpe_SingleTrial_MatchesReference() => AssertCaseMatches("case_a_single_trial");

    /// <summary>Ten trials over five years of daily data; BHY is not saturated.</summary>
    [Fact]
    public void HaircutSharpe_ModerateTrials_MatchesReference() => AssertCaseMatches("case_b_moderate_trials");

    /// <summary>One hundred trials over three years of daily data.</summary>
    [Fact]
    public void HaircutSharpe_ManyTrials_MatchesReference() => AssertCaseMatches("case_c_many_trials");

    /// <summary>
    /// Low Sharpe (0.5), 1,000 trials: the Bonferroni p-value saturates at 1,
    /// so the haircut Sharpe is 0 and the whole observed value is removed.
    /// </summary>
    [Fact]
    public void HaircutSharpe_LowSharpeThousandTrials_MatchesReference()
    {
        var result = AssertCaseMatches("case_d_low_sharpe_many_trials");
        result.HaircutAmount.Should().BeGreaterThanOrEqualTo(0.0, "a positive Sharpe is never haircut upward");
    }

    /// <summary>
    /// Negative observed Sharpe: the paper's test is two-sided, so the sign is
    /// irrelevant to significance and the haircut shrinks the magnitude toward
    /// zero (here all the way, because p_M saturates at 1).
    /// </summary>
    [Fact]
    public void HaircutSharpe_NegativeSharpe_ShrinksTowardZero()
    {
        var result = AssertCaseMatches("case_e_negative_sharpe");
        result.BhyPValue.Should().BeApproximately(1.0, 1e-12, "p-value saturates at 1 for a weak signal among 50 trials");
    }

    /// <summary>
    /// Very high Sharpe (3.0), ten years of daily data, 500 trials: p_S ≈ 5e-21
    /// is still representable, so the haircut is small but non-zero.
    /// </summary>
    [Fact]
    public void HaircutSharpe_HighSharpe_MatchesReference() => AssertCaseMatches("case_f_high_sharpe");

    /// <summary>
    /// Harvey &amp; Liu (2015) worked example: twenty years of monthly returns
    /// (T = 240), annual Sharpe 0.75, 200 trials. The vector is the paper's
    /// equations evaluated exactly (p_S = 0.000926, p_M = 0.185, HSR = 0.297);
    /// the paper prints 0.0008 / 0.15 / 0.32. The residual gap is the
    /// normal approximation the printed figures follow (normal two-sided
    /// p_S = 0.00080); the bands below admit that gap and nothing wider.
    /// </summary>
    [Fact]
    public void HaircutSharpe_HarveyLiuWorkedExample_MatchesPaper()
    {
        var result = AssertCaseMatches("case_g_harvey_liu_example");

        (result.BonferroniPValue / 200.0).Should().BeApproximately(0.0008, 2e-4, "paper: single-test p-value 0.0008");
        result.BonferroniPValue.Should().BeApproximately(0.15, 0.04, "paper: p_M = 0.15 for N = 200");
        result.HaircutSharpe.Should().BeApproximately(0.32, 0.03, "paper: HSR = 0.32");
    }

    private static HaircutSharpeResult<double> AssertCaseMatches(string caseName)
    {
        using var doc = LoadVector("haircut_fisher");
        var c = doc.RootElement.GetProperty("haircut_sharpe").GetProperty(caseName);

        var result = HaircutSharpe<double>.Compute(
            c.GetProperty("observed_sharpe").GetDouble(),
            c.GetProperty("num_trials").GetInt32(),
            c.GetProperty("backtest_years").GetDouble(),
            c.GetProperty("trading_days_per_year").GetInt32());

        AssertRelative(result.BonferroniPValue, c.GetProperty("p_bonferroni").GetDouble(), $"Bonferroni p ({caseName})");
        AssertRelative(result.HolmPValue, c.GetProperty("p_holm").GetDouble(), $"Holm p ({caseName})");
        AssertRelative(result.BhyPValue, c.GetProperty("p_bhy").GetDouble(), $"BHY p ({caseName})");
        AssertScalarWithin(result.HaircutSharpe, c.GetProperty("haircut_sharpe").GetDouble(), HaircutSharpeTolerance, $"haircut SR ({caseName})");
        AssertScalarWithin(result.HaircutAmount, c.GetProperty("haircut_amount").GetDouble(), HaircutSharpeTolerance, $"haircut amount ({caseName})");
        return result;
    }

    private static void AssertRelative(double actual, double expected, string label)
    {
        var relative = Math.Abs(actual - expected) / Math.Max(Math.Abs(expected), double.Epsilon);
        Assert.True(
            relative <= PValueRelativeTolerance,
            $"{label}: expected {expected:R}, got {actual:R}, relative diff={relative:E3} > tol={PValueRelativeTolerance}");
    }

    // ---------------------------------------------------------------
    //  FisherZTransform — Forward / Inverse / ConfidenceInterval
    // ---------------------------------------------------------------

    /// <summary>
    /// Forward transform z = atanh(r) at six r values from −0.9 to 0.95.
    /// Both Python (numpy.arctanh via math.log) and C# use the closed-form
    /// 0.5·ln((1+r)/(1−r)); tolerance 1e-10 is achievable because no
    /// approximation is involved — only standard IEEE 754 log.
    /// </summary>
    [Fact]
    public void FisherZTransform_Forward_MatchesReference()
    {
        using var doc = LoadVector("haircut_fisher");
        var cases = doc.RootElement.GetProperty("fisher_z").GetProperty("forward_cases");

        foreach (var item in cases.EnumerateArray())
        {
            var r = item.GetProperty("r").GetDouble();
            var expectedZ = item.GetProperty("z").GetDouble();

            var z = FisherZTransform<double>.Forward(r);
            AssertScalarWithin(z, expectedZ, PrecisionExact, $"Forward(r={r})");
        }
    }

    /// <summary>
    /// Inverse transform r = tanh(z) at five z values from −2 to 2.
    /// Tolerance 1e-10; both sides call tanh directly with no approximation.
    /// </summary>
    [Fact]
    public void FisherZTransform_Inverse_MatchesReference()
    {
        using var doc = LoadVector("haircut_fisher");
        var cases = doc.RootElement.GetProperty("fisher_z").GetProperty("inverse_cases");

        foreach (var item in cases.EnumerateArray())
        {
            var z = item.GetProperty("z").GetDouble();
            var expectedR = item.GetProperty("r").GetDouble();

            var r = FisherZTransform<double>.Inverse(z);
            AssertScalarWithin(r, expectedR, PrecisionExact, $"Inverse(z={z})");
        }
    }

    /// <summary>
    /// Forward–Inverse roundtrip: Forward(r) → Inverse → r. Verifies
    /// that tanh(atanh(r)) ≈ r within machine epsilon at all six test points.
    /// </summary>
    [Fact]
    public void FisherZTransform_ForwardInverseRoundtrip_WithinMachineEpsilon()
    {
        using var doc = LoadVector("haircut_fisher");
        var cases = doc.RootElement.GetProperty("fisher_z").GetProperty("forward_cases");

        foreach (var item in cases.EnumerateArray())
        {
            var r = item.GetProperty("r").GetDouble();
            var roundtrip = FisherZTransform<double>.Inverse(FisherZTransform<double>.Forward(r));
            AssertScalarWithin(roundtrip, r, PrecisionExact, $"Roundtrip Forward→Inverse(r={r})");
        }
    }

    /// <summary>
    /// Inverse–Forward roundtrip: Inverse(z) → Forward → z. Verifies
    /// that atanh(tanh(z)) ≈ z within machine epsilon at five test points.
    /// </summary>
    [Fact]
    public void FisherZTransform_InverseForwardRoundtrip_WithinMachineEpsilon()
    {
        using var doc = LoadVector("haircut_fisher");
        var cases = doc.RootElement.GetProperty("fisher_z").GetProperty("inverse_cases");

        foreach (var item in cases.EnumerateArray())
        {
            var z = item.GetProperty("z").GetDouble();
            var roundtrip = FisherZTransform<double>.Forward(FisherZTransform<double>.Inverse(z));
            AssertScalarWithin(roundtrip, z, PrecisionExact, $"Roundtrip Inverse→Forward(z={z})");
        }
    }

    /// <summary>
    /// Two-sided confidence intervals for six (r, n, confidence_level)
    /// combinations. Lower and upper bounds match the scipy reference to
    /// within 1e-8 — the tighter than PrecisionNumeric (1e-6) floor is
    /// achievable because both sides use the same atanh/tanh path and the
    /// InverseCdf precision is the only source of divergence.
    /// </summary>
    [Fact]
    public void FisherZTransform_ConfidenceInterval_MatchesReference()
    {
        const double CiTolerance = 1e-8;

        using var doc = LoadVector("haircut_fisher");
        var cases = doc.RootElement.GetProperty("fisher_z").GetProperty("ci_cases");

        foreach (var item in cases.EnumerateArray())
        {
            var r = item.GetProperty("r").GetDouble();
            var n = item.GetProperty("n").GetInt32();
            var cl = item.GetProperty("confidence_level").GetDouble();
            var expectedLo = item.GetProperty("lower").GetDouble();
            var expectedHi = item.GetProperty("upper").GetDouble();

            var (lo, hi) = FisherZTransform<double>.ConfidenceInterval(r, n, cl);

            AssertScalarWithin(lo, expectedLo, CiTolerance, $"CI lower (r={r}, n={n}, cl={cl})");
            AssertScalarWithin(hi, expectedHi, CiTolerance, $"CI upper (r={r}, n={n}, cl={cl})");
        }
    }

    /// <summary>
    /// For r = 0 the Fisher CI must be exactly symmetric around 0:
    /// lower = −upper. Python and C# both produce this by construction
    /// (z = 0, lo = −hi in z-domain, tanh is odd). Tolerance 1e-15.
    /// </summary>
    [Fact]
    public void FisherZTransform_CiAtZeroCorrelation_IsSymmetric()
    {
        using var doc = LoadVector("haircut_fisher");
        var cases = doc.RootElement.GetProperty("fisher_z").GetProperty("ci_cases");

        // Fifth entry: r = 0, n = 20, cl = 0.95.
        var zeroCase = cases.EnumerateArray().Single(e => e.GetProperty("r").GetDouble() == 0.0);
        var n = zeroCase.GetProperty("n").GetInt32();
        var cl = zeroCase.GetProperty("confidence_level").GetDouble();

        var (lo, hi) = FisherZTransform<double>.ConfidenceInterval(0.0, n, cl);

        AssertScalarWithin(lo + hi, 0.0, 1e-15, "CI symmetry at r=0: lower + upper = 0");
    }
}
