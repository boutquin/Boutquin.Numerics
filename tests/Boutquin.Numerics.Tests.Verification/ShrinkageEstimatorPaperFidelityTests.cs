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

namespace Boutquin.Numerics.Tests.Verification;

/// <summary>
/// Paper-fidelity checks for the Ledoit-Wolf linear shrinkage family and QIS.
/// The goldens in <c>covariance.json</c> come from independent sources, not from
/// ports of the C# classes: <c>sklearn.covariance.ledoit_wolf</c> (JMVA 2004 eq. 14),
/// numpy ports of the authors' published <c>covCor.m</c> (Honey 2004 App. B) and
/// <c>covMarket.m</c> (LW 2003 Theorem 2), and the LW 2022 Appendix A QIS code.
/// Every regime is compared with a scale-relative bound so the small-unit regime
/// is held to the same standard as the unit-scale ones.
/// </summary>
public sealed class ShrinkageEstimatorPaperFidelityTests : CrossLanguageVerificationBase
{
    // The linear-shrinkage intensities are arithmetic in T; decimal against the
    // double oracle agrees to ~1e-13. A formula error moves entries by percent.
    private const decimal LinearShrinkageTolerance = 1e-9m;

    // QIS goes through an eigendecomposition (Jacobi here, LAPACK in the oracle).
    private const decimal QisTolerance = 1e-6m;

    public static TheoryData<string> Regimes => new()
    {
        "small_well_conditioned",
        "small_scale_oas",
        "moderate",
        "overconcentrated",
    };

    [Theory]
    [MemberData(nameof(Regimes))]
    public void LedoitWolfScaledIdentity_MatchesJmvaEquation14(string regime)
    {
        AssertRegime(regime, "ledoit_wolf_scaled_identity", new LedoitWolfShrinkageEstimator<decimal>(), LinearShrinkageTolerance);
    }

    [Theory]
    [MemberData(nameof(Regimes))]
    public void LedoitWolfConstantCorrelation_MatchesHoneyAppendixB(string regime)
    {
        AssertRegime(regime, "lw_constant_correlation", new LedoitWolfConstantCorrelationEstimator<decimal>(), LinearShrinkageTolerance);
    }

    [Theory]
    [MemberData(nameof(Regimes))]
    public void LedoitWolfSingleFactor_MatchesLw2003Theorem2(string regime)
    {
        AssertRegime(regime, "lw_single_factor", new LedoitWolfSingleFactorEstimator<decimal>(), LinearShrinkageTolerance);
    }

    [Theory]
    [MemberData(nameof(Regimes))]
    public void QuadraticInverseShrinkage_MatchesLw2022AppendixA(string regime)
    {
        AssertRegime(regime, "qis", new QuadraticInverseShrinkageEstimator<decimal>(), QisTolerance);
    }

    private static void AssertRegime(string regime, string key, ICovarianceEstimator<decimal> estimator, decimal tolerance)
    {
        using var doc = LoadVector("covariance");
        var block = doc.RootElement.GetProperty("regimes").GetProperty(regime);
        var returns = GetDecimal2D(block.GetProperty("returns"));
        var expected = GetDouble2D(block.GetProperty(key));

        var actual = estimator.Estimate(returns);

        Assert.Equal(expected.GetLength(0), actual.GetLength(0));
        Assert.Equal(expected.GetLength(1), actual.GetLength(1));

        var scale = 0m;
        foreach (var value in expected)
        {
            scale = Math.Max(scale, Math.Abs((decimal)value));
        }

        var worst = 0m;
        var worstI = 0;
        var worstJ = 0;
        for (var i = 0; i < expected.GetLength(0); i++)
        {
            for (var j = 0; j < expected.GetLength(1); j++)
            {
                var error = Math.Abs(actual[i, j] - (decimal)expected[i, j]) / scale;
                if (error > worst)
                {
                    worst = error;
                    worstI = i;
                    worstJ = j;
                }
            }
        }

        Assert.True(
            worst <= tolerance,
            $"{key}/{regime}: scale-relative error {worst:E3} at [{worstI},{worstJ}] exceeds {tolerance:E0} " +
            $"(actual {actual[worstI, worstJ]:E10}, expected {expected[worstI, worstJ]:E10})");
    }
}
