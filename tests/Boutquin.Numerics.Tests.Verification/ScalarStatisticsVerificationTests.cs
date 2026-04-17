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
using Boutquin.Numerics.Statistics;

namespace Boutquin.Numerics.Tests.Verification;

/// <summary>
/// Python cross-checks for the five scalar-statistic public types
/// (NeweyWest HAC variance, Probability of Backtest Overfitting,
/// Generalization Score, Harrell-Davis percentile, P-Square streaming
/// percentile). See <c>tests/Verification/generate_scalar_stats_vectors.py</c>
/// for the reference generators — each is a numpy / scipy port of the
/// published paper formula; no library calls are mocked.
/// </summary>
public sealed class ScalarStatisticsVerificationTests : CrossLanguageVerificationBase
{
    /// <summary>
    /// Bartlett-kernel HAC variance on an artificially AR(1)-contaminated
    /// series. Python reference is a hand-port of the C# formula (γ₀, γ_ℓ,
    /// Bartlett weight, 1/T normalisation) so both sides compute identical
    /// arithmetic; tolerance 1e-10 absolute is the spec's §2.3 bar.
    /// </summary>
    [Fact]
    public void NeweyWestVariance_MatchesReference()
    {
        using var doc = LoadVector("scalar_stats");
        var nw = doc.RootElement.GetProperty("newey_west");
        var series = GetDecimalArray(nw.GetProperty("series"));
        var lags = nw.GetProperty("lags").GetInt32();
        var expected = nw.GetProperty("variance").GetDouble();

        var actual = (double)NeweyWestVariance.MeanVariance(series, lags);

        Assert.True(
            Math.Abs(actual - expected) <= 1e-10,
            $"NeweyWest: expected {expected}, got {actual}, diff={Math.Abs(actual - expected)}");
    }

    /// <summary>
    /// Combinatorially-Symmetric Cross-Validation PBO on a 10-strategy ×
    /// 128-observation synthetic panel, exhaustively enumerating the
    /// C(16, 8) = 12,870 folds. Python reference is a direct port of
    /// Bailey et al. 2014; the stable sort for OOS rank breaks ties the
    /// same way the C# <c>OrderBy</c> does, so the logit sequence is
    /// bit-identical across implementations.
    /// </summary>
    [Fact]
    public void Pbo_MatchesReference()
    {
        using var doc = LoadVector("scalar_stats");
        var pbo = doc.RootElement.GetProperty("pbo");
        var returns = GetDecimal2D(pbo.GetProperty("returns"));
        var expectedPbo = pbo.GetProperty("pbo").GetDouble();
        var expectedMedian = pbo.GetProperty("logit_median").GetDouble();
        var splitCount = pbo.GetProperty("split_count").GetInt32();

        var result = ProbabilityOfBacktestOverfitting.Compute(returns, splitCount);

        Assert.True(
            Math.Abs((double)result.Pbo - expectedPbo) <= 1e-8,
            $"PBO: expected {expectedPbo}, got {result.Pbo}");
        Assert.True(
            Math.Abs((double)result.LogitMedian - expectedMedian) <= 1e-8,
            $"PBO logit median: expected {expectedMedian}, got {result.LogitMedian}");
    }

    /// <summary>
    /// GT-Score on a 504-observation synthetic daily return series.
    /// Verifies all four components (performance / significance /
    /// consistency / downside) plus the composite score at 1e-10 absolute.
    /// </summary>
    [Fact]
    public void GeneralizationScore_MatchesReference()
    {
        using var doc = LoadVector("scalar_stats");
        var gs = doc.RootElement.GetProperty("generalization_score");
        var returnsDecimal = GetDoubleArray(gs.GetProperty("returns"));

        var result = GeneralizationScore.Compute(returnsDecimal);

        const double Tolerance = 1e-10;
        AssertScalarWithin(result.Score, gs.GetProperty("score").GetDouble(), Tolerance, "score");
        AssertScalarWithin(result.PerformanceComponent, gs.GetProperty("performance_component").GetDouble(), Tolerance, "performance");
        AssertScalarWithin(result.SignificanceComponent, gs.GetProperty("significance_component").GetDouble(), Tolerance, "significance");
        AssertScalarWithin(result.ConsistencyComponent, gs.GetProperty("consistency_component").GetDouble(), Tolerance, "consistency");
        AssertScalarWithin(result.DownsideRiskComponent, gs.GetProperty("downside_risk_component").GetDouble(), Tolerance, "downside");
    }

    /// <summary>
    /// Harrell-Davis percentile at 5 quantiles (10/25/50/75/90). Python
    /// reference uses <c>scipy.special.betainc</c>; the C# implementation
    /// uses a Numerical-Recipes continued-fraction expansion. Both target
    /// double-precision incomplete-beta agreement — the tolerance floor
    /// at 1e-8 relative absorbs the small residual between the two
    /// numerical routines, which is far below the Harrell-Davis estimator's
    /// own sampling variance.
    /// </summary>
    [Theory]
    [InlineData("0.1")]
    [InlineData("0.25")]
    [InlineData("0.5")]
    [InlineData("0.75")]
    [InlineData("0.9")]
    public void HarrellDavisPercentile_MatchesReference(string percentileKey)
    {
        using var doc = LoadVector("scalar_stats");
        var hd = doc.RootElement.GetProperty("harrell_davis");
        var sorted = GetDecimalArray(hd.GetProperty("sorted_sample"));
        var percentile = double.Parse(percentileKey, System.Globalization.CultureInfo.InvariantCulture);
        var expected = hd.GetProperty("percentile_values").GetProperty(percentileKey).GetDouble();

        var actual = (double)HarrellDavisPercentile.Compute(sorted, percentile);

        // Max(|expected|, 1)-relative tolerance; 1e-8 is achievable because
        // both routines target full double precision on the regularised
        // incomplete beta and weighted-sum accumulation is in decimal.
        var denom = Math.Max(Math.Abs(expected), 1.0);
        var relativeError = Math.Abs(actual - expected) / denom;
        Assert.True(
            relativeError <= 1e-8,
            $"HarrellDavis p={percentileKey}: expected {expected}, got {actual}, relErr={relativeError}");
    }

    /// <summary>
    /// P² streaming percentile over a 1000-sample stream at three
    /// percentiles (0.5, 0.9, 0.99). The estimator is deterministic given
    /// the input sequence, so both the Python port and the C# class should
    /// produce bit-identical marker state after each observation; tolerance
    /// 1e-10 absolute verifies the state transitions match.
    /// </summary>
    [Theory]
    [InlineData("0.5")]
    [InlineData("0.9")]
    [InlineData("0.99")]
    public void PSquareEstimator_MatchesReference(string percentileKey)
    {
        using var doc = LoadVector("scalar_stats");
        var ps = doc.RootElement.GetProperty("psquare");
        var stream = GetDoubleArray(ps.GetProperty("stream"));
        var percentile = double.Parse(percentileKey, System.Globalization.CultureInfo.InvariantCulture);
        var expected = ps.GetProperty("estimates").GetProperty(percentileKey).GetDouble();

        var estimator = new PSquareEstimator(percentile);
        foreach (var obs in stream)
        {
            estimator.Add(obs);
        }

        AssertScalarWithin(estimator.Estimate, expected, 1e-10, $"PSquare p={percentileKey}");
    }
}
