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

using System.Numerics;

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Result of a GT-Score computation, decomposed into its four components.
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <param name="Score">Composite GT-Score value.</param>
/// <param name="PerformanceComponent">Annualized Sharpe ratio contribution (weighted).</param>
/// <param name="SignificanceComponent">t-statistic contribution (weighted).</param>
/// <param name="ConsistencyComponent">Fraction of positive sub-periods (weighted).</param>
/// <param name="DownsideRiskComponent">Maximum drawdown penalty (weighted, subtracted).</param>
public sealed record GtScoreResult<T>(
    T Score,
    T PerformanceComponent,
    T SignificanceComponent,
    T ConsistencyComponent,
    T DownsideRiskComponent)
    where T : IFloatingPointIeee754<T>;

/// <summary>
/// Generalization Threshold Score (GT-Score): a composite objective combining
/// annualized performance, statistical significance, sub-period consistency,
/// and a drawdown penalty, designed to embed anti-overfitting directly into
/// the fitness function rather than deflating a post-hoc Sharpe. Reference:
/// Sheppert (2026), arXiv:2602.00080.
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <remarks>
/// <para>
/// Complementary to <see cref="DeflatedSharpeRatio{T}"/> and
/// <see cref="ProbabilityOfBacktestOverfitting{T}"/>: those are <em>diagnostic</em>
/// metrics applied after search; GT-Score is a <em>search</em> objective chosen
/// so that optimizers chase generalizable performance in the first place.
/// The reference reports ~98% improvement in the generalization ratio
/// (OOS Sharpe / IS Sharpe) vs. optimizing Sharpe alone on walk-forward
/// validation benchmarks.
/// </para>
/// <para>
/// Composite form:
/// <c>GT = w₁ · Performance + w₂ · Significance + w₃ · Consistency − w₄ · DownsideRisk</c>,
/// with component semantics:
/// <list type="bullet">
///   <item><c>Performance</c> — annualized Sharpe ratio.</item>
///   <item><c>Significance</c> — t-statistic normalized by <c>√T</c>, so its scale is commensurate with Performance rather than diverging with sample length.</item>
///   <item><c>Consistency</c> — fraction of equally-sized sub-periods with positive cumulative return, in <c>[0, 1]</c>.</item>
///   <item><c>DownsideRisk</c> — maximum peak-to-trough drawdown, subtracted.</item>
/// </list>
/// </para>
/// <para>
/// Default weights <c>(0.3, 0.3, 0.2, 0.2)</c> are from the reference's
/// ablation study, balancing raw performance (Sharpe) with statistical
/// confidence (t-stat) at equal weight, with structural sanity checks
/// (consistency, drawdown) at two-thirds weight each. The weights are
/// exposed as parameters; practitioners with asymmetric loss preferences
/// should override rather than renormalize post-hoc.
/// </para>
/// <para>
/// Tier B: Fully transcendental computation.
/// </para>
/// </remarks>
public static class GeneralizationScore<T>
    where T : IFloatingPointIeee754<T>
{
    /// <summary>
    /// Computes the GT-Score for a daily return series.
    /// </summary>
    /// <param name="returns">Daily return observations (at least 2 required).</param>
    /// <param name="subPeriodCount">Number of equal sub-periods for consistency calculation.</param>
    /// <param name="tradingDaysPerYear">Trading days per year for annualization.</param>
    /// <param name="performanceWeight">Weight for annualized Sharpe component.</param>
    /// <param name="significanceWeight">Weight for t-statistic component.</param>
    /// <param name="consistencyWeight">Weight for sub-period consistency.</param>
    /// <param name="downsideWeight">Weight for maximum drawdown penalty.</param>
    /// <returns>Decomposed GT-Score result.</returns>
    /// <exception cref="ArgumentException">Fewer than 2 returns provided.</exception>
    public static GtScoreResult<T> Compute(
        ReadOnlySpan<T> returns,
        int subPeriodCount,
        T tradingDaysPerYear,
        T performanceWeight,
        T significanceWeight,
        T consistencyWeight,
        T downsideWeight)
    {
        if (returns.Length < 2)
        {
            throw new ArgumentException("At least 2 return observations are required.", nameof(returns));
        }

        var (mean, stdDev) = ComputeMeanAndStdDev(returns);
        var sharpe = stdDev > T.Zero ? mean / stdDev * T.Sqrt(tradingDaysPerYear) : T.Zero;
        var performanceComponent = performanceWeight * sharpe;

        var tStat = stdDev > T.Zero ? mean / (stdDev / T.Sqrt(T.CreateChecked(returns.Length))) : T.Zero;
        var normalizedTStat = T.Max(T.Zero, tStat) / T.Sqrt(T.CreateChecked(returns.Length));
        var annualizedTStat = normalizedTStat * T.Sqrt(tradingDaysPerYear);
        var significanceComponent = significanceWeight * annualizedTStat;

        var consistency = ComputeConsistency(returns, subPeriodCount);
        var consistencyComponent = consistencyWeight * consistency;

        var maxDrawdown = ComputeMaxDrawdown(returns);
        var downsideComponent = downsideWeight * maxDrawdown;

        var score = performanceComponent + significanceComponent + consistencyComponent - downsideComponent;

        return new GtScoreResult<T>(score, performanceComponent, significanceComponent,
            consistencyComponent, downsideComponent);
    }

    private static (T Mean, T StdDev) ComputeMeanAndStdDev(ReadOnlySpan<T> values)
    {
        var sum = T.Zero;
        for (var i = 0; i < values.Length; i++)
        {
            sum += values[i];
        }

        var mean = sum / T.CreateChecked(values.Length);

        var sumSq = T.Zero;
        for (var i = 0; i < values.Length; i++)
        {
            var diff = values[i] - mean;
            sumSq += diff * diff;
        }

        var stdDev = T.Sqrt(sumSq / T.CreateChecked(values.Length - 1));
        return (mean, stdDev);
    }

    private static T ComputeConsistency(ReadOnlySpan<T> returns, int subPeriodCount)
    {
        if (subPeriodCount <= 0 || returns.Length < subPeriodCount)
        {
            var totalSum = T.Zero;
            for (var i = 0; i < returns.Length; i++)
            {
                totalSum += returns[i];
            }

            return totalSum > T.Zero ? T.One : T.Zero;
        }

        var periodSize = returns.Length / subPeriodCount;
        var positiveCount = 0;

        for (var p = 0; p < subPeriodCount; p++)
        {
            var start = p * periodSize;
            var end = p == subPeriodCount - 1 ? returns.Length : start + periodSize;
            var periodSum = T.Zero;
            for (var i = start; i < end; i++)
            {
                periodSum += returns[i];
            }

            if (periodSum > T.Zero)
            {
                positiveCount++;
            }
        }

        return T.CreateChecked(positiveCount) / T.CreateChecked(subPeriodCount);
    }

    private static T ComputeMaxDrawdown(ReadOnlySpan<T> returns)
    {
        var peak = T.One;
        var cumulative = T.One;
        var maxDd = T.Zero;

        for (var i = 0; i < returns.Length; i++)
        {
            cumulative *= T.One + returns[i];
            if (cumulative > peak)
            {
                peak = cumulative;
            }

            var dd = (peak - cumulative) / peak;
            if (dd > maxDd)
            {
                maxDd = dd;
            }
        }

        return maxDd;
    }
}

/// <summary>
/// Result of a GT-Score computation, decomposed into its four components.
/// </summary>
/// <param name="Score">Composite GT-Score value.</param>
/// <param name="PerformanceComponent">Annualized Sharpe ratio contribution (weighted).</param>
/// <param name="SignificanceComponent">t-statistic contribution (weighted).</param>
/// <param name="ConsistencyComponent">Fraction of positive sub-periods (weighted).</param>
/// <param name="DownsideRiskComponent">Maximum drawdown penalty (weighted, subtracted).</param>
public sealed record GtScoreResult(
    double Score,
    double PerformanceComponent,
    double SignificanceComponent,
    double ConsistencyComponent,
    double DownsideRiskComponent);

/// <summary>
/// Generalization Threshold Score (GT-Score): a composite objective combining
/// annualized performance, statistical significance, sub-period consistency,
/// and a drawdown penalty, designed to embed anti-overfitting directly into
/// the fitness function rather than deflating a post-hoc Sharpe. Reference:
/// Sheppert (2026), arXiv:2602.00080.
/// </summary>
/// <remarks>
/// <para>
/// Complementary to <see cref="DeflatedSharpeRatio"/> and
/// <see cref="ProbabilityOfBacktestOverfitting"/>: those are <em>diagnostic</em>
/// metrics applied after search; GT-Score is a <em>search</em> objective chosen
/// so that optimizers chase generalizable performance in the first place.
/// The reference reports ~98% improvement in the generalization ratio
/// (OOS Sharpe / IS Sharpe) vs. optimizing Sharpe alone on walk-forward
/// validation benchmarks.
/// </para>
/// <para>
/// Composite form:
/// <c>GT = w₁ · Performance + w₂ · Significance + w₃ · Consistency − w₄ · DownsideRisk</c>,
/// with component semantics:
/// <list type="bullet">
///   <item><c>Performance</c> — annualized Sharpe ratio.</item>
///   <item><c>Significance</c> — t-statistic normalized by <c>√T</c>, so its scale is commensurate with Performance rather than diverging with sample length.</item>
///   <item><c>Consistency</c> — fraction of equally-sized sub-periods with positive cumulative return, in <c>[0, 1]</c>.</item>
///   <item><c>DownsideRisk</c> — maximum peak-to-trough drawdown, subtracted.</item>
/// </list>
/// </para>
/// <para>
/// Default weights <c>(0.3, 0.3, 0.2, 0.2)</c> are from the reference's
/// ablation study, balancing raw performance (Sharpe) with statistical
/// confidence (t-stat) at equal weight, with structural sanity checks
/// (consistency, drawdown) at two-thirds weight each. The weights are
/// exposed as parameters; practitioners with asymmetric loss preferences
/// should override rather than renormalize post-hoc.
/// </para>
/// <para>
/// Legacy facade: delegates to <see cref="GeneralizationScore{T}"/> at <c>T = double</c>.
/// </para>
/// </remarks>
public static class GeneralizationScore
{
    /// <summary>
    /// Computes the GT-Score for a daily return series.
    /// </summary>
    /// <param name="returns">Daily return observations (at least 2 required).</param>
    /// <param name="subPeriodCount">Number of equal sub-periods for consistency calculation.</param>
    /// <param name="tradingDaysPerYear">Trading days per year for annualization.</param>
    /// <param name="performanceWeight">Weight for annualized Sharpe component.</param>
    /// <param name="significanceWeight">Weight for t-statistic component.</param>
    /// <param name="consistencyWeight">Weight for sub-period consistency.</param>
    /// <param name="downsideWeight">Weight for maximum drawdown penalty.</param>
    /// <returns>Decomposed GT-Score result.</returns>
    /// <exception cref="ArgumentException">Fewer than 2 returns provided.</exception>
    public static GtScoreResult Compute(
        ReadOnlySpan<double> returns,
        int subPeriodCount = 12,
        double tradingDaysPerYear = 252.0,
        double performanceWeight = 0.3,
        double significanceWeight = 0.3,
        double consistencyWeight = 0.2,
        double downsideWeight = 0.2)
    {
        var result = GeneralizationScore<double>.Compute(
            returns, subPeriodCount, tradingDaysPerYear,
            performanceWeight, significanceWeight, consistencyWeight, downsideWeight);
        return new GtScoreResult(
            result.Score,
            result.PerformanceComponent,
            result.SignificanceComponent,
            result.ConsistencyComponent,
            result.DownsideRiskComponent);
    }
}
