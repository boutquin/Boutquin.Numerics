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
using Boutquin.Numerics.Distributions;

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Result of the Harvey-Liu-Zhu (2016) Sharpe-ratio haircut. Contains the
/// originally observed Sharpe, the haircut applied (in Sharpe units), the
/// haircut Sharpe ratio, and the multiple-testing-adjusted p-value under
/// each of the three correction methods.
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <param name="ObservedSharpe">Originally observed annualized Sharpe.</param>
/// <param name="HaircutSharpe">Sharpe net of the multiple-testing haircut.</param>
/// <param name="HaircutAmount">Reduction (Sharpe units) applied to the observed value.</param>
/// <param name="BonferroniPValue">Multiple-testing-adjusted p-value via Bonferroni correction.</param>
/// <param name="HolmPValue">Adjusted p-value via Holm's step-down procedure.</param>
/// <param name="BhyPValue">Adjusted p-value via the Benjamini-Hochberg-Yekutieli false discovery rate procedure.</param>
public sealed record HaircutSharpeResult<T>(
    T ObservedSharpe,
    T HaircutSharpe,
    T HaircutAmount,
    T BonferroniPValue,
    T HolmPValue,
    T BhyPValue)
    where T : IFloatingPoint<T>;

/// <summary>
/// Harvey-Liu-Zhu (2016) "Haircut Sharpe Ratio" — adjusts an observed Sharpe
/// for multiple-testing bias using three correction methods (Bonferroni,
/// Holm, Benjamini-Hochberg-Yekutieli) and reports the resulting haircut.
/// More stringent than the Deflated Sharpe Ratio for large trial counts.
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <remarks>
/// <para>
/// Reference: Harvey, C. R., Liu, Y. &amp; Zhu, H. (2016). "...and the
/// Cross-Section of Expected Returns." Review of Financial Studies, 29(1),
/// 5–68.
/// </para>
/// <para>
/// The haircut converts each adjusted p-value into the Sharpe ratio that
/// would have produced it under the null. Subtracting the haircut from
/// the observed Sharpe gives the multiple-testing-corrected estimate.
/// </para>
/// <para>
/// Tier A with transcendental tail: arithmetic body, scalar Sqrt/InverseCdf finish.
/// </para>
/// </remarks>
public static class HaircutSharpe<T>
    where T : IFloatingPoint<T>
{
    /// <summary>
    /// Computes the Harvey-Liu-Zhu Sharpe haircut.
    /// </summary>
    /// <param name="observedSharpe">Observed annualized Sharpe ratio.</param>
    /// <param name="numTrials">Number of trials in the search.</param>
    /// <param name="backTestYears">Backtest length in years.</param>
    /// <param name="tradingDaysPerYear">Trading days per year. Default 252.</param>
    public static HaircutSharpeResult<T> Compute(
        T observedSharpe,
        int numTrials,
        T backTestYears,
        int tradingDaysPerYear = 252)
    {
        if (numTrials < 1)
        {
            numTrials = 1;
        }

        if (backTestYears <= T.Zero)
        {
            backTestYears = T.One;
        }

        var tDbl = double.CreateChecked(backTestYears * T.CreateChecked(tradingDaysPerYear));
        var srDbl = double.CreateChecked(observedSharpe);

        var z = srDbl * Math.Sqrt(tDbl);
        var pSingle = 1.0 - NormalDistribution<double>.Cdf(z);

        var pBonferroni = Math.Min(1.0, numTrials * pSingle);

        var pHolm = pBonferroni;

        var harmonic = 0.0;
        for (var i = 1; i <= numTrials; i++)
        {
            harmonic += 1.0 / i;
        }

        var pBhy = Math.Min(1.0, pSingle * harmonic);

        var qInput = Math.Clamp(1.0 - pBhy, 1e-15, 1.0 - 1e-15);
        var srAdjusted = NormalDistribution<double>.InverseCdf(qInput) / Math.Sqrt(tDbl);
        var haircutSr = T.CreateChecked(srAdjusted);
        if (haircutSr > observedSharpe)
        {
            haircutSr = observedSharpe;
        }

        var haircutAmount = observedSharpe - haircutSr;

        return new HaircutSharpeResult<T>(
            ObservedSharpe: observedSharpe,
            HaircutSharpe: haircutSr,
            HaircutAmount: haircutAmount,
            BonferroniPValue: T.CreateChecked(pBonferroni),
            HolmPValue: T.CreateChecked(pHolm),
            BhyPValue: T.CreateChecked(pBhy));
    }
}

/// <summary>
/// Result of the Harvey-Liu-Zhu (2016) Sharpe-ratio haircut. Contains the
/// originally observed Sharpe, the haircut applied (in Sharpe units), the
/// haircut Sharpe ratio, and the multiple-testing-adjusted p-value under
/// each of the three correction methods.
/// </summary>
/// <param name="ObservedSharpe">Originally observed annualized Sharpe.</param>
/// <param name="HaircutSharpe">Sharpe net of the multiple-testing haircut.</param>
/// <param name="HaircutAmount">Reduction (Sharpe units) applied to the observed value.</param>
/// <param name="BonferroniPValue">Multiple-testing-adjusted p-value via Bonferroni correction.</param>
/// <param name="HolmPValue">Adjusted p-value via Holm's step-down procedure.</param>
/// <param name="BhyPValue">Adjusted p-value via the Benjamini-Hochberg-Yekutieli false discovery rate procedure.</param>
public sealed record HaircutSharpeResult(
    decimal ObservedSharpe,
    decimal HaircutSharpe,
    decimal HaircutAmount,
    decimal BonferroniPValue,
    decimal HolmPValue,
    decimal BhyPValue);

/// <summary>
/// Harvey-Liu-Zhu (2016) "Haircut Sharpe Ratio" — adjusts an observed Sharpe
/// for multiple-testing bias using three correction methods (Bonferroni,
/// Holm, Benjamini-Hochberg-Yekutieli) and reports the resulting haircut.
/// More stringent than the Deflated Sharpe Ratio for large trial counts.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Harvey, C. R., Liu, Y. &amp; Zhu, H. (2016). "...and the
/// Cross-Section of Expected Returns." Review of Financial Studies, 29(1),
/// 5–68.
/// </para>
/// <para>
/// The haircut converts each adjusted p-value into the Sharpe ratio that
/// would have produced it under the null. Subtracting the haircut from
/// the observed Sharpe gives the multiple-testing-corrected estimate.
/// </para>
/// <para>
/// Legacy facade: delegates to <see cref="HaircutSharpe{T}"/> at <c>T = decimal</c>.
/// </para>
/// </remarks>
public static class HaircutSharpe
{
    /// <summary>
    /// Computes the Harvey-Liu-Zhu Sharpe haircut.
    /// </summary>
    /// <param name="observedSharpe">Observed annualized Sharpe ratio.</param>
    /// <param name="numTrials">Number of trials in the search.</param>
    /// <param name="backTestYears">Backtest length in years.</param>
    /// <param name="tradingDaysPerYear">Trading days per year. Default 252.</param>
    public static HaircutSharpeResult Compute(
        decimal observedSharpe,
        int numTrials,
        decimal backTestYears,
        int tradingDaysPerYear = 252)
    {
        var result = HaircutSharpe<decimal>.Compute(observedSharpe, numTrials, backTestYears, tradingDaysPerYear);
        return new HaircutSharpeResult(
            result.ObservedSharpe,
            result.HaircutSharpe,
            result.HaircutAmount,
            result.BonferroniPValue,
            result.HolmPValue,
            result.BhyPValue);
    }
}
