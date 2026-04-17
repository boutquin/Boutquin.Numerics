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
using Boutquin.Numerics.Internal;

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Deflated Sharpe Ratio result: observed SR minus the expected maximum SR
/// under the null, together with its p-value and the trial count used for the
/// deflation.
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <param name="DeflatedSharpe">Observed SR minus E[max SR] under the null.</param>
/// <param name="PValue">Probability of observing an SR at least this large under the null, clamped to [0, 1].</param>
/// <param name="NumTrials">Number of independent configurations tested (used for the multiple-testing bound).</param>
/// <param name="ExpectedMaxSharpe">Asymptotic expectation of the maximum SR across <paramref name="NumTrials"/> trials under the null.</param>
public sealed record DsrResult<T>(
    T DeflatedSharpe,
    T PValue,
    int NumTrials,
    T ExpectedMaxSharpe)
    where T : IFloatingPoint<T>;

/// <summary>
/// Deflated Sharpe Ratio (Bailey &amp; López de Prado, 2014). Adjusts an
/// observed Sharpe for multiple-testing bias: the more configurations tested,
/// the higher the expected maximum Sharpe even under a null of zero mean.
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <remarks>
/// <para>
/// The expected maximum Sharpe across N independent trials is approximated by
/// the extreme-value statistic
/// E[max SR] ≈ √(2·ln N)·(1 − γ/ln N) + γ/√(2·ln N)
/// where γ ≈ 0.5772 is the Euler-Mascheroni constant. The p-value is
/// computed from the standard normal CDF evaluated at
/// (SR − E[max SR]) / SE(SR), with SE(SR) adjusting for skewness and
/// excess kurtosis per Mertens (2002).
/// </para>
/// <para>
/// Reference: Bailey, D. &amp; López de Prado, M. (2014). "The Deflated
/// Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and
/// Non-Normality." Journal of Portfolio Management, 40(5), 94–107.
/// </para>
/// <para>
/// Tier A with transcendental tail: arithmetic body, scalar Log/Sqrt/Cdf finish.
/// </para>
/// </remarks>
public static class DeflatedSharpeRatio<T>
    where T : IFloatingPoint<T>
{
    private const double EulerMascheroni = 0.5772156649015329;

    /// <summary>
    /// Convenience overload that accepts a return series and computes the
    /// observed Sharpe, sample skewness, and sample kurtosis internally
    /// before delegating to <see cref="Compute(T, int, T, T, T, int)"/>.
    /// </summary>
    /// <param name="returns">Periodic return series (e.g. daily returns).</param>
    /// <param name="numTrials">Number of trials in the search.</param>
    /// <param name="tradingDaysPerYear">Trading days per year for annualization. Default 252.</param>
    public static DsrResult<T> ComputeFromReturns(T[] returns, int numTrials, int tradingDaysPerYear = 252)
    {
        ArgumentNullException.ThrowIfNull(returns);
        if (returns.Length < 4)
        {
            throw new ArgumentException("Need at least 4 return observations.", nameof(returns));
        }

        var n = T.CreateChecked(returns.Length);
        T mean = T.Zero;
        for (var i = 0; i < returns.Length; i++)
        {
            mean += returns[i];
        }

        mean /= n;

        T m2 = T.Zero, m3 = T.Zero, m4 = T.Zero;
        for (var i = 0; i < returns.Length; i++)
        {
            var d = returns[i] - mean;
            var d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }

        m2 /= n;
        m3 /= n;
        m4 /= n;

        var sd = NumericPrecision<T>.Sqrt(m2);
        if (sd == T.Zero)
        {
            throw new ArgumentException("Returns have zero variance — Sharpe is undefined.", nameof(returns));
        }

        var dailySharpe = mean / sd;
        var sqrtTradingDays = NumericPrecision<T>.Sqrt(T.CreateChecked(tradingDaysPerYear));
        var annualSharpe = dailySharpe * sqrtTradingDays;
        var skew = m3 / (sd * sd * sd);
        var kurt = m4 / (m2 * m2);

        var backtestYears = n / T.CreateChecked(tradingDaysPerYear);
        return Compute(annualSharpe, numTrials, backtestYears, skew, kurt, tradingDaysPerYear);
    }

    /// <summary>
    /// Computes the deflated Sharpe ratio and its p-value.
    /// </summary>
    /// <param name="observedSharpe">Best observed annualized Sharpe ratio.</param>
    /// <param name="numTrials">Number of configurations tried. Values below 1 are clamped to 1.</param>
    /// <param name="backTestYears">Backtest length in years. Values ≤ 0 are clamped to 1.</param>
    /// <param name="skewness">Sample skewness of the return distribution.</param>
    /// <param name="kurtosis">Sample kurtosis of the return distribution (3 for normal).</param>
    /// <param name="tradingDaysPerYear">Trading days per year. Default 252 (US).</param>
    public static DsrResult<T> Compute(
        T observedSharpe,
        int numTrials,
        T backTestYears,
        T skewness,
        T kurtosis,
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

        double expectedMaxSharpe;
        if (numTrials == 1)
        {
            expectedMaxSharpe = 0.0;
        }
        else
        {
            var logN = Math.Log(numTrials);
            var sqrtTwoLogN = Math.Sqrt(2.0 * logN);
            expectedMaxSharpe =
                sqrtTwoLogN * (1.0 - EulerMascheroni / logN) +
                EulerMascheroni / sqrtTwoLogN;
        }

        var tDbl = double.CreateChecked(backTestYears * T.CreateChecked(tradingDaysPerYear));
        var srDbl = double.CreateChecked(observedSharpe);
        var skewDbl = double.CreateChecked(skewness);
        var kurtDbl = double.CreateChecked(kurtosis);

        var srVariance = (1.0 - skewDbl * srDbl + (kurtDbl - 1.0) / 4.0 * srDbl * srDbl) / tDbl;
        if (srVariance < 0.0)
        {
            srVariance = 1.0 / tDbl;
        }

        var srStdError = Math.Sqrt(srVariance);
        var testStat = srStdError > 0.0 ? (srDbl - expectedMaxSharpe) / srStdError : 0.0;
        var pValue = 1.0 - NormalDistribution<double>.Cdf(testStat);
        var deflated = srDbl - expectedMaxSharpe;

        return new DsrResult<T>(
            DeflatedSharpe: T.CreateChecked(deflated),
            PValue: T.Clamp(T.CreateChecked(pValue), T.Zero, T.One),
            NumTrials: numTrials,
            ExpectedMaxSharpe: T.CreateChecked(expectedMaxSharpe));
    }
}

/// <summary>
/// Deflated Sharpe Ratio result: observed SR minus the expected maximum SR
/// under the null, together with its p-value and the trial count used for the
/// deflation.
/// </summary>
/// <param name="DeflatedSharpe">Observed SR minus E[max SR] under the null.</param>
/// <param name="PValue">Probability of observing an SR at least this large under the null, clamped to [0, 1].</param>
/// <param name="NumTrials">Number of independent configurations tested (used for the multiple-testing bound).</param>
/// <param name="ExpectedMaxSharpe">Asymptotic expectation of the maximum SR across <paramref name="NumTrials"/> trials under the null.</param>
public sealed record DsrResult(
    decimal DeflatedSharpe,
    decimal PValue,
    int NumTrials,
    decimal ExpectedMaxSharpe);

/// <summary>
/// Deflated Sharpe Ratio (Bailey &amp; López de Prado, 2014). Adjusts an
/// observed Sharpe for multiple-testing bias: the more configurations tested,
/// the higher the expected maximum Sharpe even under a null of zero mean.
/// </summary>
/// <remarks>
/// <para>
/// The expected maximum Sharpe across N independent trials is approximated by
/// the extreme-value statistic
/// E[max SR] ≈ √(2·ln N)·(1 − γ/ln N) + γ/√(2·ln N)
/// where γ ≈ 0.5772 is the Euler-Mascheroni constant. The p-value is
/// computed from the standard normal CDF evaluated at
/// (SR − E[max SR]) / SE(SR), with SE(SR) adjusting for skewness and
/// excess kurtosis per Mertens (2002).
/// </para>
/// <para>
/// Reference: Bailey, D. &amp; López de Prado, M. (2014). "The Deflated
/// Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and
/// Non-Normality." Journal of Portfolio Management, 40(5), 94–107.
/// </para>
/// <para>
/// Legacy facade: delegates to <see cref="DeflatedSharpeRatio{T}"/> at <c>T = decimal</c>.
/// </para>
/// </remarks>
public static class DeflatedSharpeRatio
{
    /// <summary>
    /// Convenience overload that accepts a return series and computes the
    /// observed Sharpe, sample skewness, and sample kurtosis internally
    /// before delegating to <see cref="Compute(decimal, int, decimal, decimal, decimal, int)"/>.
    /// </summary>
    /// <param name="returns">Periodic return series (e.g. daily returns).</param>
    /// <param name="numTrials">Number of trials in the search.</param>
    /// <param name="tradingDaysPerYear">Trading days per year for annualization. Default 252.</param>
    public static DsrResult ComputeFromReturns(decimal[] returns, int numTrials, int tradingDaysPerYear = 252)
    {
        var result = DeflatedSharpeRatio<decimal>.ComputeFromReturns(returns, numTrials, tradingDaysPerYear);
        return new DsrResult(result.DeflatedSharpe, result.PValue, result.NumTrials, result.ExpectedMaxSharpe);
    }

    /// <summary>
    /// Computes the deflated Sharpe ratio and its p-value.
    /// </summary>
    /// <param name="observedSharpe">Best observed annualized Sharpe ratio.</param>
    /// <param name="numTrials">Number of configurations tried. Values below 1 are clamped to 1.</param>
    /// <param name="backTestYears">Backtest length in years. Values ≤ 0 are clamped to 1.</param>
    /// <param name="skewness">Sample skewness of the return distribution.</param>
    /// <param name="kurtosis">Sample kurtosis of the return distribution (3 for normal).</param>
    /// <param name="tradingDaysPerYear">Trading days per year. Default 252 (US).</param>
    public static DsrResult Compute(
        decimal observedSharpe,
        int numTrials,
        decimal backTestYears,
        decimal skewness,
        decimal kurtosis,
        int tradingDaysPerYear = 252)
    {
        var result = DeflatedSharpeRatio<decimal>.Compute(
            observedSharpe, numTrials, backTestYears, skewness, kurtosis, tradingDaysPerYear);
        return new DsrResult(result.DeflatedSharpe, result.PValue, result.NumTrials, result.ExpectedMaxSharpe);
    }
}
