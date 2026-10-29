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
/// Deflated Sharpe Ratio result: the rejection threshold SR̂₀ (the expected
/// maximum Sharpe across the trials under the null), the observed Sharpe's
/// excess over it, and the p-value 1 − DSR.
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <param name="DeflatedSharpe">Observed annualized SR minus the annualized threshold SR̂₀. A convenience difference; the paper's statistic is DSR = 1 − <paramref name="PValue"/>.</param>
/// <param name="PValue">1 − DSR: the probability, under the null that the true Sharpe does not exceed SR̂₀, of an estimate at least this large; clamped to [0, 1].</param>
/// <param name="NumTrials">Number of independent configurations tested (used for the multiple-testing bound).</param>
/// <param name="ExpectedMaxSharpe">Rejection threshold SR̂₀ (Bailey &amp; López de Prado 2014, eq. 1): the expected maximum Sharpe across <paramref name="NumTrials"/> trials under the null, in the annualized units of the observed Sharpe.</param>
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
/// The rejection threshold is the expected maximum Sharpe across N independent
/// trials (eq. 1): SR̂₀ = √V[{SR̂ₙ}]·((1 − γ)·Z⁻¹[1 − 1/N] + γ·Z⁻¹[1 − e⁻¹/N]),
/// where γ ≈ 0.5772 is the Euler-Mascheroni constant and V[{SR̂ₙ}] is the
/// variance of the trials' Sharpe ratios. The statistic (eq. 2) is
/// DSR = Z[(SR̂ − SR̂₀)·√(T − 1) / √(1 − γ̂₃·SR̂ + (γ̂₄ − 1)/4·SR̂²)], with SR̂,
/// SR̂₀ and the sample length T all at the per-period frequency of the returns:
/// the annualized inputs are divided by √tradingDaysPerYear, the trials'
/// variance by tradingDaysPerYear, and T = backTestYears · tradingDaysPerYear.
/// When the trials' variance is not supplied it takes its null value
/// 1/(T − 1), the sampling variance of a per-period Sharpe estimate with no
/// skill. γ̂₄ is <b>raw (Pearson) kurtosis</b> per Mertens (2002) — kurtosis equals 3 for
/// a normal distribution, not 0. Callers holding an excess-kurtosis value
/// (e.g. from <see cref="SampleExcessKurtosis{T}"/>, which equals
/// 0 for normal) must add 3 before passing it as <c>kurtosis</c>; passing
/// excess kurtosis directly understates SE(SR) and biases the p-value.
/// </para>
/// <para>
/// Reference: Bailey, D. &amp; López de Prado, M. (2014). "The Deflated
/// Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and
/// Non-Normality." Journal of Portfolio Management, 40(5), 94–107.
/// </para>
/// <para>
/// Tier A with transcendental tail: arithmetic body, scalar Sqrt/InverseCdf/Cdf finish.
/// </para>
/// </remarks>
public static class DeflatedSharpeRatio<T>
    where T : IFloatingPoint<T>
{
    private const double EulerMascheroni = 0.5772156649015329;

    /// <summary>
    /// Convenience overload that accepts a return series and computes the
    /// observed Sharpe, population skewness and Pearson kurtosis (N-divisor moments) internally
    /// before delegating to <see cref="Compute(T, int, T, T, T, int)"/>.
    /// </summary>
    /// <param name="returns">Periodic return series (e.g. daily returns).</param>
    /// <param name="numTrials">Number of trials in the search.</param>
    /// <param name="tradingDaysPerYear">Trading days per year for annualization. Default 252.</param>
    public static DsrResult<T> ComputeFromReturns(ReadOnlySpan<T> returns, int numTrials, int tradingDaysPerYear = 252)
    {
        if (returns.Length < 4)
        {
            throw new ArgumentException("Need at least 4 return observations.", nameof(returns));
        }

        if (tradingDaysPerYear <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tradingDaysPerYear), tradingDaysPerYear,
                "Trading days per year must be positive.");
        }

        for (var i = 0; i < returns.Length; i++)
        {
            if (!T.IsFinite(returns[i]))
            {
                throw new ArgumentOutOfRangeException(nameof(returns), returns[i], "Return observations must be finite.");
            }
        }

        T mean, sd;
        double skew, kurt;
        if (typeof(T) == typeof(Half))
        {
            // T = Half can neither hold the raw observation count once it
            // exceeds 65,504 (T.CreateChecked silently returns +Infinity
            // rather than throwing, F-L1-03) nor accumulate return-scale
            // increments into a growing sum without losing them once the sum
            // has moved a few binades away — Half's ~10-bit mantissa makes a
            // per-term-scaled running mean lose precision long before a
            // realistic sample is exhausted. Run the whole moment pass in
            // double instead, matching GtInspiredScore.ComputeMeanAndStdDev's
            // existing widening, and round only the finished mean and
            // standard deviation back to Half.
            var n = returns.Length;
            var meanDouble = 0.0;
            for (var i = 0; i < n; i++)
            {
                meanDouble += double.CreateChecked(returns[i]);
            }

            meanDouble /= n;

            var maxAbsDeviationDouble = 0.0;
            for (var i = 0; i < n; i++)
            {
                maxAbsDeviationDouble = Math.Max(maxAbsDeviationDouble, Math.Abs(double.CreateChecked(returns[i]) - meanDouble));
            }

            double sdDouble, skewDouble, kurtDouble;
            if (maxAbsDeviationDouble == 0.0)
            {
                sdDouble = 0.0;
                skewDouble = 0.0;
                kurtDouble = 0.0;
            }
            else
            {
                var m2nDouble = 0.0;
                var m3nDouble = 0.0;
                var m4nDouble = 0.0;
                for (var i = 0; i < n; i++)
                {
                    var dn = (double.CreateChecked(returns[i]) - meanDouble) / maxAbsDeviationDouble;
                    var dn2 = dn * dn;
                    m2nDouble += dn2;
                    m3nDouble += dn2 * dn;
                    m4nDouble += dn2 * dn2;
                }

                m2nDouble /= n;
                m3nDouble /= n;
                m4nDouble /= n;

                var sqrtM2nDouble = Math.Sqrt(m2nDouble);
                sdDouble = maxAbsDeviationDouble * sqrtM2nDouble;
                skewDouble = m3nDouble / (m2nDouble * sqrtM2nDouble);
                kurtDouble = m4nDouble / (m2nDouble * m2nDouble);
            }

            mean = T.CreateChecked(meanDouble);
            sd = T.CreateChecked(sdDouble);
            // Skewness and kurtosis stay in double: eq. 2 is evaluated in double, and a
            // Pearson kurtosis can reach n, above Half.MaxValue for a long series with a
            // dominant outlier, where narrowing would give +Infinity and a PValue of 0.5.
            skew = skewDouble;
            kurt = kurtDouble;
        }
        else
        {
            var n = T.CreateChecked(returns.Length);
            var minReturn = returns[0];
            var maxReturn = returns[0];
            for (var i = 1; i < returns.Length; i++)
            {
                minReturn = T.Min(minReturn, returns[i]);
                maxReturn = T.Max(maxReturn, returns[i]);
            }

            // Remove a common location before scaling: dividing raw levels by their maximum
            // rounds a large shared level into a few ulps of 1 and erases the representable
            // spread (a decimal level of 7e28 with offsets 0..6 loses every offset). With one
            // sign throughout, differences from the first return cannot overflow. A mixed-sign
            // series has a spread comparable to its range, so it is normalized about zero, where
            // opposite-sign differences cannot overflow either.
            var anchor = minReturn >= T.Zero || maxReturn <= T.Zero ? returns[0] : T.Zero;
            var returnScale = T.Max(maxReturn - anchor, anchor - minReturn);

            if (returnScale == T.Zero)
            {
                throw new ArgumentException("Returns have zero variance — Sharpe is undefined.", nameof(returns));
            }

            // Moments and Sharpe are scale-free. Normalize the anchored offsets before the mean
            // so neither its sum nor the centered differences overflow.
            T meanOffset = T.Zero;
            for (var i = 0; i < returns.Length; i++)
            {
                meanOffset += (returns[i] - anchor) / returnScale;
            }

            meanOffset /= n;

            // Scale-normalize deviations by their maximum absolute value
            // before forming the second/third/fourth central moments.
            // Skewness and kurtosis are scale-free (skew = m3n / m2n^1.5,
            // kurt = m4n / m2n^2, independent of the normalizing scale — the
            // scale factor cancels exactly), but the RAW fourth-order
            // products (d^4, m2^2) are not: for T = decimal, ordinary return
            // magnitudes make these underflow to zero while sd itself stays
            // representable, producing a spurious divide-by-zero kurtosis
            // (F-L1-01 / F-L5-01).
            T maxAbsDeviation = T.Zero;
            for (var i = 0; i < returns.Length; i++)
            {
                var absDeviation = T.Abs((returns[i] - anchor) / returnScale - meanOffset);
                if (absDeviation > maxAbsDeviation)
                {
                    maxAbsDeviation = absDeviation;
                }
            }

            T sdT, skewT, kurtT;
            if (maxAbsDeviation == T.Zero)
            {
                sdT = T.Zero;
                skewT = T.Zero;
                kurtT = T.Zero;
            }
            else
            {
                T m2n = T.Zero, m3n = T.Zero, m4n = T.Zero;
                for (var i = 0; i < returns.Length; i++)
                {
                    var dn = ((returns[i] - anchor) / returnScale - meanOffset) / maxAbsDeviation;
                    var dn2 = dn * dn;
                    m2n += dn2;
                    m3n += dn2 * dn;
                    m4n += dn2 * dn2;
                }

                m2n /= n;
                m3n /= n;
                m4n /= n;

                var sqrtM2n = NumericPrecision<T>.Sqrt(m2n);
                sdT = maxAbsDeviation * sqrtM2n;
                skewT = m3n / (m2n * sqrtM2n);
                kurtT = m4n / (m2n * m2n);
            }

            // Mean and standard deviation share the units of returnScale, so their quotient is
            // the unscaled per-period Sharpe with the anchor's level kept in the numerator.
            mean = anchor / returnScale + meanOffset;
            sd = sdT;
            skew = double.CreateChecked(skewT);
            kurt = double.CreateChecked(kurtT);
        }

        if (sd == T.Zero)
        {
            throw new ArgumentException("Returns have zero variance — Sharpe is undefined.", nameof(returns));
        }

        var dailySharpe = mean / sd;
        // A positive int frequency can exceed Half's range even though its root fits.
        var sqrtTradingDays = typeof(T) == typeof(Half)
            ? T.CreateChecked(Math.Sqrt(tradingDaysPerYear))
            : NumericPrecision<T>.Sqrt(T.CreateChecked(tradingDaysPerYear));
        var annualSharpe = dailySharpe * sqrtTradingDays;

        RequireFiniteObservedSharpe(annualSharpe);

        // The observation count is exact in double. A years fraction narrowed to T would not be:
        // Half rounds 3e-9 years to zero, and the one-year fallback then replaces the sample length.
        return EvaluateObservations(
            annualSharpe, numTrials, hasTrialVariance: false, T.Zero, T.Zero,
            returns.Length, skew, kurt, tradingDaysPerYear);
    }

    /// <summary>
    /// Computes the deflated Sharpe ratio and its p-value when the variance of
    /// the trials' Sharpe ratios is not known. The trials' variance is taken at
    /// its null value: when no trial has skill, the dispersion of their Sharpe
    /// estimates is pure estimation noise, with per-period variance 1/(T − 1).
    /// </summary>
    /// <param name="observedSharpe">Best observed annualized Sharpe ratio.</param>
    /// <param name="numTrials">Number of configurations tried. Values below 1 are clamped to 1.</param>
    /// <param name="backTestYears">Backtest length in years. Values ≤ 0 are clamped to 1.</param>
    /// <param name="skewness">Sample skewness of the return distribution.</param>
    /// <param name="kurtosis">Pearson (raw) kurtosis of the return distribution. Equals 3 for a normal distribution. This is NOT excess kurtosis (which equals 0 for normal); passing excess kurtosis will produce an incorrect SE estimate.</param>
    /// <param name="tradingDaysPerYear">Return observations per year, used to convert the annualized inputs to the per-period frequency of the returns. Default 252 (US).</param>
    public static DsrResult<T> Compute(
        T observedSharpe,
        int numTrials,
        T backTestYears,
        T skewness,
        T kurtosis,
        int tradingDaysPerYear = 252)
        => Evaluate(observedSharpe, numTrials, hasTrialVariance: false, T.Zero, T.Zero, backTestYears, skewness, kurtosis, tradingDaysPerYear);

    /// <summary>
    /// Computes the zero-trial-mean compatibility DSR and its p-value from the variance of
    /// the trials' Sharpe ratios, as in Bailey &amp; López de Prado (2014), eqs. 1–2.
    /// </summary>
    /// <param name="observedSharpe">Best observed annualized Sharpe ratio.</param>
    /// <param name="numTrials">Number of independent configurations tried. Values below 1 are clamped to 1.</param>
    /// <param name="trialSharpeVariance">Variance across the trials of their annualized Sharpe ratios, V[{SR̂ₙ}]. Must be finite and non-negative.</param>
    /// <param name="backTestYears">Backtest length in years. Values ≤ 0 are clamped to 1.</param>
    /// <param name="skewness">Sample skewness of the return distribution.</param>
    /// <param name="kurtosis">Pearson (raw) kurtosis of the return distribution. Equals 3 for a normal distribution.</param>
    /// <param name="tradingDaysPerYear">Return observations per year, used to convert the annualized inputs to the per-period frequency of the returns.</param>
    public static DsrResult<T> ComputeWithTrialVariance(
        T observedSharpe,
        int numTrials,
        T trialSharpeVariance,
        T backTestYears,
        T skewness,
        T kurtosis,
        int tradingDaysPerYear)
    {
        if (!T.IsFinite(trialSharpeVariance) || trialSharpeVariance < T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(trialSharpeVariance), trialSharpeVariance, "Trial Sharpe variance must be finite and non-negative.");
        }

        return Evaluate(observedSharpe, numTrials, hasTrialVariance: true, trialSharpeVariance, T.Zero, backTestYears, skewness, kurtosis, tradingDaysPerYear);
    }

    /// <summary>
    /// Computes the expected-maximum threshold including the trials' mean (Bailey
    /// and Lopez de Prado 2014, equation 1), then evaluates PSR at that threshold.
    /// The paper's zero-mean null DSR is recovered when trialSharpeMean is zero.
    /// </summary>
    /// <param name="observedSharpe">Observed annualized Sharpe.</param>
    /// <param name="numTrials">Independent trial count; values below one use one.</param>
    /// <param name="trialSharpeMean">Finite mean of the trials' annualized Sharpes.</param>
    /// <param name="trialSharpeVariance">Finite nonnegative variance of the trials' annualized Sharpes.</param>
    /// <param name="backTestYears">Backtest years; nonpositive values use one.</param>
    /// <param name="skewness">Return skewness.</param>
    /// <param name="kurtosis">Raw Pearson return kurtosis.</param>
    /// <param name="tradingDaysPerYear">Return observations per year.</param>
    /// <returns>The annualized expected maximum, observed excess, and complementary PSR.</returns>
    public static DsrResult<T> ComputeWithTrialMoments(
        T observedSharpe, int numTrials, T trialSharpeMean, T trialSharpeVariance,
        T backTestYears, T skewness, T kurtosis, int tradingDaysPerYear = 252)
    {
        if (!T.IsFinite(trialSharpeMean))
        {
            throw new ArgumentOutOfRangeException(nameof(trialSharpeMean));
        }

        if (!T.IsFinite(trialSharpeVariance) || trialSharpeVariance < T.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(trialSharpeVariance));
        }

        return Evaluate(observedSharpe, numTrials, hasTrialVariance: true, trialSharpeVariance, trialSharpeMean,
            backTestYears, skewness, kurtosis, tradingDaysPerYear);
    }

    private static DsrResult<T> Evaluate(
        T observedSharpe,
        int numTrials,
        bool hasTrialVariance,
        T trialSharpeVariance,
        T trialSharpeMean,
        T backTestYears,
        T skewness,
        T kurtosis,
        int tradingDaysPerYear)
    {
        if (tradingDaysPerYear <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(tradingDaysPerYear), tradingDaysPerYear, "Trading days per year must be positive.");
        }

        RequireFiniteObservedSharpe(observedSharpe);

        if (!T.IsFinite(backTestYears))
        {
            throw new ArgumentOutOfRangeException(nameof(backTestYears), backTestYears, "Backtest years must be finite.");
        }

        if (!T.IsFinite(skewness))
        {
            throw new ArgumentOutOfRangeException(nameof(skewness), skewness, "Skewness must be finite.");
        }

        if (!T.IsFinite(kurtosis))
        {
            throw new ArgumentOutOfRangeException(nameof(kurtosis), kurtosis, "Kurtosis must be finite.");
        }

        if (backTestYears <= T.Zero)
        {
            backTestYears = T.One;
        }

        // Multiply in double, not in T: for T = Half, backTestYears *
        // T.CreateChecked(tradingDaysPerYear) can overflow Half's ~65,504
        // ceiling even though backTestYears and tradingDaysPerYear are each
        // individually representable (F-L1-03) — e.g. a 70,000-observation
        // series at 252 trading days/year gives backTestYears ~= 277.8, whose
        // product with 252 is ~69,972, silently rounding to +Infinity in Half
        // arithmetic before this line's own double.CreateChecked ever runs.
        return EvaluateObservations(
            observedSharpe, numTrials, hasTrialVariance, trialSharpeVariance, trialSharpeMean,
            double.CreateChecked(backTestYears) * tradingDaysPerYear, double.CreateChecked(skewness),
            double.CreateChecked(kurtosis), tradingDaysPerYear);
    }

    private static void RequireFiniteObservedSharpe(T observedSharpe)
    {
        if (!T.IsFinite(observedSharpe))
        {
            throw new ArgumentOutOfRangeException(nameof(observedSharpe), observedSharpe, "Observed Sharpe must be finite.");
        }
    }

    /// <summary>
    /// Evaluates eqs. 1–2 for already validated inputs and a sample length given as an observation
    /// count in double, so no years fraction is narrowed to <typeparamref name="T"/> on the way in.
    /// Skewness and kurtosis also arrive in double, so a moment computed in double is never narrowed.
    /// </summary>
    private static DsrResult<T> EvaluateObservations(
        T observedSharpe,
        int numTrials,
        bool hasTrialVariance,
        T trialSharpeVariance,
        T trialSharpeMean,
        double observationCount,
        double skewDbl,
        double kurtDbl,
        int tradingDaysPerYear)
    {
        if (numTrials < 1)
        {
            numTrials = 1;
        }

        // Eq. 2 needs SR̂ and T at the same, non-annualized frequency: convert
        // the annualized inputs to per-period units of the return series.
        var periodsPerYear = (double)tradingDaysPerYear;
        var sqrtPeriodsPerYear = Math.Sqrt(periodsPerYear);
        var degreesOfFreedom = Math.Max(observationCount - 1.0, 1.0);
        var srAnnual = double.CreateChecked(observedSharpe);
        var srPerPeriod = srAnnual / sqrtPeriodsPerYear;

        // V[{SR̂ₙ}] per period; without a supplied value, the null sampling
        // variance of a per-period Sharpe estimate.
        var trialVariancePerPeriod = hasTrialVariance
            ? double.CreateChecked(trialSharpeVariance) / periodsPerYear
            : 1.0 / degreesOfFreedom;

        // Eq. 1: SR̂₀ = √V·((1 − γ)·Z⁻¹[1 − 1/N] + γ·Z⁻¹[1 − e⁻¹/N]).
        var expectedMaxStandardNormal = numTrials == 1
            ? 0.0
            : (1.0 - EulerMascheroni) * NormalDistribution<double>.InverseCdf(1.0 - 1.0 / numTrials)
              + EulerMascheroni * NormalDistribution<double>.InverseCdf(1.0 - Math.Exp(-1.0) / numTrials);
        var thresholdPerPeriod = double.CreateChecked(trialSharpeMean) / sqrtPeriodsPerYear
            + Math.Sqrt(trialVariancePerPeriod) * expectedMaxStandardNormal;

        // Eq. 2: DSR = Z[(SR̂ − SR̂₀)·√(T − 1) / √(1 − γ̂₃·SR̂ + (γ̂₄ − 1)/4·SR̂²)].
        var nonNormality = 1.0 - skewDbl * srPerPeriod + (kurtDbl - 1.0) / 4.0 * srPerPeriod * srPerPeriod;
        double testStat;
        var scaledVariance = 0.0;
        var scaledSharpe = 0.0;
        var srScale = 1.0;
        if (!double.IsFinite(nonNormality))
        {
            // The skew or quadratic term overflowed: +Infinity alone, -Infinity from the skew term,
            // or NaN when opposite infinities meet. Normalize only the overflowing terms. Scaling a
            // finite constant variance would instead underflow it when the quadratic term is zero.
            srScale = Math.Max(1.0, Math.Abs(srPerPeriod));
            var inverseScale = 1.0 / srScale;
            scaledSharpe = srPerPeriod / srScale;
            scaledVariance = inverseScale * inverseScale - skewDbl * scaledSharpe * inverseScale
                + (kurtDbl - 1.0) / 4.0 * scaledSharpe * scaledSharpe;
            if (!(scaledVariance > 0.0))
            {
                // A non-positive variance estimate takes the Gaussian value, as below.
                nonNormality = 1.0;
            }
        }
        else if (nonNormality <= 0.0)
        {
            nonNormality = 1.0;
        }

        if (scaledVariance > 0.0)
        {
            testStat = (scaledSharpe - thresholdPerPeriod / srScale) * Math.Sqrt(degreesOfFreedom)
                / Math.Sqrt(scaledVariance);
        }
        else
        {
            // Taking roots first preserves a representable standard error when the
            // positive variance quotient is smaller than the least subnormal.
            var srStdError = Math.Sqrt(nonNormality) / Math.Sqrt(degreesOfFreedom);
            testStat = srStdError > 0.0 ? (srPerPeriod - thresholdPerPeriod) / srStdError : 0.0;
        }

        var pValue = 1.0 - NormalDistribution<double>.Cdf(testStat);
        var thresholdAnnual = thresholdPerPeriod * sqrtPeriodsPerYear;

        return new DsrResult<T>(
            DeflatedSharpe: T.CreateChecked(srAnnual - thresholdAnnual),
            PValue: T.Clamp(T.CreateChecked(pValue), T.Zero, T.One),
            NumTrials: numTrials,
            ExpectedMaxSharpe: T.CreateChecked(thresholdAnnual));
    }
}

/// <summary>
/// Deflated Sharpe Ratio result: the rejection threshold SR̂₀ (the expected
/// maximum Sharpe across the trials under the null), the observed Sharpe's
/// excess over it, and the p-value 1 − DSR.
/// </summary>
/// <param name="DeflatedSharpe">Observed annualized SR minus the annualized threshold SR̂₀. A convenience difference; the paper's statistic is DSR = 1 − <paramref name="PValue"/>.</param>
/// <param name="PValue">1 − DSR: the probability, under the null that the true Sharpe does not exceed SR̂₀, of an estimate at least this large; clamped to [0, 1].</param>
/// <param name="NumTrials">Number of independent configurations tested (used for the multiple-testing bound).</param>
/// <param name="ExpectedMaxSharpe">Rejection threshold SR̂₀ (Bailey &amp; López de Prado 2014, eq. 1): the expected maximum Sharpe across <paramref name="NumTrials"/> trials under the null, in the annualized units of the observed Sharpe.</param>
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
/// The rejection threshold is the expected maximum Sharpe across N independent
/// trials (eq. 1): SR̂₀ = √V[{SR̂ₙ}]·((1 − γ)·Z⁻¹[1 − 1/N] + γ·Z⁻¹[1 − e⁻¹/N]),
/// where γ ≈ 0.5772 is the Euler-Mascheroni constant and V[{SR̂ₙ}] is the
/// variance of the trials' Sharpe ratios. The statistic (eq. 2) is
/// DSR = Z[(SR̂ − SR̂₀)·√(T − 1) / √(1 − γ̂₃·SR̂ + (γ̂₄ − 1)/4·SR̂²)], with SR̂,
/// SR̂₀ and the sample length T all at the per-period frequency of the returns:
/// the annualized inputs are divided by √tradingDaysPerYear, the trials'
/// variance by tradingDaysPerYear, and T = backTestYears · tradingDaysPerYear.
/// When the trials' variance is not supplied it takes its null value
/// 1/(T − 1), the sampling variance of a per-period Sharpe estimate with no
/// skill. γ̂₄ is <b>raw (Pearson) kurtosis</b> per Mertens (2002) — kurtosis equals 3 for
/// a normal distribution, not 0. Callers holding an excess-kurtosis value
/// (e.g. from <see cref="SampleExcessKurtosis"/>, which equals 0 for normal)
/// must add 3 before passing it as <c>kurtosis</c>; passing excess kurtosis
/// directly understates SE(SR) and biases the p-value.
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
[Obsolete(
    "Use DeflatedSharpeRatio<decimal> instead (the facade preserves the original decimal precision; use <double> deliberately if you want IEEE precision). This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class DeflatedSharpeRatio
{
    /// <summary>
    /// Convenience overload that accepts a return series and computes the
    /// observed Sharpe, population skewness and Pearson kurtosis (N-divisor moments) internally
    /// before delegating to <see cref="Compute(decimal, int, decimal, decimal, decimal, int)"/>.
    /// </summary>
    /// <param name="returns">Periodic return series (e.g. daily returns).</param>
    /// <param name="numTrials">Number of trials in the search.</param>
    /// <param name="tradingDaysPerYear">Trading days per year for annualization. Default 252.</param>
    public static DsrResult ComputeFromReturns(ReadOnlySpan<decimal> returns, int numTrials, int tradingDaysPerYear = 252)
    {
        var result = DeflatedSharpeRatio<decimal>.ComputeFromReturns(returns, numTrials, tradingDaysPerYear);
        return new DsrResult(result.DeflatedSharpe, result.PValue, result.NumTrials, result.ExpectedMaxSharpe);
    }

    /// <summary>
    /// Computes the zero-trial-mean compatibility DSR and its p-value.
    /// </summary>
    /// <param name="observedSharpe">Best observed annualized Sharpe ratio.</param>
    /// <param name="numTrials">Number of configurations tried. Values below 1 are clamped to 1.</param>
    /// <param name="backTestYears">Backtest length in years. Values ≤ 0 are clamped to 1.</param>
    /// <param name="skewness">Sample skewness of the return distribution.</param>
    /// <param name="kurtosis">Pearson (raw) kurtosis of the return distribution. Equals 3 for a normal distribution. This is NOT excess kurtosis (which equals 0 for normal); passing excess kurtosis will produce an incorrect SE estimate.</param>
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
