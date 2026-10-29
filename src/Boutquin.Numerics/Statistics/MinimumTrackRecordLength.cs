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
/// Minimum Track Record Length (MinTRL) — Bailey &amp; López de Prado (2012).
/// The smallest sample length T* such that an observed Sharpe ratio is
/// statistically significantly greater than a benchmark, accounting for
/// non-normality (skewness, raw kurtosis).
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <remarks>
/// <para>
/// Reference: Bailey, D. H. &amp; López de Prado, M. (2012). "The Sharpe
/// Ratio Efficient Frontier." Journal of Risk, 15(2), 13–44.
/// </para>
/// <para>
/// Formula:
/// <c>T* = 1 + (1 − γ₃·SR + (γ₄−1)/4 · SR²) · (z_α / (SR − SR*))²</c>
/// where SR is the observed Sharpe and SR* the benchmark (typically 0), both at
/// the per-period frequency of the returns, γ₃ is sample skewness, γ₄ is raw
/// (Pearson) sample kurtosis (3 for a normal distribution), and z_α is the
/// standard normal quantile at significance level α (the closed form holds for confidence 1 − α above one half;
/// at α ≥ 1/2 a single observation already attains PSR ≥ 1 − α, so the minimum is one observation). The inputs are annualized
/// Sharpe ratios; they are divided by √tradingDaysPerYear before the formula is
/// applied, as the formula requires non-annualized Sharpe ratios.
/// </para>
/// <para>
/// The result T* is a <b>number of observations</b> (return samples at the
/// per-period frequency), per the canonical convention — not a count of years.
/// Divide by <c>tradingDaysPerYear</c> to express it in years.
/// </para>
/// <para>
/// MinTRL is the inverse of the Probabilistic Sharpe Ratio at a specified benchmark: PSR asks "is the
/// observed value significant given my sample length?", MinTRL asks "how
/// long must my sample be for this value to become significant?". Use
/// MinTRL when planning a new strategy or evaluating whether an existing
/// track record is long enough to support claims of skill.
/// </para>
/// <para>
/// Tier B: Fully transcendental computation.
/// </para>
/// </remarks>
public static class MinimumTrackRecordLength<T>
    where T : IFloatingPointIeee754<T>
{
    /// <summary>
    /// Convenience overload supplying the standard defaults the legacy facade
    /// exposes — benchmark Sharpe 0, skewness 0, kurtosis 3 (Gaussian),
    /// significance 0.05, and 252 trading days per year. Expressed as fixed-arity
    /// overloads (rather than optional parameters) because <typeparamref name="T"/>-typed
    /// default values cannot be written as compile-time constants in a generic
    /// method, and the public-API analyzer requires the optional-parameter overload
    /// to be the longest among the public overloads.
    /// </summary>
    /// <param name="observedSharpe">Observed annualized Sharpe ratio.</param>
    /// <returns>Minimum track record length expressed as a number of observations. Returns <c>T.PositiveInfinity</c> if the observed Sharpe is at or below the benchmark.</returns>
    public static T Compute(T observedSharpe)
        => Compute(observedSharpe, significanceLevel: 0.05, tradingDaysPerYear: 252);

    /// <summary>
    /// Convenience overload supplying default benchmark Sharpe 0, skewness 0, and
    /// kurtosis 3 (Gaussian), with caller-chosen significance and trading days.
    /// </summary>
    /// <param name="observedSharpe">Observed annualized Sharpe ratio.</param>
    /// <param name="significanceLevel">Significance α in (0, 1). For α ≥ 1/2 (confidence 1 − α at most one half) the result is one observation when the observed Sharpe exceeds the benchmark.</param>
    /// <param name="tradingDaysPerYear">Return observations per year; the annualized Sharpe ratios are divided by its square root to obtain the per-period Sharpe ratios the formula requires. Must be positive.</param>
    /// <returns>Minimum track record length expressed as a number of observations. Returns <c>T.PositiveInfinity</c> if the observed Sharpe is at or below the benchmark.</returns>
    public static T Compute(
        T observedSharpe,
        double significanceLevel,
        int tradingDaysPerYear)
        => Compute(
            observedSharpe,
            benchmarkSharpe: T.Zero,
            skewness: T.Zero,
            kurtosis: T.CreateChecked(3),
            significanceLevel,
            tradingDaysPerYear);

    /// <summary>
    /// Computes the minimum track record length required for the observed
    /// Sharpe to be statistically distinguishable from the benchmark.
    /// </summary>
    /// <param name="observedSharpe">Observed annualized Sharpe ratio.</param>
    /// <param name="benchmarkSharpe">Annualized benchmark Sharpe (typically 0). Default 0.</param>
    /// <param name="skewness">Sample skewness. Default 0 (Gaussian).</param>
    /// <param name="kurtosis">Raw (Pearson) sample kurtosis (3 = normal). Default 3.</param>
    /// <param name="significanceLevel">Significance α in (0, 1). Default 0.05 (95% confidence). For α ≥ 1/2 (confidence 1 − α at most one half) the result is one observation when the observed Sharpe exceeds the benchmark: PSR already equals 1/2 at a single observation, so eq. 12's minimum is 1.</param>
    /// <param name="tradingDaysPerYear">Return observations per year; the annualized Sharpe ratios are divided by its square root to obtain the per-period Sharpe ratios the formula requires. Must be positive. Default 252.</param>
    /// <returns>Minimum track record length expressed as a number of observations. Returns <c>T.PositiveInfinity</c> if the observed Sharpe is at or below the benchmark.</returns>
    public static T Compute(
        T observedSharpe,
        T benchmarkSharpe,
        T skewness,
        T kurtosis,
        double significanceLevel,
        int tradingDaysPerYear)
    {
        if (!double.IsFinite(significanceLevel) || significanceLevel <= 0.0 || significanceLevel >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(significanceLevel), significanceLevel, "Significance level must lie in (0, 1).");
        }

        if (tradingDaysPerYear <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tradingDaysPerYear), tradingDaysPerYear, "Trading days per year must be positive.");
        }

        if (!T.IsFinite(observedSharpe))
        {
            throw new ArgumentOutOfRangeException(nameof(observedSharpe), observedSharpe, "Observed Sharpe must be finite.");
        }

        if (!T.IsFinite(benchmarkSharpe))
        {
            throw new ArgumentOutOfRangeException(nameof(benchmarkSharpe), benchmarkSharpe, "Benchmark Sharpe must be finite.");
        }

        if (!T.IsFinite(skewness))
        {
            throw new ArgumentOutOfRangeException(nameof(skewness), skewness, "Skewness must be finite.");
        }

        if (!T.IsFinite(kurtosis))
        {
            throw new ArgumentOutOfRangeException(nameof(kurtosis), kurtosis, "Kurtosis must be finite.");
        }

        if (observedSharpe <= benchmarkSharpe)
        {
            return T.PositiveInfinity;
        }

        // Eq. 12 defines MinTRL as the smallest n with PSR(n) >= 1 - alpha, and eq. 13 is its
        // closed form only for confidence 1 - alpha above one half (z_alpha > 0). At one
        // observation PSR = Phi(0) = 1/2, so for confidence at most one half the minimum is
        // exactly one observation. Squaring the non-positive quantile instead would be
        // symmetric in alpha and 1 - alpha and grow as the target confidence falls.
        if (significanceLevel >= 0.5)
        {
            return T.One;
        }

        // Eq. 13 takes SR̂ and SR* in the original, non-annualized sampling
        // frequency (BLdP 2012, fn. 5) and returns a NUMBER OF OBSERVATIONS at
        // that frequency. Convert the annualized inputs to per-period Sharpes.
        // Every positive int has a square root representable in Half, even when
        // the original frequency is too large to convert to Half directly.
        var sqrtPeriodsPerYear = T.CreateChecked(Math.Sqrt(tradingDaysPerYear));
        var srPerPeriod = observedSharpe / sqrtPeriodsPerYear;
        var benchmarkPerPeriod = benchmarkSharpe / sqrtPeriodsPerYear;

        // Significance is supplied in double. Evaluate its scalar quantile before
        // narrowing, and use normal symmetry so tiny alpha never rounds 1-alpha to 1.
        var z = T.CreateChecked(-NormalDistribution<double>.InverseCdf(significanceLevel));
        if (z == T.Zero)
        {
            // A quantile too close to the median to survive narrowing to T makes the
            // correction exactly zero even when the reciprocal Sharpe gap would overflow.
            return T.One;
        }

        // Cancel the common scale before squaring; both numerator and denominator
        // can overflow even when their ratio is a small finite observation count.
        var scale = T.Max(T.One, T.Max(T.Abs(srPerPeriod), T.Abs(benchmarkPerPeriod)));
        var inverseScale = T.One / scale;
        var scaledSharpe = srPerPeriod / scale;
        var srGap = scaledSharpe - benchmarkPerPeriod / scale;
        var nonGaussianity = inverseScale * inverseScale - skewness * scaledSharpe * inverseScale
            + (kurtosis - T.One) / T.CreateChecked(4) * scaledSharpe * scaledSharpe;
        if (nonGaussianity < T.Zero)
        {
            nonGaussianity = inverseScale * inverseScale;
        }

        var ratio = T.Sqrt(nonGaussianity) / srGap * z;
        return T.One + ratio * ratio;
    }
}

/// <summary>
/// Minimum Track Record Length (MinTRL) — Bailey &amp; López de Prado (2012).
/// The smallest sample length T* such that an observed Sharpe ratio is
/// statistically significantly greater than a benchmark, accounting for
/// non-normality (skewness, raw kurtosis).
/// </summary>
/// <remarks>
/// <para>
/// Reference: Bailey, D. H. &amp; López de Prado, M. (2012). "The Sharpe
/// Ratio Efficient Frontier." Journal of Risk, 15(2), 13–44.
/// </para>
/// <para>
/// Formula:
/// <c>T* = 1 + (1 − γ₃·SR + (γ₄−1)/4 · SR²) · (z_α / (SR − SR*))²</c>
/// where SR is the observed Sharpe and SR* the benchmark (typically 0), both at
/// the per-period frequency of the returns, γ₃ is sample skewness, γ₄ is raw
/// (Pearson) sample kurtosis (3 for a normal distribution), and z_α is the
/// standard normal quantile at significance level α (the closed form holds for confidence 1 − α above one half;
/// at α ≥ 1/2 a single observation already attains PSR ≥ 1 − α, so the minimum is one observation). The inputs are annualized
/// Sharpe ratios; they are divided by √tradingDaysPerYear before the formula is
/// applied, as the formula requires non-annualized Sharpe ratios.
/// </para>
/// <para>
/// MinTRL is the inverse of the Probabilistic Sharpe Ratio at a specified benchmark: PSR asks "is the
/// observed value significant given my sample length?", MinTRL asks "how
/// long must my sample be for this value to become significant?". Use
/// MinTRL when planning a new strategy or evaluating whether an existing
/// track record is long enough to support claims of skill.
/// </para>
/// <para>
/// Legacy facade: delegates to <see cref="MinimumTrackRecordLength{T}"/> at <c>T = double</c>.
/// </para>
/// </remarks>
[Obsolete(
    "Use MinimumTrackRecordLength<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class MinimumTrackRecordLength
{
    /// <summary>
    /// Computes the minimum track record length required for the observed
    /// Sharpe to be statistically distinguishable from the benchmark.
    /// </summary>
    /// <param name="observedSharpe">Observed annualized Sharpe ratio.</param>
    /// <param name="benchmarkSharpe">Annualized benchmark Sharpe (typically 0). Default 0.</param>
    /// <param name="skewness">Sample skewness. Default 0 (Gaussian).</param>
    /// <param name="kurtosis">Raw (Pearson) sample kurtosis (3 = normal). Default 3.</param>
    /// <param name="significanceLevel">Significance α in (0, 1). Default 0.05 (95% confidence). For α ≥ 1/2 (confidence 1 − α at most one half) the result is one observation when the observed Sharpe exceeds the benchmark: PSR already equals 1/2 at a single observation, so eq. 12's minimum is 1.</param>
    /// <param name="tradingDaysPerYear">Return observations per year; the annualized Sharpe ratios are divided by its square root to obtain the per-period Sharpe ratios the formula requires. Must be positive. Default 252.</param>
    /// <returns>Minimum track record length expressed as a number of observations. Returns <see cref="double.PositiveInfinity"/> if the observed Sharpe is at or below the benchmark.</returns>
    public static double Compute(
        double observedSharpe,
        double benchmarkSharpe = 0.0,
        double skewness = 0.0,
        double kurtosis = 3.0,
        double significanceLevel = 0.05,
        int tradingDaysPerYear = 252)
        => MinimumTrackRecordLength<double>.Compute(
            observedSharpe, benchmarkSharpe, skewness, kurtosis, significanceLevel, tradingDaysPerYear);
}
