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
/// non-normality (skewness, excess kurtosis).
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <remarks>
/// <para>
/// Reference: Bailey, D. H. &amp; López de Prado, M. (2012). "The Sharpe
/// Ratio Efficient Frontier." Journal of Risk, 15(2), 13–44.
/// arXiv:1205.1480.
/// </para>
/// <para>
/// Formula:
/// <c>T* = 1 + (1 − γ₃·SR + (γ₄−1)/4 · SR²) · (z_α / (SR − SR*))²</c>
/// where SR is the observed annualized Sharpe, SR* is the benchmark
/// (typically 0), γ₃ is sample skewness, γ₄ is sample kurtosis, and z_α
/// is the standard normal quantile at significance level α.
/// </para>
/// <para>
/// MinTRL is the inverse of the Deflated Sharpe Ratio: DSR asks "is the
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
    /// Computes the minimum track record length required for the observed
    /// Sharpe to be statistically distinguishable from the benchmark.
    /// </summary>
    /// <param name="observedSharpe">Observed annualized Sharpe ratio.</param>
    /// <param name="benchmarkSharpe">Benchmark Sharpe (typically 0). Default 0.</param>
    /// <param name="skewness">Sample skewness. Default 0 (Gaussian).</param>
    /// <param name="kurtosis">Sample kurtosis (3 = normal). Default 3.</param>
    /// <param name="significanceLevel">Significance α in (0, 1). Default 0.05 (95% confidence).</param>
    /// <param name="tradingDaysPerYear">Trading days per year for time-unit conversion. Default 252.</param>
    /// <returns>Minimum track record length in years. Returns <c>T.PositiveInfinity</c> if the observed Sharpe is at or below the benchmark.</returns>
    public static T Compute(
        T observedSharpe,
        T benchmarkSharpe,
        T skewness,
        T kurtosis,
        T significanceLevel,
        int tradingDaysPerYear)
    {
        if (significanceLevel <= T.Zero || significanceLevel >= T.One)
        {
            throw new ArgumentOutOfRangeException(nameof(significanceLevel), significanceLevel, "Significance level must lie in (0, 1).");
        }

        if (tradingDaysPerYear <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tradingDaysPerYear), tradingDaysPerYear, "Trading days per year must be positive.");
        }

        if (observedSharpe <= benchmarkSharpe)
        {
            return T.PositiveInfinity;
        }

        var z = NormalDistribution<T>.InverseCdf(T.One - significanceLevel);
        var srGap = observedSharpe - benchmarkSharpe;
        var nonGaussianity = T.One - skewness * observedSharpe + (kurtosis - T.One) / T.CreateChecked(4) * observedSharpe * observedSharpe;
        if (nonGaussianity < T.Zero)
        {
            nonGaussianity = T.One;
        }

        var tStarObs = T.One + nonGaussianity * (z * z) / (srGap * srGap);
        return tStarObs / T.CreateChecked(tradingDaysPerYear);
    }
}

/// <summary>
/// Minimum Track Record Length (MinTRL) — Bailey &amp; López de Prado (2012).
/// The smallest sample length T* such that an observed Sharpe ratio is
/// statistically significantly greater than a benchmark, accounting for
/// non-normality (skewness, excess kurtosis).
/// </summary>
/// <remarks>
/// <para>
/// Reference: Bailey, D. H. &amp; López de Prado, M. (2012). "The Sharpe
/// Ratio Efficient Frontier." Journal of Risk, 15(2), 13–44.
/// arXiv:1205.1480.
/// </para>
/// <para>
/// Formula:
/// <c>T* = 1 + (1 − γ₃·SR + (γ₄−1)/4 · SR²) · (z_α / (SR − SR*))²</c>
/// where SR is the observed annualized Sharpe, SR* is the benchmark
/// (typically 0), γ₃ is sample skewness, γ₄ is sample kurtosis, and z_α
/// is the standard normal quantile at significance level α.
/// </para>
/// <para>
/// MinTRL is the inverse of the Deflated Sharpe Ratio: DSR asks "is the
/// observed value significant given my sample length?", MinTRL asks "how
/// long must my sample be for this value to become significant?". Use
/// MinTRL when planning a new strategy or evaluating whether an existing
/// track record is long enough to support claims of skill.
/// </para>
/// <para>
/// Legacy facade: delegates to <see cref="MinimumTrackRecordLength{T}"/> at <c>T = double</c>.
/// </para>
/// </remarks>
public static class MinimumTrackRecordLength
{
    /// <summary>
    /// Computes the minimum track record length required for the observed
    /// Sharpe to be statistically distinguishable from the benchmark.
    /// </summary>
    /// <param name="observedSharpe">Observed annualized Sharpe ratio.</param>
    /// <param name="benchmarkSharpe">Benchmark Sharpe (typically 0). Default 0.</param>
    /// <param name="skewness">Sample skewness. Default 0 (Gaussian).</param>
    /// <param name="kurtosis">Sample kurtosis (3 = normal). Default 3.</param>
    /// <param name="significanceLevel">Significance α in (0, 1). Default 0.05 (95% confidence).</param>
    /// <param name="tradingDaysPerYear">Trading days per year for time-unit conversion. Default 252.</param>
    /// <returns>Minimum track record length in years. Returns <see cref="double.PositiveInfinity"/> if the observed Sharpe is at or below the benchmark.</returns>
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
