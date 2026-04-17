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
/// Heteroskedasticity- and Autocorrelation-Consistent (HAC) variance
/// estimation using the Newey-West estimator with the Bartlett kernel.
/// Produces unbiased standard errors when residuals are autocorrelated
/// and/or heteroskedastic — both of which are universal properties of
/// daily financial return series.
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <remarks>
/// <para>
/// References:
/// <list type="bullet">
/// <item><description>Newey, W. K. &amp; West, K. D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." Econometrica, 55(3), 703–708.</description></item>
/// <item><description>Newey, W. K. &amp; West, K. D. (1994). "Automatic Lag Selection in Covariance Matrix Estimation." Review of Economic Studies, 61(4), 631–653.</description></item>
/// </list>
/// </para>
/// <para>
/// Estimator: <c>HAC = γ₀ + 2 Σ_{ℓ=1}^L w(ℓ, L) γ_ℓ</c> where γ_ℓ is the
/// sample autocovariance at lag ℓ and w(ℓ, L) = 1 − ℓ/(L+1) is the Bartlett
/// kernel. The Bartlett kernel guarantees a positive HAC estimate.
/// </para>
/// <para>
/// Tier A: Arithmetic-only computation.
/// </para>
/// </remarks>
public static class NeweyWestVariance<T>
    where T : IFloatingPoint<T>
{
    /// <summary>
    /// Computes the Newey-West HAC variance of the sample mean of <paramref name="series"/>.
    /// </summary>
    /// <param name="series">Input series.</param>
    /// <param name="lags">Truncation lag L. Must satisfy 0 ≤ L &lt; series.Length.</param>
    /// <returns>HAC variance estimate (scaled by 1/T to match Var(mean)).</returns>
    public static T MeanVariance(T[] series, int lags)
    {
        ArgumentNullException.ThrowIfNull(series);
        if (lags < 0 || lags >= series.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(lags), lags, $"Lags must lie in [0, {series.Length - 1}].");
        }

        var t = series.Length;
        var tVal = T.CreateChecked(t);
        var mean = T.Zero;
        for (var i = 0; i < t; i++)
        {
            mean += series[i];
        }

        mean /= tVal;

        var deviations = new T[t];
        for (var i = 0; i < t; i++)
        {
            deviations[i] = series[i] - mean;
        }

        var gamma0 = T.Zero;
        for (var i = 0; i < t; i++)
        {
            gamma0 += deviations[i] * deviations[i];
        }

        gamma0 /= tVal;

        var hac = gamma0;
        var two = T.CreateChecked(2);
        for (var lag = 1; lag <= lags; lag++)
        {
            var gamma = T.Zero;
            for (var i = lag; i < t; i++)
            {
                gamma += deviations[i] * deviations[i - lag];
            }

            gamma /= tVal;
            var weight = T.One - T.CreateChecked(lag) / T.CreateChecked(lags + 1);
            hac += two * weight * gamma;
        }

        return hac / tVal;
    }

    /// <summary>
    /// Newey-West (1994) automatic lag selection. Returns ⌊4·(T/100)^(2/9)⌋,
    /// the recommended default truncation lag.
    /// </summary>
    public static int AutomaticLags(int t)
        => Math.Max(1, (int)Math.Floor(4.0 * Math.Pow(t / 100.0, 2.0 / 9.0)));
}

/// <summary>
/// Heteroskedasticity- and Autocorrelation-Consistent (HAC) variance
/// estimation using the Newey-West estimator with the Bartlett kernel.
/// Produces unbiased standard errors when residuals are autocorrelated
/// and/or heteroskedastic — both of which are universal properties of
/// daily financial return series.
/// </summary>
/// <remarks>
/// <para>
/// References:
/// <list type="bullet">
/// <item><description>Newey, W. K. &amp; West, K. D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." Econometrica, 55(3), 703–708.</description></item>
/// <item><description>Newey, W. K. &amp; West, K. D. (1994). "Automatic Lag Selection in Covariance Matrix Estimation." Review of Economic Studies, 61(4), 631–653.</description></item>
/// </list>
/// </para>
/// <para>
/// Estimator: <c>HAC = γ₀ + 2 Σ_{ℓ=1}^L w(ℓ, L) γ_ℓ</c> where γ_ℓ is the
/// sample autocovariance at lag ℓ and w(ℓ, L) = 1 − ℓ/(L+1) is the Bartlett
/// kernel. The Bartlett kernel guarantees a positive HAC estimate.
/// </para>
/// <para>
/// Legacy facade: delegates to <see cref="NeweyWestVariance{T}"/> at <c>T = decimal</c>.
/// </para>
/// </remarks>
public static class NeweyWestVariance
{
    /// <summary>
    /// Computes the Newey-West HAC variance of the sample mean of <paramref name="series"/>.
    /// </summary>
    /// <param name="series">Input series.</param>
    /// <param name="lags">Truncation lag L. Must satisfy 0 ≤ L &lt; series.Length.</param>
    /// <returns>HAC variance estimate (scaled by 1/T to match Var(mean)).</returns>
    public static decimal MeanVariance(decimal[] series, int lags)
        => NeweyWestVariance<decimal>.MeanVariance(series, lags);

    /// <summary>
    /// Newey-West (1994) automatic lag selection. Returns ⌊4·(T/100)^(2/9)⌋,
    /// the recommended default truncation lag.
    /// </summary>
    public static int AutomaticLags(int t)
        => NeweyWestVariance<decimal>.AutomaticLags(t);
}
