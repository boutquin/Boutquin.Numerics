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
using Boutquin.Numerics.Internal;

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Heteroskedasticity- and Autocorrelation-Consistent (HAC) variance
/// estimation using the Newey-West estimator with the Bartlett kernel.
/// Estimates the variance of the sample mean with a Bartlett-weighted HAC
/// correction for autocorrelation and heteroskedasticity. HAC consistency is
/// asymptotic under suitable assumptions; finite-sample estimates are not
/// guaranteed to be unbiased.
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
/// kernel. The Bartlett kernel guarantees a non-negative HAC estimate (zero for a constant series).
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
    public static T MeanVariance(ReadOnlySpan<T> series, int lags)
    {
        if (lags < 0 || lags >= series.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(lags), lags, $"Lags must lie in [0, {series.Length - 1}].");
        }

        if (typeof(T) == typeof(Half))
        {
            return MeanVarianceHalf(series, lags);
        }

        var t = series.Length;
        var tVal = T.CreateChecked(t);
        var minimum = series[0];
        var maximum = series[0];
        for (var i = 0; i < t; i++)
        {
            if (!T.IsFinite(series[i]))
            {
                throw new ArgumentOutOfRangeException(nameof(series), series[i], "Series observations must be finite.");
            }

            minimum = T.Min(minimum, series[i]);
            maximum = T.Max(maximum, series[i]);
        }

        var two = T.CreateChecked(2);
        // Center relative to a bounded anchor. Keep the mean offset separate so
        // adding it back to a large absolute level cannot erase small deviations.
        var anchor = minimum / two + maximum / two;
        var meanOffset = T.Zero;
        var correction = T.Zero;
        for (var i = 0; i < t; i++)
        {
            AddCompensated(ref meanOffset, ref correction, (series[i] - anchor) / tVal);
        }

        var deviations = new T[t];
        var scale = T.Zero;
        for (var i = 0; i < t; i++)
        {
            deviations[i] = (series[i] - anchor) - meanOffset;
            scale = T.Max(scale, T.Abs(deviations[i]));
        }

        if (scale == T.Zero)
        {
            return T.Zero;
        }

        // Dimensionless products cannot overflow. Restore the scale only after
        // both sample-count divisions, without ever forming scale squared.
        var hac = T.Zero;
        correction = T.Zero;
        for (var i = 0; i < t; i++)
        {
            deviations[i] /= scale;
            AddCompensated(ref hac, ref correction, deviations[i] * deviations[i]);
        }

        for (var lag = 1; lag <= lags; lag++)
        {
            var gamma = T.Zero;
            var gammaCorrection = T.Zero;
            for (var i = lag; i < t; i++)
            {
                AddCompensated(ref gamma, ref gammaCorrection, deviations[i] * deviations[i - lag]);
            }

            var weight = T.One - T.CreateChecked(lag) / T.CreateChecked(lags + 1);
            AddCompensated(ref hac, ref correction, two * weight * gamma);
        }

        return (hac / tVal * scale) * (scale / tVal);
    }

    /// <summary>
    /// Estimates the long-run covariance of regression moments using the complete
    /// Newey-West (1994), equation (2-2), Bartlett selector and VAR(1) recolouring.
    /// </summary>
    /// <param name="instruments">Rows are observations; columns are instruments Z.</param>
    /// <param name="residuals">Regression residuals u; moments are h = Z u, without additional centering.</param>
    /// <param name="weights">Weights for the scalar pilot. For a leading constant instrument the authors propose (0,1,...,1); callers must supply their choice explicitly.</param>
    /// <param name="pilotLags">Optional nonnegative pilot window for sensitivity analysis; default is floor(4 (T/100)^(2/9)).</param>
    /// <returns>The recoloured long-run covariance S (not divided by T), and the selected truncation lag m.</returns>
    /// <remarks>
    /// Tier A arithmetic computes moments, VAR coefficients, residual covariances and
    /// recolouring in T. Only the scalar bandwidth decision uses double cube-root
    /// arithmetic; its integer result selects Bartlett weights subsequently computed
    /// in T. Singular VAR normal equations, singular recolouring and a zero pilot
    /// spectral sum are rejected rather than silently selecting a different estimator.
    /// A zero weighted residual series selects m=0. No stationarity clipping is applied.
    /// The moment series h = Z u is divided by its largest magnitude before the VAR normal
    /// equations are formed, and the squared scale is restored on the recoloured covariance,
    /// so a representable long-run covariance is not rejected because the raw Gram sum of
    /// squares overflows. The scalar pilot series and its spectral sums stay in the same unit
    /// coordinates, because only their scale-free ratio selects the bandwidth.
    /// </remarks>
    /// <exception cref="ArgumentException">Shapes are incompatible or fewer than two observations are supplied.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An input or residual observation count is nonfinite in T, or the pilot window is negative.</exception>
    /// <exception cref="InvalidOperationException">A required inverse or finite bandwidth/covariance is unavailable.</exception>
    public static (T[,] Covariance, int Lags) LongRunCovariance(
        T[,] instruments, ReadOnlySpan<T> residuals, ReadOnlySpan<T> weights, int? pilotLags = null)
    {
        return LongRunCovarianceScaled(instruments, residuals, weights, pilotLags, T.One);
    }

    private static (T[,] Covariance, int Lags) LongRunCovarianceScaled(
        T[,] instruments, ReadOnlySpan<T> residuals, ReadOnlySpan<T> weights, int? pilotLags, T outputDivisor)
    {
        try
        {
            return LongRunCovarianceCore(instruments, residuals, weights, pilotLags, outputDivisor);
        }
        catch (OverflowException exception)
        {
            throw new InvalidOperationException("The required long-run covariance is not representable in the working type.", exception);
        }
    }

    private static (T[,] Covariance, int Lags) LongRunCovarianceCore(
        T[,] instruments, ReadOnlySpan<T> residuals, ReadOnlySpan<T> weights, int? pilotLags, T outputDivisor)
    {
        ArgumentNullException.ThrowIfNull(instruments);
        var count = instruments.GetLength(0);
        var dimension = instruments.GetLength(1);
        if (count < 2 || dimension == 0 || residuals.Length != count || weights.Length != dimension)
        {
            throw new ArgumentException("Require at least two observations and matching nonempty instrument, residual and weight dimensions.");
        }

        if (pilotLags < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pilotLags));
        }

        var divisor = T.CreateChecked(count - 1);
        if (!T.IsFinite(divisor) || divisor <= T.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(instruments), "The residual observation count must be finite and positive in the working type.");
        }

        var h = new T[count, dimension];
        for (var j = 0; j < dimension; j++)
        {
            if (!T.IsFinite(weights[j]))
            {
                throw new ArgumentOutOfRangeException(nameof(weights));
            }

            for (var t = 0; t < count; t++)
            {
                if (!T.IsFinite(instruments[t, j]) || !T.IsFinite(residuals[t]))
                {
                    throw new ArgumentOutOfRangeException(nameof(instruments), "Instruments and residuals must be finite.");
                }

                h[t, j] = instruments[t, j] * residuals[t];
            }
        }

        // Fit the VAR in unit-magnitude coordinates g = h / c, where c is the largest magnitude
        // in h. The raw Gram sum of squares overflows (and, for decimal, loses its resolution)
        // long before the long-run variance does; dimensionless moments cannot overflow. The
        // estimator is homogeneous of degree two in h, so c^2 is restored on the recoloured
        // covariance below, as MeanVariance restores its scale.
        var momentScale = T.Zero;
        for (var t = 0; t < count; t++)
        {
            for (var j = 0; j < dimension; j++)
            {
                momentScale = T.Max(momentScale, T.Abs(h[t, j]));
            }
        }

        if (momentScale == T.Zero)
        {
            momentScale = T.One;
        }

        var g = new T[count, dimension];
        for (var t = 0; t < count; t++)
        {
            for (var j = 0; j < dimension; j++)
            {
                g[t, j] = h[t, j] / momentScale;
            }
        }

        var gram = new T[dimension, dimension];
        var cross = new T[dimension, dimension];
        for (var i = 0; i < dimension; i++)
        {
            for (var j = 0; j < dimension; j++)
            {
                for (var t = 1; t < count; t++)
                {
                    gram[i, j] += g[t - 1, i] * g[t - 1, j];
                    cross[i, j] += g[t, i] * g[t - 1, j];
                }
            }
        }

        var a = Multiply(cross, Invert(gram));
        var e = new T[count - 1, dimension];
        var weighted = new T[count - 1];
        for (var t = 1; t < count; t++)
        {
            for (var i = 0; i < dimension; i++)
            {
                var value = g[t, i];
                for (var j = 0; j < dimension; j++)
                {
                    value -= a[i, j] * g[t - 1, j];
                }

                e[t - 1, i] = value;

                // The pilot series stays in unit coordinates: only the scale-free ratio s1 / s0 is consumed,
                // so restoring momentScale here would square it into the spectral sums for no benefit.
                weighted[t - 1] += weights[i] * value;
            }
        }

        var n = pilotLags ?? (int)Math.Floor(4.0 * Math.Pow(count / 100.0, 2.0 / 9.0));
        var s0 = T.Zero;
        var s1 = T.Zero;
        var sigma0 = T.Zero;
        for (var lag = 0; lag <= Math.Min(n, count - 2); lag++)
        {
            var sigma = T.Zero;
            for (var t = lag; t < count - 1; t++)
            {
                // Divide one factor first: the raw product can exceed the range while its quotient does not.
                sigma += (weighted[t] / divisor) * weighted[t - lag];
            }

            if (lag == 0)
            {
                sigma0 = sigma;
                s0 = sigma;
            }
            else
            {
                s0 += T.CreateChecked(2) * sigma;
                s1 += T.CreateChecked(2) * T.CreateChecked(lag) * sigma;
            }
        }

        if (!T.IsFinite(s0) || !T.IsFinite(s1) || (s0 == T.Zero && sigma0 != T.Zero))
        {
            throw new InvalidOperationException("The pilot spectral ratio is undefined.");
        }

        var bandwidth = sigma0 == T.Zero ? 0.0
            : 1.1447 * Math.Pow(Math.Abs(double.CreateChecked(s1 / s0)), 2.0 / 3.0) * Math.Cbrt(count);
        if (!double.IsFinite(bandwidth) || bandwidth >= int.MaxValue)
        {
            throw new InvalidOperationException("The selected bandwidth is not representable.");
        }

        var m = (int)Math.Floor(bandwidth);
        var star = new T[dimension, dimension];
        for (var i = 0; i < dimension; i++)
        {
            for (var j = 0; j < dimension; j++)
            {
                for (var t = 0; t < count - 1; t++)
                {
                    star[i, j] += e[t, i] * e[t, j] / divisor;
                }

                for (var lag = 1; lag <= Math.Min(m, count - 2); lag++)
                {
                    var weight = T.One - T.CreateChecked(lag) / T.CreateChecked(m + 1);
                    for (var t = lag; t < count - 1; t++)
                    {
                        star[i, j] += weight * (e[t, i] * e[t - lag, j] + e[t, j] * e[t - lag, i]) / divisor;
                    }
                }
            }
        }

        var inverseArgument = new T[dimension, dimension];
        for (var i = 0; i < dimension; i++)
        {
            for (var j = 0; j < dimension; j++)
            {
                inverseArgument[i, j] = (i == j ? T.One : T.Zero) - a[i, j];
            }
        }

        var recolour = Invert(inverseArgument);
        var left = Multiply(recolour, star);
        var covariance = new T[dimension, dimension];
        for (var i = 0; i < dimension; i++)
        {
            for (var j = 0; j < dimension; j++)
            {
                var unitScale = T.Zero;
                for (var k = 0; k < dimension; k++)
                {
                    unitScale += left[i, k] * recolour[j, k];
                }

                covariance[i, j] = ScaledArithmetic<T>.ProductQuotient(momentScale, momentScale, unitScale, outputDivisor);
                if (!T.IsFinite(covariance[i, j]))
                {
                    throw new InvalidOperationException("The recoloured covariance is not finite.");
                }
            }
        }

        return (covariance, m);
    }

    /// <summary>
    /// Computes variance of a scalar sample mean using the complete data-dependent
    /// Newey-West (1994) prewhitened and recoloured estimator, S/T.
    /// </summary>
    /// <param name="series">At least two finite observations; the count must be finite in T. Constant series with a valid count return zero.</param>
    /// <param name="pilotLags">Optional pilot window for sensitivity analysis.</param>
    /// <returns>The estimated variance of the mean.</returns>
    public static T MeanVarianceAutomatic(ReadOnlySpan<T> series, int? pilotLags = null)
    {
        if (series.Length < 2)
        {
            throw new ArgumentException("At least two observations are required.", nameof(series));
        }

        if (pilotLags < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pilotLags));
        }

        var offset = T.Zero;
        var count = T.CreateChecked(series.Length);
        if (!T.IsFinite(count) || count <= T.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(series), "The observation count must be finite and positive in the working type.");
        }

        for (var i = 0; i < series.Length; i++)
        {
            if (!T.IsFinite(series[i]))
            {
                throw new ArgumentOutOfRangeException(nameof(series));
            }

            offset += (series[i] - series[0]) / count;
        }

        var residuals = new T[series.Length];
        var instruments = new T[series.Length, 1];
        var nonzero = false;
        for (var i = 0; i < series.Length; i++)
        {
            residuals[i] = (series[i] - series[0]) - offset;
            instruments[i, 0] = T.One;
            nonzero |= residuals[i] != T.Zero;
        }

        return nonzero ? LongRunCovarianceScaled(instruments, residuals, [T.One], pilotLags, count).Covariance[0, 0] : T.Zero;
    }

    private static T[,] Multiply(T[,] left, T[,] right)
    {
        var n = left.GetLength(0);
        var result = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                for (var k = 0; k < n; k++)
                {
                    result[i, j] += left[i, k] * right[k, j];
                }
            }
        }

        return result;
    }

    private static T[,] Invert(T[,] matrix)
    {
        try
        {
            return InvertCore(matrix);
        }
        catch (OverflowException exception)
        {
            throw new InvalidOperationException(
                "The VAR or recolouring matrix is rank-deficient or numerically noninvertible in the working type.", exception);
        }
    }

    // Equilibrate both axes before judging rank: a small pivot in absolute
    // units may merely reflect a valid instrument's different measurement units.
    private static void RequireFullRank(T[,] matrix)
    {
        var n = matrix.GetLength(0);
        var a = (T[,])matrix.Clone();
        for (var i = 0; i < n; i++)
        {
            var scale = T.Zero;
            for (var j = 0; j < n; j++)
            {
                scale = T.Max(scale, T.Abs(a[i, j]));
            }
            if (scale == T.Zero || !T.IsFinite(scale))
            {
                throw new InvalidOperationException("The VAR or recolouring matrix is singular or nonfinite.");
            }
            for (var j = 0; j < n; j++)
            {
                a[i, j] /= scale;
            }
        }
        for (var j = 0; j < n; j++)
        {
            var scale = T.Zero;
            for (var i = 0; i < n; i++)
            {
                scale = T.Max(scale, T.Abs(a[i, j]));
            }
            if (scale == T.Zero)
            {
                throw new InvalidOperationException("The VAR or recolouring matrix is singular.");
            }
            for (var i = 0; i < n; i++)
            {
                a[i, j] /= scale;
            }
        }
        var epsilon = T.CreateChecked(typeof(T) == typeof(decimal) ? 1e-28
            : typeof(T) == typeof(double) ? 2.2204460492503131e-16
            : typeof(T) == typeof(float) ? 1.1920928955078125e-7 : 0.0009765625);
        var tolerance = T.CreateChecked(16 * n) * epsilon;
        for (var k = 0; k < n; k++)
        {
            var pivot = k;
            for (var i = k + 1; i < n; i++)
            {
                if (T.Abs(a[i, k]) > T.Abs(a[pivot, k]))
                {
                    pivot = i;
                }
            }
            if (T.Abs(a[pivot, k]) <= tolerance)
            {
                throw new InvalidOperationException("The VAR or recolouring matrix is numerically rank-deficient.");
            }
            for (var j = k; j < n; j++)
            {
                (a[k, j], a[pivot, j]) = (a[pivot, j], a[k, j]);
            }
            for (var i = k + 1; i < n; i++)
            {
                var factor = a[i, k] / a[k, k];
                for (var j = k + 1; j < n; j++)
                {
                    a[i, j] -= factor * a[k, j];
                }
            }
        }
    }

    private static T[,] InvertCore(T[,] matrix)
    {
        var n = matrix.GetLength(0);
        var a = (T[,])matrix.Clone();
        var inverse = new T[n, n];
        RequireFullRank(matrix);
        for (var i = 0; i < n; i++)
        {
            inverse[i, i] = T.One;
        }

        for (var k = 0; k < n; k++)
        {
            var pivot = k;
            for (var i = k + 1; i < n; i++)
            {
                if (T.Abs(a[i, k]) > T.Abs(a[pivot, k]))
                {
                    pivot = i;
                }
            }

            if (a[pivot, k] == T.Zero || !T.IsFinite(a[pivot, k]))
            {
                throw new InvalidOperationException("The VAR or recolouring matrix is singular or nonfinite.");
            }

            for (var j = 0; j < n; j++)
            {
                (a[k, j], a[pivot, j]) = (a[pivot, j], a[k, j]);
                (inverse[k, j], inverse[pivot, j]) = (inverse[pivot, j], inverse[k, j]);
            }

            var scale = a[k, k];
            for (var j = 0; j < n; j++)
            {
                a[k, j] /= scale;
                inverse[k, j] /= scale;
            }

            for (var i = 0; i < n; i++)
            {
                if (i == k)
                {
                    continue;
                }

                var factor = a[i, k];
                for (var j = 0; j < n; j++)
                {
                    a[i, j] -= factor * a[k, j];
                    inverse[i, j] -= factor * inverse[k, j];
                }
            }
        }

        return inverse;
    }

    private static void AddCompensated(ref T sum, ref T correction, T value)
    {
        var adjusted = value - correction;
        var next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }

    private static T MeanVarianceHalf(ReadOnlySpan<T> series, int lags)
    {
        var sum = 0.0;
        for (var i = 0; i < series.Length; i++)
        {
            if (!T.IsFinite(series[i]))
            {
                throw new ArgumentOutOfRangeException(nameof(series), series[i], "Series observations must be finite.");
            }

            sum += double.CreateChecked(series[i]);
        }

        var mean = sum / series.Length;
        var deviations = new double[series.Length];
        var gamma0 = 0.0;
        for (var i = 0; i < series.Length; i++)
        {
            deviations[i] = double.CreateChecked(series[i]) - mean;
            gamma0 += deviations[i] * deviations[i];
        }

        var hac = gamma0 / series.Length;
        for (var lag = 1; lag <= lags; lag++)
        {
            var gamma = 0.0;
            for (var i = lag; i < series.Length; i++)
            {
                gamma += deviations[i] * deviations[i - lag];
            }

            hac += 2.0 * (1.0 - (double)lag / (lags + 1)) * gamma / series.Length;
        }

        return T.CreateChecked(hac / series.Length);
    }

    /// <summary>
    /// Sample-size pilot-window heuristic ⌊4·(T/100)^(2/9)⌋, capped below the sample length.
    /// This is the pilot window used in Newey-West (1994), not the complete
    /// data-dependent selector with prewhitening and estimated spectral moments.
    /// </summary>
    public static int AutomaticLags(int t)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(t);
        return Math.Min(t - 1, Math.Max(1, (int)Math.Floor(4.0 * Math.Pow(t / 100.0, 2.0 / 9.0))));
    }
}

/// <summary>
/// Heteroskedasticity- and Autocorrelation-Consistent (HAC) variance
/// estimation using the Newey-West estimator with the Bartlett kernel.
/// Estimates the variance of the sample mean with a Bartlett-weighted HAC
/// correction for autocorrelation and heteroskedasticity. HAC consistency is
/// asymptotic under suitable assumptions; finite-sample estimates are not
/// guaranteed to be unbiased.
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
/// kernel. The Bartlett kernel guarantees a non-negative HAC estimate (zero for a constant series).
/// </para>
/// <para>
/// Legacy facade: delegates to <see cref="NeweyWestVariance{T}"/> at <c>T = decimal</c>.
/// </para>
/// </remarks>
[Obsolete(
    "Use NeweyWestVariance<decimal> instead (the facade preserves the original decimal precision; use <double> deliberately if you want IEEE precision). This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class NeweyWestVariance
{
    /// <summary>
    /// Computes the Newey-West HAC variance of the sample mean of <paramref name="series"/>.
    /// </summary>
    /// <param name="series">Input series.</param>
    /// <param name="lags">Truncation lag L. Must satisfy 0 ≤ L &lt; series.Length.</param>
    /// <returns>HAC variance estimate (scaled by 1/T to match Var(mean)).</returns>
    public static decimal MeanVariance(ReadOnlySpan<decimal> series, int lags)
        => NeweyWestVariance<decimal>.MeanVariance(series, lags);

    /// <summary>
    /// Sample-size pilot-window heuristic ⌊4·(T/100)^(2/9)⌋, capped below the sample length.
    /// This is the pilot window used in Newey-West (1994), not the complete
    /// data-dependent selector with prewhitening and estimated spectral moments.
    /// </summary>
    public static int AutomaticLags(int t)
        => NeweyWestVariance<decimal>.AutomaticLags(t);
}
