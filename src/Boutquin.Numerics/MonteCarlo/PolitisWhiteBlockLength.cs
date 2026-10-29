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

namespace Boutquin.Numerics.MonteCarlo;

/// <summary>
/// Politis-White (2004) data-driven optimal block length selection for
/// the stationary bootstrap. Uses a flat-top kernel estimator of the
/// spectral density at frequency zero to balance bias against variance,
/// producing the asymptotically optimal block length under MSE.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Politis, D. N. &amp; White, H. (2004). "Automatic Block-Length
/// Selection for the Dependent Bootstrap." Econometric Reviews, 23(1),
/// 53–70. Patton, A., Politis, D. N. &amp; White, H. (2009), Econometric
/// Reviews, 28(4), 372–375, corrected the stationary-bootstrap constant to
/// <c>D̂_SB = 2·ĝ(0)²</c>; this implementation applies that correction.
/// </para>
/// <para>
/// Algorithm (high level):
/// <list type="number">
/// <item><description>Compute sample autocorrelations ρ̂(k) for k = 1..m_max, with the search horizon m_max = ⌈√T⌉ + K_N of Patton's reference code (the paper gives none).</description></item>
/// <item><description>Find the smallest m̂ such that |ρ̂(m̂+k)| &lt; 2·√(log₁₀T / T) for k = 0..K_N−1, K_N = max(5, √log₁₀T), following the corrected companion selector; when no window converges within the horizon, m̂ is the largest significant lag. M = min(2·m̂, m_max).</description></item>
/// <item><description>Recompute sample covariances R̂(k) on the common M-truncated rows; estimate G = Σ |k|·R̂(k)·w(k/M) and ĝ(0) = Σ R̂(k)·w(k/M) using the flat-top kernel.</description></item>
/// <item><description>Optimal block length: b̂ = (2 G² / D̂_SB)^(1/3) · T^(1/3) with D̂_SB = 2·ĝ(0)², capped above at B_max with Patton's cap B_max = ⌈min(3√T, T/3)⌉.</description></item>
/// </list>
/// </para>
/// <para>
/// <b>Working type: <see cref="double"/>.</b> The single entry point
/// <see cref="EstimateDouble"/> takes a <see cref="double"/> span and returns a
/// <see cref="double"/>; there is no generic or <see cref="decimal"/> overload.
/// The stationary bootstrap resampler in this namespace takes this double-valued
/// length as its mean block length, which it requires to be at least one. The estimate
/// has no lower floor, so floor it with <c>Math.Max(1.0, estimate)</c> before constructing
/// the resampler. This non-generic double-working transcendental
/// selector uses logarithms and cube roots; it is not a Tier A generic algorithm.
/// </para>
/// <para>
/// Convention note: this implementation uses <c>Math.Log10</c> in the threshold
/// coefficient c and the stability window K_N. Politis-White
/// (2004) specify base-10 logarithms for the threshold and window, and
/// <c>arch.bootstrap.optimal_block_length</c> uses them too. The verification
/// suite (<c>BootstrapVerificationTests.PolitisWhiteBlockLength_MatchesPaperReference</c>)
/// asserts agreement to 1e-10 with a NumPy implementation of the corrected
/// author listing on a positively and a negatively
/// autocorrelated series.
/// </para>
/// <para>
/// The lag-window indexing follows the corrected companion selector
/// <c>opt_block_length_REV_dec07.m</c>, beginning at lag m̂. The 2004 paper
/// describes a window beginning at m̂+1. Correlations and autocovariances use
/// the author's common truncated rows, separate column means and sample covariance
/// denominator. There is no lower floor: estimates below one can occur.
/// Constant input returns one. Each lag column is translated by the midpoint of its
/// own extrema and scaled by a power of two of its own spread. Both preserve the
/// mathematical estimator, and they keep a large common offset or a remote extreme
/// from erasing or underflowing the spread of columns that exclude it.
/// </para>
/// </remarks>
public static class PolitisWhiteBlockLength
{
    /// <summary>
    /// Estimates the optimal stationary-bootstrap block length for a double-precision series.
    /// </summary>
    public static double EstimateDouble(ReadOnlySpan<double> series)
    {
        var t = series.Length;
        if (t < 32)
        {
            throw new ArgumentException(
                "Politis-White block length estimation requires at least 32 observations.",
                nameof(series));
        }

        for (var i = 0; i < t; i++)
        {
            if (!double.IsFinite(series[i]))
            {
                throw new ArgumentOutOfRangeException(
                    nameof(series), series[i], "Series observations must be finite.");
            }
        }

        var constant = true;
        for (var i = 1; i < t && constant; i++)
        {
            constant = series[i] == series[0];
        }

        if (constant)
        {
            return 1.0;
        }

        // Each lag column is conditioned by its own extrema (see ConditionColumn), so a
        // remote extreme cannot erase or underflow the spread of columns that exclude it.
        // A pair buffer holds the conditioned lag column in its first half and the
        // conditioned common-row column in its second half, so PairedCovariance with
        // start = lag = rows reads exactly those two columns.
        var buffer = new double[2 * t];

        // Author FIRST STEP: all lag columns use the same truncated rows.
        var kn = Math.Max(5, (int)Math.Ceiling(Math.Sqrt(Math.Log10((double)t))));
        var mMax = (int)Math.Ceiling(Math.Sqrt(t)) + kn;
        var rho = new double[mMax + 1];
        var rows = t - mMax;
        var pair = buffer.AsSpan(0, 2 * rows);
        ConditionColumn(series[mMax..], pair[rows..]);
        for (var k = 1; k <= mMax; k++)
        {
            ConditionColumn(series.Slice(mMax - k, rows), pair[..rows]);
            var covariance = PairedCovariance(pair, rows, rows, out var leftVariance, out var rightVariance);
            rho[k] = covariance / Math.Sqrt(leftVariance * rightVariance);
        }

        // Step 3: corrected companion-selector indexing: the first window of
        // K_N insignificant autocorrelations begins at m̂, not at m̂+1.
        var c = 2.0 * Math.Sqrt(Math.Log10((double)t) / t);

        // Find the smallest lag k whose K_N autocorrelations are all below the
        // threshold c. When such a window exists, the empirical lag is that k. When
        // no window converges within the horizon, fall back to the largest significant
        // lag, as Patton's code does.
        var m = 0;
        for (var k = 1; k <= mMax - kn + 1; k++)
        {
            var allBelow = true;
            for (var j = 0; j < kn; j++)
            {
                if (!(Math.Abs(rho[k + j]) < c))
                {
                    allBelow = false;
                    break;
                }
            }

            if (allBelow)
            {
                m = k;
                break;
            }
        }

        if (m == 0)
        {
            // No window converged, so some lag in the last window is significant.
            for (var k = mMax; k >= 1; k--)
            {
                if (Math.Abs(rho[k]) > c)
                {
                    m = k;
                    break;
                }
            }
        }

        // Politis-White: M = 2·m̂, capped at the search horizon.
        m *= 2;
        m = Math.Min(m, mMax);

        if (m <= 0)
        {
            return 1.0;
        }

        // Author SECOND STEP: M-truncated rows and separate column means. Covariance k
        // equals 2^(leftExponent + exponents[k]) * scaled[k]; the common left exponent
        // cancels in G / g(0), and the rest is aligned to the largest contributing term.
        rows = t - m;
        pair = buffer.AsSpan(0, 2 * rows);
        ConditionColumn(series[m..], pair[rows..]);
        var scaled = new double[m + 1];
        var exponents = new int[m + 1];
        var weights = new double[m + 1];
        var common = int.MinValue;
        for (var k = 0; k <= m; k++)
        {
            var x = (double)k / m;
            weights[k] = x <= 0.5 ? 1.0 : 2.0 * (1.0 - x);
            exponents[k] = ConditionColumn(series.Slice(m - k, rows), pair[..rows]);
            scaled[k] = PairedCovariance(pair, rows, rows, out _, out _);
            if (weights[k] != 0.0 && scaled[k] != 0.0)
            {
                common = Math.Max(common, exponents[k] + Math.ILogB(scaled[k]));
            }
        }

        var g = 0.0;
        var dSpec = 0.0;
        if (common != int.MinValue)
        {
            dSpec = Math.ScaleB(scaled[0], exponents[0] - common);
            for (var k = 1; k <= m; k++)
            {
                if (weights[k] == 0.0)
                {
                    // The kernel vanishes at k = M; its column may hold an extreme that
                    // the alignment would overflow, and it contributes nothing.
                    continue;
                }

                var covariance = Math.ScaleB(scaled[k], exponents[k] - common);
                g += 2.0 * weights[k] * k * covariance;
                dSpec += 2.0 * weights[k] * covariance;
            }
        }

        // Author FINAL STEP has an upper cap but no lower floor.
        var optimal = StationaryOptimum(g, dSpec, t);
        var bMax = Math.Ceiling(Math.Min(3.0 * Math.Sqrt(t), t / 3.0));
        if (optimal > bMax)
        {
            return bMax;
        }

        return optimal;
    }

    // Writes (column - c) / 2^e to conditioned and returns e. The translation c is the
    // midpoint of the column's own extrema (halved before adding so it stays finite),
    // so it cannot erase spread that this column retains; 2^e brings the largest
    // deviation into [1, 2), and dividing by a power of two is exact. Column covariances
    // and variances are therefore 2^e times those of the conditioned column. A constant
    // column is written as zeros and returns zero.
    private static int ConditionColumn(ReadOnlySpan<double> column, Span<double> conditioned)
    {
        var minimum = column[0];
        var maximum = column[0];
        for (var i = 1; i < column.Length; i++)
        {
            minimum = Math.Min(minimum, column[i]);
            maximum = Math.Max(maximum, column[i]);
        }

        var center = minimum / 2.0 + maximum / 2.0;
        var spread = Math.Max(maximum - center, center - minimum);
        if (spread == 0.0)
        {
            conditioned.Clear();
            return 0;
        }

        var exponent = Math.ILogB(spread);
        for (var i = 0; i < column.Length; i++)
        {
            conditioned[i] = Math.ScaleB(column[i] - center, -exponent);
        }

        return exponent;
    }

    private static double PairedCovariance(ReadOnlySpan<double> data, int start, int lag,
        out double leftVariance, out double rightVariance)
    {
        var count = data.Length - start;
        var leftMean = 0.0;
        var rightMean = 0.0;
        for (var i = start; i < data.Length; i++)
        {
            leftMean += data[i];
            rightMean += data[i - lag];
        }

        leftMean /= count;
        rightMean /= count;
        var covariance = 0.0;
        leftVariance = 0.0;
        rightVariance = 0.0;
        for (var i = start; i < data.Length; i++)
        {
            var left = data[i] - leftMean;
            var right = data[i - lag] - rightMean;
            covariance += left * right;
            leftVariance += left * left;
            rightVariance += right * right;
        }

        leftVariance /= count - 1;
        rightVariance /= count - 1;
        return covariance / (count - 1);
    }

    // Shared population/plug-in formula: PW (2004) eq. (6), with the PPW (2009)
    // correction D_SB = 2 g(0)^2. The caller applies finite-sample bounds.
    internal static double StationaryOptimum(double covarianceMoment, double spectralSum, int sampleLength)
    {
        var dSb = 2.0 * spectralSum * spectralSum;
        return Math.Pow(2.0 * covarianceMoment * covarianceMoment / dSb * sampleLength, 1.0 / 3.0);
    }
}
