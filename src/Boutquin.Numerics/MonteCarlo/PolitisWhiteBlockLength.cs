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
/// 53–70. Patton, A., Politis, D. N. &amp; White, H. (2009) provided a
/// correction to one constant; we apply the corrected formula.
/// </para>
/// <para>
/// Algorithm (high level):
/// <list type="number">
/// <item><description>Compute sample autocorrelations ρ̂(k) for k = 1..K_max.</description></item>
/// <item><description>Find the smallest K such that |ρ̂(K+1)| &lt; 2·√(log T / T) for the next K_lag consecutive lags.</description></item>
/// <item><description>Estimate G = Σ k·|ρ̂(k)| · w(k/M) and D = Σ ρ̂(k) · w(k/M) using the flat-top kernel.</description></item>
/// <item><description>Optimal block length: b̂ = (2 G² / D²)^(1/3) · T^(1/3).</description></item>
/// </list>
/// </para>
/// <para>
/// Tier A: Arithmetic autocorrelation operations on floating-point types.
/// </para>
/// </remarks>
public static class PolitisWhiteBlockLength
{
    /// <summary>
    /// Estimates the optimal stationary-bootstrap block length for the given series.
    /// </summary>
    /// <param name="series">Input time series (length must be ≥ 32).</param>
    /// <returns>Estimated optimal block length, clamped to [1, T/2].</returns>
    public static double Estimate<T>(T[] series)
        where T : IFloatingPoint<T>
    {
        ArgumentNullException.ThrowIfNull(series);
        if (series.Length < 32)
        {
            throw new ArgumentException(
                "Politis-White block length estimation requires at least 32 observations.",
                nameof(series));
        }

        var doubles = new double[series.Length];
        for (var i = 0; i < series.Length; i++)
        {
            doubles[i] = double.CreateChecked(series[i]);
        }

        return EstimateDouble(doubles);
    }

    /// <summary>
    /// Estimates the optimal stationary-bootstrap block length for a double-precision series.
    /// </summary>
    public static double EstimateDouble(double[] series)
    {
        ArgumentNullException.ThrowIfNull(series);
        var t = series.Length;
        if (t < 32)
        {
            throw new ArgumentException(
                "Politis-White block length estimation requires at least 32 observations.",
                nameof(series));
        }

        // Step 1: sample mean and variance.
        var mean = 0.0;
        for (var i = 0; i < t; i++)
        {
            mean += series[i];
        }

        mean /= t;

        var variance = 0.0;
        for (var i = 0; i < t; i++)
        {
            var d = series[i] - mean;
            variance += d * d;
        }

        variance /= t;
        if (variance <= 0.0)
        {
            return 1.0;
        }

        // Step 2: autocorrelations up to K_max ≈ ⌈log10(T)⌉ * 5 (Politis-White recommendation).
        var kMax = Math.Max(5, (int)Math.Ceiling(Math.Log10(t) * 5.0));
        kMax = Math.Min(kMax, t / 2);

        var rho = new double[kMax + 1];
        rho[0] = 1.0;
        for (var k = 1; k <= kMax; k++)
        {
            var acc = 0.0;
            for (var i = 0; i < t - k; i++)
            {
                acc += (series[i] - mean) * (series[i + k] - mean);
            }

            rho[k] = acc / (t * variance);
        }

        // Step 3: empirical lag length M — first K such that |ρ̂(K)| < c·√(log T / T)
        // for K_n = max(5, ceil(log10(T))) consecutive subsequent lags.
        var c = 2.0 * Math.Sqrt(Math.Log10((double)t) / t);
        var kn = Math.Max(5, (int)Math.Ceiling(Math.Log10((double)t)));

        var m = 1;
        for (var k = 1; k <= kMax - kn; k++)
        {
            var allBelow = true;
            for (var j = 1; j <= kn; j++)
            {
                if (Math.Abs(rho[k + j]) >= c)
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

            m = k;
        }

        // Politis-White recommend M = 2 * empirical lag.
        m *= 2;
        m = Math.Min(m, kMax);

        // Step 4: G and D using flat-top kernel.
        // w(x) = 1 if |x| ≤ 0.5, 2(1−|x|) if 0.5 < |x| ≤ 1, 0 otherwise.
        var g = 0.0;
        var dSpec = 0.0;
        for (var k = 1; k <= m; k++)
        {
            var x = (double)k / m;
            double w;
            if (x <= 0.5)
            {
                w = 1.0;
            }
            else if (x <= 1.0)
            {
                w = 2.0 * (1.0 - x);
            }
            else
            {
                w = 0.0;
            }

            g += w * k * Math.Abs(rho[k]);
            dSpec += w * rho[k];
        }

        g *= 2.0; // double-sided sum.
        dSpec = 2.0 * dSpec * variance + variance; // include lag-0 (= variance) once.
        dSpec *= dSpec;
        if (dSpec <= 1e-18)
        {
            return 1.0;
        }

        var optimal = Math.Pow(2.0 * g * g / dSpec * t, 1.0 / 3.0);
        // Clamp to [1, T/2].
        if (!double.IsFinite(optimal) || optimal < 1.0)
        {
            return 1.0;
        }

        if (optimal > t / 2.0)
        {
            return t / 2.0;
        }

        return optimal;
    }
}
