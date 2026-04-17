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
/// Harrell-Davis percentile estimator. Each order statistic is given a
/// Beta-distribution weight and the percentile is the weighted average of
/// all order statistics. Has substantially lower variance than the
/// linear-interpolation estimator (NumPy's <c>linear</c>) for small
/// samples — particularly important for bootstrap confidence intervals
/// where the bootstrap distribution is itself a small sample.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Harrell, F. E. &amp; Davis, C. E. (1982). "A New Distribution-Free
/// Quantile Estimator." Biometrika, 69(3), 635–640.
/// </para>
/// <para>
/// Weight for the i-th order statistic (1-based): w_i = I(i/n; (n+1)·p, (n+1)·(1−p)) − I((i−1)/n; (n+1)·p, (n+1)·(1−p))
/// where I is the regularized incomplete beta function. This is computed
/// via a numerical incomplete beta evaluation; for stability the weights
/// are normalized to sum to one.
/// </para>
/// <para>
/// Tier A: Uses double internally for Beta weights but accumulates in T.
/// </para>
/// </remarks>
public static class HarrellDavisPercentile
{
    /// <summary>
    /// Computes the Harrell-Davis percentile of a sorted array.
    /// </summary>
    /// <param name="sorted">Sorted (ascending) sample.</param>
    /// <param name="p">Percentile in [0, 1].</param>
    public static T Compute<T>(T[] sorted, double p)
        where T : IFloatingPoint<T>
    {
        ArgumentNullException.ThrowIfNull(sorted);
        if (sorted.Length == 0)
        {
            return T.Zero;
        }

        if (sorted.Length == 1)
        {
            return sorted[0];
        }

        p = Math.Clamp(p, 0.0, 1.0);
        var n = sorted.Length;

        var alpha = (n + 1) * p;
        var beta = (n + 1) * (1 - p);

        var weights = new double[n];
        var weightSum = 0.0;
        var prev = 0.0;
        for (var i = 1; i <= n; i++)
        {
            var current = RegularizedIncompleteBeta((double)i / n, alpha, beta);
            weights[i - 1] = current - prev;
            weightSum += weights[i - 1];
            prev = current;
        }

        if (weightSum <= 0)
        {
            return Percentile.Compute(sorted, T.CreateChecked(p));
        }

        T acc = T.Zero;
        for (var i = 0; i < n; i++)
        {
            acc += sorted[i] * T.CreateChecked(weights[i] / weightSum);
        }

        return acc;
    }

    private static double RegularizedIncompleteBeta(double x, double a, double b)
    {
        if (x <= 0.0)
        {
            return 0.0;
        }

        if (x >= 1.0)
        {
            return 1.0;
        }

        var bt = Math.Exp(LogGamma(a + b) - LogGamma(a) - LogGamma(b)
                          + a * Math.Log(x) + b * Math.Log(1.0 - x));

        if (x < (a + 1.0) / (a + b + 2.0))
        {
            return bt * BetaContinuedFraction(x, a, b) / a;
        }

        return 1.0 - bt * BetaContinuedFraction(1.0 - x, b, a) / b;
    }

    private static double BetaContinuedFraction(double x, double a, double b)
    {
        const int maxIter = 200;
        const double eps = 3e-7;
        const double fpmin = 1e-30;

        var qab = a + b;
        var qap = a + 1.0;
        var qam = a - 1.0;
        var c = 1.0;
        var d = 1.0 - qab * x / qap;
        if (Math.Abs(d) < fpmin)
        {
            d = fpmin;
        }

        d = 1.0 / d;
        var h = d;

        for (var m = 1; m <= maxIter; m++)
        {
            var m2 = 2 * m;
            var aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (Math.Abs(d) < fpmin)
            {
                d = fpmin;
            }

            c = 1.0 + aa / c;
            if (Math.Abs(c) < fpmin)
            {
                c = fpmin;
            }

            d = 1.0 / d;
            h *= d * c;

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (Math.Abs(d) < fpmin)
            {
                d = fpmin;
            }

            c = 1.0 + aa / c;
            if (Math.Abs(c) < fpmin)
            {
                c = fpmin;
            }

            d = 1.0 / d;
            var del = d * c;
            h *= del;
            if (Math.Abs(del - 1.0) < eps)
            {
                return h;
            }
        }

        return h;
    }

    private static double LogGamma(double x)
    {
        double[] coef = [
            76.18009172947146,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.1208650973866179e-2,
            -0.5395239384953e-5,
        ];
        var y = x;
        var t = x + 5.5;
        t -= (x + 0.5) * Math.Log(t);
        var ser = 1.000000000190015;
        for (var j = 0; j < 6; j++)
        {
            y += 1.0;
            ser += coef[j] / y;
        }

        return -t + Math.Log(2.5066282746310005 * ser / x);
    }
}
