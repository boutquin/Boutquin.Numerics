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
/// Distance correlation (Székely, Rizzo &amp; Bakirov 2007). Detects
/// arbitrary nonlinear dependence — including dependencies that yield
/// zero Pearson correlation. dCor(X, Y) = 0 if and only if X and Y are
/// independent (under finite first-moment assumptions).
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <remarks>
/// <para>
/// Reference: Székely, G. J., Rizzo, M. L. &amp; Bakirov, N. K. (2007).
/// "Measuring and Testing Dependence by Correlation of Distances."
/// Annals of Statistics, 35(6), 2769–2794.
/// arXiv:0803.4101.
/// </para>
/// <para>
/// Implementation: O(n²) memory and arithmetic via the doubly-centered
/// distance matrix. For very large samples, use the O(n log n) algorithm
/// of Huo &amp; Székely (2016) — not currently exposed.
/// </para>
/// <para>
/// Tier B: Fully transcendental computation.
/// </para>
/// </remarks>
public static class DistanceCorrelation<T>
    where T : IFloatingPointIeee754<T>
{
    /// <summary>Sample distance correlation in [0, 1].</summary>
    public static T Compute(ReadOnlySpan<T> x, ReadOnlySpan<T> y)
    {
        var n = Math.Min(x.Length, y.Length);
        if (n < 2)
        {
            return T.Zero;
        }

        var dCovXY = DistanceCovariance(x, y, n);
        var dVarX = DistanceVariance(x, n);
        var dVarY = DistanceVariance(y, n);
        var denom = T.Sqrt(dVarX * dVarY);
        var epsilon = T.CreateChecked(1e-18);
        if (denom <= epsilon)
        {
            return T.Zero;
        }

        var corr = T.Sqrt(dCovXY / denom);
        if (corr > T.One)
        {
            return T.One;
        }

        return corr;
    }

    private static T DistanceCovariance(ReadOnlySpan<T> x, ReadOnlySpan<T> y, int n)
    {
        var aBar = DoublyCenter(x, n);
        var bBar = DoublyCenter(y, n);

        var sum = T.Zero;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                sum += aBar[i, j] * bBar[i, j];
            }
        }

        return sum / T.CreateChecked(n * n);
    }

    private static T DistanceVariance(ReadOnlySpan<T> x, int n)
    {
        var aBar = DoublyCenter(x, n);
        var sum = T.Zero;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                sum += aBar[i, j] * aBar[i, j];
            }
        }

        return sum / T.CreateChecked(n * n);
    }

    private static T[,] DoublyCenter(ReadOnlySpan<T> x, int n)
    {
        var a = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                var d = T.Abs(x[i] - x[j]);
                a[i, j] = d;
                a[j, i] = d;
            }
        }

        var rowMean = new T[n];
        var colMean = new T[n];
        var grand = T.Zero;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                rowMean[i] += a[i, j];
                colMean[j] += a[i, j];
                grand += a[i, j];
            }
        }

        var nVal = T.CreateChecked(n);
        for (var i = 0; i < n; i++)
        {
            rowMean[i] /= nVal;
            colMean[i] /= nVal;
        }

        grand /= T.CreateChecked(n * n);

        var aBar = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                aBar[i, j] = a[i, j] - rowMean[i] - colMean[j] + grand;
            }
        }

        return aBar;
    }
}

/// <summary>
/// Distance correlation (Székely, Rizzo &amp; Bakirov 2007). Detects
/// arbitrary nonlinear dependence — including dependencies that yield
/// zero Pearson correlation. dCor(X, Y) = 0 if and only if X and Y are
/// independent (under finite first-moment assumptions).
/// </summary>
/// <remarks>
/// <para>
/// Reference: Székely, G. J., Rizzo, M. L. &amp; Bakirov, N. K. (2007).
/// "Measuring and Testing Dependence by Correlation of Distances."
/// Annals of Statistics, 35(6), 2769–2794.
/// arXiv:0803.4101.
/// </para>
/// <para>
/// Implementation: O(n²) memory and arithmetic via the doubly-centered
/// distance matrix. For very large samples, use the O(n log n) algorithm
/// of Huo &amp; Székely (2016) — not currently exposed.
/// </para>
/// <para>
/// Legacy facade: delegates to <see cref="DistanceCorrelation{T}"/> at <c>T = double</c>.
/// </para>
/// </remarks>
public static class DistanceCorrelation
{
    /// <summary>Sample distance correlation in [0, 1].</summary>
    public static double Compute(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
        => DistanceCorrelation<double>.Compute(x, y);
}
