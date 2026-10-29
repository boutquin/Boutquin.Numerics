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
/// zero Pearson correlation. The population distance correlation is zero if
/// and only if the random variables are independent under finite first-moment
/// assumptions. The returned finite-sample statistic does not establish
/// independence of the underlying population.
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
/// Tier B: Fully transcendental computation. Half inputs use double working arithmetic
/// for distance centering and accumulation; the final correlation is rounded to Half.
/// </para>
/// </remarks>
public static class DistanceCorrelation<T>
    where T : IFloatingPointIeee754<T>
{
    /// <summary>Sample distance correlation in [0, 1].</summary>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths.</exception>
    public static T Compute(ReadOnlySpan<T> x, ReadOnlySpan<T> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException(
                $"Spans must have equal length (x.Length = {x.Length}, y.Length = {y.Length}).",
                nameof(y));
        }

        ValidateFinite(x, nameof(x));
        ValidateFinite(y, nameof(y));

        var n = x.Length;
        if (n < 3)
        {
            // Library minimum-sample convention: return zero for n < 3, matching
            // PearsonCorrelation and RankCorrelation. The paper's biased statistic
            // is defined and equals one for two nonconstant paired observations.
            return T.Zero;
        }

        // Half cannot reliably accumulate n² distances or products: even unit terms
        // stop changing the sum at 2048. Promote before normalization and centering,
        // and round only the final correlation back to the caller's type.
        if (typeof(T) == typeof(Half))
        {
            var promotedX = new double[n];
            var promotedY = new double[n];
            for (var i = 0; i < n; i++)
            {
                promotedX[i] = double.CreateChecked(x[i]);
                promotedY[i] = double.CreateChecked(y[i]);
            }

            return T.CreateChecked(DistanceCorrelation<double>.Compute(promotedX, promotedY));
        }

        // Doubly-center each variable once and reuse; the covariance and both variances are
        // inner products over the same two centered matrices (was 4 centerings, now 2).
        var aBarX = DoublyCenter(Normalize(x), n);
        var aBarY = DoublyCenter(Normalize(y), n);
        var dCovXY = MeanProduct(aBarX, aBarY, n);
        var dVarX = MeanProduct(aBarX, aBarX, n);
        var dVarY = MeanProduct(aBarY, aBarY, n);
        if (dVarX == T.Zero || dVarY == T.Zero)
        {
            return T.Zero;
        }

        // dCov² is a V-statistic that is exactly zero when the empirical joint law factorises
        // (e.g. a product design); rounding in the doubly-centred sums can leave it a tiny
        // negative, whose square root would be NaN. Map it to the documented lower bound.
        if (dCovXY <= T.Zero)
        {
            return T.Zero;
        }

        var corr = T.Sqrt(dCovXY / T.Sqrt(dVarX) / T.Sqrt(dVarY));
        if (corr > T.One)
        {
            return T.One;
        }

        return corr;
    }

    private static T[] Normalize(ReadOnlySpan<T> values)
    {
        var minimum = values[0];
        var maximum = values[0];
        for (var i = 1; i < values.Length; i++)
        {
            minimum = T.Min(minimum, values[i]);
            maximum = T.Max(maximum, values[i]);
        }

        // Remove the common offset before scaling so representable differences survive.
        // Halving each endpoint first keeps the midpoint finite even at opposite extremes.
        var two = T.One + T.One;
        var center = minimum / two + maximum / two;
        var normalized = new T[values.Length];
        var scale = T.Zero;
        for (var i = 0; i < values.Length; i++)
        {
            normalized[i] = values[i] - center;
            scale = T.Max(scale, T.Abs(normalized[i]));
        }

        if (scale != T.Zero)
        {
            for (var i = 0; i < values.Length; i++)
            {
                normalized[i] /= scale;
            }
        }

        return normalized;
    }

    private static void ValidateFinite(ReadOnlySpan<T> values, string parameterName)
    {
        for (var i = 0; i < values.Length; i++)
        {
            if (!T.IsFinite(values[i]))
            {
                throw new ArgumentOutOfRangeException(
                    parameterName, values[i], "Distance-correlation observations must be finite.");
            }
        }
    }

    // Mean of the element-wise product of two doubly-centered distance matrices:
    // the distance covariance (a == x-centered, b == y-centered) or a distance variance
    // (a == b). Identical arithmetic and summation order to the previous separate methods.
    private static T MeanProduct(T[,] a, T[,] b, int n)
    {
        var sum = T.Zero;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                sum += a[i, j] * b[i, j];
            }
        }

        var nVal = T.CreateChecked(n);
        return sum / nVal / nVal;
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

        grand /= nVal;
        grand /= nVal;

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
/// zero Pearson correlation. The population distance correlation is zero if
/// and only if the random variables are independent under finite first-moment
/// assumptions. The returned finite-sample statistic does not establish
/// independence of the underlying population.
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
[Obsolete(
    "Use DistanceCorrelation<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class DistanceCorrelation
{
    /// <summary>Sample distance correlation in [0, 1].</summary>
    public static double Compute(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
        => DistanceCorrelation<double>.Compute(x, y);
}
