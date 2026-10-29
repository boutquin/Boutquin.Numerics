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
/// Generic rank correlation primitives — Spearman rho and Kendall tau.
/// Both are robust to outliers and detect monotonic (not just linear)
/// dependence; the trade-off is lower power than Pearson under the bivariate
/// normal assumption.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="NumericPrecision{T}.Sqrt"/> to support <c>decimal</c> (which lacks
/// <see cref="IRootFunctions{TSelf}"/>).
/// </para>
/// <para>
/// Spearman = Pearson correlation applied to ranks. Tied values receive
/// average ranks (fractional ranking). Returns 0 for fewer than three
/// observations or when either series is constant.
/// </para>
/// <para>
/// Kendall tau-b counts concordant minus discordant pairs, normalized by
/// sqrt((P - T_x)(P - T_y)) where P = n(n-1)/2 and T_x, T_y are tie corrections.
/// tau-b is the standard variant when ties are present; reduces to tau-a when
/// no ties exist.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class RankCorrelation<T>
    where T : IFloatingPoint<T>
{
    /// <summary>Spearman rank correlation coefficient rho in [-1, 1].</summary>
    public static T Spearman(ReadOnlySpan<T> x, ReadOnlySpan<T> y)
    {
        ValidateFinite(x, nameof(x));
        ValidateFinite(y, nameof(y));
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Series must have the same length.", nameof(y));
        }

        var n = x.Length;
        if (n < 3)
        {
            return T.Zero;
        }

        var rx = AverageRanks(x, n);
        var ry = AverageRanks(y, n);

        return PearsonCorrelation<T>.Compute(rx, ry);
    }

    /// <summary>
    /// Kendall tau-b (tie-corrected). Returns 0 for fewer than two
    /// observations or when all pairs are tied in either coordinate.
    /// </summary>
    public static T KendallTauB(ReadOnlySpan<T> x, ReadOnlySpan<T> y)
    {
        ValidateFinite(x, nameof(x));
        ValidateFinite(y, nameof(y));
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Series must have the same length.", nameof(y));
        }

        var n = x.Length;
        if (n < 2)
        {
            return T.Zero;
        }

        long concordant = 0;
        long discordant = 0;
        long tiedX = 0;
        long tiedY = 0;

        for (var i = 0; i < n - 1; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                var orderX = x[i].CompareTo(x[j]);
                var orderY = y[i].CompareTo(y[j]);
                if (orderX == 0 && orderY == 0)
                {
                    // Mutual tie — counts as tied in both.
                    tiedX++;
                    tiedY++;
                }
                else if (orderX == 0)
                {
                    tiedX++;
                }
                else if (orderY == 0)
                {
                    tiedY++;
                }
                else if ((orderX > 0 && orderY > 0) || (orderX < 0 && orderY < 0))
                {
                    concordant++;
                }
                else
                {
                    discordant++;
                }
            }
        }

        var totalPairs = (long)n * (n - 1) / 2;
        var denomX = totalPairs - tiedX;
        var denomY = totalPairs - tiedY;
        if (denomX <= 0 || denomY <= 0)
        {
            return T.Zero;
        }

        // Counts fit the guaranteed working types throughout the supported length
        // domain. Divide in T so a small double ratio never narrows through decimal.
        var pairScale = T.CreateChecked(totalPairs);
        var normalizedNumerator = T.CreateChecked(concordant - discordant) / pairScale;
        var normalizedX = T.CreateChecked(denomX) / pairScale;
        var normalizedY = T.CreateChecked(denomY) / pairScale;
        var tau = normalizedNumerator
            / NumericPrecision<T>.Sqrt(normalizedX)
            / NumericPrecision<T>.Sqrt(normalizedY);
        return T.Clamp(tau, -T.One, T.One);
    }

    private static void ValidateFinite(ReadOnlySpan<T> values, string parameterName)
    {
        for (var i = 0; i < values.Length; i++)
        {
            if (!T.IsFinite(values[i]))
            {
                throw new ArgumentOutOfRangeException(
                    parameterName, values[i], "Rank-correlation observations must be finite.");
            }
        }
    }

    private static T[] AverageRanks(ReadOnlySpan<T> values, int n)
    {
        var indexed = new (T Value, int Index)[n];
        for (var i = 0; i < n; i++)
        {
            indexed[i] = (values[i], i);
        }

        Array.Sort(indexed, (a, b) => a.Value.CompareTo(b.Value));

        var ranks = new T[n];
        var i2 = 0;
        while (i2 < n)
        {
            var j = i2;
            while (j + 1 < n && indexed[j + 1].Value == indexed[i2].Value)
            {
                j++;
            }

            // Average rank for the tie group [i2, j].
            var rank = ((long)i2 + j + 2) / 2.0; // 1-based ranks.
            // Pearson is invariant under positive scaling. Scale before converting
            // to Half, whose finite range can be smaller than the observation count.
            var avgRank = T.CreateChecked(typeof(T) == typeof(Half) ? rank / n : rank);
            for (var k = i2; k <= j; k++)
            {
                ranks[indexed[k].Index] = avgRank;
            }

            i2 = j + 1;
        }

        return ranks;
    }
}

/// <summary>
/// Rank correlation primitives — Spearman rho and Kendall tau.
/// Both are robust to outliers and detect monotonic (not just linear)
/// dependence; the trade-off is lower power than Pearson under the bivariate
/// normal assumption.
/// </summary>
/// <remarks>
/// <para>
/// Spearman = Pearson correlation applied to ranks. Tied values receive
/// average ranks (fractional ranking). Returns 0 for fewer than three
/// observations or when either series is constant.
/// </para>
/// <para>
/// Kendall tau-b counts concordant minus discordant pairs, normalized by
/// sqrt((P - T_x)(P - T_y)) where P = n(n-1)/2 and T_x, T_y are tie corrections.
/// tau-b is the standard variant when ties are present; reduces to tau-a when
/// no ties exist.
/// </para>
/// </remarks>
[Obsolete(
    "Use RankCorrelation<decimal> instead (the facade preserves the original decimal precision; use <double> deliberately if you want IEEE precision). This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class RankCorrelation
{
    /// <summary>Spearman rank correlation coefficient rho in [-1, 1].</summary>
    public static decimal Spearman(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y)
        => RankCorrelation<decimal>.Spearman(x, y);

    /// <summary>
    /// Kendall tau-b (tie-corrected). Returns 0 for fewer than two
    /// observations or when all pairs are tied in either coordinate.
    /// </summary>
    public static decimal KendallTauB(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y)
        => RankCorrelation<decimal>.KendallTauB(x, y);
}
