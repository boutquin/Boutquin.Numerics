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
    private static readonly T s_two = T.CreateChecked(2);

    /// <summary>Spearman rank correlation coefficient rho in [-1, 1].</summary>
    public static T Spearman(ReadOnlySpan<T> x, ReadOnlySpan<T> y)
    {
        var n = Math.Min(x.Length, y.Length);
        if (n < 3)
        {
            return T.Zero;
        }

        var rx = AverageRanks(x, n);
        var ry = AverageRanks(y, n);

        return PearsonCorrelation<T>.Compute(rx, ry);
    }

    /// <summary>
    /// Kendall tau-b (tie-corrected). Returns 0 for fewer than three
    /// observations or when all pairs are tied in either coordinate.
    /// </summary>
    public static T KendallTauB(ReadOnlySpan<T> x, ReadOnlySpan<T> y)
    {
        var n = Math.Min(x.Length, y.Length);
        if (n < 3)
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
                var dx = x[i] - x[j];
                var dy = y[i] - y[j];
                if (dx == T.Zero && dy == T.Zero)
                {
                    // Mutual tie — counts as tied in both.
                    tiedX++;
                    tiedY++;
                }
                else if (dx == T.Zero)
                {
                    tiedX++;
                }
                else if (dy == T.Zero)
                {
                    tiedY++;
                }
                else if ((T.IsPositive(dx) && T.IsPositive(dy)) || (T.IsNegative(dx) && T.IsNegative(dy)))
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

        var denom = NumericPrecision<T>.Sqrt(T.CreateChecked(denomX) * T.CreateChecked(denomY));
        var tau = T.CreateChecked(concordant - discordant) / denom;
        return T.Clamp(tau, -T.One, T.One);
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
            var avgRank = T.CreateChecked(i2 + j + 2) / s_two; // 1-based ranks.
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
public static class RankCorrelation
{
    /// <summary>Spearman rank correlation coefficient rho in [-1, 1].</summary>
    public static decimal Spearman(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y)
        => RankCorrelation<decimal>.Spearman(x, y);

    /// <summary>
    /// Kendall tau-b (tie-corrected). Returns 0 for fewer than three
    /// observations or when all pairs are tied in either coordinate.
    /// </summary>
    public static decimal KendallTauB(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y)
        => RankCorrelation<decimal>.KendallTauB(x, y);
}
