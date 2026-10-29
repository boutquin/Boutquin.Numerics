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

using System.Globalization;
using System.Numerics;
using Boutquin.Numerics.Internal;
using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// The conditioning guard of <see cref="ActiveSetQpSolver{T}"/> refuses a solve exactly when <c>48·n·u·κ₁ ≥ 1</c> for the
/// equilibrated block that decides the result, on blocks where the eigenvalue bound of the guard is at its weakest:
/// <c>A = I − (1 − δ)·v̂v̂′</c> with <c>v = (√m, 1, …, 1)</c>, whose inverse is dominated by <c>v̂v̂′/δ</c>, so that
/// <c>‖A⁻¹‖₁</c> is about <c>√m/2</c> times <c>‖A⁻¹‖₂ = 1/λ_min(A)</c>. A lower bound on <c>λ_min</c> proves the margin
/// only through <c>‖A⁻¹‖₁ ≤ √m·‖A⁻¹‖₂</c>; a bound that dropped or shrank the <c>√m</c> would accept these blocks past
/// the margin. The optimum of each problem is strictly inside the box, so the whole matrix decides the result, and the
/// expected decision comes from this test's own integer computation of <c>κ₁</c> on the equilibrated matrix.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpConditioningMarginControlTests
{
    private const string Refusal = "numerically singular at the working precision";

    // (order, target of 48·n·u·κ₁ before the per-order factor of the family, seed). The double blocks just past the
    // margin (48·n·u·κ₁ from 1.00003 to 1.0006) are the ones the exact norm witness cannot refuse; the decimal blocks run
    // from just inside the margin to 1.8 times past it, where no witness exists.
    public static TheoryData<int, double, int> DoubleBlocks => new()
    {
        { 5, 1.0, 1 },
        { 13, 0.85, 1 },
        { 40, 0.5, 1 },
        { 13, 0.9045, 2 },
        { 13, 0.906, 1 },
        { 20, 0.7505, 1 },
        { 30, 0.6283, 1 },
        { 40, 0.5532, 3 },
        { 5, 2.0, 1 },
        { 40, 1.0, 1 },
    };

    public static TheoryData<int, double, int> DecimalBlocks => new()
    {
        { 5, 1.0, 1 },
        { 13, 0.85, 1 },
        { 40, 0.5, 1 },
        { 11, 1.0, 1 },
        { 13, 0.95, 2 },
        { 16, 1.0, 1 },
        { 24, 1.0, 1 },
        { 40, 1.0, 1 },
        { 5, 4.0, 1 },
    };

    // (order, target, seed) placing 48·n·u·κ₁ within 3% of the margin on both sides: accepted blocks between 0.98 and
    // 0.995, refused blocks between 1.005 and 1.02. The constant 48 is then pinned from both sides to about 2%: a margin
    // of 49 (refusing from 48/49 ≈ 0.9796) or 50 refuses the accepted rows, and 47 (accepting below 48/47 ≈ 1.0213)
    // accepts the refused ones, where the rows above sit a factor of up to 0.85 or 1.8 from the bound.
    public static TheoryData<int, double, int> DoubleNearMarginBlocks => new()
    {
        { 2, 1.7268, 1 },
        { 5, 1.3517, 1 },
        { 8, 1.1036, 1 },
        { 2, 1.7672, 1 },
        { 5, 1.3735, 1 },
        { 8, 1.1405, 1 },
    };

    public static TheoryData<int, double, int> DecimalNearMarginBlocks => new()
    {
        { 2, 1.7483, 1 },
        { 5, 1.3471, 1 },
        { 8, 1.1026, 1 },
        { 2, 1.7765, 1 },
        { 5, 1.3688, 1 },
        { 8, 1.1395, 1 },
    };

    [Theory]
    [MemberData(nameof(DoubleNearMarginBlocks))]
    public void ConditioningGuard_OnDoubleBlocksWithinThreePercentOfTheMargin_DecidesAsTheExactConditionNumber(int n, double target, int seed)
    {
        AssertDecidedAsTheExactConditionNumber<double>(n, target, seed);
    }

    [Theory]
    [MemberData(nameof(DecimalNearMarginBlocks))]
    public void ConditioningGuard_OnDecimalBlocksWithinThreePercentOfTheMargin_DecidesAsTheExactConditionNumber(int n, double target, int seed)
    {
        AssertDecidedAsTheExactConditionNumber<decimal>(n, target, seed);
    }

    // The near-margin rows stay where they pin the constant: every one within (0.97, 0.995] or [1.005, 1.03) of the bound
    // by the exact κ₁, with at least three on each side per type.
    [Fact]
    public void NearMarginBlocks_SitWithinThreePercentOnBothSidesOfTheMargin()
    {
        foreach (var (name, ratios) in new[] { ("double", Ratios<double>(DoubleNearMarginBlocks)), ("decimal", Ratios<decimal>(DecimalNearMarginBlocks)) })
        {
            var outside = ratios.Where(r => !(r.Ratio > 0.97 && r.Ratio <= 0.995 && !r.Refuses) && !(r.Ratio >= 1.005 && r.Ratio < 1.03 && r.Refuses)).ToList();
            Assert.True(outside.Count == 0, name + ": rows outside the two windows: " + string.Join(", ", outside.Select(r => r.Ratio.ToString("G6", CultureInfo.InvariantCulture))));
            Assert.True(ratios.Count(r => r.Refuses) >= 3 && ratios.Count(r => !r.Refuses) >= 3, name + ": both sides need three rows");
        }
    }

    private static List<(bool Refuses, double Ratio)> Ratios<T>(TheoryData<int, double, int> cases)
        where T : IFloatingPoint<T> =>
        cases.Select(row => ExactlyOutsideMargin(Crafted<T>((int)row[0], (double)row[1], (int)row[2]))).ToList();

    [Theory]
    [MemberData(nameof(DoubleBlocks))]
    public void ConditioningGuard_OnDoubleBlocksWithASlackSqrtMBound_DecidesAsTheExactConditionNumber(int n, double target, int seed)
    {
        AssertDecidedAsTheExactConditionNumber<double>(n, target, seed);
    }

    [Theory]
    [MemberData(nameof(DecimalBlocks))]
    public void ConditioningGuard_OnDecimalBlocksWithASlackSqrtMBound_DecidesAsTheExactConditionNumber(int n, double target, int seed)
    {
        AssertDecidedAsTheExactConditionNumber<decimal>(n, target, seed);
    }

    // Both sides of the margin are represented in each type's cases, so a guard that refused or accepted everything
    // would fail one of them.
    [Fact]
    public void CraftedBlocks_PopulateBothSidesOfTheMargin()
    {
        foreach (var refusesSome in new[] { Sides<double>(DoubleBlocks), Sides<decimal>(DecimalBlocks) })
        {
            Assert.True(refusesSome.Refused >= 3, "refused blocks " + refusesSome.Refused);
            Assert.True(refusesSome.Accepted >= 3, "accepted blocks " + refusesSome.Accepted);
        }
    }

    private static (int Refused, int Accepted) Sides<T>(TheoryData<int, double, int> cases)
        where T : IFloatingPoint<T>
    {
        int refused = 0, accepted = 0;
        foreach (var row in cases)
        {
            var (refuses, _) = ExactlyOutsideMargin(Crafted<T>((int)row[0], (double)row[1], (int)row[2]));
            refused += refuses ? 1 : 0;
            accepted += refuses ? 0 : 1;
        }

        return (refused, accepted);
    }

    private static void AssertDecidedAsTheExactConditionNumber<T>(int n, double target, int seed)
        where T : IFloatingPoint<T>
    {
        var cov = Crafted<T>(n, target, seed);
        var (refuses, ratio) = ExactlyOutsideMargin(cov);
        var label = string.Create(CultureInfo.InvariantCulture, $"{typeof(T).Name} n={n} target={target} seed={seed} 48·n·u·κ₁={ratio:G8}");
        var expected = refuses ? "refused" : "accepted";
        var minVariance = Outcome(() => ActiveSetQpSolver<T>.SolveMinVariance(cov, T.Zero, T.One), label);
        var meanVariance = Outcome(() => ActiveSetQpSolver<T>.SolveMeanVariance(cov, new T[n], T.One, T.Zero, T.One), label);
        Assert.True(expected == minVariance && expected == meanVariance,
            label + ": SolveMinVariance " + minVariance + ", SolveMeanVariance " + meanVariance + ", the exact κ₁ " + (refuses ? "refuses" : "accepts"));
    }

    private static string Outcome<T>(Func<T[]> solve, string label)
        where T : IFloatingPoint<T>
    {
        T[] weights;
        try
        {
            weights = solve();
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains(Refusal, StringComparison.Ordinal))
        {
            return "refused";
        }

        // The deciding block is the whole matrix only for an interior optimum.
        Assert.All(weights, w => Assert.True(w > T.Zero && w < T.One, label + ": weight on the box " + w));
        return "accepted";
    }

    /// <summary>
    /// <c>I − (1 − δ)·v̂v̂′</c> with <c>v = (√m, 1 + ε₂, …, 1 + ε_m)</c> normalised (<c>|ε_i| ≤ 0.005</c> from
    /// <paramref name="seed"/>); <c>δ</c> puts <c>48·n·u·κ₁</c> at <paramref name="target"/> times a factor that grows with
    /// the order (about 0.73 at 5, 1.1 at 13 and 1.8 at 40). For <c>decimal</c> the vector is normalised in
    /// <c>decimal</c>, since its <c>double</c> image would carry an error far above <c>δ</c>.
    /// </summary>
    private static T[,] Crafted<T>(int m, double target, int seed)
        where T : IFloatingPoint<T>
    {
        var rng = new System.Random(seed);
        var v = new double[m];
        v[0] = Math.Sqrt(m);
        for (var i = 1; i < m; i++)
        {
            v[i] = 1.0 + (0.01 * (rng.NextDouble() - 0.5));
        }

        var norm = Math.Sqrt(v.Sum(x => x * x));
        for (var i = 0; i < m; i++)
        {
            v[i] /= norm;
        }

        var unit = typeof(T) == typeof(decimal) ? 1e-28 : Math.ScaleB(1.0, -53);
        var delta = 2.0 * Math.Abs(v[0]) * v.Sum(Math.Abs) * 48.0 * m * unit / target;
        var a = new T[m, m];
        if (typeof(T) == typeof(decimal))
        {
            var vd = v.Select(x => (decimal)x).ToArray();
            var root = NumericPrecision<decimal>.Sqrt(vd.Sum(x => x * x));
            for (var i = 0; i < m; i++)
            {
                vd[i] /= root;
            }

            var c = 1m - (decimal)delta;
            for (var i = 0; i < m; i++)
            {
                for (var j = 0; j <= i; j++)
                {
                    a[i, j] = a[j, i] = T.CreateChecked((i == j ? 1m : 0m) - (c * vd[i] * vd[j]));
                }
            }
        }
        else
        {
            for (var i = 0; i < m; i++)
            {
                for (var j = 0; j <= i; j++)
                {
                    a[i, j] = a[j, i] = T.CreateChecked((i == j ? 1.0 : 0.0) - ((1.0 - delta) * v[i] * v[j]));
                }
            }
        }

        return a;
    }

    /// <summary>
    /// Whether <c>48·n·u·κ₁(Σ̃) ≥ 1</c> for the equilibrated covariance <c>Σ̃</c>, <c>u</c> the unit roundoff of
    /// <typeparamref name="T"/> (2⁻⁵³ or 10⁻²⁸), decided in integers: <c>M</c> is <c>Σ̃</c> on the finest common radix grid of
    /// its entries, fraction-free Gauss–Jordan elimination of <c>[M | I]</c> leaves <c>det M</c> on the diagonal and
    /// <c>adj M</c> on the right, and <c>κ₁ = ‖M‖₁·‖adj M‖₁/det M</c> (a condition number does not depend on the grid).
    /// Also returns <c>48·n·u·κ₁</c> rounded, for the failure message.
    /// </summary>
    private static (bool Refuses, double Ratio) ExactlyOutsideMargin<T>(T[,] full)
        where T : IFloatingPoint<T>
    {
        var scaled = QpScaling<T>.EquilibrateCovariance(full).Covariance;
        var n = scaled.GetLength(0);
        var isDecimal = typeof(T) == typeof(decimal);
        var mantissas = new BigInteger[n, n];
        var exponents = new int[n, n];
        var finest = int.MaxValue;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                (mantissas[i, j], exponents[i, j]) = isDecimal ? Split(decimal.CreateChecked(scaled[i, j])) : Split(double.CreateChecked(scaled[i, j]));
                if (!mantissas[i, j].IsZero)
                {
                    finest = Math.Min(finest, exponents[i, j]);
                }
            }
        }

        var radix = new BigInteger(isDecimal ? 10 : 2);
        var original = new BigInteger[n, n];
        var work = new BigInteger[n, 2 * n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                original[i, j] = mantissas[i, j] * BigInteger.Pow(radix, exponents[i, j] - finest);
                work[i, j] = original[i, j];
            }

            work[i, n + i] = BigInteger.One;
        }

        var previous = BigInteger.One;
        for (var k = 0; k < n; k++)
        {
            var pivot = work[k, k];
            Assert.True(pivot.Sign > 0, "the crafted covariance is not positive definite");
            for (var i = 0; i < n; i++)
            {
                if (i == k)
                {
                    continue;
                }

                var factor = work[i, k];
                for (var j = 0; j < 2 * n; j++)
                {
                    work[i, j] = ((pivot * work[i, j]) - (factor * work[k, j])) / previous;
                }
            }

            previous = pivot;
        }

        var determinant = work[n - 1, n - 1];
        BigInteger normM = BigInteger.Zero, normAdjugate = BigInteger.Zero;
        for (var c = 0; c < n; c++)
        {
            BigInteger column = BigInteger.Zero, adjugateColumn = BigInteger.Zero;
            for (var r = 0; r < n; r++)
            {
                column += BigInteger.Abs(original[r, c]);
                adjugateColumn += BigInteger.Abs(work[r, n + c]);
            }

            normM = BigInteger.Max(normM, column);
            normAdjugate = BigInteger.Max(normAdjugate, adjugateColumn);
        }

        var inverseUnit = isDecimal ? BigInteger.Pow(10, 28) : BigInteger.One << 53;
        var left = 48 * n * normM * normAdjugate;
        var right = determinant * inverseUnit;
        return (left >= right, Math.Exp(BigInteger.Log(left) - BigInteger.Log(right)));
    }

    private static (BigInteger Mantissa, int Exponent) Split(double x)
    {
        if (x == 0.0)
        {
            return (BigInteger.Zero, 0);
        }

        var bits = BitConverter.DoubleToInt64Bits(x);
        var biased = (int)((bits >> 52) & 0x7FF);
        var fraction = bits & 0xFFFFFFFFFFFFFL;
        var mantissa = new BigInteger(biased == 0 ? fraction : fraction | (1L << 52));
        return (x < 0.0 ? -mantissa : mantissa, (biased == 0 ? 1 : biased) - 1075);
    }

    private static (BigInteger Mantissa, int Exponent) Split(decimal x)
    {
        var bits = decimal.GetBits(x);
        var magnitude = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | new BigInteger((uint)bits[0]);
        return (bits[3] < 0 ? -magnitude : magnitude, -((bits[3] >> 16) & 0xFF));
    }
}
