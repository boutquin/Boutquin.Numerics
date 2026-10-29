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

using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// The <c>riskAversion = 0</c> greedy LP of <see cref="ActiveSetQpSolver{T}.SolveMeanVariance"/> on boxes whose budget
/// <c>1 − n·minWeight</c> or capacity <c>maxWeight − minWeight</c> is too large to be carried in the working type: the
/// unit budget must not be absorbed by rounding. Each result is compared with the greedy optimum computed here in exact
/// rational arithmetic (every weight at the lower bound, the budget distributed up to the upper bound in descending mean
/// order): every returned weight must lie in the box and be a faithful rounding of the exact optimum's weight.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpMaxReturnLpBudgetTests
{
    [Fact]
    public void SolveMeanVariance_ZeroRiskAversion_ShortBoundAtTwoToThe53_ReturnsTheMaximumReturnVertex()
    {
        var w = ActiveSetQpSolver<double>.SolveMeanVariance(
            Identity<double>(2), [0.0, 1.0], 0.0, -9007199254740992.0, 1.0);

        Assert.Equal(new[] { 0.0, 1.0 }, w);
    }

    [Fact]
    public void SolveMeanVariance_ZeroRiskAversion_SixteenAssetsShortBoundMinus1e15_ReturnsTheExactGreedyAllocation()
    {
        const int n = 16;
        var means = new double[n];
        for (var i = 0; i < n; i++)
        {
            means[i] = i / 16.0;
        }

        var w = ActiveSetQpSolver<double>.SolveMeanVariance(Identity<double>(n), means, 0.0, -1e15, 1.0);

        var expected = new double[n];
        expected[0] = -14.0;
        for (var i = 1; i < n; i++)
        {
            expected[i] = 1.0;
        }

        Assert.Equal(expected, w);
    }

    [Fact]
    public void SolveMeanVariance_ZeroRiskAversion_DoubleBoxesFrom1e15To1e100_MatchTheExactGreedyOptimum()
    {
        var failures = Sweep<double>(
            [1e15, 3e15, 1e16, 1e20, 1e50, 1e100], seed: 20261022, reps: 3, IntegerMagnitude<double>);

        Assert.True(failures.Count == 0, failures.Count + " solves differ from the exact greedy optimum: " + string.Join(" | ", failures.Take(5)));
    }

    [Fact]
    public void SolveMeanVariance_ZeroRiskAversion_FloatShortBoundMinus1e8_ReturnsTheMaximumReturnVertex()
    {
        var w = ActiveSetQpSolver<float>.SolveMeanVariance(Identity<float>(2), [0f, 1f], 0f, -1e8f, 1f);

        Assert.Equal(new[] { 0f, 1f }, w);
    }

    [Fact]
    public void SolveMeanVariance_ZeroRiskAversion_FloatBoxesFrom1e8To1e30_MatchTheExactGreedyOptimum()
    {
        var failures = Sweep<float>([1e8, 1e10, 1e15, 1e30], seed: 20261023, reps: 2, IntegerMagnitude<float>);

        Assert.True(failures.Count == 0, failures.Count + " solves differ from the exact greedy optimum: " + string.Join(" | ", failures.Take(5)));
    }

    [Fact]
    public void SolveMeanVariance_ZeroRiskAversion_DecimalIntegerBoxesUpTo1e14_MatchTheExactGreedyOptimum()
    {
        var failures = Sweep<decimal>([1e6, 1e10, 1e13, 1e14], seed: 20261024, reps: 2, IntegerMagnitude<decimal>);

        Assert.True(failures.Count == 0, failures.Count + " solves differ from the exact greedy optimum: " + string.Join(" | ", failures.Take(5)));
    }

    [Fact]
    public void SolveMeanVariance_ZeroRiskAversion_DecimalTwentyEightDigitBoundsNear1e14_MatchTheExactGreedyOptimum()
    {
        // A lower bound with 28 significant digits at about 1e13–1e14: n·minWeight needs more digits than decimal
        // carries, so a budget formed in decimal loses its trailing digits before the fill.
        var failures = Sweep<decimal>([1e13, 1e14], seed: 20261025, reps: 3, FullPrecisionDecimal);

        Assert.True(failures.Count == 0, failures.Count + " solves differ from the exact greedy optimum: " + string.Join(" | ", failures.Take(5)));
    }

    // Intermediate budget scales also satisfy the exact allocation oracle.
    [Fact]
    public void SolveMeanVariance_ZeroRiskAversion_DoubleNonIntegerShortBoundsAtVariedBudgetScales_MatchTheExactGreedyOptimum()
    {
        var failures = BudgetScaleSweep<double>(seed: 20261026, reps: 3, DoubleBound);

        Assert.True(failures.Count == 0, failures.Count + " solves differ from the exact greedy optimum: " + string.Join(" | ", failures.Take(5)));
    }

    [Fact]
    public void SolveMeanVariance_ZeroRiskAversion_DecimalTwentyEightDigitShortBoundsAtVariedBudgetScales_MatchTheExactGreedyOptimum()
    {
        var failures = BudgetScaleSweep<decimal>(seed: 20261027, reps: 3, DecimalBound);

        Assert.True(failures.Count == 0, failures.Count + " solves differ from the exact greedy optimum: " + string.Join(" | ", failures.Take(5)));
    }

    // Ordinary short and long-only boxes, including large asset counts.
    [Theory]
    [InlineData(3, -68.616, 1.0)]
    [InlineData(10, -8.518666666666666, 1.0)]
    [InlineData(100, -0.1, 0.2)]
    [InlineData(60, -0.3, 0.5)]
    [InlineData(200, -0.05, 0.2)]
    [InlineData(1023, 0.0, 0.01)]
    [InlineData(1500, 0.0002, 0.0026666666666666666)]
    [InlineData(2000, 0.000225, 0.002)]
    public void SolveMeanVariance_ZeroRiskAversion_OrdinaryBoxes_ReturnTheExactGreedyOptimum(int n, double lower, double upper)
    {
        var means = new double[n];
        for (var i = 0; i < n; i++)
        {
            means[i] = (i * 7919) % n;
        }

        var w = ActiveSetQpSolver<double>.SolveMeanVariance(Identity<double>(n), means, 0.0, lower, upper);

        var problem = Check(w, ExactGreedy(means, lower, upper), lower, upper);
        Assert.True(problem is null, problem);
        Assert.Contains(w, x => x == lower || x == upper);
    }

    // Preserve varied intermediate budgets, including values around the former path boundary.
    private static readonly double[] s_budgetScaleMultiples = [0.9, 0.98, 1.02, 1.1, 1.5, 8, 100, 1000, 20000, 60000];

    private static readonly int[] s_budgetScaleAssetCounts = [2, 3, 4, 5, 8, 13, 16, 24, 32, 40];

    private static List<string> BudgetScaleSweep<T>(int seed, int reps, Func<double, System.Random, T> bound)
        where T : IFloatingPoint<T>
    {
        var rng = new System.Random(seed);
        var failures = new List<string>();
        foreach (var n in s_budgetScaleAssetCounts)
        {
            foreach (var multiple in s_budgetScaleMultiples)
            {
                foreach (var symmetric in new[] { true, false })
                {
                    for (var rep = 0; rep < reps; rep++)
                    {
                        // |1 − n·minWeight| = 1 + n·B, so B = (1024·multiple/(n + 2) − 1)/n; jitter keeps the digits arbitrary
                        // while the product stays within half a percent of the intended multiple.
                        var target = (((1024 * multiple) / (n + 2)) - 1) / n;
                        var b = bound(target * (1 + ((rng.NextDouble() - 0.5) * 0.004)), rng);
                        var lower = -b;
                        var upper = symmetric ? b : T.One;

                        var means = new T[n];
                        for (var i = 0; i < n; i++)
                        {
                            means[i] = T.CreateChecked((rng.NextDouble() * 2) - 1);
                        }

                        var w = ActiveSetQpSolver<T>.SolveMeanVariance(Identity<T>(n), means, T.Zero, lower, upper);
                        var problem = Check(w, ExactGreedy(means, lower, upper), lower, upper);
                        if (problem is not null)
                        {
                            failures.Add($"{typeof(T).Name} n={n} multiple={multiple} B={b} {(symmetric ? "[-B,B]" : "[-B,1]")} rep={rep}: {problem}");
                        }
                    }
                }
            }
        }

        return failures;
    }

    // A non-integer double: the target with its low bits drawn at random.
    private static double DoubleBound(double target, System.Random rng)
        => BitConverter.Int64BitsToDouble(BitConverter.DoubleToInt64Bits(target) ^ (long)rng.Next(1, 1 << 20));

    // A decimal with 28 significant digits: the target's leading digits, the rest drawn at random.
    private static decimal DecimalBound(double target, System.Random rng)
    {
        var leading = ((decimal)target).ToString(CultureInfo.InvariantCulture);
        if (!leading.Contains('.', StringComparison.Ordinal))
        {
            leading += ".";
        }

        var significant = leading.Replace(".", string.Empty, StringComparison.Ordinal).TrimStart('0').Length;
        var digits = new char[Math.Max(0, 28 - significant)];
        for (var i = 0; i < digits.Length; i++)
        {
            digits[i] = (char)('1' + rng.Next(9));
        }

        return decimal.Parse(leading + new string(digits), NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    // ------------------------------------------------------------------------------------------
    // Sweep and exact greedy oracle
    // ------------------------------------------------------------------------------------------

    private static List<string> Sweep<T>(double[] magnitudes, int seed, int reps, Func<double, System.Random, T> magnitude)
        where T : IFloatingPoint<T>
    {
        var rng = new System.Random(seed);
        var failures = new List<string>();
        foreach (var mag in magnitudes)
        {
            foreach (var symmetric in new[] { true, false })
            {
                for (var n = 2; n <= 40; n += n < 8 ? 1 : 8)
                {
                    for (var rep = 0; rep < reps; rep++)
                    {
                        var b = magnitude(mag, rng);
                        var lower = -b;
                        var upper = symmetric ? b : T.One;
                        var means = new T[n];
                        for (var i = 0; i < n; i++)
                        {
                            means[i] = T.CreateChecked((rng.NextDouble() * 2) - 1);
                        }

                        var w = ActiveSetQpSolver<T>.SolveMeanVariance(Identity<T>(n), means, T.Zero, lower, upper);
                        var problem = Check(w, ExactGreedy(means, lower, upper), lower, upper);
                        if (problem is not null)
                        {
                            failures.Add($"{typeof(T).Name} B={mag:G3} n={n} {(symmetric ? "[-B,B]" : "[-B,1]")} rep={rep}: {problem}");
                        }
                    }
                }
            }
        }

        return failures;
    }

    private static T IntegerMagnitude<T>(double magnitude, System.Random rng)
        where T : IFloatingPoint<T> => T.CreateChecked(magnitude);

    private static decimal FullPrecisionDecimal(double magnitude, System.Random rng)
    {
        // 28 significant digits, the leading one at the magnitude's decade.
        var digits = new char[28];
        digits[0] = (char)('1' + rng.Next(9));
        for (var i = 1; i < digits.Length; i++)
        {
            digits[i] = (char)('0' + rng.Next(10));
        }

        var integerDigits = (int)Math.Round(Math.Log10(magnitude));
        var text = new string(digits, 0, integerDigits) + "." + new string(digits, integerDigits, digits.Length - integerDigits);
        return decimal.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    // The greedy LP optimum in exact rational arithmetic: every weight at the lower bound, then, in descending mean
    // order (index order among equal means), each weight raised by min(upper − lower, remaining budget).
    private static Q[] ExactGreedy<T>(T[] means, T lower, T upper)
        where T : IFloatingPoint<T>
    {
        var n = means.Length;
        var order = Enumerable.Range(0, n).OrderByDescending(i => means[i]).ToArray();
        var lowerQ = Q.From(lower);
        var capacity = Q.From(upper) - lowerQ;
        var remaining = Q.One - (new Q(n, 1) * lowerQ);
        var optimum = Enumerable.Repeat(lowerQ, n).ToArray();
        foreach (var i in order)
        {
            if (remaining.Sign <= 0)
            {
                break;
            }

            var add = remaining.CompareTo(capacity) < 0 ? remaining : capacity;
            optimum[i] += add;
            remaining -= add;
        }

        return optimum;
    }

    // Null when every weight is in the box and is a faithful rounding of the exact weight (for decimal: within 1e-24 of
    // it relative to max(1, |exact|)); otherwise a description of the first failure.
    private static string? Check<T>(T[] w, Q[] exact, T lower, T upper)
        where T : IFloatingPoint<T>
    {
        for (var i = 0; i < w.Length; i++)
        {
            if (w[i] < lower || w[i] > upper)
            {
                return $"w[{i}]={w[i]} outside [{lower}, {upper}]";
            }

            if (!Close(w[i], exact[i]))
            {
                return $"w[{i}]={w[i]}, exact {exact[i].ToDouble():R}; w=[{string.Join(", ", w.Take(6))}{(w.Length > 6 ? ", ..." : "")}]";
            }
        }

        return null;
    }

    private static bool Close<T>(T value, Q exact)
        where T : IFloatingPoint<T>
    {
        if (value is double d)
        {
            return Q.From(Math.BitDecrement(d)).CompareTo(exact) <= 0 && exact.CompareTo(Q.From(Math.BitIncrement(d))) <= 0;
        }

        if (value is float f)
        {
            return Q.From(MathF.BitDecrement(f)).CompareTo(exact) <= 0 && exact.CompareTo(Q.From(MathF.BitIncrement(f))) <= 0;
        }

        var error = (Q.From(value) - exact).Abs();
        var scale = exact.Abs().CompareTo(Q.One) > 0 ? exact.Abs() : Q.One;
        return error.CompareTo(new Q(1, BigInteger.Pow(10, 24)) * scale) <= 0;
    }

    private static T[,] Identity<T>(int n)
        where T : IFloatingPoint<T>
    {
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            cov[i, i] = T.One;
        }

        return cov;
    }

    // An exact rational number with a positive denominator.
    private readonly struct Q : IComparable<Q>
    {
        public Q(BigInteger numerator, BigInteger denominator)
        {
            if (denominator.Sign < 0)
            {
                numerator = -numerator;
                denominator = -denominator;
            }

            var g = BigInteger.GreatestCommonDivisor(numerator, denominator);
            if (g > BigInteger.One)
            {
                numerator /= g;
                denominator /= g;
            }

            N = numerator;
            D = denominator;
        }

        public static Q One => new(1, 1);

        public BigInteger N { get; }

        public BigInteger D { get; }

        public int Sign => N.Sign;

        public static Q From<T>(T value)
            where T : IFloatingPoint<T>
        {
            if (value is decimal m)
            {
                Span<int> bits = stackalloc int[4];
                decimal.GetBits(m, bits);
                var magnitude = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | new BigInteger((uint)bits[0]);
                var scale = (bits[3] >> 16) & 0xFF;
                return new Q(bits[3] < 0 ? -magnitude : magnitude, BigInteger.Pow(10, scale));
            }

            var x = value is float f ? f : (double)(object)value!;
            if (x == 0)
            {
                return new Q(0, 1);
            }

            var raw = BitConverter.DoubleToInt64Bits(x);
            var biased = (int)((raw >> 52) & 0x7FF);
            var fraction = raw & 0xFFFFFFFFFFFFFL;
            if (biased == 0)
            {
                biased = 1;
            }
            else
            {
                fraction |= 1L << 52;
            }

            var exponent = biased - 1075;
            BigInteger mantissa = raw < 0 ? -fraction : fraction;
            return exponent >= 0 ? new Q(mantissa << exponent, 1) : new Q(mantissa, BigInteger.One << -exponent);
        }

        public static Q operator +(Q a, Q b) => new((a.N * b.D) + (b.N * a.D), a.D * b.D);

        public static Q operator -(Q a, Q b) => new((a.N * b.D) - (b.N * a.D), a.D * b.D);

        public static Q operator *(Q a, Q b) => new(a.N * b.N, a.D * b.D);

        public Q Abs() => N.Sign < 0 ? new Q(-N, D) : this;

        public int CompareTo(Q other) => (N * other.D).CompareTo(other.N * D);

        // For messages only: the quotient to about 64 significant bits, then scaled.
        public double ToDouble()
        {
            if (N.IsZero)
            {
                return 0.0;
            }

            var shift = 64 - (int)(BigInteger.Abs(N).GetBitLength() - D.GetBitLength());
            var q = shift >= 0 ? (BigInteger.Abs(N) << shift) / D : BigInteger.Abs(N) / (D << -shift);
            return Math.ScaleB((double)q, -shift) * N.Sign;
        }
    }
}
