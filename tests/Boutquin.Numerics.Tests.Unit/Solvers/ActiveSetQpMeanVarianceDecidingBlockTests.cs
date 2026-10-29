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
using System.Text;
using System.Text.Json;

using Boutquin.Numerics.Internal;

using HeldOutOutcome = Boutquin.Numerics.Tests.Unit.Solvers.ActiveSetQpCorpusTests.HeldOutOutcome;
using HeldOutProblem = Boutquin.Numerics.Tests.Unit.Solvers.ActiveSetQpCorpusTests.HeldOutProblem;
using HeldOutResult = Boutquin.Numerics.Tests.Unit.Solvers.ActiveSetQpCorpusTests.HeldOutResult;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// AC-4.1 for the deciding block of design step 4 (<c>B = F*</c>, the variables strictly inside the box at the exact
/// optimum) through <c>SolveMeanVariance</c>. Each member is a qualified min-variance case of the committed corpus
/// whose exact optimum puts a variable the exact oracle leaves free exactly on its bound (degenerate at the bound),
/// and whose label depends on that choice: the interior block meets the margin <c>48·n·u·κ_eq &lt; 1</c> while the
/// oracle's free block, which also holds the variable on its bound, does not. Each is posed as a mean-variance problem
/// with a constant mean vector (<c>μ = 0</c> or <c>μ = 0.0625·1</c>) and <c>λ = 1</c> or <c>2</c>: under the budget
/// <c>μᵀw</c> is constant, so the exact optimum and the covariance-only qualification are those of the min-variance
/// case, while the solver's gradient carries the linear term. The shared rational oracle of
/// <see cref="ActiveSetQpCorpusTests"/> solves the mean-variance problem itself and judges the result: no member is
/// refused and none returns a vector outside the AC-3.1 bound.
/// Falsifiability: a deciding block that kept the variable on its bound for mean variance only crosses the margin on
/// these members and refuses them naming stationarity, which the min-variance witnesses and the well-conditioned
/// mean-variance degenerate-at-bound census cannot see.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpMeanVarianceDecidingBlockTests
{
    // Label-sensitive members per type in the committed corpus: 30 double and 13 decimal (43 in all).
    private const int MinimumDoubleMembers = 25;
    private const int MinimumDecimalMembers = 10;

    // Constructed upper-bound degenerate members per type.
    private const int UpperBoundMembersPerType = 40;

    // The constant mean vectors and risk aversions each member is posed with (exact in both types).
    private static readonly (string Mean, string Lambda)[] s_variants = [("0", "1"), ("0", "2"), ("0.0625", "1"), ("0.0625", "2")];

    [Fact]
    public void ActiveSetQpMeanVariance_LabelSensitiveDegenerateAtBoundMembers_WithConstantMeans_NoQualifiedRefusalNoWrongVector()
    {
        var members = LabelSensitiveMembers("double").Concat(LabelSensitiveMembers("decimal")).ToList();
        var doubles = members.Count(m => m.Type == "double");
        var decimals = members.Count - doubles;
        Assert.True(
            doubles >= MinimumDoubleMembers && decimals >= MinimumDecimalMembers,
            "the census must hold at least " + MinimumDoubleMembers + " double and " + MinimumDecimalMembers
            + " decimal label-sensitive degenerate-at-bound members, found " + doubles + " and " + decimals);

        var results = new List<HeldOutResult>();
        var problems = new StringBuilder();
        foreach (var member in members)
        {
            foreach (var (mean, lambda) in s_variants)
            {
                var posed = member with
                {
                    Name = member.Name + " μ=" + mean + "·1 λ=" + lambda,
                    Mu = Enumerable.Repeat(mean, member.Cov.Length).ToArray(),
                    Lambda = lambda,
                };
                var result = ActiveSetQpCorpusTests.SolveHeldOut(posed);
                results.Add(result);
                if (!result.Qualified)
                {
                    problems.Append(posed.Name).Append(": the interior block qualifies the min-variance case but the shared oracle labels the mean-variance problem refusal-permitted: ")
                        .Append(result.Detail).Append(" | ");
                }
                else if (result.Outcome != HeldOutOutcome.Ok)
                {
                    problems.Append(result.Outcome).Append(' ').Append(result.Detail).Append(" | ");
                }
            }
        }

        ActiveSetQpCorpusTests.AssertCensus(results, results.Count);
        Assert.True(problems.Length == 0, problems.ToString());
    }

    /// <summary>
    /// The same deciding-block property at the <b>upper</b> bound, which the committed corpus's degenerate members (all on
    /// the lower bound) do not reach, posed as a min-variance problem and with the constant means of
    /// <see cref="s_variants"/>. Each member is built exactly: <c>A = diag(dᵢ)</c> with <c>dᵢ ∈ {8, 16}</c>, interior
    /// weights <c>w_F = p/2048</c> summing to <c>1/2</c> in the box <c>[0, 1/2]</c>, the last variable on its upper bound
    /// <c>1/2</c>, and <c>c = 2(ν·1 − A·w_F)</c>, <c>s = 4·Σ dᵢwᵢ²</c> chosen so that <c>Σw = ν·1</c> exactly (every
    /// gradient equals the multiplier <c>ν</c>, so the bound variable's multiplier is zero) with the Schur complement
    /// <c>s − cᵀA⁻¹c = 4ν(1 − νΣ1/dᵢ)</c>; a tiny <c>ν</c> makes the block of the interior variables and the bound
    /// variable nearly singular while <c>A</c> is perfectly conditioned. The optimum is certified by the exact KKT
    /// check, and a member is kept only when its interior block meets the margin and the block with the bound variable
    /// crosses it (both in exact rationals after the production covariance-only equilibration).
    /// Falsifiability: a deciding block whose at-bound test saw only the lower bound keeps the variable on its upper
    /// bound and refuses these members naming stationarity.
    /// </summary>
    [Fact]
    public void ActiveSet_UpperBoundDegenerateMembers_AsMinVarianceAndWithConstantMeans_NoQualifiedRefusalNoWrongVector()
    {
        var members = UpperBoundMembers("double", 2026102501, UpperBoundMembersPerType).Concat(UpperBoundMembers("decimal", 2026102502, UpperBoundMembersPerType)).ToList();
        Assert.True(
            members.Count(m => m.Type == "double") == UpperBoundMembersPerType && members.Count(m => m.Type == "decimal") == UpperBoundMembersPerType,
            "the construction must give " + UpperBoundMembersPerType + " members per type, found " + members.Count);

        var results = new List<HeldOutResult>();
        var problems = new StringBuilder();
        foreach (var member in members)
        {
            var posings = new List<HeldOutProblem> { member with { Name = member.Name + " min-variance" } };
            posings.AddRange(s_variants.Select(v => member with
            {
                Name = member.Name + " μ=" + v.Mean + "·1 λ=" + v.Lambda,
                Mu = Enumerable.Repeat(v.Mean, member.Cov.Length).ToArray(),
                Lambda = v.Lambda,
            }));
            foreach (var posed in posings)
            {
                var result = ActiveSetQpCorpusTests.SolveHeldOut(posed);
                results.Add(result);
                if (!result.Qualified)
                {
                    problems.Append(posed.Name).Append(": the shared oracle does not label the member qualified: ").Append(result.Detail).Append(" | ");
                }
                else if (result.Outcome != HeldOutOutcome.Ok)
                {
                    problems.Append(result.Outcome).Append(' ').Append(result.Detail).Append(" | ");
                }
            }
        }

        ActiveSetQpCorpusTests.AssertCensus(results, results.Count);
        Assert.True(problems.Length == 0, problems.ToString());
    }

    private static List<HeldOutProblem> UpperBoundMembers(string type, int seed, int count)
    {
        var rng = new System.Random(seed);
        var isDouble = type == "double";
        var unit = isDouble ? Fraction.Of(BigInteger.One, BigInteger.One << 53) : Fraction.Of(BigInteger.One, BigInteger.Pow(10, 28));
        var half = Fraction.Of(1, 2);
        var members = new List<HeldOutProblem>();
        for (var attempt = 0; members.Count < count && attempt < 50 * count; attempt++)
        {
            var k = rng.Next(2, 7);
            var n = k + 1;
            var diagonal = Enumerable.Range(0, k).Select(_ => Fraction.FromInt(rng.Next(2) == 0 ? 8 : 16)).ToArray();
            var p = Enumerable.Range(0, k - 1).Select(_ => (long)rng.Next(1, 301)).ToList();
            if (p.Sum() >= 1024)
            {
                continue;
            }

            p.Add(1024 - p.Sum());
            var w = p.Select(x => Fraction.Of(x, 2048)).Append(half).ToArray();

            // ν = 2^-e (double) or 10^-e (decimal): the Schur complement 4ν(1 − νΣ1/dᵢ) against s of order one.
            var nu = isDouble
                ? Fraction.Of(BigInteger.One, BigInteger.One << rng.Next(46, 50))
                : Fraction.Of(BigInteger.One, BigInteger.Pow(10, rng.Next(26, 28)));
            var sigma = new Fraction[n, n];
            var s = Fraction.Zero;
            for (var i = 0; i < k; i++)
            {
                for (var j = 0; j < k; j++)
                {
                    sigma[i, j] = i == j ? diagonal[i] : Fraction.Zero;
                }

                sigma[i, k] = sigma[k, i] = Fraction.FromInt(2) * (nu - (diagonal[i] * w[i]));
                s += Fraction.FromInt(4) * diagonal[i] * w[i] * w[i];
            }

            sigma[k, k] = s;
            var perm = Enumerable.Range(0, n).ToArray();
            for (var i = n - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (perm[i], perm[j]) = (perm[j], perm[i]);
            }

            var cov = new string[n][];
            var permuted = new Fraction[n, n];
            for (var i = 0; i < n; i++)
            {
                cov[i] = new string[n];
                for (var j = 0; j < n; j++)
                {
                    permuted[i, j] = sigma[perm[i], perm[j]];
                    cov[i][j] = Literal(isDouble, permuted[i, j]);
                }
            }

            if (cov.Any(row => row.Any(x => x.Length == 0)))
            {
                continue;
            }

            var optimum = Enumerable.Range(0, n).Select(i => w[perm[i]]).ToArray();
            Assert.True(IsKktPoint(permuted, new Fraction[n].Select(_ => Fraction.Zero).ToArray(), Fraction.One, half, optimum), "the constructed point fails the exact KKT conditions");
            var interior = Enumerable.Range(0, n).Where(i => optimum[i] < half).ToArray();
            var scaled = isDouble ? Equilibrated<double>(cov, 2) : Equilibrated<decimal>(cov, 10);
            var margin = Fraction.FromInt(48 * n) * unit;
            var interiorCondition = Fraction.Condition1(scaled, interior);
            var boundCondition = Fraction.Condition1(scaled, Enumerable.Range(0, n).ToArray());
            if (interiorCondition is null || !(margin * interiorCondition.Value < Fraction.One)
                || (boundCondition is not null && margin * boundCondition.Value < Fraction.One))
            {
                continue;
            }

            members.Add(new HeldOutProblem("upperdeg-" + type + "-" + seed + "-" + members.Count, type, cov, "0", "0.5"));
        }

        return members;
    }

    /// <summary>
    /// The qualified in-domain min-variance cases of the committed corpus of <paramref name="type"/> whose label
    /// depends on the deciding block: a variable the exact oracle leaves free (pattern <c>0</c>) has its exact optimum
    /// on a bound, the committed flag (judged on the interior block) is qualified, and the oracle's free block crosses
    /// the margin. The margin of the oracle's block is recomputed here in exact rationals after the covariance-only
    /// equilibration the solver's guard measures (<c>QpScaling&lt;T&gt;.EquilibrateCovariance</c>).
    /// </summary>
    private static IEnumerable<HeldOutProblem> LabelSensitiveMembers(string type)
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Verification", "qp-corpus", "corpus-" + type + ".json");
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var unit = type == "double" ? Fraction.Of(BigInteger.One, BigInteger.One << 53) : Fraction.Of(BigInteger.One, BigInteger.Pow(10, 28));
        var members = new List<HeldOutProblem>();
        foreach (var c in document.RootElement.GetProperty("cases").EnumerateArray())
        {
            if (c.GetProperty("mode").GetString() != "minvar" || !c.GetProperty("in_domain").GetBoolean() || !c.GetProperty("qualified").GetBoolean())
            {
                continue;
            }

            var cov = c.GetProperty("cov").EnumerateArray().Select(row => row.EnumerateArray().Select(x => x.GetString()!).ToArray()).ToArray();
            var lb = c.GetProperty("lb").GetString()!;
            var ub = c.GetProperty("ub").GetString()!;
            var pattern = c.GetProperty("pattern").GetString()!;
            var optimum = c.GetProperty("optimum").EnumerateArray().Select(x => Fraction.Parse(x.GetString()!)).ToArray();
            var n = cov.Length;
            var lower = Fraction.Parse(lb);
            var upper = Fraction.Parse(ub);
            var oracleFree = Enumerable.Range(0, n).Where(i => pattern[i] == '0').ToArray();
            if (!oracleFree.Any(i => optimum[i] == lower || optimum[i] == upper))
            {
                continue;
            }

            var scaled = type == "double" ? Equilibrated<double>(cov, 2) : Equilibrated<decimal>(cov, 10);
            var margin = Fraction.FromInt(48 * n) * unit;
            var cond = Fraction.Condition1(scaled, oracleFree);
            if (cond is not null && margin * cond.Value < Fraction.One)
            {
                continue;
            }

            members.Add(new HeldOutProblem(c.GetProperty("id").GetString()!, type, cov, lb, ub));
        }

        return members;
    }

    // D⁻¹ΣD⁻¹ exactly, with D = diag(radix^eᵢ) the exponents of the production covariance-only scaling (shared with
    // ActiveSetQpGeneralMeanDecidingBlockTests).
    internal static Fraction[,] Equilibrated<T>(string[][] rows, int radix)
        where T : IFloatingPoint<T>
    {
        var n = rows.Length;
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = T.Parse(rows[i][j], NumberStyles.Float, CultureInfo.InvariantCulture);
            }
        }

        var exponents = QpScaling<T>.EquilibrateCovariance(cov).Exponents;
        var scaled = new Fraction[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var k = exponents[i] + exponents[j];
                var power = Fraction.Of(BigInteger.Pow(radix, Math.Abs(k)), BigInteger.One);
                var entry = Fraction.Parse(rows[i][j]);
                scaled[i, j] = k >= 0 ? entry / power : entry * power;
            }
        }

        return scaled;
    }

    // Exact KKT (shared with ActiveSetQpGeneralMeanDecidingBlockTests) for max μ′w − (λ/2)w′Σw s.t. 1′w = 1, 0 ≤ w ≤ ub: feasibility; g = λΣw − μ equal to ν on the variables
    // strictly inside the box; g − ν ≥ 0 at the lower bound and ≤ 0 at the upper. With Σ positive definite the KKT point
    // is the unique optimum.
    internal static bool IsKktPoint(Fraction[,] sigma, Fraction[] mu, Fraction lambda, Fraction upper, Fraction[] w)
    {
        var n = w.Length;
        var total = Fraction.Zero;
        foreach (var x in w)
        {
            if (x < Fraction.Zero || x > upper)
            {
                return false;
            }

            total += x;
        }

        if (total != Fraction.One)
        {
            return false;
        }

        var g = new Fraction[n];
        for (var i = 0; i < n; i++)
        {
            var sum = Fraction.Zero;
            for (var j = 0; j < n; j++)
            {
                sum += sigma[i, j] * w[j];
            }

            g[i] = (lambda * sum) - mu[i];
        }

        var inside = Enumerable.Range(0, n).Where(i => w[i] > Fraction.Zero && w[i] < upper).ToArray();
        if (inside.Length == 0)
        {
            return false;
        }

        var nu = g[inside[0]];
        return inside.All(i => g[i] == nu)
            && Enumerable.Range(0, n).All(i => (w[i] == Fraction.Zero && !(g[i] < nu)) || (w[i] == upper && !(g[i] > nu)) || inside.Contains(i));
    }

    // The literal of an exact value in the working type, or "" when the value is not exactly representable. A double
    // literal is the exact decimal expansion of the dyadic value (the shortest round-trip form is not the value).
    internal static string Literal(bool isDouble, Fraction value)
    {
        string text;
        if (isDouble)
        {
            var e = (int)value.Den.GetBitLength() - 1;
            if (value.Den != BigInteger.One << e || BigInteger.Abs(value.Num).GetBitLength() > 53)
            {
                return "";
            }

            var digits = BigInteger.Abs(value.Num * BigInteger.Pow(5, e)).ToString(CultureInfo.InvariantCulture).PadLeft(e + 1, '0');
            // A numerator of at most 53 bits over a power of two is a double in the range used here.
            text = (value.Num.Sign < 0 ? "-" : "") + (e == 0 ? digits : digits[..^e] + "." + digits[^e..]);
        }
        else
        {
            if (BigInteger.Abs(value.Num) > new BigInteger(decimal.MaxValue) || value.Den > new BigInteger(decimal.MaxValue))
            {
                return "";
            }

            text = ((decimal)value.Num / (decimal)value.Den).ToString(CultureInfo.InvariantCulture);
        }

        return Fraction.Parse(text) == value ? text : "";
    }

    // An exact rational with a positive denominator in lowest terms.
    internal readonly record struct Fraction(BigInteger Num, BigInteger Den)
    {
        public static Fraction Zero => new(BigInteger.Zero, BigInteger.One);

        public static Fraction One => new(BigInteger.One, BigInteger.One);

        public static Fraction FromInt(long value) => new(value, BigInteger.One);

        public static Fraction Of(BigInteger num, BigInteger den)
        {
            if (den.Sign < 0)
            {
                num = -num;
                den = -den;
            }

            var g = BigInteger.GreatestCommonDivisor(num, den);
            return g.IsOne || g.IsZero ? new Fraction(num, den) : new Fraction(num / g, den / g);
        }

        // A decimal literal with an optional exponent, read exactly.
        public static Fraction Parse(string s)
        {
            s = s.Trim();
            var negative = s.StartsWith('-');
            if (negative || s.StartsWith('+'))
            {
                s = s[1..];
            }

            var exponent = 0;
            var e = s.IndexOfAny(['E', 'e']);
            if (e >= 0)
            {
                exponent = int.Parse(s[(e + 1)..], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
                s = s[..e];
            }

            var dot = s.IndexOf('.');
            if (dot >= 0)
            {
                exponent -= s.Length - dot - 1;
                s = s.Remove(dot, 1);
            }

            var mantissa = BigInteger.Parse(s, CultureInfo.InvariantCulture);
            if (negative)
            {
                mantissa = -mantissa;
            }

            return exponent >= 0 ? Of(mantissa * BigInteger.Pow(10, exponent), BigInteger.One) : Of(mantissa, BigInteger.Pow(10, -exponent));
        }

        public Fraction Abs() => new(BigInteger.Abs(Num), Den);

        public static Fraction operator +(Fraction a, Fraction b) => Of((a.Num * b.Den) + (b.Num * a.Den), a.Den * b.Den);

        public static Fraction operator -(Fraction a, Fraction b) => Of((a.Num * b.Den) - (b.Num * a.Den), a.Den * b.Den);

        public static Fraction operator *(Fraction a, Fraction b) => Of(a.Num * b.Num, a.Den * b.Den);

        public static Fraction operator /(Fraction a, Fraction b) => Of(a.Num * b.Den, a.Den * b.Num);

        public static bool operator <(Fraction a, Fraction b) => a.Num * b.Den < b.Num * a.Den;

        public static bool operator >(Fraction a, Fraction b) => a.Num * b.Den > b.Num * a.Den;

        // The exact 1-norm condition number of the principal block idx of a; null when that block is singular.
        public static Fraction? Condition1(Fraction[,] a, int[] idx)
        {
            var k = idx.Length;
            if (k == 0)
            {
                return One;
            }

            var m = new Fraction[k, 2 * k];
            for (var i = 0; i < k; i++)
            {
                for (var j = 0; j < k; j++)
                {
                    m[i, j] = a[idx[i], idx[j]];
                    m[i, k + j] = i == j ? One : Zero;
                }
            }

            for (var col = 0; col < k; col++)
            {
                var pivot = Enumerable.Range(col, k - col).FirstOrDefault(r => !m[r, col].Num.IsZero, -1);
                if (pivot < 0)
                {
                    return null;
                }

                for (var j = 0; j < 2 * k; j++)
                {
                    (m[col, j], m[pivot, j]) = (m[pivot, j], m[col, j]);
                }

                var p = m[col, col];
                for (var j = 0; j < 2 * k; j++)
                {
                    m[col, j] /= p;
                }

                for (var r = 0; r < k; r++)
                {
                    if (r == col || m[r, col].Num.IsZero)
                    {
                        continue;
                    }

                    var f = m[r, col];
                    for (var j = 0; j < 2 * k; j++)
                    {
                        m[r, j] -= f * m[col, j];
                    }
                }
            }

            Fraction Norm(Func<int, int, Fraction> entry)
            {
                var best = Zero;
                for (var j = 0; j < k; j++)
                {
                    var sum = Zero;
                    for (var i = 0; i < k; i++)
                    {
                        sum += entry(i, j).Abs();
                    }

                    if (sum > best)
                    {
                        best = sum;
                    }
                }

                return best;
            }

            return Norm((i, j) => a[idx[i], idx[j]]) * Norm((i, j) => m[i, k + j]);
        }
    }
}
