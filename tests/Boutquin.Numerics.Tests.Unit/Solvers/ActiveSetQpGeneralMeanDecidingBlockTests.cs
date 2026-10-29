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

using Fraction = Boutquin.Numerics.Tests.Unit.Solvers.ActiveSetQpMeanVarianceDecidingBlockTests.Fraction;
using HeldOutOutcome = Boutquin.Numerics.Tests.Unit.Solvers.ActiveSetQpCorpusTests.HeldOutOutcome;
using HeldOutProblem = Boutquin.Numerics.Tests.Unit.Solvers.ActiveSetQpCorpusTests.HeldOutProblem;
using HeldOutResult = Boutquin.Numerics.Tests.Unit.Solvers.ActiveSetQpCorpusTests.HeldOutResult;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// AC-4.1 for the deciding block of design step 4 (<c>B = F*</c>, the variables strictly inside the box at the exact
/// optimum) through <c>SolveMeanVariance</c> with a <b>non-constant</b> mean vector, where the constant-mean members of
/// <see cref="ActiveSetQpMeanVarianceDecidingBlockTests"/> cannot reach: a deciding block that changes only when the
/// means differ passes those. Each member is built exactly, in the working type, with a known optimum that puts one
/// variable exactly on a bound with an exactly zero multiplier:
/// <c>Σ = [[A, c], [cᵀ, aᵀAa + δ]]</c> with <c>A</c> a diagonally dominant integer block, <c>c = A·a</c> for a small
/// integer vector <c>a</c> and <c>δ</c> one unit of the last representable place, so that <c>Σ</c> is positive definite
/// and its block over the interior variables and the bound variable is nearly singular; <c>w = (w_F, 0)</c> with
/// <c>w_F = p/1024 &gt; 0</c> summing to 1 (or the bound variable on the upper bound <c>1/2</c> and <c>w_F = p/2048</c>), and <c>μ = λ·Σ·w</c>, which makes the gradient <c>λΣw − μ</c> exactly
/// zero, so the multiplier of the bound variable is zero and <c>ν = 0</c>. The expected optimum comes from that
/// construction and is certified here by an exact KKT check (and independently by the shared rational oracle of
/// <see cref="ActiveSetQpCorpusTests"/>, which judges the result); it is never the solver's. Every member is kept only
/// when its interior block meets the margin <c>48·n·u·κ_eq &lt; 1</c> and the interior block with the bound variable
/// crosses it, recomputed here in exact rationals after the production covariance-only equilibration, so each is
/// qualified and label sensitive. No member may be refused, and none may return a vector outside the AC-3.1 bound.
/// Falsifiability: a deciding block that kept the variable on its bound only for non-constant means refuses these
/// members naming stationarity.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpGeneralMeanDecidingBlockTests
{
    private const int MembersPerType = 24;
    private const int Seed = 20261025;

    [Theory]
    [InlineData("double")]
    [InlineData("decimal")]
    public void SolveMeanVariance_GeneralMeanDegenerateAtBoundMembers_WhoseBoundVariableWouldCrossTheMargin_AreAnsweredWithinTheBound(string type)
    {
        var members = Members(type, Seed, MembersPerType);
        Assert.Equal(MembersPerType, members.Count);

        var results = new List<HeldOutResult>();
        var problems = new List<string>();
        foreach (var member in members)
        {
            var result = ActiveSetQpCorpusTests.SolveHeldOut(member);
            results.Add(result);
            if (!result.Qualified)
            {
                problems.Add(member.Name + ": the shared oracle does not label the member qualified: " + result.Detail);
            }
            else if (result.Outcome != HeldOutOutcome.Ok)
            {
                problems.Add(result.Outcome + " " + result.Detail);
            }
        }

        ActiveSetQpCorpusTests.AssertCensus(results, results.Count);
        Assert.True(problems.Count == 0, string.Join(" | ", problems.Take(6)));
    }

    // The members themselves: non-constant means, a certified optimum with a variable exactly on a bound (both bounds
    // represented), a qualified interior block and a crossing block with that variable.
    [Theory]
    [InlineData("double")]
    [InlineData("decimal")]
    public void GeneralMeanMembers_AreQualifiedLabelSensitiveAndHaveNonConstantMeans(string type)
    {
        var members = Members(type, Seed, MembersPerType);
        Assert.Equal(MembersPerType, members.Count);
        Assert.All(members, m => Assert.True(m.Mu!.Distinct(StringComparer.Ordinal).Count() > 1, m.Name + ": constant means"));
        Assert.True(
            members.Count(m => m.Name.EndsWith("-lower", StringComparison.Ordinal)) >= 6 && members.Count(m => m.Name.EndsWith("-upper", StringComparison.Ordinal)) >= 6,
            "both bounds need at least six members: " + string.Join(", ", members.Select(m => m.Name)));
    }

    /// <summary>
    /// Builds <paramref name="count"/> members of <paramref name="type"/> from <paramref name="seed"/>; each is certified
    /// (exact KKT at the constructed optimum) and kept only when its interior block qualifies and the interior block
    /// with the bound variable crosses the margin.
    /// </summary>
    private static List<HeldOutProblem> Members(string type, int seed, int count)
    {
        var rng = new System.Random(seed);
        var members = new List<HeldOutProblem>();
        var isDouble = type == "double";
        var unit = isDouble ? Fraction.Of(BigInteger.One, BigInteger.One << 53) : Fraction.Of(BigInteger.One, BigInteger.Pow(10, 28));
        for (var attempt = 0; members.Count < count && attempt < 50 * count; attempt++)
        {
            var k = rng.Next(2, 7);
            var a = new long[k, k];
            for (var i = 0; i < k; i++)
            {
                for (var j = 0; j <= i; j++)
                {
                    a[i, j] = a[j, i] = i == j ? rng.Next(8, 17) : rng.Next(-1, 2);
                }
            }

            var p = new long[k];
            for (var i = 0; i < k - 1; i++)
            {
                p[i] = rng.Next(1, 401);
            }

            var head = p.Take(k - 1).Sum();
            var shift = new long[k];
            for (var i = 0; i < k; i++)
            {
                shift[i] = rng.Next(-3, 4);
            }

            if (head >= 1024 || shift.All(x => x == 0))
            {
                continue;
            }

            p[k - 1] = 1024 - head;
            var c = new long[k];
            for (var i = 0; i < k; i++)
            {
                for (var j = 0; j < k; j++)
                {
                    c[i] += a[i, j] * shift[j];
                }
            }

            long norm = 0;
            for (var i = 0; i < k; i++)
            {
                norm += shift[i] * c[i];
            }

            // δ is one unit in the last place of aᵀAa + δ: 2^-e with e = 52 − bits(N) − r for double, 10^-e with
            // e = 27 − digits(N) − r for decimal, so the entry is exact in the working type.
            var delta = isDouble
                ? Fraction.Of(BigInteger.One, BigInteger.One << (52 - (int)new BigInteger(norm).GetBitLength() - rng.Next(0, 4)))
                : Fraction.Of(BigInteger.One, BigInteger.Pow(10, 27 - norm.ToString(CultureInfo.InvariantCulture).Length - rng.Next(0, 3)));
            var lambda = rng.Next(1, 3);
            // The bound variable sits at the lower bound 0 (box [0, 1] or [0, 2]) or at the upper bound 1/2 (box [0, 1/2],
            // the interior weights then sum to 1/2).
            var atUpper = rng.Next(2) == 1;
            var upper = atUpper ? Fraction.Of(1, 2) : Fraction.FromInt(rng.Next(1, 3));
            var n = k + 1;
            var sigma = new Fraction[n, n];
            for (var i = 0; i < k; i++)
            {
                for (var j = 0; j < k; j++)
                {
                    sigma[i, j] = Fraction.FromInt(a[i, j]);
                }

                sigma[i, k] = sigma[k, i] = Fraction.FromInt(c[i]);
            }

            sigma[k, k] = Fraction.FromInt(norm) + delta;
            var w = new Fraction[n];
            for (var i = 0; i < k; i++)
            {
                w[i] = Fraction.Of(p[i], atUpper ? 2048 : 1024);
            }

            w[k] = atUpper ? upper : Fraction.Zero;

            // μ = λ·Σ·w: the gradient λΣw − μ is zero, so the bound variable's multiplier and ν are zero.
            var mu = new Fraction[n];
            for (var i = 0; i < n; i++)
            {
                var sum = Fraction.Zero;
                for (var j = 0; j < n; j++)
                {
                    sum += sigma[i, j] * w[j];
                }

                mu[i] = Fraction.FromInt(lambda) * sum;
            }

            var perm = Enumerable.Range(0, n).ToArray();
            for (var i = n - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (perm[i], perm[j]) = (perm[j], perm[i]);
            }

            var cov = new string[n][];
            var permutedSigma = new Fraction[n, n];
            for (var i = 0; i < n; i++)
            {
                cov[i] = new string[n];
                for (var j = 0; j < n; j++)
                {
                    permutedSigma[i, j] = sigma[perm[i], perm[j]];
                    cov[i][j] = ActiveSetQpMeanVarianceDecidingBlockTests.Literal(isDouble, permutedSigma[i, j]);
                }
            }

            var means = Enumerable.Range(0, n).Select(i => mu[perm[i]]).ToArray();
            var optimum = Enumerable.Range(0, n).Select(i => w[perm[i]]).ToArray();
            if (means.Distinct().Count() == 1 || cov.Any(row => row.Any(x => x.Length == 0)) || means.Any(x => ActiveSetQpMeanVarianceDecidingBlockTests.Literal(isDouble, x).Length == 0))
            {
                continue;
            }

            Assert.True(
                ActiveSetQpMeanVarianceDecidingBlockTests.IsKktPoint(permutedSigma, means, Fraction.FromInt(lambda), upper, optimum),
                "the constructed point fails the exact KKT conditions");

            var interior = Enumerable.Range(0, n).Where(i => optimum[i] > Fraction.Zero && optimum[i] < upper).ToArray();
            var withBound = Enumerable.Range(0, n).ToArray();
            var scaled = isDouble
                ? ActiveSetQpMeanVarianceDecidingBlockTests.Equilibrated<double>(cov, 2)
                : ActiveSetQpMeanVarianceDecidingBlockTests.Equilibrated<decimal>(cov, 10);
            var margin = Fraction.FromInt(48 * n) * unit;
            var interiorCondition = Fraction.Condition1(scaled, interior);
            var boundCondition = Fraction.Condition1(scaled, withBound);
            var qualifies = interiorCondition is not null && margin * interiorCondition.Value < Fraction.One;
            var crosses = boundCondition is null || !(margin * boundCondition.Value < Fraction.One);
            if (!qualifies || !crosses)
            {
                continue;
            }

            members.Add(new HeldOutProblem(
                "gmdeg-" + type + "-" + seed + "-" + members.Count + (atUpper ? "-upper" : "-lower"),
                type,
                cov,
                "0",
                ActiveSetQpMeanVarianceDecidingBlockTests.Literal(isDouble, upper),
                means.Select(x => ActiveSetQpMeanVarianceDecidingBlockTests.Literal(isDouble, x)).ToArray(),
                lambda.ToString(CultureInfo.InvariantCulture)));
        }

        return members;
    }

}
