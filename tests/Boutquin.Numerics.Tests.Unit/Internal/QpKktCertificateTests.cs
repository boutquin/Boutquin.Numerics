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
using System.Reflection;
using System.Text.Json;

using Boutquin.Numerics.Internal;
using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Tests.Unit.Internal;

/// <summary>
/// The independent KKT certificate of the active-set QP rewrite (<c>Internal/QpKktCertificate&lt;T&gt;</c>):
/// it accepts the correctly rounded exact optimum of every qualified case of the committed regression corpus at
/// <see cref="double"/> and <see cref="decimal"/> and, at the nominal tolerance scale κ = 1, rejects every copy of
/// that optimum with one weight perturbed by max(4× the case's sensitivity bound, max(4, n+1) ulps of the perturbed
/// weight), naming the failed component (AC-2.1, AC-2.2 as amended 2026-10-08).
/// </summary>
/// <remarks>
/// The certificate is reached through reflection so that a missing or misnamed unit fails at an assertion
/// rather than at compile time. Expected values never come from the certificate: the exact optimum is the
/// committed 60-digit fixture value (verified against the <see cref="BigInteger"/> rational oracle by
/// <c>ActiveSetQpCorpusTests</c>), rounded to the working type here with a minimal rational helper copied from
/// that file (its oracle types are private to it).
/// </remarks>
public sealed class QpKktCertificateTests
{
    // Floor on the qualified cases each type's corpus must hold, re-derived from the regenerated corpus (qualified when
    // 48·n·u·κ_eq < 1 with κ_eq after the covariance-only equilibration QpScaling<T>.EquilibrateCovariance, the block
    // the solver's conditioning guard measures; spec amendment #11): it holds 949 qualified double and 894 qualified
    // decimal cases, and the floor sits about 5% below the smaller count, as the earlier floor of 400 sat below the
    // 421 / 424 qualified cases of the first corpus, so that a regeneration that silently loses the qualified set fails.
    private const int QualifiedCaseFloor = 850;

    // ------------------------------------------------------------------------------------------
    // AC-2.1  the certificate accepts the correctly rounded exact optimum of every qualified case
    // ------------------------------------------------------------------------------------------

    [Fact]
    public void Accepts_TheCorrectlyRoundedExactOptimum_OfEveryQualifiedCorpusCase_Double()
    {
        var cases = Corpus.Load("double").Where(c => c.Qualified).ToList();
        Assert.True(cases.Count >= QualifiedCaseFloor, "the qualified double corpus must be present, found " + cases.Count);

        var rejected = new List<string>();
        foreach (var c in cases)
        {
            var w = c.RoundedOptimum<double>();
            var (accepted, component, detail) = Certificate.Check(c.Cov<double>(), c.Mu<double>(), c.Lambda<double>(), c.Lb<double>(), c.Ub<double>(), w, 1.0);
            if (!accepted)
            {
                rejected.Add(c.Id + " (" + component + ": " + detail + ")");
            }
        }

        Assert.True(rejected.Count == 0, rejected.Count + " of " + cases.Count + " rounded optima rejected: " + string.Join("; ", rejected.Take(10)));
    }

    [Fact]
    public void Accepts_TheCorrectlyRoundedExactOptimum_OfEveryQualifiedCorpusCase_Decimal()
    {
        var cases = Corpus.Load("decimal").Where(c => c.Qualified).ToList();
        Assert.True(cases.Count >= QualifiedCaseFloor, "the qualified decimal corpus must be present, found " + cases.Count);

        var rejected = new List<string>();
        foreach (var c in cases)
        {
            var w = c.RoundedOptimum<decimal>();
            var (accepted, component, detail) = Certificate.Check(c.Cov<decimal>(), c.Mu<decimal>(), c.Lambda<decimal>(), c.Lb<decimal>(), c.Ub<decimal>(), w, 1m);
            if (!accepted)
            {
                rejected.Add(c.Id + " (" + component + ": " + detail + ")");
            }
        }

        Assert.True(rejected.Count == 0, rejected.Count + " of " + cases.Count + " rounded optima rejected: " + string.Join("; ", rejected.Take(10)));
    }

    // Hand-built problems with a closed-form exact optimum, checked at both guaranteed types.
    //   minvar: Σ = diag(1, 2, 4), box [0.2, 1]: w* = (8/15, 4/15, 1/5), the third weight at its lower bound.
    //   meanvar: Σ = I, μ = (1, 0), λ = 1, box [0, 0.8]: w* = (4/5, 1/5), the first weight at its upper bound.
    //   meanvar: Σ = [[2, 1], [1, 2]], μ = (2, 1), λ = 2, box [-1, 2]: w* = (3/4, 1/4), both free
    //   (μ − λΣw* = (−3/2, −3/2)).
    public static TheoryData<string> HandBuiltCases => new() { "minvar-lower", "meanvar-upper", "meanvar-free" };

    [Theory]
    [MemberData(nameof(HandBuiltCases))]
    public void Accepts_TheCorrectlyRoundedOptimum_OfHandBuiltCases_AtDoubleAndDecimal(string name)
    {
        var c = HandBuilt.Get(name);

        var d = Certificate.Check(c.Cov<double>(), c.Mu<double>(), c.Lambda<double>(), c.Lb<double>(), c.Ub<double>(), c.RoundedOptimum<double>(), 1.0);
        Assert.True(d.Accepted, name + " at double: " + d.Component + ": " + d.Detail);

        var m = Certificate.Check(c.Cov<decimal>(), c.Mu<decimal>(), c.Lambda<decimal>(), c.Lb<decimal>(), c.Ub<decimal>(), c.RoundedOptimum<decimal>(), 1m);
        Assert.True(m.Accepted, name + " at decimal: " + m.Component + ": " + m.Detail);
    }

    // Advisory, not an acceptance criterion: how many cases that are not qualified (48·n·u·κ_eq ≥ 1 for the
    // equilibrated free block) but whose exact optimum is representable the certificate also accepts.
    // Written next to the test binaries as qp-certificate-advisory.json for the solver rewrite.
    [Fact]
    public void Advisory_MeasuresAcceptanceOnRepresentableCasesThatAreNotQualified()
    {
        var report = new SortedDictionary<string, int[]>(StringComparer.Ordinal);
        foreach (var type in new[] { "double", "decimal" })
        {
            foreach (var c in Corpus.Load(type).Where(c => !c.Qualified && c.Representable))
            {
                var accepted = type == "double"
                    ? Certificate.Check(c.Cov<double>(), c.Mu<double>(), c.Lambda<double>(), c.Lb<double>(), c.Ub<double>(), c.RoundedOptimum<double>(), 1.0).Accepted
                    : Certificate.Check(c.Cov<decimal>(), c.Mu<decimal>(), c.Lambda<decimal>(), c.Lb<decimal>(), c.Ub<decimal>(), c.RoundedOptimum<decimal>(), 1m).Accepted;
                foreach (var key in new[] { type + "|*", type + "|" + c.Family })
                {
                    var row = report.TryGetValue(key, out var r) ? r : report[key] = new int[4];
                    row[0]++;
                    row[1] += accepted ? 1 : 0;
                    row[2] += c.InDomain ? 1 : 0;
                    row[3] += c.InDomain && accepted ? 1 : 0;
                }
            }
        }

        Assert.True(report["double|*"][0] > 0 && report["decimal|*"][0] > 0, "the corpus must contain non-qualified representable cases");
        var json = JsonSerializer.Serialize(
            report.ToDictionary(kv => kv.Key, kv => new { cases = kv.Value[0], accepted = kv.Value[1], in_domain = kv.Value[2], in_domain_accepted = kv.Value[3] }),
            new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(AppContext.BaseDirectory, "qp-certificate-advisory.json"), json);
    }

    // ------------------------------------------------------------------------------------------
    // AC-2.2  at κ = 1, every weight perturbed by max(4x sensitivity bound, max(4, n+1) ulps) is rejected
    // ------------------------------------------------------------------------------------------

    [Fact]
    public void Rejects_TheOptimumWithOneWeightPerturbedByFourTimesTheSensitivityBound_Double()
        => AssertEveryMutationRejected<double>("double");

    [Fact]
    public void Rejects_TheOptimumWithOneWeightPerturbedByFourTimesTheSensitivityBound_Decimal()
        => AssertEveryMutationRejected<decimal>("decimal");

    // The mutation check has teeth: each rejection is a tolerance decision, so loosening every row tolerance far
    // enough beyond the 4x perturbation turns the same copy into an accepted one, and the rejection assertion above
    // would then fail. The loosening factor needed is always above 1 (the default tolerance rejects).
    [Fact]
    public void MutationCheck_FailsOnceTheToleranceIsLoosenedBeyondTheFourTimesBound_Double()
        => AssertLooseningAcceptsEveryMutation<double>("double", maxDoublings: 400);

    [Fact]
    public void MutationCheck_FailsOnceTheToleranceIsLoosenedBeyondTheFourTimesBound_Decimal()
        => AssertLooseningAcceptsEveryMutation<decimal>("decimal", maxDoublings: 95);

    // Each failure names its component. Hand-built vectors that violate exactly one KKT condition.
    //   Bound: the box-relaxed optimum of Σ = diag(1, 2, 4) (4/7, 2/7, 1/7) with lb = 0.2 (budget and stationarity hold).
    //   Budget: 1.000001 · w* of the same problem with lb = 0 (stationarity and bounds hold).
    //   Stationarity: w* + 1e-6·(e1 - e2) with lb = 0 (budget and bounds hold, gradients of free variables differ).
    //   MultiplierSign: Σ = I, box [0.1, 0.9], w = (0.9, 0.1): no free variable, the upper one has the smaller
    //   gradient, so no budget multiplier satisfies both bound signs (budget and bounds hold).
    public static TheoryData<string, string> ComponentCases => new()
    {
        { "bound", "Bound" },
        { "budget", "Budget" },
        { "stationarity", "Stationarity" },
        { "multiplier", "MultiplierSign" },
    };

    [Theory]
    [MemberData(nameof(ComponentCases))]
    public void Rejection_NamesTheFailedComponent_AtDoubleAndDecimal(string name, string component)
    {
        var (c, w) = ComponentViolation.Get(name);

        var d = Certificate.Check(c.Cov<double>(), c.Mu<double>(), c.Lambda<double>(), c.Lb<double>(), c.Ub<double>(), w.Select(Rounding.ToDouble).ToArray(), 1.0);
        Assert.False(d.Accepted, name + " at double must be rejected");
        Assert.Equal(component, d.Component);
        Assert.StartsWith(ComponentWord(component), d.Detail, StringComparison.Ordinal);

        var m = Certificate.Check(c.Cov<decimal>(), c.Mu<decimal>(), c.Lambda<decimal>(), c.Lb<decimal>(), c.Ub<decimal>(), w.Select(Rounding.ToDecimal).ToArray(), 1m);
        Assert.False(m.Accepted, name + " at decimal must be rejected");
        Assert.Equal(component, m.Component);
        Assert.StartsWith(ComponentWord(component), m.Detail, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------------------------------
    // Design step 4  the tolerance scale κ ≥ 1; AC-2.1 and AC-2.2 hold at the nominal κ = 1
    // ------------------------------------------------------------------------------------------

    // The overload without κ is the nominal κ = 1: the same verdict and the same detail on every qualified
    // corpus case's rounded optimum and on every one of its AC-2.2 copies, at both guaranteed types.
    [Fact]
    public void Check_WithoutAToleranceScale_IsTheNominalKappaOne_OnEveryCorpusOptimumAndCopy()
    {
        AssertNominalIsKappaOne<double>("double");
        AssertNominalIsKappaOne<decimal>("decimal");
    }

    // κ < 1 would tighten a tolerance below the rounding bound of a correctly rounded optimum (AC-2.1 then
    // fails); the certificate refuses it instead of silently rejecting optima.
    [Theory]
    [InlineData("0")]
    [InlineData("0.5")]
    [InlineData("0.9999")]
    [InlineData("-1")]
    public void Check_RefusesAToleranceScaleBelowOne_AtDoubleAndDecimal(string kappa)
    {
        var c = HandBuilt.Get("minvar-lower");
        var d = Assert.Throws<ArgumentOutOfRangeException>(() => Certificate.Check(c.Cov<double>(), c.Mu<double>(), c.Lambda<double>(), c.Lb<double>(), c.Ub<double>(), c.RoundedOptimum<double>(), double.Parse(kappa, CultureInfo.InvariantCulture)));
        Assert.Equal("toleranceScale", d.ParamName);
        var m = Assert.Throws<ArgumentOutOfRangeException>(() => Certificate.Check(c.Cov<decimal>(), c.Mu<decimal>(), c.Lambda<decimal>(), c.Lb<decimal>(), c.Ub<decimal>(), c.RoundedOptimum<decimal>(), decimal.Parse(kappa, CultureInfo.InvariantCulture)));
        Assert.Equal("toleranceScale", m.ParamName);
    }

    private static void AssertNominalIsKappaOne<T>(string type)
        where T : IFloatingPoint<T>
    {
        var cases = Corpus.Load(type).Where(c => c.Qualified).ToList();
        var differences = new List<string>();
        foreach (var c in cases)
        {
            foreach (var w in Mutations<T>(c).Select(m => m.Weights).Prepend(c.RoundedOptimum<T>()))
            {
                var nominal = Certificate.CheckNominal(c.Cov<T>(), c.Mu<T>(), c.Lambda<T>(), c.Lb<T>(), c.Ub<T>(), w);
                var one = Certificate.Check(c.Cov<T>(), c.Mu<T>(), c.Lambda<T>(), c.Lb<T>(), c.Ub<T>(), w, T.One);
                if (nominal != one)
                {
                    differences.Add(c.Id + ": nominal " + nominal + " vs kappa 1 " + one);
                }
            }
        }

        Assert.True(differences.Count == 0, type + ": " + differences.Count + " verdicts differ: " + string.Join("; ", differences.Take(5)));
    }

    // ------------------------------------------------------------------------------------------
    // AC-2.1 / AC-2.2 off the corpus: held-out regressions and fresh seeded problems (n = 3..5)
    // ------------------------------------------------------------------------------------------

    // Held-out regressions, inputs exactly as the verifier found them; the exact optimum and the one-ulp sensitivity
    // are recomputed here by the rational oracle below, and every weight is perturbed in both directions.
    //   dec-n5-minvar … dec-n3-meanvar: an earlier tolerance (1e-28 per weight summed over the budget row, which
    //   reaches the 4e-28 floor delta once n >= 3) accepted the floor-perturbed copy.
    //   vb-2592-n5-minvar, vb-107-n5-meanvar: qualified n = 5 problems (fresh generator case
    //   vm2r2-fresh-20261008-2592 and input '/#107', seed 2026100822) on which the half-ulp certificate accepted
    //   w[0..2] moved by 4 ulps (rounded residual −2 / +2 units); the amended (n+1)-ulp copies must be rejected.
    public static TheoryData<string> HeldOutWitnesses => new()
    {
        "dec-n5-minvar",
        "dec-n4-meanvar",
        "dec-n3-minvar",
        "dec-n3-meanvar",
        "vb-2592-n5-minvar",
        "vb-107-n5-meanvar",
    };

    [Theory]
    [MemberData(nameof(HeldOutWitnesses))]
    public void HeldOutWitness_AcceptsTheRoundedOptimum_AndRejectsTheFloorPerturbedCopy_Decimal(string name)
    {
        var c = Oracle.Complete<decimal>(HeldOut.Get(name));
        Assert.True(c.Qualified, name + " must be a qualified problem");
        var failures = AcceptRejectFailures<decimal>(c);
        Assert.True(failures.Count == 0, name + ": " + string.Join("; ", failures));
    }

    // Fresh problems from a seeded generator that is not the corpus: Σ = AA′ + shift·I with small integer A,
    // rescaled per row by radix powers, globally, and by a non-radix factor; minimum variance or mean-variance;
    // assorted boxes. Only qualified problems (in-domain inputs, exact free-block 1-norm condition number below
    // 1/(4u), as in the corpus definition) are asserted.
    [Fact]
    public void SeededFreshProblems_AcceptTheRoundedOptimum_AndRejectThePerturbedCopy_Decimal()
        => AssertSeededProperty<decimal>(seed: 20261008, count: 150);

    [Fact]
    public void SeededFreshProblems_AcceptTheRoundedOptimum_AndRejectThePerturbedCopy_Double()
        => AssertSeededProperty<double>(seed: 20261008, count: 150);

    private static void AssertSeededProperty<T>(int seed, int count)
        where T : IFloatingPoint<T>
    {
        var rng = new System.Random(seed);
        var failures = new List<string>();
        var sizes = new int[6];
        for (var q = 0; q < count; q++)
        {
            var c = Oracle.Complete<T>(Generator.Next<T>(rng, "seed" + seed + "-" + q));
            if (!c.Qualified)
            {
                continue;
            }

            sizes[c.N]++;
            failures.AddRange(AcceptRejectFailures<T>(c));
        }

        Assert.True(sizes[3] > 0 && sizes[4] > 0 && sizes[5] > 0, "the generator must cover qualified problems at n = 3, 4 and 5");
        Assert.True(sizes.Sum() >= count * 3 / 4, "at least three quarters of the generated problems must be qualified, found " + sizes.Sum());
        Assert.True(failures.Count == 0, failures.Count + " failures over " + count + " problems: " + string.Join("; ", failures.Take(12)));
    }

    // Why the AC-2.2 floor is max(4, n+1) ulps (n = 4 with an exact tie, n = 5 without one). Mean-variance,
    // λ = 9.5, Σ = 0.95·11′ + diag(4, …, 4, 0): the last asset's covariance column is constant, so moving its weight
    // shifts every stationarity row by the same amount, which the free budget multiplier absorbs; only the budget
    // row sees the move. μ_j = λ·4·w*_j and μ_last = 0 make w* (all free) the exact optimum. The other weights are
    // chosen so that their rounding errors add up to +2 units: the rounded optimum's budget residual is +2e-28.
    // A 4-unit move of the last weight down (below the amended floor) gives residual −2e-28 and the same
    // stationarity residuals, so no residual certificate that accepts the rounded optimum (AC-2.1) can reject it;
    // the amended floor's (n+1)-unit move gives residual −(n−1)e-28, beyond the grid-rounded budget tolerance
    // ⌊n/2⌋e-28, and the certificate rejects it at κ = 1 as a budget failure. Every other AC-2.2 copy of these
    // problems (every weight, both directions) is rejected as well.
    public static TheoryData<string> FloorLimitWitnesses => new() { "limit-n5", "limit-n4-tie" };

    [Theory]
    [MemberData(nameof(FloorLimitWitnesses))]
    public void FloorLimit_RejectsTheAmendedFloorCopy_WhereTheFourUnitCopyIsIndistinguishable_Decimal(string name)
    {
        var n = name == "limit-n5" ? 5 : 4;
        var small = Rational.ParseScientific("0.04999999999999999999999999995");
        var last = Rational.Of(1, 1) - (Rational.Of(n - 1, 1) * small);
        var cov = new string[n][];
        for (var i = 0; i < n; i++)
        {
            cov[i] = Enumerable.Range(0, n).Select(j => i == j && i < n - 1 ? "4.95" : "0.95").ToArray();
        }

        var c = Oracle.Complete<decimal>(new QpCase
        {
            Id = name,
            Mode = "meanvar",
            CovText = cov,
            MuText = Enumerable.Range(0, n).Select(i => i < n - 1 ? "1.8999999999999999999999999981" : "0").ToArray(),
            LambdaText = "9.5",
            LbText = "0",
            UbText = "1",
        });
        Assert.True(c.Qualified, name + " must be qualified");
        Assert.True(c.Optimum.Take(n - 1).All(x => x.CompareTo(small) == 0) && c.Optimum[n - 1].CompareTo(last) == 0, name + ": unexpected exact optimum");

        var rounded = c.RoundedOptimum<decimal>();
        var unit = Rational.Of(BigInteger.One, BigInteger.Pow(10, 28));
        static Rational Residual(decimal[] w) => w.Select(Rational.FromDecimal).Aggregate(default(Rational), (x, y) => x + y) - Rational.Of(1, 1);
        Assert.Equal(0, Residual(rounded).CompareTo(Rational.Of(2, 1) * unit));

        var accepted = Certificate.Check(c.Cov<decimal>(), c.Mu<decimal>(), c.Lambda<decimal>(), c.Lb<decimal>(), c.Ub<decimal>(), rounded, 1m);
        Assert.True(accepted.Accepted, name + ": AC-2.1 requires the rounded optimum to be accepted: " + accepted.Detail);

        // Below the floor: the 4-unit copy is indistinguishable from the rounded optimum (residual −2 units).
        var fourUnits = (decimal[])rounded.Clone();
        fourUnits[n - 1] -= 4e-28m;
        Assert.Equal(0, Residual(fourUnits).CompareTo(Rational.Of(-2, 1) * unit));
        var below = Certificate.Check(c.Cov<decimal>(), c.Mu<decimal>(), c.Lambda<decimal>(), c.Lb<decimal>(), c.Ub<decimal>(), fourUnits, 1m);
        Assert.True(below.Accepted, name + ": the 4-unit copy (below the amended floor) mirrors the rounded optimum, yet was rejected as " + below.Component);

        // At the amended floor: the AC-2.2 copy of the last weight (the exact optimum moved down by n+1 units, then
        // rounded; at n = 4 both ends are exact ties, so the rounded move is 6 units) is rejected.
        var down = Mutations<decimal>(c).Single(m => m.Label == "w[" + (n - 1) + "]-delta").Weights;
        var move = Rational.FromDecimal(rounded[n - 1]) - Rational.FromDecimal(down[n - 1]);
        Assert.False(move < Rational.Of(n + 1, 1) * unit, name + ": the amended-floor move must be at least n+1 units");
        Assert.Equal(0, Residual(down).CompareTo((Rational.Of(2, 1) * unit) - move));
        Assert.True(((Rational.Of(2, 1) * unit) - move).Abs() > Rational.Of(n / 2, 1) * unit, name + ": the copy's residual must exceed the grid-rounded budget tolerance");
        var copy = Certificate.Check(c.Cov<decimal>(), c.Mu<decimal>(), c.Lambda<decimal>(), c.Lb<decimal>(), c.Ub<decimal>(), down, 1m);
        Assert.False(copy.Accepted, name + ": the amended-floor copy must be rejected at kappa = 1");
        Assert.Equal("Budget", copy.Component);

        var failures = AcceptRejectFailures<decimal>(c);
        Assert.True(failures.Count == 0, name + ": " + string.Join("; ", failures));
    }

    // ------------------------------------------------------------------------------------------
    // Bound pairs: a variable at its upper bound must not have a larger gradient than one at its lower bound
    // ------------------------------------------------------------------------------------------

    // Held-out witness k7e-935007-135 (double mean-variance, n = 20, box ±1e89, λ = 1; repeated 2×2 blocks with a
    // rank-one coupling): the solver's rounded route once returned the correctly rounded exact optimum with weights 0
    // and 15, two near-identical tiny-variance assets, on swapped bounds (error 2e89). Their exact ascent components
    // differ by about 1.29e-8 while the two rows' tolerances at κ = 1 sum to about 6.75e-9, so the copy violates
    // max g(upper) ≤ min g(lower) and the certificate rejects it as a bound pair at κ = 1. The exact optimum is
    // given by its active pattern (L lower bound, U upper bound, F free) and recomputed here by the rational oracle,
    // which returns it only if every KKT condition holds exactly; its correct rounding is pinned below.
    private const string BoundPairWitnessPattern = "LFFFFFFFLFFFFUFFFUFL";

    private static readonly string[] s_boundPairWitnessRounded =
    [
        "-1E+89", "-5337906011259001", "-2.3249859736482796E+56", "-1.5237028076901365E+61", "-1.7081299236028803E+17",
        "-2.3249859736482796E+56", "-1.5237028076901365E+61", "-1.7081299236028803E+17", "-1E+89", "8.388020526187662E+67",
        "-1.5602716750746998E+64", "-5337906011259001", "-2.5207548436277905E+37", "1E+89", "-3.9942954881912315E+66",
        "1E+89", "-2.5207548436277905E+37", "1E+89", "-1.5602716750746998E+64", "-1E+89",
    ];

    [Fact]
    public void BoundPairWitness_AcceptsTheCorrectlyRoundedExactOptimum_AtKappaOne_Double()
    {
        var c = BoundPairWitness();
        var rounded = c.RoundedOptimum<double>();
        Assert.Equal(s_boundPairWitnessRounded.Select(s => double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture)).ToArray(), rounded);

        var (accepted, component, detail) = Certificate.Check(c.Cov<double>(), c.Mu<double>(), c.Lambda<double>(), c.Lb<double>(), c.Ub<double>(), rounded, 1.0);
        Assert.True(accepted, "k7e-935007-135: the correctly rounded exact optimum must be accepted at kappa = 1: " + component + ": " + detail);
    }

    [Fact]
    public void BoundPairWitness_RejectsTheOptimumWithTheBoundPairSwapped_AtKappaOne_NamingThePair_Double()
    {
        var c = BoundPairWitness();
        var swapped = c.RoundedOptimum<double>();
        (swapped[0], swapped[15]) = (swapped[15], swapped[0]);
        Assert.Equal(1e89, swapped[0]);
        Assert.Equal(-1e89, swapped[15]);

        var (accepted, component, detail) = Certificate.Check(c.Cov<double>(), c.Mu<double>(), c.Lambda<double>(), c.Lb<double>(), c.Ub<double>(), swapped, 1.0);
        Assert.False(accepted, "k7e-935007-135: the copy with weights 0 and 15 on swapped bounds must be rejected at kappa = 1");
        Assert.Equal("MultiplierSign", component);
        Assert.StartsWith("multiplier sign: bound pair: variable 0 at its upper bound", detail, StringComparison.Ordinal);
        Assert.Contains("variable 15 at its lower bound", detail, StringComparison.Ordinal);
    }

    // The bound-pair check measures each pair against the rounding of its two rows, not against an exact
    // comparison: at a tie across a bound pair (hand-built minvar-tied-pair, exact gradients all equal, w*₀ at
    // the lower bound and w*₁ at the upper bound) the rounding of the free weights 1/3 and 1/6 leaves the lower
    // variable's ascent component above the upper one's at both types, and the correctly rounded optimum must
    // still be accepted (AC-2.1). The test first asserts that the rounding does invert the pair.
    [Fact]
    public void BoundPairCheck_AcceptsTheRoundedOptimumOfATieAcrossABoundPair_WithinTheTwoRowsRounding_AtDoubleAndDecimal()
    {
        AssertTiedPairAccepted<double>();
        AssertTiedPairAccepted<decimal>();
    }

    private static void AssertTiedPairAccepted<T>()
        where T : IFloatingPoint<T>
    {
        var c = HandBuilt.Get("minvar-tied-pair");
        var w = c.RoundedOptimum<T>();
        var cov = c.Cov<T>();
        Rational Ascent(int i) => default(Rational) - Enumerable.Range(0, c.N).Select(j => Rational.From(cov[i, j]) * Rational.From(w[j])).Aggregate((x, y) => x + y);
        Assert.Equal(T.Zero, w[0]);
        Assert.Equal(c.Ub<T>(), w[1]);
        Assert.True(Ascent(0) > Ascent(1), typeof(T).Name + ": the rounded free weights must leave the lower variable's ascent component above the upper one's");

        var (accepted, component, detail) = Certificate.Check(cov, c.Mu<T>(), c.Lambda<T>(), c.Lb<T>(), c.Ub<T>(), w, T.One);
        Assert.True(accepted, typeof(T).Name + ": the correctly rounded optimum of a tie across a bound pair must be accepted at kappa = 1: " + component + ": " + detail);
    }

    private static QpCase BoundPairWitness()
    {
        var problem = new QpCase
        {
            Id = "k7e-935007-135",
            Mode = "meanvar",
            CovText = [.. BoundPairWitnessCovariance.Split(';').Select(r => r.Split(','))],
            MuText = BoundPairWitnessMeans.Split(','),
            LambdaText = "1",
            LbText = "-1E+89",
            UbText = "1E+89",
        };
        var pattern = BoundPairWitnessPattern.Select(ch => ch == 'L' ? -1 : ch == 'U' ? 1 : 0).ToArray();
        var optimum = Oracle.ForPattern<double>(problem, pattern);
        Assert.True(optimum is not null, "k7e-935007-135: the pattern " + BoundPairWitnessPattern + " must be the exact KKT point");
        return new QpCase
        {
            Id = problem.Id,
            Mode = problem.Mode,
            CovText = problem.CovText,
            MuText = problem.MuText,
            LambdaText = problem.LambdaText,
            LbText = problem.LbText,
            UbText = problem.UbText,
            Optimum = optimum!,
        };
    }

    private const string BoundPairWitnessMeans =
        "2.7898078576873015E-23,65786056145275.47,1.3680945509740797E+46,1.442072031180521E-65,782698977.8941387,-5.027643857555136E+33,-5.073521044469966E-41,1.1730013789008158E-34,-7657735.799725139,3.3521461772944588E+62,-165004514.717017,-1.062148446019181E+58,5.892334686484354E+58,-3.0239576739635205E-67,9.991998628905614E-26,1.2858277832244846E-08,-1.7045892701684966E-29,2.5319993504335914E-48,-3.1279917678347036E-10,-4.0922531357646206E+63";

    private const string BoundPairWitnessCovariance =
        "1.8273181228672674E-114,2.3283064365386963E-10,5.345529420184391E-51,8.156630584998156E-56,7.275957614183426E-12,5.345529420184391E-51,8.156630584998156E-56,7.275957614183426E-12,1.5340917079055395E-92,3.111507638930571E-61,7.965459555662261E-59,2.3283064365386963E-10,4.930380657631324E-32,1.8909140209225187E-124,3.111507638930571E-61,2.0303534698525194E-115,4.930380657631324E-32,1.8909140209225187E-124,7.965459555662261E-59,1.5340917079055395E-92;"
        + "2.3283064365386963E-10,2.402985415411024E+96,6.129982163463556E+54,9.353610478917779E+49,8.343699359066055E+93,6.129982163463556E+54,9.353610478917779E+49,8.343699359066055E+93,17592186044416,3.5681192317649E+44,9.134385233318143E+46,2.6699837949011376E+95,5.65391060729083E+73,2.168404344971009E-19,3.5681192317649E+44,2.3283064365386963E-10,5.65391060729083E+73,2.168404344971009E-19,9.134385233318143E+46,17592186044416;"
        + "5.345529420184391E-51,6.129982163463556E+54,1266637395197952,2147483648,1.915619426082361E+53,140737488355328,2147483648,1.915619426082361E+53,4.0389678347315804E-28,8192,2097152,6.129982163463556E+54,1.298074214633707E+33,4.9784122222889134E-60,8192,5.345529420184391E-51,1.298074214633707E+33,4.9784122222889134E-60,2097152,4.0389678347315804E-28;"
        + "8.156630584998156E-56,9.353610478917779E+49,2147483648,294912,2.923003274661806E+48,2147483648,32768,2.923003274661806E+48,6.162975822039155E-33,0.125,32,9.353610478917779E+49,1.9807040628566084E+28,7.596454196607839E-65,0.125,8.156630584998156E-56,1.9807040628566084E+28,7.596454196607839E-65,32,6.162975822039155E-33;"
        + "7.275957614183426E-12,8.343699359066055E+93,1.915619426082361E+53,2.923003274661806E+48,2.346665444737328E+93,1.915619426082361E+53,2.923003274661806E+48,2.6074060497081422E+92,549755813888,1.1150372599265312E+43,2.85449538541192E+45,8.343699359066055E+93,1.7668470647783843E+72,6.776263578034403E-21,1.1150372599265312E+43,7.275957614183426E-12,1.7668470647783843E+72,6.776263578034403E-21,2.85449538541192E+45,549755813888;"
        + "5.345529420184391E-51,6.129982163463556E+54,140737488355328,2147483648,1.915619426082361E+53,1266637395197952,2147483648,1.915619426082361E+53,4.0389678347315804E-28,8192,2097152,6.129982163463556E+54,1.298074214633707E+33,4.9784122222889134E-60,8192,5.345529420184391E-51,1.298074214633707E+33,4.9784122222889134E-60,2097152,4.0389678347315804E-28;"
        + "8.156630584998156E-56,9.353610478917779E+49,2147483648,32768,2.923003274661806E+48,2147483648,294912,2.923003274661806E+48,6.162975822039155E-33,0.125,32,9.353610478917779E+49,1.9807040628566084E+28,7.596454196607839E-65,0.125,8.156630584998156E-56,1.9807040628566084E+28,7.596454196607839E-65,32,6.162975822039155E-33;"
        + "7.275957614183426E-12,8.343699359066055E+93,1.915619426082361E+53,2.923003274661806E+48,2.6074060497081422E+92,1.915619426082361E+53,2.923003274661806E+48,2.346665444737328E+93,549755813888,1.1150372599265312E+43,2.85449538541192E+45,8.343699359066055E+93,1.7668470647783843E+72,6.776263578034403E-21,1.1150372599265312E+43,7.275957614183426E-12,1.7668470647783843E+72,6.776263578034403E-21,2.85449538541192E+45,549755813888;"
        + "1.5340917079055395E-92,17592186044416,4.0389678347315804E-28,6.162975822039155E-33,549755813888,4.0389678347315804E-28,6.162975822039155E-33,549755813888,1.0432142298808373E-68,2.350988701644575E-38,6.018531076210112E-36,17592186044416,3.725290298461914E-09,1.4287342391028437E-101,2.350988701644575E-38,1.5340917079055395E-92,3.725290298461914E-09,1.4287342391028437E-101,6.018531076210112E-36,1.1591269220898192E-69;"
        + "3.111507638930571E-61,3.5681192317649E+44,8192,0.125,1.1150372599265312E+43,8192,0.125,1.1150372599265312E+43,2.350988701644575E-38,4.291534423828125E-06,0.0001220703125,3.5681192317649E+44,7.555786372591432E+22,2.897817305224548E-70,4.76837158203125E-07,3.111507638930571E-61,7.555786372591432E+22,2.897817305224548E-70,0.0001220703125,2.350988701644575E-38;"
        + "7.965459555662261E-59,9.134385233318143E+46,2097152,32,2.85449538541192E+45,2097152,32,2.85449538541192E+45,6.018531076210112E-36,0.0001220703125,0.28125,9.134385233318143E+46,1.9342813113834067E+25,7.418412301374843E-68,0.0001220703125,7.965459555662261E-59,1.9342813113834067E+25,7.418412301374843E-68,0.03125,6.018531076210112E-36;"
        + "2.3283064365386963E-10,2.6699837949011376E+95,6.129982163463556E+54,9.353610478917779E+49,8.343699359066055E+93,6.129982163463556E+54,9.353610478917779E+49,8.343699359066055E+93,17592186044416,3.5681192317649E+44,9.134385233318143E+46,2.402985415411024E+96,5.65391060729083E+73,2.168404344971009E-19,3.5681192317649E+44,2.3283064365386963E-10,5.65391060729083E+73,2.168404344971009E-19,9.134385233318143E+46,17592186044416;"
        + "4.930380657631324E-32,5.65391060729083E+73,1.298074214633707E+33,1.9807040628566084E+28,1.7668470647783843E+72,1.298074214633707E+33,1.9807040628566084E+28,1.7668470647783843E+72,3.725290298461914E-09,7.555786372591432E+22,1.9342813113834067E+25,5.65391060729083E+73,1.077535927171328E+53,4.591774807899561E-41,7.555786372591432E+22,4.930380657631324E-32,1.1972621413014757E+52,4.591774807899561E-41,1.9342813113834067E+25,3.725290298461914E-09;"
        + "1.8909140209225187E-124,2.168404344971009E-19,4.9784122222889134E-60,7.596454196607839E-65,6.776263578034403E-21,4.9784122222889134E-60,7.596454196607839E-65,6.776263578034403E-21,1.4287342391028437E-101,2.897817305224548E-70,7.418412301374843E-68,2.168404344971009E-19,4.591774807899561E-41,1.5849458229078602E-132,2.897817305224548E-70,1.8909140209225187E-124,4.591774807899561E-41,1.761050914342067E-133,7.418412301374843E-68,1.4287342391028437E-101;"
        + "3.111507638930571E-61,3.5681192317649E+44,8192,0.125,1.1150372599265312E+43,8192,0.125,1.1150372599265312E+43,2.350988701644575E-38,4.76837158203125E-07,0.0001220703125,3.5681192317649E+44,7.555786372591432E+22,2.897817305224548E-70,4.291534423828125E-06,3.111507638930571E-61,7.555786372591432E+22,2.897817305224548E-70,0.0001220703125,2.350988701644575E-38;"
        + "2.0303534698525194E-115,2.3283064365386963E-10,5.345529420184391E-51,8.156630584998156E-56,7.275957614183426E-12,5.345529420184391E-51,8.156630584998156E-56,7.275957614183426E-12,1.5340917079055395E-92,3.111507638930571E-61,7.965459555662261E-59,2.3283064365386963E-10,4.930380657631324E-32,1.8909140209225187E-124,3.111507638930571E-61,1.8273181228672674E-114,4.930380657631324E-32,1.8909140209225187E-124,7.965459555662261E-59,1.5340917079055395E-92;"
        + "4.930380657631324E-32,5.65391060729083E+73,1.298074214633707E+33,1.9807040628566084E+28,1.7668470647783843E+72,1.298074214633707E+33,1.9807040628566084E+28,1.7668470647783843E+72,3.725290298461914E-09,7.555786372591432E+22,1.9342813113834067E+25,5.65391060729083E+73,1.1972621413014757E+52,4.591774807899561E-41,7.555786372591432E+22,4.930380657631324E-32,1.077535927171328E+53,4.591774807899561E-41,1.9342813113834067E+25,3.725290298461914E-09;"
        + "1.8909140209225187E-124,2.168404344971009E-19,4.9784122222889134E-60,7.596454196607839E-65,6.776263578034403E-21,4.9784122222889134E-60,7.596454196607839E-65,6.776263578034403E-21,1.4287342391028437E-101,2.897817305224548E-70,7.418412301374843E-68,2.168404344971009E-19,4.591774807899561E-41,1.761050914342067E-133,2.897817305224548E-70,1.8909140209225187E-124,4.591774807899561E-41,1.5849458229078602E-132,7.418412301374843E-68,1.4287342391028437E-101;"
        + "7.965459555662261E-59,9.134385233318143E+46,2097152,32,2.85449538541192E+45,2097152,32,2.85449538541192E+45,6.018531076210112E-36,0.0001220703125,0.03125,9.134385233318143E+46,1.9342813113834067E+25,7.418412301374843E-68,0.0001220703125,7.965459555662261E-59,1.9342813113834067E+25,7.418412301374843E-68,0.28125,6.018531076210112E-36;"
        + "1.5340917079055395E-92,17592186044416,4.0389678347315804E-28,6.162975822039155E-33,549755813888,4.0389678347315804E-28,6.162975822039155E-33,549755813888,1.1591269220898192E-69,2.350988701644575E-38,6.018531076210112E-36,17592186044416,3.725290298461914E-09,1.4287342391028437E-101,2.350988701644575E-38,1.5340917079055395E-92,3.725290298461914E-09,1.4287342391028437E-101,6.018531076210112E-36,1.0432142298808373E-68";

    private static List<string> AcceptRejectFailures<T>(QpCase c)
        where T : IFloatingPoint<T>
    {
        var failures = new List<string>();
        var accept = Certificate.Check(c.Cov<T>(), c.Mu<T>(), c.Lambda<T>(), c.Lb<T>(), c.Ub<T>(), c.RoundedOptimum<T>(), T.One);
        if (!accept.Accepted)
        {
            failures.Add(c.Id + " (n=" + c.N + "): rounded optimum rejected as " + accept.Component + " (" + accept.Detail + ")");
        }

        foreach (var (w, outward, label) in Mutations<T>(c))
        {
            var (accepted, component, detail) = Certificate.Check(c.Cov<T>(), c.Mu<T>(), c.Lambda<T>(), c.Lb<T>(), c.Ub<T>(), w, T.One);
            if (accepted)
            {
                failures.Add(c.Id + " (n=" + c.N + ") " + label + ": accepted");
            }
            else if (outward && component != "Bound")
            {
                failures.Add(c.Id + " (n=" + c.N + ") " + label + ": rejected as " + component + " (" + detail + ")");
            }
        }

        return failures;
    }

    private static class HeldOut
    {
        public static QpCase Get(string name) => name switch
        {
            "dec-n5-minvar" => new QpCase
            {
                Id = name,
                Mode = "minvar",
                CovText =
                [
                    ["39.0000", "-14.0000", "30.0000", "-2500000.00", "-0.20000"],
                    ["-14.0000", "70.0000", "25.0000", "-800000.00", "0.40000"],
                    ["30.0000", "25.0000", "52.0000", "-2500000.00", "0.10000"],
                    ["-2500000.00", "-800000.00", "-2500000.00", "510000000000", "50000.000"],
                    ["-0.20000", "0.40000", "0.10000", "50000.000", "0.170000"],
                ],
                LbText = "-0.25",
                UbText = "1",
            },
            "dec-n4-meanvar" => new QpCase
            {
                Id = name,
                Mode = "meanvar",
                CovText =
                [
                    ["25.0000", "-190000.00", "14000.00", "-1700.00"],
                    ["-190000.00", "4700000000", "120000000", "37000000"],
                    ["14000.00", "120000000", "32000000", "400000"],
                    ["-1700.00", "37000000", "400000", "400000"],
                ],
                MuText = ["2500.00", "2100.00", "400.00", "2800.00"],
                LambdaText = "0.5",
                LbText = "-0.25",
                UbText = "2",
            },
            "dec-n3-minvar" => new QpCase
            {
                Id = name,
                Mode = "minvar",
                CovText = [["133000000.0", "280000.0", "4900.000"], ["280000.0", "15400.0", "-70.000"], ["4900.000", "-70.000", "1.12000"]],
                LbText = "-0.25",
                UbText = "1",
            },
            "dec-n3-meanvar" => new QpCase
            {
                Id = name,
                Mode = "meanvar",
                CovText = [["29900.0", "-208.00", "156000.0"], ["-208.00", "7.670", "-3380.00"], ["156000.0", "-3380.00", "2860000.0"]],
                MuText = ["0.70", "1.90", "-1.00"],
                LambdaText = "10",
                LbText = "0",
                UbText = "1",
            },
            "vb-2592-n5-minvar" => new QpCase
            {
                Id = name,
                Mode = "minvar",
                CovText =
                [
                    ["21700000.0", "1540000.0", "-1890000.0", "1610.0000", "77000000.0"],
                    ["1540000.0", "434000.0", "-343000.0", "231.0000", "-13300000.0"],
                    ["-1890000.0", "-343000.0", "364000.0", "-182.0000", "8400000.0"],
                    ["1610.0000", "231.0000", "-182.0000", "0.2310000", "0.0000"],
                    ["77000000.0", "-13300000.0", "8400000.0", "0.0000", "1750000000.0"],
                ],
                LbText = "-0.25",
                UbText = "2",
            },
            "vb-107-n5-meanvar" => new QpCase
            {
                Id = name,
                Mode = "meanvar",
                CovText =
                [
                    ["138000000", "-7800.000", "-39000000", "-11700000000", "-3900000000"],
                    ["-7800.000", "3.090000", "-4500.000", "1200000.000", "1260000.000"],
                    ["-39000000", "-4500.000", "81000000", "2700000000", "-3300000000"],
                    ["-11700000000", "1200000.000", "2700000000", "2460000000000", "390000000000"],
                    ["-3900000000", "1260000.000", "-3300000000", "390000000000", "780000000000"],
                ],
                MuText = ["2300.00", "1100.00", "700.00", "1500.00", "-2200.00"],
                LambdaText = "0.5",
                LbText = "-0.25",
                UbText = "1",
            },
            _ => throw new ArgumentOutOfRangeException(nameof(name)),
        };
    }

    // Seeded problem generator (not the corpus generator): integer Gram matrix plus a diagonal shift, scaled per row
    // and globally by radix powers and by a non-radix factor so equilibration and the absolute decimal grid matter.
    private static class Generator
    {
        public static QpCase Next<T>(System.Random rng, string id)
            where T : IFloatingPoint<T>
        {
            var isDecimal = typeof(T) == typeof(decimal);
            var n = rng.Next(3, 6);
            var a = new int[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    a[i, j] = rng.Next(-5, 6);
                }
            }

            var shift = rng.Next(1, 4);
            var factors = new[] { "1", "3", "0.7", "1.3" };
            var c = Parse<T>(factors[rng.Next(factors.Length)]);
            var s = new T[n];
            for (var i = 0; i < n; i++)
            {
                s[i] = isDecimal ? Parse<T>("1E" + rng.Next(-3, 4).ToString(CultureInfo.InvariantCulture)) : T.CreateChecked(Math.ScaleB(1.0, rng.Next(-12, 13)));
            }

            var g = isDecimal ? Parse<T>("1E" + rng.Next(-4, 5).ToString(CultureInfo.InvariantCulture)) : T.CreateChecked(Math.ScaleB(1.0, rng.Next(-30, 31)));
            var cov = new string[n][];
            for (var i = 0; i < n; i++)
            {
                cov[i] = new string[n];
            }

            for (var i = 0; i < n; i++)
            {
                for (var j = i; j < n; j++)
                {
                    long gram = i == j ? shift : 0;
                    for (var q = 0; q < n; q++)
                    {
                        gram += a[i, q] * a[j, q];
                    }

                    var v = T.CreateChecked(gram) * c * s[i] * s[j] * g;
                    cov[i][j] = Text(v);
                    cov[j][i] = cov[i][j];
                }
            }

            var meanVariance = rng.Next(2) == 1;
            string[] mu = [];
            var lambda = "";
            if (meanVariance)
            {
                mu = new string[n];
                for (var i = 0; i < n; i++)
                {
                    mu[i] = Text(Parse<T>((rng.Next(-30, 31) / 100.0).ToString("0.00", CultureInfo.InvariantCulture)) * g);
                }

                var lambdas = new[] { "1", "2", "0.5", "10" };
                lambda = lambdas[rng.Next(lambdas.Length)];
            }

            var lbs = new[] { "0", "0.05", "-0.25" };
            var ubs = new[] { "1", "0.5", "0.35", "2" };
            var lb = lbs[rng.Next(lbs.Length)];
            var ub = ubs[rng.Next(ubs.Length)];
            var lbR = Rational.ParseScientific(lb);
            var ubR = Rational.ParseScientific(ub);
            var nR = Rational.Of(n, 1);
            if (!(lbR * nR < Rational.Of(1, 1)) || !(ubR * nR > Rational.Of(1, 1)))
            {
                lb = "0";
                ub = "1";
            }

            return new QpCase
            {
                Id = id,
                Mode = meanVariance ? "meanvar" : "minvar",
                CovText = cov,
                MuText = mu,
                LambdaText = lambda,
                LbText = lb,
                UbText = ub,
            };
        }

        private static string Text<T>(T v)
            where T : IFloatingPoint<T>
            => typeof(T) == typeof(double) ? ((double)(object)v).ToString("R", CultureInfo.InvariantCulture) : ((decimal)(object)v).ToString(CultureInfo.InvariantCulture);

        private static T Parse<T>(string s)
            where T : IFloatingPoint<T> => T.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    // Exact rational KKT oracle for small problems (active-set pattern enumeration) and the one-ulp input
    // sensitivity of the corpus definition: the largest max|Δw*| / max|w*| over single-entry ±1-ulp
    // perturbations of the working-type inputs (covariance entries, bounds, means, risk aversion).
    private static class Oracle
    {
        public static QpCase Complete<T>(QpCase c)
            where T : IFloatingPoint<T>
        {
            var n = c.N;
            var covT = c.Cov<T>();
            var s = new Rational[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    s[i, j] = Rational.From(covT[i, j]);
                }
            }

            var muT = c.Mu<T>();
            var mu = muT?.Select(Rational.From).ToArray();
            var lambdaT = c.Lambda<T>();
            var lambda = Rational.From(lambdaT);
            var lbT = c.Lb<T>();
            var ubT = c.Ub<T>();
            var lb = Rational.From(lbT);
            var ub = Rational.From(ubT);
            var solved = Solve(s, mu, lambda, lb, ub, null);
            Assert.True(solved is not null, c.Id + ": the oracle found no KKT point");
            var (w, pattern) = solved!.Value;
            var free = Enumerable.Range(0, n).Where(i => pattern[i] == 0).ToArray();
            var unitRoundoff = typeof(T) == typeof(double) ? Rational.Of(BigInteger.One, BigInteger.One << 53) : Rational.Of(BigInteger.One, BigInteger.Pow(10, 28));
            var qualified = InDomain(covT, muT, lambdaT, lbT, ubT) && Rational.Of(4, 1) * unitRoundoff * Cond1(s, free) < Rational.Of(1, 1);

            var scale = w.Select(x => x.Abs()).Aggregate((x, y) => x > y ? x : y);
            var worst = default(Rational);
            void Update(Rational[,] s2, Rational[]? mu2, Rational lambda2, Rational lb2, Rational ub2)
            {
                if (!IsPositiveDefinite(s2))
                {
                    return;
                }

                var r = Solve(s2, mu2, lambda2, lb2, ub2, pattern);
                if (r is null)
                {
                    return;
                }

                var d = Enumerable.Range(0, n).Select(i => (r.Value.W[i] - w[i]).Abs()).Aggregate((x, y) => x > y ? x : y) / scale;
                if (d > worst)
                {
                    worst = d;
                }
            }

            for (var i = 0; i < n; i++)
            {
                for (var j = i; j < n; j++)
                {
                    foreach (var v in Neighbours(covT[i, j]))
                    {
                        var s2 = (Rational[,])s.Clone();
                        s2[i, j] = v;
                        s2[j, i] = v;
                        Update(s2, mu, lambda, lb, ub);
                    }
                }
            }

            foreach (var v in Neighbours(lbT))
            {
                Update(s, mu, lambda, v, ub);
            }

            foreach (var v in Neighbours(ubT))
            {
                Update(s, mu, lambda, lb, v);
            }

            if (mu is not null)
            {
                for (var i = 0; i < n; i++)
                {
                    foreach (var v in Neighbours(muT![i]))
                    {
                        var mu2 = (Rational[])mu.Clone();
                        mu2[i] = v;
                        Update(s, mu2, lambda, lb, ub);
                    }
                }

                foreach (var v in Neighbours(lambdaT))
                {
                    if (v.Sign > 0)
                    {
                        Update(s, mu, v, lb, ub);
                    }
                }
            }

            return new QpCase
            {
                Id = c.Id,
                Mode = c.Mode,
                CovText = c.CovText,
                MuText = c.MuText,
                LambdaText = c.LambdaText,
                LbText = c.LbText,
                UbText = c.UbText,
                Optimum = w,
                Qualified = qualified,
                Representable = true,
                InDomain = true,
                Sensitivity = ToDouble(worst).ToString("R", CultureInfo.InvariantCulture),
            };
        }

        private static double ToDouble(Rational r) => r.Sign == 0 ? 0.0 : Rounding.ToDouble(r);

        // Operating domain of the supported-types contract: double 1e-150 <= |x| <= 1e150, decimal |x| <= 1e14.
        private static bool InDomain<T>(T[,] cov, T[]? mu, T lambda, T lb, T ub)
            where T : IFloatingPoint<T>
        {
            var values = cov.Cast<T>().Concat([lb, ub]).Concat(mu is null ? [] : mu.Append(lambda));
            foreach (var x in values)
            {
                var v = Rational.From(x).Abs();
                if (v.Sign == 0)
                {
                    continue;
                }

                var ok = typeof(T) == typeof(double)
                    ? !(v < Rational.Of(BigInteger.One, BigInteger.Pow(10, 150))) && !(v > Rational.Of(BigInteger.Pow(10, 150), BigInteger.One))
                    : !(v > Rational.Of(BigInteger.Pow(10, 14), BigInteger.One));
                if (!ok)
                {
                    return false;
                }
            }

            return true;
        }

        // Exact 1-norm condition number ||A||1 ||A^-1||1 of the free block (1 for an empty block).
        private static Rational Cond1(Rational[,] s, int[] free)
        {
            var k = free.Length;
            if (k == 0)
            {
                return Rational.Of(1, 1);
            }

            var a = new Rational[k, 2 * k];
            for (var r = 0; r < k; r++)
            {
                for (var col = 0; col < k; col++)
                {
                    a[r, col] = s[free[r], free[col]];
                }

                a[r, k + r] = Rational.Of(1, 1);
            }

            for (var col = 0; col < k; col++)
            {
                var pivot = col;
                while (a[pivot, col].Sign == 0)
                {
                    pivot++;
                }

                for (var q = 0; q < 2 * k; q++)
                {
                    (a[col, q], a[pivot, q]) = (a[pivot, q], a[col, q]);
                }

                var d = a[col, col];
                for (var q = 0; q < 2 * k; q++)
                {
                    a[col, q] = a[col, q] / d;
                }

                for (var r = 0; r < k; r++)
                {
                    if (r == col || a[r, col].Sign == 0)
                    {
                        continue;
                    }

                    var f = a[r, col];
                    for (var q = 0; q < 2 * k; q++)
                    {
                        a[r, q] -= f * a[col, q];
                    }
                }
            }

            Rational Norm(int offset)
            {
                var best = default(Rational);
                for (var col = 0; col < k; col++)
                {
                    var sum = default(Rational);
                    for (var r = 0; r < k; r++)
                    {
                        sum += offset == 0 ? s[free[r], free[col]].Abs() : a[r, k + col].Abs();
                    }

                    best = sum > best ? sum : best;
                }

                return best;
            }

            return Norm(0) * Norm(k);
        }

        private static IEnumerable<Rational> Neighbours<T>(T x)
            where T : IFloatingPoint<T>
        {
            if (typeof(T) == typeof(double))
            {
                var d = (double)(object)x;
                yield return Rational.FromDouble(Math.BitIncrement(d));
                yield return Rational.FromDouble(Math.BitDecrement(d));
                yield break;
            }

            var v = Rational.From(x);
            var unit = DecimalUnit(Math.Abs((decimal)(object)x));
            yield return v + unit;
            yield return v - unit;
        }

        // The exact optimum for a given active pattern (−1 lower bound, +1 upper bound, 0 free), or null unless every
        // KKT condition holds exactly at it: for problems too large for the pattern enumeration of Complete.
        public static Rational[]? ForPattern<T>(QpCase c, int[] pattern)
            where T : IFloatingPoint<T>
        {
            var n = c.N;
            var covT = c.Cov<T>();
            var s = new Rational[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    s[i, j] = Rational.From(covT[i, j]);
                }
            }

            var mu = c.Mu<T>()?.Select(Rational.From).ToArray();
            return TryPattern(s, mu, Rational.From(c.Lambda<T>()), Rational.From(c.Lb<T>()), Rational.From(c.Ub<T>()), pattern);
        }

        private static (Rational[] W, int[] Pattern)? Solve(Rational[,] s, Rational[]? mu, Rational lambda, Rational lb, Rational ub, int[]? hint)
        {
            var n = s.GetLength(0);
            if (hint is not null)
            {
                var r = TryPattern(s, mu, lambda, lb, ub, hint);
                if (r is not null)
                {
                    return (r, hint);
                }
            }

            var total = (int)Math.Pow(3, n);
            for (var code = 0; code < total; code++)
            {
                var p = new int[n];
                var x = code;
                for (var i = 0; i < n; i++)
                {
                    p[i] = (x % 3) - 1;
                    x /= 3;
                }

                var r = TryPattern(s, mu, lambda, lb, ub, p);
                if (r is not null)
                {
                    return (r, p);
                }
            }

            return null;
        }

        // Minimise (λ/2)w′Σw − μ′w (minimum variance: μ = 0, λ = 1) with the variables of pattern p fixed at their
        // bounds (−1 lower, +1 upper) and the rest free; null unless the KKT conditions hold exactly.
        private static Rational[]? TryPattern(Rational[,] s, Rational[]? mu, Rational lambda, Rational lb, Rational ub, int[] p)
        {
            var n = p.Length;
            var w = new Rational[n];
            var free = Enumerable.Range(0, n).Where(i => p[i] == 0).ToArray();
            var fixedSum = default(Rational);
            for (var i = 0; i < n; i++)
            {
                if (p[i] != 0)
                {
                    w[i] = p[i] < 0 ? lb : ub;
                    fixedSum += w[i];
                }
            }

            var one = Rational.Of(1, 1);
            Rational nu;
            if (free.Length == 0)
            {
                if ((fixedSum - one).Sign != 0)
                {
                    return null;
                }

                Rational? low = null;
                Rational? high = null;
                for (var i = 0; i < n; i++)
                {
                    var gi = Gradient(s, mu, lambda, w, i);
                    if (p[i] > 0)
                    {
                        low = low is null || gi > low.Value ? gi : low;
                    }
                    else
                    {
                        high = high is null || gi < high.Value ? gi : high;
                    }
                }

                return low is not null && high is not null && low.Value > high.Value ? null : w;
            }

            var m = free.Length + 1;
            var a = new Rational[m, m + 1];
            for (var r = 0; r < free.Length; r++)
            {
                var i = free[r];
                for (var col = 0; col < free.Length; col++)
                {
                    a[r, col] = lambda * s[i, free[col]];
                }

                a[r, m - 1] = Rational.Of(-1, 1);
                var rhs = mu is null ? default : mu[i];
                for (var j = 0; j < n; j++)
                {
                    if (p[j] != 0)
                    {
                        rhs -= lambda * s[i, j] * w[j];
                    }
                }

                a[r, m] = rhs;
            }

            for (var col = 0; col < free.Length; col++)
            {
                a[m - 1, col] = one;
            }

            a[m - 1, m] = one - fixedSum;
            var x = Gauss(a, m);
            if (x is null)
            {
                return null;
            }

            for (var r = 0; r < free.Length; r++)
            {
                w[free[r]] = x[r];
                if (x[r] < lb || x[r] > ub)
                {
                    return null;
                }
            }

            nu = x[m - 1];
            for (var i = 0; i < n; i++)
            {
                if (p[i] == 0)
                {
                    continue;
                }

                var d = Gradient(s, mu, lambda, w, i) - nu;
                if ((p[i] < 0 && d.Sign < 0) || (p[i] > 0 && d.Sign > 0))
                {
                    return null;
                }
            }

            return w;
        }

        private static Rational Gradient(Rational[,] s, Rational[]? mu, Rational lambda, Rational[] w, int i)
        {
            var sum = default(Rational);
            for (var j = 0; j < w.Length; j++)
            {
                sum += s[i, j] * w[j];
            }

            return (lambda * sum) - (mu is null ? default : mu[i]);
        }

        private static Rational[]? Gauss(Rational[,] a, int m)
        {
            for (var col = 0; col < m; col++)
            {
                var pivot = -1;
                for (var r = col; r < m; r++)
                {
                    if (a[r, col].Sign != 0)
                    {
                        pivot = r;
                        break;
                    }
                }

                if (pivot < 0)
                {
                    return null;
                }

                if (pivot != col)
                {
                    for (var k = 0; k <= m; k++)
                    {
                        (a[col, k], a[pivot, k]) = (a[pivot, k], a[col, k]);
                    }
                }

                for (var r = 0; r < m; r++)
                {
                    if (r == col || a[r, col].Sign == 0)
                    {
                        continue;
                    }

                    var f = a[r, col] / a[col, col];
                    for (var k = col; k <= m; k++)
                    {
                        a[r, k] -= f * a[col, k];
                    }
                }
            }

            var x = new Rational[m];
            for (var r = 0; r < m; r++)
            {
                x[r] = a[r, m] / a[r, r];
            }

            return x;
        }

        private static bool IsPositiveDefinite(Rational[,] s)
        {
            var n = s.GetLength(0);
            var a = (Rational[,])s.Clone();
            for (var col = 0; col < n; col++)
            {
                if (a[col, col].Sign <= 0)
                {
                    return false;
                }

                for (var r = col + 1; r < n; r++)
                {
                    var f = a[r, col] / a[col, col];
                    for (var k = col; k < n; k++)
                    {
                        a[r, k] -= f * a[col, k];
                    }
                }
            }

            return true;
        }
    }

    private static string ComponentWord(string component) => component switch
    {
        "Bound" => "bound",
        "Budget" => "budget",
        "Stationarity" => "stationarity",
        _ => "multiplier sign",
    };

    private static void AssertEveryMutationRejected<T>(string type)
        where T : IFloatingPoint<T>
    {
        var cases = Corpus.Load(type).Where(c => c.Qualified).ToList();
        Assert.True(cases.Count >= QualifiedCaseFloor, "the qualified " + type + " corpus must be present, found " + cases.Count);

        var failures = new List<string>();
        var checkedCopies = 0;
        foreach (var c in cases)
        {
            foreach (var (w, outward, label) in Mutations<T>(c))
            {
                checkedCopies++;
                var (accepted, component, detail) = Certificate.Check(c.Cov<T>(), c.Mu<T>(), c.Lambda<T>(), c.Lb<T>(), c.Ub<T>(), w, T.One);
                if (accepted)
                {
                    failures.Add(c.Id + " " + label + ": accepted");
                }
                else if (component == "None" || (outward && component != "Bound"))
                {
                    failures.Add(c.Id + " " + label + ": rejected as " + component + " (" + detail + ")");
                }
            }
        }

        Assert.Equal(2 * cases.Sum(c => c.N), checkedCopies);
        Assert.True(failures.Count == 0, failures.Count + " of " + checkedCopies + " perturbed copies not rejected correctly: " + string.Join("; ", failures.Take(12)));
    }

    private static void AssertLooseningAcceptsEveryMutation<T>(string type, int maxDoublings)
        where T : IFloatingPoint<T>
    {
        var cases = Corpus.Load(type).Where(c => c.Qualified).ToList();
        Assert.True(cases.Count >= QualifiedCaseFloor, "the qualified " + type + " corpus must be present, found " + cases.Count);

        var neverAccepted = new List<string>();
        var beyondRange = new List<string>();
        var acceptedAtDefault = new List<string>();
        var firstAcceptance = new SortedDictionary<int, int>();
        var two = T.One + T.One;
        foreach (var c in cases)
        {
            // An outward move past a bound the optimum sits on is a bound decision (asserted as Bound above); its
            // tolerance is half a unit of the bound itself, which at a zero bound is the subnormal half-step, beyond
            // any representable loosening factor at double. The teeth are measured on the inward copies.
            foreach (var (w, _, label) in Mutations<T>(c).Where(m => !m.Outward))
            {
                var scale = T.One;
                var accepted = false;
                for (var p = 0; p <= maxDoublings && !accepted; p++)
                {
                    if (p > 0)
                    {
                        scale *= two;
                    }

                    accepted = Certificate.Check(c.Cov<T>(), c.Mu<T>(), c.Lambda<T>(), c.Lb<T>(), c.Ub<T>(), w, scale).Accepted;
                    if (accepted)
                    {
                        firstAcceptance[p] = firstAcceptance.GetValueOrDefault(p) + 1;
                        if (p == 0)
                        {
                            acceptedAtDefault.Add(c.Id + " " + label);
                        }
                    }
                }

                if (!accepted && typeof(T) == typeof(decimal))
                {
                    // At decimal the loosening factor is itself a decimal: after 2^95 the only larger factor left is
                    // decimal's maximum (about 2^96). Acceptance is monotone in the factor (every component tolerance is
                    // multiplied by it), so a copy still rejected there needs a loosening no decimal can represent.
                    scale = T.CreateChecked(decimal.MaxValue);
                    var (atMaximum, component, _) = Certificate.Check(c.Cov<T>(), c.Mu<T>(), c.Lambda<T>(), c.Lb<T>(), c.Ub<T>(), w, scale);
                    accepted = atMaximum;
                    if (!accepted && component != "Bound" && component != "None")
                    {
                        // The regenerated corpus qualifies range problems (equilibration removes their spread) whose 4x
                        // move is resolved with a margin beyond every representable factor. Such a copy is rejected by a
                        // component whose tolerance the factor scales (budget, stationarity, multiplier sign), never by a
                        // bound; the exact set of such copies is pinned below, so no other copy can join it unnoticed.
                        beyondRange.Add(c.Id + " " + label + " (" + component + ")");
                        continue;
                    }
                }

                if (!accepted)
                {
                    neverAccepted.Add(c.Id + " " + label);
                }
            }
        }

        File.WriteAllText(
            Path.Combine(AppContext.BaseDirectory, "qp-certificate-teeth-beyond-range-" + type + ".txt"),
            beyondRange.Count + " copies rejected beyond the largest representable loosening" + Environment.NewLine + string.Join(Environment.NewLine, beyondRange));

        // Advisory: the histogram of log2(loosening factor) at which each mutation is first accepted.
        File.WriteAllText(
            Path.Combine(AppContext.BaseDirectory, "qp-certificate-teeth-" + type + ".json"),
            JsonSerializer.Serialize(firstAcceptance.ToDictionary(kv => kv.Key.ToString(CultureInfo.InvariantCulture), kv => kv.Value)));

        Assert.True(acceptedAtDefault.Count == 0, "accepted at the default tolerance: " + string.Join("; ", acceptedAtDefault.Take(12)));
        Assert.True(neverAccepted.Count == 0, "still rejected with every tolerance loosened by 2^" + maxDoublings + ": " + string.Join("; ", neverAccepted.Take(12)));
        var expectedBeyondRange = typeof(T) == typeof(decimal) ? s_decimalCopiesBeyondTheRepresentableLoosening : [];
        Assert.True(
            beyondRange.SequenceEqual(expectedBeyondRange, StringComparer.Ordinal),
            "the copies rejected at every representable loosening differ from the pinned set: found [" + string.Join("; ", beyondRange)
            + "], pinned [" + string.Join("; ", expectedBeyondRange) + "]");
    }

    // The decimal copies still rejected with every tolerance loosened by decimal's maximum (the largest representable
    // factor), each by a component the factor scales. All are r5-dec-zc2 range problems: the equilibration qualifies
    // them, and the 4x move of a weight far below the resolution of the dominant one is resolved with a margin beyond
    // 7.9e28. (Stopping at 2^95 left 30 such copies; 24 of them are accepted at decimal's maximum.) The set is pinned,
    // in corpus order, so the teeth check stays strict for every other copy.
    private static readonly string[] s_decimalCopiesBeyondTheRepresentableLoosening =
    [
        "r5-dec-zc2-13 w[3]+delta (Stationarity)", "r5-dec-zc2-13 w[3]-delta (Stationarity)",
        "r5-dec-zc2-29 w[3]+delta (Stationarity)", "r5-dec-zc2-29 w[3]-delta (Stationarity)",
        "r5-dec-zc2-38 w[3]+delta (Stationarity)", "r5-dec-zc2-38 w[3]-delta (Stationarity)",
    ];

    // AC-2.2 (amended 2026-10-08): every weight k of the correctly rounded optimum, in turn, moved in each direction
    // by delta_k = max(4 x the case's sensitivity bound, max(4, n+1) ulps of the perturbed weight), the moved value
    // rounded to nearest. The sensitivity bound is sensitivity x max|w*| (the corpus sensitivity is relative to
    // max|w*|, as in the AC-3.1 accuracy bound); the ulp is that of the rounded weight w_k (decimal: one unit in
    // its 28th significant digit, at least 1e-28, as in ActiveSetQpCorpusTests). The n+1 term is the smallest move
    // a residual certificate can resolve: a correctly rounded optimum's budget residual is up to n/2 grid units.
    private static IEnumerable<(T[] Weights, bool Outward, string Label)> Mutations<T>(QpCase c)
        where T : IFloatingPoint<T>
    {
        var rounded = c.RoundedOptimum<T>();
        var m = c.Optimum.Select(x => x.Abs()).Aggregate((x, y) => x > y ? x : y);
        var sensitivity = double.Parse(c.Sensitivity, NumberStyles.Float, CultureInfo.InvariantCulture);
        Assert.True(double.IsFinite(sensitivity), c.Id + ": qualified case without a finite sensitivity");
        var scaled = Rational.Of(4, 1) * Rational.FromDouble(sensitivity) * m;
        var units = Rational.Of(Math.Max(4, c.N + 1), 1);
        var lb = Rational.ParseScientific(c.LbText);
        var ub = Rational.ParseScientific(c.UbText);
        for (var k = 0; k < c.N; k++)
        {
            var floor = units * Ulp(rounded[k]);
            var delta = scaled > floor ? scaled : floor;
            foreach (var sign in new[] { 1, -1 })
            {
                var target = sign > 0 ? c.Optimum[k] + delta : c.Optimum[k] - delta;
                var copy = (T[])rounded.Clone();
                copy[k] = Rounding.ToNearest<T>(target);
                // A move that leaves the box is a bound decision: from an optimum on the bound, or from one closer to the
                // bound than delta (the regenerated corpus qualifies range problems whose optimum puts weights far below
                // the resolution of the dominant one, a hair inside a zero bound).
                var outward = target.CompareTo(lb) < 0 || target.CompareTo(ub) > 0;
                yield return (copy, outward, "w[" + k + "]" + (sign > 0 ? "+" : "-") + "delta");
            }
        }
    }

    // One ulp of a weight of the working type (double: the spacing above |w|; decimal: DecimalUnit).
    private static Rational Ulp<T>(T weight)
        where T : IFloatingPoint<T>
    {
        if (typeof(T) == typeof(double))
        {
            var md = Math.Abs((double)(object)weight);
            return Rational.FromDouble(Math.BitIncrement(md) - md);
        }

        return DecimalUnit(Math.Abs((decimal)(object)weight));
    }

    // One decimal unit in the 28th significant digit (at least 1e-28), as in ActiveSetQpCorpusTests.
    private static Rational DecimalUnit(decimal d)
    {
        if (d == 0m)
        {
            return Rational.Of(BigInteger.One, BigInteger.Pow(10, 28));
        }

        var bits = decimal.GetBits(d);
        var magnitude = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | new BigInteger((uint)bits[0]);
        var scale = (bits[3] >> 16) & 0xFF;
        var adjusted = magnitude.ToString(CultureInfo.InvariantCulture).Length - 1 - scale;
        var exponent = Math.Max(adjusted - 27, -28);
        return exponent >= 0 ? Rational.Of(BigInteger.Pow(10, exponent), BigInteger.One) : Rational.Of(BigInteger.One, BigInteger.Pow(10, -exponent));
    }

    private static class ComponentViolation
    {
        public static (QpCase Case, Rational[] Weights) Get(string name)
        {
            QpCase Diag124(string lb) => new()
            {
                Id = name,
                Mode = "minvar",
                CovText = [["1", "0", "0"], ["0", "2", "0"], ["0", "0", "4"]],
                LbText = lb,
                UbText = "1",
            };

            var relaxed = new[] { Rational.Of(4, 7), Rational.Of(2, 7), Rational.Of(1, 7) };
            var eps = Rational.Of(1, 1000000);
            return name switch
            {
                "bound" => (Diag124("0.2"), relaxed),
                "budget" => (Diag124("0"), relaxed.Select(x => x * (Rational.Of(1, 1) + eps)).ToArray()),
                "stationarity" => (Diag124("0"), [relaxed[0] + eps, relaxed[1] - eps, relaxed[2]]),
                "multiplier" => (new QpCase
                {
                    Id = name,
                    Mode = "minvar",
                    CovText = [["1", "0"], ["0", "1"]],
                    LbText = "0.1",
                    UbText = "0.9",
                }, [Rational.Of(9, 10), Rational.Of(1, 10)]),
                _ => throw new ArgumentOutOfRangeException(nameof(name)),
            };
        }
    }

    // ------------------------------------------------------------------------------------------
    // Stage work: each check records one run of its loops on the thread it runs on
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// Every check records one run of the certificate, with the loop bodies it executed, on the stage-work record of the
    /// thread it runs on. A certified candidate of n assets runs every loop of the check to its end: the scan of the
    /// covariance and the stationarity products n² bodies each, and the stationarity rows and the passes over the weights
    /// (bounds, budget, free variables, bound pairs, multiplier signs) n each, <c>2n² + 6n</c> in all, counted here from
    /// the loop structure the remarks describe. A rejected candidate stops at its failed component, below that. A solve
    /// records every check it makes, at least the one of the result it returns, each within <c>2n² + 6n</c>.
    /// </summary>
    [Fact]
    public void Check_RecordsOneRunOfItsLoopBodiesOnTheThreadItRunsOn()
    {
        AssertCertificateWork<double>("double");
        AssertCertificateWork<decimal>("decimal");
    }

    private static void AssertCertificateWork<T>(string type)
        where T : IFloatingPoint<T>
    {
        var cases = Corpus.Load(type).Where(c => c.Qualified).Take(40).ToList();
        Assert.NotEmpty(cases);
        foreach (var c in cases)
        {
            var n = c.N;
            var full = (2L * n * n) + (6L * n);
            var accepted = false;
            var work = CertificateWork.Run<T>(() => accepted = Certificate.Check(c.Cov<T>(), c.Mu<T>(), c.Lambda<T>(), c.Lb<T>(), c.Ub<T>(), c.RoundedOptimum<T>(), T.One).Accepted);
            Assert.True(accepted, c.Id);
            Assert.True(work == (1, 0, full, (long)n * n), $"{type} {c.Id} (n = {n}): certified check recorded {work}, expected one run of {full} loop bodies");

            // One weight moved by a unit plus its own magnitude fails the budget row (or a bound) and ends the check early.
            var moved = c.RoundedOptimum<T>();
            moved[0] += T.One + T.Abs(moved[0]);
            var rejected = CertificateWork.Run<T>(() => accepted = Certificate.Check(c.Cov<T>(), c.Mu<T>(), c.Lambda<T>(), c.Lb<T>(), c.Ub<T>(), moved, T.One).Accepted);
            Assert.False(accepted, c.Id);
            Assert.True(rejected.Runs == 1 && rejected.Iterations < full, $"{type} {c.Id}: rejected check recorded {rejected}");
        }

        foreach (var c in cases.Where(c => c.Mode == "minvar" || c.Lambda<T>() > T.Zero).Take(8))
        {
            var n = c.N;
            var full = (2L * n * n) + (6L * n);
            var work = CertificateWork.Run<T>(() =>
            {
                try
                {
                    _ = c.Mode == "minvar"
                        ? ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(c.Cov<T>(), c.Lb<T>(), c.Ub<T>())
                        : ActiveSetQpSolver<T>.SolveMeanVarianceUsingCore(c.Cov<T>(), c.Mu<T>()!, c.Lambda<T>(), c.Lb<T>(), c.Ub<T>());
                }
                catch (InvalidOperationException)
                {
                    // A documented refusal still certified (or refused) a candidate first.
                }
            });
            Assert.True(work.Runs >= 1 && work.Iterations >= full && work.Iterations <= work.Runs * full && work.OrderSquares == (long)work.Runs * n * n, $"{type} {c.Id} solve (n = {n}): {work}");
        }
    }

    // Runs an action with a fresh stage-work record installed for the certificate on this thread.
    private static class CertificateWork
    {
        public static (int Runs, int RepeatedRuns, long Iterations, long OrderSquares) Run<T>(Action action)
            where T : IFloatingPoint<T>
        {
            var work = new ActiveSetQpSolver<T>.StageWork();
            QpKktCertificate<T>.ThreadWork = work;
            try
            {
                action();
            }
            finally
            {
                QpKktCertificate<T>.ThreadWork = null;
            }

            return (work.Runs, work.RepeatedRuns, work.Iterations, work.OrderSquares);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Reflection accessor: Check(covariance, means-or-null, riskAversion, lower, upper, weights, toleranceScale)
    // ------------------------------------------------------------------------------------------

    private static class Certificate
    {
        private static readonly Type? s_open = typeof(NumericPrecision<>).Assembly.GetType("Boutquin.Numerics.Internal.QpKktCertificate`1");

        public static (bool Accepted, string Component, string Detail) Check<T>(T[,] covariance, T[]? means, T riskAversion, T lower, T upper, T[] weights, T toleranceScale)
            where T : IFloatingPoint<T>
        {
            Assert.True(s_open is not null, "Boutquin.Numerics.Internal.QpKktCertificate<T> does not exist");
            var closed = s_open!.MakeGenericType(typeof(T));
            Assert.False(closed.IsPublic, "QpKktCertificate<T> must be internal (PublicAPI unchanged)");
            var method = closed.GetMethod(
                "Check",
                BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic,
                [typeof(T[,]), typeof(T[]), typeof(T), typeof(T), typeof(T), typeof(T[]), typeof(T)]);
            Assert.True(method is not null, "QpKktCertificate<T>.Check(T[,], T[]?, T, T, T, T[], T) does not exist");
            return Unpack(Invoke(method!, [covariance, means, riskAversion, lower, upper, weights, toleranceScale]));
        }

        // The overload without a tolerance scale: the nominal κ = 1 of design step 4.
        public static (bool Accepted, string Component, string Detail) CheckNominal<T>(T[,] covariance, T[]? means, T riskAversion, T lower, T upper, T[] weights)
            where T : IFloatingPoint<T>
        {
            Assert.True(s_open is not null, "Boutquin.Numerics.Internal.QpKktCertificate<T> does not exist");
            var method = s_open!.MakeGenericType(typeof(T)).GetMethod(
                "Check",
                BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic,
                [typeof(T[,]), typeof(T[]), typeof(T), typeof(T), typeof(T), typeof(T[])]);
            Assert.True(method is not null, "QpKktCertificate<T>.Check(T[,], T[]?, T, T, T, T[]) (the nominal kappa = 1) does not exist");
            return Unpack(Invoke(method!, [covariance, means, riskAversion, lower, upper, weights]));
        }

        // Reflection wraps a thrown exception; rethrow the certificate's own so tests can assert its type.
        private static object Invoke(MethodInfo method, object?[] arguments)
        {
            try
            {
                return method.Invoke(null, arguments)!;
            }
            catch (TargetInvocationException e) when (e.InnerException is not null)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerException).Throw();
                throw;
            }
        }

        private static (bool Accepted, string Component, string Detail) Unpack(object result)
        {
            var type = result.GetType();
            var accepted = (bool)type.GetProperty("Accepted")!.GetValue(result)!;
            var component = type.GetProperty("FailedComponent")!.GetValue(result)!.ToString()!;
            var detail = (string?)type.GetProperty("Detail")!.GetValue(result) ?? "";
            return (accepted, component, detail);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Corpus fixtures and hand-built problems
    // ------------------------------------------------------------------------------------------

    // QpCase, Corpus, Rounding and Rational are shared with QpScalingTests.
    internal sealed class QpCase
    {
        public string Id { get; init; } = "";

        public string Family { get; init; } = "";

        public string Type { get; init; } = "";

        public string Mode { get; init; } = "";

        public string[][] CovText { get; init; } = [];

        public string[] MuText { get; init; } = [];

        public string LambdaText { get; init; } = "";

        public string LbText { get; init; } = "";

        public string UbText { get; init; } = "";

        public Rational[] Optimum { get; init; } = [];

        public bool Qualified { get; init; }

        public bool Representable { get; init; }

        public bool InDomain { get; init; }

        public string Sensitivity { get; init; } = "";

        public int N => CovText.Length;

        public T[,] Cov<T>()
            where T : IFloatingPoint<T>
        {
            var cov = new T[N, N];
            for (var i = 0; i < N; i++)
            {
                for (var j = 0; j < N; j++)
                {
                    cov[i, j] = Parse<T>(CovText[i][j]);
                }
            }

            return cov;
        }

        public T[]? Mu<T>()
            where T : IFloatingPoint<T> => Mode == "minvar" ? null : MuText.Select(Parse<T>).ToArray();

        public T Lambda<T>()
            where T : IFloatingPoint<T> => Mode == "minvar" ? T.One : Parse<T>(LambdaText);

        public T Lb<T>()
            where T : IFloatingPoint<T> => Parse<T>(LbText);

        public T Ub<T>()
            where T : IFloatingPoint<T> => Parse<T>(UbText);

        public T[] RoundedOptimum<T>()
            where T : IFloatingPoint<T> => Optimum.Select(Rounding.ToNearest<T>).ToArray();

        private static T Parse<T>(string s)
            where T : IFloatingPoint<T> => T.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    internal static class Corpus
    {
        private static readonly Dictionary<string, List<QpCase>> s_cache = [];

        public static List<QpCase> Load(string type)
        {
            lock (s_cache)
            {
                if (s_cache.TryGetValue(type, out var cached))
                {
                    return cached;
                }

                var path = Path.Combine(AppContext.BaseDirectory, "Verification", "qp-corpus", "corpus-" + type + ".json");
                Assert.True(File.Exists(path), "the M-1 corpus fixture is missing: " + path);
                using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
                var cases = new List<QpCase>();
                foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray())
                {
                    var mode = c.GetProperty("mode").GetString()!;
                    cases.Add(new QpCase
                    {
                        Id = c.GetProperty("id").GetString()!,
                        Family = c.GetProperty("family").GetString()!,
                        Type = type,
                        Mode = mode,
                        CovText = c.GetProperty("cov").EnumerateArray().Select(r => r.EnumerateArray().Select(x => x.GetString()!).ToArray()).ToArray(),
                        MuText = mode == "minvar" ? [] : c.GetProperty("mu").EnumerateArray().Select(x => x.GetString()!).ToArray(),
                        LambdaText = mode == "minvar" ? "" : c.GetProperty("lambda").GetString()!,
                        LbText = c.GetProperty("lb").GetString()!,
                        UbText = c.GetProperty("ub").GetString()!,
                        Optimum = c.GetProperty("optimum").EnumerateArray().Select(x => Rational.ParseScientific(x.GetString()!)).ToArray(),
                        Qualified = c.GetProperty("qualified").GetBoolean(),
                        Representable = c.GetProperty("representable").GetBoolean(),
                        InDomain = c.GetProperty("in_domain").GetBoolean(),
                        Sensitivity = c.GetProperty("sensitivity").GetString()!,
                    });
                }

                s_cache[type] = cases;
                return cases;
            }
        }
    }

    private static class HandBuilt
    {
        public static QpCase Get(string name) => name switch
        {
            "minvar-lower" => new QpCase
            {
                Id = name,
                Mode = "minvar",
                CovText = [["1", "0", "0"], ["0", "2", "0"], ["0", "0", "4"]],
                LbText = "0.2",
                UbText = "1",
                Optimum = [Rational.Of(8, 15), Rational.Of(4, 15), Rational.Of(1, 5)],
                Qualified = true,
            },
            "meanvar-upper" => new QpCase
            {
                Id = name,
                Mode = "meanvar",
                CovText = [["1", "0"], ["0", "1"]],
                MuText = ["1", "0"],
                LambdaText = "1",
                LbText = "0",
                UbText = "0.8",
                Optimum = [Rational.Of(4, 5), Rational.Of(1, 5)],
                Qualified = true,
            },
            "meanvar-free" => new QpCase
            {
                Id = name,
                Mode = "meanvar",
                CovText = [["2", "1"], ["1", "2"]],
                MuText = ["2", "1"],
                LambdaText = "2",
                LbText = "-1",
                UbText = "2",
                Optimum = [Rational.Of(3, 4), Rational.Of(1, 4)],
                Qualified = true,
            },
            // Σ = QᵀQ + 11ᵀ with the rows of Q orthogonal to (0, 3, 2, 1), so Σw* = 1 at w* = (0, 1/2, 1/3, 1/6):
            // every gradient is equal, w*₀ sits at the lower bound and w*₁ at the upper bound, both with a zero
            // multiplier (an exact tie across the bound pair).
            "minvar-tied-pair" => new QpCase
            {
                Id = name,
                Mode = "minvar",
                CovText = [["2", "1", "1", "1"], ["1", "6", "-5", "-2"], ["1", "-5", "10", "1"], ["1", "-2", "1", "10"]],
                LbText = "0",
                UbText = "0.5",
                Optimum = [Rational.Of(0, 1), Rational.Of(1, 2), Rational.Of(1, 3), Rational.Of(1, 6)],
                Qualified = true,
            },
            _ => throw new ArgumentOutOfRangeException(nameof(name)),
        };
    }

    // ------------------------------------------------------------------------------------------
    // Correct rounding of an exact rational to the working type (round to nearest, ties to even)
    // ------------------------------------------------------------------------------------------

    internal static class Rounding
    {
        private static readonly BigInteger s_decimalLimit = (BigInteger.One << 96) - 1;

        public static T ToNearest<T>(Rational r)
            where T : IFloatingPoint<T>
        {
            if (typeof(T) == typeof(double))
            {
                return (T)(object)ToDouble(r);
            }

            if (typeof(T) == typeof(decimal))
            {
                return (T)(object)ToDecimal(r);
            }

            throw new NotSupportedException(typeof(T).Name);
        }

        public static double ToDouble(Rational r)
        {
            if (r.Sign == 0)
            {
                return 0.0;
            }

            var a = BigInteger.Abs(r.Num);
            var b = r.Den;
            // Choose e so that a / b / 2^e lies in [2^52, 2^53).
            var e = (int)(a.GetBitLength() - b.GetBitLength()) - 53;
            BigInteger q;
            BigInteger rem;
            BigInteger divisor;
            while (true)
            {
                var num = e >= 0 ? a : a << -e;
                divisor = e >= 0 ? b << e : b;
                q = BigInteger.DivRem(num, divisor, out rem);
                if (q < (BigInteger.One << 52))
                {
                    e--;
                    continue;
                }

                if (q >= (BigInteger.One << 53))
                {
                    e++;
                    continue;
                }

                break;
            }

            var twice = rem * 2;
            if (twice > divisor || (twice == divisor && !q.IsEven))
            {
                q += 1;
            }

            var value = Math.ScaleB((double)q, e);
            return r.Sign < 0 ? -value : value;
        }

        public static decimal ToDecimal(Rational r)
        {
            if (r.Sign == 0)
            {
                return 0m;
            }

            var a = BigInteger.Abs(r.Num);
            var b = r.Den;
            for (var scale = 28; scale >= 0; scale--)
            {
                var num = a * BigInteger.Pow(10, scale);
                var q = BigInteger.DivRem(num, b, out var rem);
                var twice = rem * 2;
                if (twice > b || (twice == b && !q.IsEven))
                {
                    q += 1;
                }

                if (q <= s_decimalLimit)
                {
                    var lo = (int)(uint)(q & uint.MaxValue);
                    var mid = (int)(uint)((q >> 32) & uint.MaxValue);
                    var hi = (int)(uint)((q >> 64) & uint.MaxValue);
                    return new decimal(lo, mid, hi, r.Sign < 0, (byte)scale);
                }
            }

            throw new OverflowException("the exact value is not representable as decimal");
        }
    }

    // ------------------------------------------------------------------------------------------
    // Minimal exact rational, copied from ActiveSetQpCorpusTests (private there)
    // ------------------------------------------------------------------------------------------

    internal readonly struct Rational : IComparable<Rational>
    {
        private readonly BigInteger _den;

        private Rational(BigInteger num, BigInteger den)
        {
            Num = num;
            _den = den;
        }

        public BigInteger Num { get; }

        public BigInteger Den => _den.IsZero ? BigInteger.One : _den;

        public int Sign => Num.Sign;

        public static Rational Of(BigInteger num, BigInteger den)
        {
            if (den.IsZero)
            {
                throw new DivideByZeroException();
            }

            if (den.Sign < 0)
            {
                num = -num;
                den = -den;
            }

            if (num.IsZero)
            {
                return default;
            }

            var g = BigInteger.GreatestCommonDivisor(num, den);
            return g.IsOne ? new Rational(num, den) : new Rational(num / g, den / g);
        }

        public static Rational FromDouble(double x)
        {
            var bits = BitConverter.DoubleToInt64Bits(x);
            var negative = bits < 0;
            var exponent = (int)((bits >> 52) & 0x7FF);
            var mantissa = bits & 0xFFFFFFFFFFFFFL;
            if (exponent == 0)
            {
                exponent = 1;
            }
            else
            {
                mantissa |= 1L << 52;
            }

            exponent -= 1075;
            BigInteger num = negative ? -mantissa : mantissa;
            return exponent >= 0 ? Of(num << exponent, BigInteger.One) : Of(num, BigInteger.One << -exponent);
        }

        public static Rational FromDecimal(decimal x)
        {
            var bits = decimal.GetBits(x);
            var magnitude = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | new BigInteger((uint)bits[0]);
            var scale = (bits[3] >> 16) & 0xFF;
            return Of(bits[3] < 0 ? -magnitude : magnitude, BigInteger.Pow(10, scale));
        }

        public static Rational From<T>(T x)
            where T : IFloatingPoint<T>
            => typeof(T) == typeof(double) ? FromDouble((double)(object)x) : FromDecimal((decimal)(object)x);

        public static Rational ParseScientific(string s)
        {
            s = s.Trim();
            var exponent = 0;
            var e = s.IndexOfAny(['E', 'e']);
            if (e >= 0)
            {
                exponent = int.Parse(s[(e + 1)..], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
                s = s[..e];
            }

            var negative = s.StartsWith('-');
            if (negative || s.StartsWith('+'))
            {
                s = s[1..];
            }

            var dot = s.IndexOf('.', StringComparison.Ordinal);
            if (dot >= 0)
            {
                exponent -= s.Length - dot - 1;
                s = s.Remove(dot, 1);
            }

            var digits = BigInteger.Parse(s, NumberStyles.None, CultureInfo.InvariantCulture);
            if (negative)
            {
                digits = -digits;
            }

            return exponent >= 0 ? Of(digits * BigInteger.Pow(10, exponent), BigInteger.One) : Of(digits, BigInteger.Pow(10, -exponent));
        }

        public static Rational operator +(Rational a, Rational b) => Of((a.Num * b.Den) + (b.Num * a.Den), a.Den * b.Den);

        public static Rational operator -(Rational a, Rational b) => Of((a.Num * b.Den) - (b.Num * a.Den), a.Den * b.Den);

        public static Rational operator *(Rational a, Rational b) => Of(a.Num * b.Num, a.Den * b.Den);

        public static Rational operator /(Rational a, Rational b) => Of(a.Num * b.Den, a.Den * b.Num);

        public static bool operator <(Rational a, Rational b) => a.CompareTo(b) < 0;

        public static bool operator >(Rational a, Rational b) => a.CompareTo(b) > 0;

        public Rational Abs() => Sign < 0 ? Of(-Num, Den) : this;

        public int CompareTo(Rational other) => (Num * other.Den).CompareTo(other.Num * Den);
    }
}
