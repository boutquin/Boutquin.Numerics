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

using Xunit.Abstractions;

using Rational = Boutquin.Numerics.Tests.Unit.Internal.QpKktCertificateTests.Rational;
using Rounding = Boutquin.Numerics.Tests.Unit.Internal.QpKktCertificateTests.Rounding;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// The transitions of <see cref="ActiveSetQpSolver{T}"/> that leave every result unchanged when they are switched off,
/// observed on the solve's <c>RouteTrace</c> so that switching one off fails here: the rounded route's reuse of the
/// leading Cholesky columns of the previous working set, the exact budget of the feasible start of each route (the
/// rounded route's greedy fill and the exact route's start, from the feasible fill or from the rounded iterate), and the
/// membership rule of the block that decides a certified result (a variable on a bound joins it when its exact
/// multiplier is materially negative, beyond <c>64·u</c> of its row scale). Later stages repair a start that misses the
/// budget, refactor what a skipped reuse would have copied, and accept a result the clause would not have refused, so no
/// public result shows any of them; the expected values here come from the documented rule, computed independently in
/// exact rational arithmetic, never from the solver. Every case runs at both guaranteed types through both objectives.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpFactorReuseObservationTests(ITestOutputHelper output)
{
    // ---------------------------------------------------------------------------------------------------------------
    // Rounded route: the leading Cholesky columns of the previous working set's factor
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// Every reduced Hessian the rounded route factors copies exactly the leading columns it shares with the factor of
    /// the previous working set: none when there is no previous factor or the reference changed, otherwise as many as the
    /// leading other free variables the two working sets have in common. The factor of reference <c>r</c> has one column
    /// per other free variable in index order, and its entry <c>(a, a)</c> depends only on that variable and <c>r</c>, so
    /// the expected count follows from the two working sets alone. The previous factor handed over is the last one
    /// created, and the working set never repeats it (an unchanged working set keeps its factor). On these factor models
    /// bounds bind and release while the reference stays, so the expected copies are many.
    /// </summary>
    [Theory]
    [InlineData(2026130)]
    [InlineData(2026131)]
    public void RoundedRoute_CopiesExactlyTheLeadingCholeskyColumnsItSharesWithThePreviousWorkingSet(int seed)
    {
        CheckPrefixReuse<double>(seed);
        CheckPrefixReuse<decimal>(seed);
    }

    private void CheckPrefixReuse<T>(int seed)
        where T : IFloatingPoint<T>
    {
        foreach (var meanVariance in new[] { false, true })
        {
            long expectedTotal = 0;
            long columns = 0;
            var factors = 0;
            for (var sample = 0; sample < 12; sample++)
            {
                var (cov, means, lower, upper) = FactorModel<T>((seed * 100) + sample);
                var trace = new ActiveSetQpSolver<T>.RouteTrace();
                _ = Solve(cov, meanVariance ? means : null, lower, upper, trace);
                var records = Observed.ReducedFactors(trace);
                var label = string.Create(CultureInfo.InvariantCulture, $"{typeof(T).Name} {(meanVariance ? "mean-variance" : "min-variance")} seed {(seed * 100) + sample} n={cov.GetLength(0)}");
                for (var k = 0; k < records.Count; k++)
                {
                    var record = records[k];
                    if (k == 0)
                    {
                        Assert.True(record.PreviousFree is null, label + ": the first factor of the solve was handed a previous factor");
                    }
                    else if (record.PreviousFree is not null)
                    {
                        Assert.True(records[k - 1].Free.AsSpan().SequenceEqual(record.PreviousFree) && records[k - 1].Reference == record.PreviousReference, label + $": factor {k} was handed a factor other than the last one created");
                        Assert.False(record.Free.AsSpan().SequenceEqual(record.PreviousFree), label + $": factor {k} refactored an unchanged working set");
                    }

                    var expected = ExpectedPrefix(record);
                    Assert.True(expected == record.ReusedColumns, label + $": factor {k} (free [{string.Join(",", record.Free)}], reference {record.Reference}; previous [{(record.PreviousFree is null ? "none" : string.Join(",", record.PreviousFree))}], reference {record.PreviousReference}) copied {record.ReusedColumns} leading columns, expected {expected}");
                    expectedTotal += expected;
                    columns += record.Free.Length - 1;
                }

                factors += records.Count;
            }

            output.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{typeof(T).Name} {(meanVariance ? "mean-variance" : "min-variance")} seed {seed}: {factors} factors, {columns} columns, {expectedTotal} copied"));
            Assert.True(expectedTotal > 0, "no working set of this workload kept its reference and leading variables: the control does not reach the reuse");
        }
    }

    // The leading columns a new factor shares with the previous one (see the test above).
    internal static int ExpectedPrefix(Observed.ReducedFactor record)
    {
        if (record.PreviousFree is null || record.PreviousReference != record.Reference)
        {
            return 0;
        }

        var others = record.Free.Where(i => i != record.Reference).ToArray();
        var before = record.PreviousFree.Where(i => i != record.Reference).ToArray();
        var prefix = 0;
        while (prefix < others.Length && prefix < before.Length && others[prefix] == before[prefix])
        {
            prefix++;
        }

        return prefix;
    }

    /// <summary>
    /// <c>B·B′ + diag(d)</c> with 2 to 5 factors of loadings in [-0.15, 0.35] (multiples of 1/1000) and specific variances
    /// in [0.002, 0.022] (multiples of 1/10000), n from 8 to 27, formed in <typeparamref name="T"/>; means in [-0.02, 0.02];
    /// a long-only box [0, 0.3] or a long-short box [-0.5, 0.5].
    /// </summary>
    internal static (T[,] Cov, T[] Means, T Lower, T Upper) FactorModel<T>(int seed)
        where T : IFloatingPoint<T>
    {
        var rng = new System.Random(seed);
        var n = 8 + rng.Next(0, 20);
        var f = 2 + rng.Next(0, 4);
        var thousand = T.CreateChecked(1000);
        var b = new T[n, f];
        for (var i = 0; i < n; i++)
        {
            for (var k = 0; k < f; k++)
            {
                b[i, k] = T.CreateChecked(rng.Next(-150, 351)) / thousand;
            }
        }

        var cov = new T[n, n];
        var means = new T[n];
        for (var i = 0; i < n; i++)
        {
            var specific = T.CreateChecked(20 + rng.Next(0, 201)) / T.CreateChecked(10000);
            means[i] = T.CreateChecked(rng.Next(-20, 21)) / thousand;
            for (var j = 0; j <= i; j++)
            {
                var sum = i == j ? specific : T.Zero;
                for (var k = 0; k < f; k++)
                {
                    sum += b[i, k] * b[j, k];
                }

                cov[i, j] = cov[j, i] = sum;
            }
        }

        var (lower, upper) = seed % 2 == 0 ? (T.Zero, T.CreateChecked(3) / T.CreateChecked(10)) : (-T.One / T.CreateChecked(2), T.One / T.CreateChecked(2));
        return (cov, means, lower, upper);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Feasible starts: the budget carried exactly
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// The feasible start of the rounded route (before each weight is rounded once into the working type) and the
    /// exact route's feasible start are the greedy fill carried exactly: every weight at the lower bound, the budget
    /// <c>1 − n·lb</c> distributed up to the upper bounds in index order, summing to 1 exactly. Computed here in rational
    /// arithmetic from the documented rule. The boxes include those whose budget <c>1 − n·lb</c> is not representable in
    /// the working type (<c>n = 3</c>, <c>lb = −10²⁰</c> in double, where the exact start is <c>[10²⁰, 1, −10²⁰]</c>; a
    /// 29-digit budget in decimal), where a fill whose budget is first rounded into the working type misses the unit.
    /// </summary>
    [Fact]
    public void FeasibleStarts_OfBothRoutes_CarryTheBudgetExactly()
    {
        CheckStarts(3, -1e20, 1e20);
        CheckStarts(3, decimal.Parse("-2.3333333333333333333333333333", CultureInfo.InvariantCulture), 3m);
        var rng = new System.Random(20261320);
        for (var sample = 0; sample < 10; sample++)
        {
            var (n, lower, upper) = RandomBox<double>(rng);
            CheckStarts(n, lower, upper);
            var (m, low, high) = RandomBox<decimal>(rng);
            CheckStarts(m, low, high);
        }
    }

    private void CheckStarts<T>(int n, T lower, T upper)
        where T : IFloatingPoint<T>
    {
        var cov = new T[n, n];
        var means = new T[n];
        for (var i = 0; i < n; i++)
        {
            cov[i, i] = T.One;
            means[i] = T.CreateChecked(i % 3) / T.CreateChecked(8);
        }

        var oracle = GreedyFill(n, Rational.From(lower), Rational.From(upper));
        Assert.Equal(0, Sum(oracle).CompareTo(Rational.Of(1, 1)));
        foreach (var meanVariance in new[] { false, true })
        {
            var label = string.Create(CultureInfo.InvariantCulture, $"{typeof(T).Name} {(meanVariance ? "mean-variance" : "min-variance")} n={n} box [{lower}, {upper}]");
            var rounded = new ActiveSetQpSolver<T>.RouteTrace();
            _ = TrySolve(cov, meanVariance ? means : null, lower, upper, rounded);
            var start = Observed.Start(rounded.FeasibleStart);
            Assert.True(start is not null, label + ": the rounded route recorded no feasible start");
            Assert.False(start!.Value.ReusedIterate);
            AssertSameAllocation(oracle, start.Value.Weights, label + ", rounded route");

            var exact = new ActiveSetQpSolver<T>.RouteTrace { ForceExactRoute = true };
            _ = TrySolve(cov, meanVariance ? means : null, lower, upper, exact);
            var exactStart = Observed.Start(exact.ExactStart);
            Assert.True(exactStart is not null, label + ": the exact route recorded no start");
            Assert.False(exactStart!.Value.ReusedIterate, label + ": a forced exact route has no rounded iterate to start from");
            AssertSameAllocation(oracle, exactStart.Value.Weights, label + ", exact route");
            output.WriteLine(label + ": start " + string.Join(", ", start.Value.Weights.Select(Show)));
        }
    }

    /// <summary>
    /// A solve handed to the exact route from the rounded route (an unusable working set or a refinement that does not
    /// converge) starts from the rounded iterate and its working set, the iterate's budget residual moved exactly onto a
    /// free variable that stays in the box: the start sums to 1 exactly, lies in the box, and is the rounded iterate on at
    /// least one hand-off of each type and objective (a hand-off from the exact feasible fill instead would discard the
    /// rounded route's progress and pass every public assertion).
    /// </summary>
    [Fact]
    public void ExactRoute_EnteredFromTheRoundedRoute_StartsFromItsIterateWithTheBudgetMovedExactly()
    {
        CheckHandOffStarts<double>();
        CheckHandOffStarts<decimal>();
    }

    private void CheckHandOffStarts<T>()
        where T : IFloatingPoint<T>
    {
        foreach (var meanVariance in new[] { false, true })
        {
            var handOffs = 0;
            var reused = 0;
            for (var seed = 1; seed <= 40 && reused < 3; seed++)
            {
                foreach (var n in new[] { 4, 5 })
                {
                    var (cov, means) = NearDuplicate<T>(n, seed);
                    var lower = -T.One;
                    var upper = T.One;
                    var trace = new ActiveSetQpSolver<T>.RouteTrace();
                    _ = TrySolve(cov, meanVariance ? means : null, lower, upper, trace);
                    if (trace.ExactRouteReason is ActiveSetQpSolver<T>.ExactRouteReason.None)
                    {
                        continue;
                    }

                    handOffs++;
                    var label = string.Create(CultureInfo.InvariantCulture, $"{typeof(T).Name} {(meanVariance ? "mean-variance" : "min-variance")} n={n} seed={seed} ({trace.ExactRouteReason})");
                    var start = Observed.Start(trace.ExactStart);
                    Assert.True(start is not null, label + ": the exact route recorded no start");
                    Assert.True(Sum(start!.Value.Weights).CompareTo(Rational.Of(1, 1)) == 0, label + ": the start misses the budget, sum " + Show(Sum(start.Value.Weights)));
                    Assert.All(start.Value.Weights, w => Assert.True(w.CompareTo(Rational.From(lower)) >= 0 && w.CompareTo(Rational.From(upper)) <= 0, label + ": a start weight leaves the box"));
                    reused += start.Value.ReusedIterate ? 1 : 0;
                }
            }

            output.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{typeof(T).Name} {(meanVariance ? "mean-variance" : "min-variance")}: {handOffs} hand-offs, {reused} started from the rounded iterate"));
            Assert.True(handOffs > 0, "no near-duplicate problem was handed to the exact route: the control does not reach the hand-off");
            Assert.True(reused > 0, typeof(T).Name + (meanVariance ? " mean-variance" : " min-variance") + ": no hand-off started from the rounded iterate");
        }
    }

    // Assets 0 and 1 share their loadings (multiples of 1/16) and an idiosyncratic variance `gap` far below the working
    // precision of the others' (double 2^-40 to 2^-49, decimal 1e-20 to 1e-27 by seed), the others have idiosyncratic
    // variances in [1/64, 1/8]: a near-duplicate pair whose free block the rounded route cannot use.
    internal static (T[,] Cov, T[] Means) NearDuplicate<T>(int n, int seed)
        where T : IFloatingPoint<T>
    {
        var rng = new System.Random(seed);
        var f = new T[n, 2];
        var sixteen = T.CreateChecked(16);
        for (var i = 0; i < n; i++)
        {
            for (var k = 0; k < 2; k++)
            {
                f[i, k] = T.CreateChecked(rng.Next(-8, 9)) / sixteen;
            }
        }

        for (var k = 0; k < 2; k++)
        {
            f[1, k] = f[0, k];
        }

        var gap = typeof(T) == typeof(double)
            ? (T)(object)Math.ScaleB(1.0, -40 - (seed % 10))
            : (T)(object)(1m / (decimal)Math.Pow(10, 20 + (seed % 8)));
        var meanDraws = new System.Random(seed + 7919);
        var cov = new T[n, n];
        var means = new T[n];
        for (var i = 0; i < n; i++)
        {
            means[i] = T.CreateChecked(meanDraws.Next(-4, 5)) / T.CreateChecked(64);
            var specific = i < 2 ? gap : T.CreateChecked(1 + rng.Next(0, 8)) / T.CreateChecked(64);
            for (var j = 0; j <= i; j++)
            {
                var sum = i == j ? specific : T.Zero;
                for (var k = 0; k < 2; k++)
                {
                    sum += f[i, k] * f[j, k];
                }

                cov[i, j] = cov[j, i] = sum;
            }
        }

        return (cov, means);
    }

    // A box whose budget 1 − n·lb carries more digits than the working type (n from 2 to 9): a lower bound with a full
    // mantissa (double: (1 + r)·10^k, k up to 17; decimal: 28 decimal places), an upper bound above −lb.
    internal static (int N, T Lower, T Upper) RandomBox<T>(System.Random rng)
        where T : IFloatingPoint<T>
    {
        var n = 2 + rng.Next(0, 8);
        if (typeof(T) == typeof(double))
        {
            var lower = -(1.0 + rng.NextDouble()) * Math.Pow(10, rng.Next(0, 18));
            return (n, (T)(object)lower, (T)(object)(-lower * (1.0 + rng.NextDouble())));
        }

        var digits = string.Concat(Enumerable.Range(0, 28).Select(_ => rng.Next(0, 10).ToString(CultureInfo.InvariantCulture)));
        var low = decimal.Parse("-" + (1 + rng.Next(0, 3)).ToString(CultureInfo.InvariantCulture) + "." + digits, CultureInfo.InvariantCulture);
        return (n, (T)(object)low, (T)(object)Math.Round(-low * (1m + (decimal)rng.NextDouble()), 10));
    }

    // The documented greedy fill, exactly: every weight at the lower bound, the budget distributed in index order.
    internal static Rational[] GreedyFill(int n, Rational lower, Rational upper)
    {
        var weights = new Rational[n];
        var remaining = Rational.Of(1, 1) - (Rational.Of(n, 1) * lower);
        var capacity = upper - lower;
        for (var i = 0; i < n; i++)
        {
            var add = remaining.Sign <= 0 ? Rational.Of(0, 1) : remaining < capacity ? remaining : capacity;
            weights[i] = lower + add;
            remaining -= add;
        }

        return weights;
    }

    private static void AssertSameAllocation(Rational[] expected, Rational[] actual, string label)
    {
        Assert.True(expected.Length == actual.Length, label + ": the start has " + actual.Length + " weights");
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.True(expected[i].CompareTo(actual[i]) == 0, label + $": start weight {i} is {Show(actual[i])}, the exact fill gives {Show(expected[i])} (start sum {Show(Sum(actual))})");
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // The deciding block: materially negative multipliers beyond 64·u of the row scale
    // ---------------------------------------------------------------------------------------------------------------

    public enum BlockBranch
    {
        Interior,
        AllAtBound,
    }

    public enum TestedKind
    {
        Excluded,
        FreeOnBound,
    }

    /// <summary>
    /// A variable on a bound joins the deciding block exactly when its exact multiplier at the subspace minimiser is
    /// materially negative: below <c>−64·u</c> times its row scale <c>|ν| + λ·Σⱼ|Σᵢⱼ||xⱼ| + |μ̃ᵢ|</c> with a free variable
    /// strictly inside the box (<c>ν</c> the gradient of the interior reference, the interior free variable of smallest
    /// variance), and, at a vertex, when its gradient passes the other end of the multiplier interval by more than
    /// <c>64·u</c> of the two rows' scales. Each case puts the tested variable's ratio within a unit roundoff of
    /// <c>−65·u</c> (it joins) or of <c>−63·u</c> (it does not), at the lower and at the upper bound, as a variable the
    /// exclusion rule excluded or as a free variable sitting on its bound. The whole block is compared with the one the
    /// documented rule gives, computed here in exact rational arithmetic, never with the solver's.
    /// </summary>
    [Theory]
    [InlineData(BlockBranch.Interior, -1)]
    [InlineData(BlockBranch.Interior, 1)]
    [InlineData(BlockBranch.AllAtBound, -1)]
    [InlineData(BlockBranch.AllAtBound, 1)]
    public void DecidingBlock_AddsABoundVariableExactlyWhenItsMultiplierPassesSixtyFourUnitsOfItsRowScale(BlockBranch branch, int side)
    {
        foreach (var kind in new[] { TestedKind.Excluded, TestedKind.FreeOnBound })
        {
            foreach (var meanVariance in new[] { false, true })
            {
                foreach (var units in new[] { 65, 63 })
                {
                    CheckDecidingBlock<double>(branch, side, kind, meanVariance, units);
                    CheckDecidingBlock<decimal>(branch, side, kind, meanVariance, units);
                }
            }
        }
    }

    private void CheckDecidingBlock<T>(BlockBranch branch, int side, TestedKind kind, bool meanVariance, int units)
        where T : IFloatingPoint<T>
    {
        var state = DecidingBlockCase.Base(branch, side, kind, meanVariance);
        var label = string.Create(CultureInfo.InvariantCulture, $"{typeof(T).Name} {branch} {(side < 0 ? "lower" : "upper")} {kind} {(meanVariance ? "mean-variance" : "min-variance")} at -{units}u");
        var u = DecidingBlockCase.UnitRoundoff<T>();
        var tuned = DecidingBlockCase.Tune<T>(state, Rational.Of(units, 1) * u, label);
        var ratio = DecidingBlockCase.Ratio(tuned);
        Assert.True((ratio - (Rational.Of(units, 1) * u)).Abs().CompareTo(u) <= 0, label + ": the case's ratio " + Show(ratio / u) + "u is not within a unit roundoff of the target");
        var expected = DecidingBlockCase.Block(tuned, u);
        Assert.Equal(units > 64, expected.Contains(tuned.Tested));

        var trace = new ActiveSetQpSolver<T>.RouteTrace();
        var block = Observed.DecidingBlock(trace, tuned.ToT<T>());
        output.WriteLine(label + $": ratio {Show(ratio / u)}u, block [{(block is null ? "none" : string.Join(",", block))}], expected [{string.Join(",", expected)}]");
        Assert.True(block is not null, label + ": the conditioning guard recorded no deciding block");
        Assert.True(expected.SequenceEqual(block!), label + $": deciding block [{string.Join(",", block!)}], the documented rule gives [{string.Join(",", expected)}]");
    }

    /// <summary>An exit state handed to the conditioning guard, exactly (the tested variable's variance is tuned).</summary>
    internal sealed record DecidingBlockCase(
        Rational[,] Cov, Rational[]? Means, Rational Lambda, Rational Lower, Rational Upper, Rational[] X, int[] Status, bool[] Stalled, int Tested, int Side)
    {
        public int N => X.Length;

        // u of the working type: double 2⁻⁵³, decimal 10⁻²⁸.
        public static Rational UnitRoundoff<T>()
            where T : IFloatingPoint<T> =>
            typeof(T) == typeof(decimal) ? Rational.Of(1, BigInteger.Pow(10, 28)) : Rational.Of(1, BigInteger.One << 53);

        /// <summary>
        /// The four-asset exit states. Interior: three interior free variables (the reference is asset 1, of smallest
        /// variance among them) and asset 0 on a bound. At a vertex: every asset on a bound, the tested one 2 (lower) or 0
        /// (upper), its gradient near the opposite end of the multiplier interval (asset 0, respectively 2). Every value is
        /// a short binary fraction, exact in double and in decimal; the tested variance is a placeholder for the tuning.
        /// </summary>
        public static DecidingBlockCase Base(BlockBranch branch, int side, TestedKind kind, bool meanVariance)
        {
            static Rational Q(long num, long den = 1) => Rational.Of(num, den);
            Rational[]? means = meanVariance ? [Q(1, 8), Q(-1, 8), Q(1, 16), Q(0)] : null;
            var lambda = meanVariance ? Q(2) : Q(1);
            var cov = new Rational[4, 4];
            Rational lower, upper;
            Rational[] x;
            int[] status;
            int tested;
            if (branch == BlockBranch.Interior)
            {
                tested = 0;
                if (side < 0)
                {
                    (lower, upper) = (Q(-1, 2), Q(2));
                    x = [Q(-1, 2), Q(-1, 4), Q(1), Q(3, 4)];
                    Diagonal(cov, Q(1), Q(2), Q(3), Q(3));
                    cov[0, 1] = cov[1, 0] = Q(1, 8);
                    cov[2, 3] = cov[3, 2] = Q(1, 4);
                }
                else
                {
                    (lower, upper) = (Q(-1, 2), Q(1));
                    x = [Q(1), Q(1, 4), Q(-1, 8), Q(-1, 8)];
                    Diagonal(cov, Q(1), Q(4), Q(5), Q(5));
                    cov[0, 1] = cov[1, 0] = Q(1, 16);
                }

                status = [kind == TestedKind.Excluded ? side : 0, 0, 0, 0];
            }
            else
            {
                (lower, upper) = (Q(1, 8), Q(3, 8));
                x = [Q(3, 8), Q(3, 8), Q(1, 8), Q(1, 8)];
                tested = side < 0 ? 2 : 0;
                if (side < 0)
                {
                    Diagonal(cov, Q(1), Q(1, 2), Q(3), Q(4));
                }
                else
                {
                    Diagonal(cov, Q(2), Q(1), Q(11, 2), Q(8));
                }

                status = [1, 1, -1, -1];
                status[tested] = kind == TestedKind.Excluded ? side : 0;
            }

            var stalled = new bool[4];
            stalled[tested] = kind == TestedKind.Excluded;
            return new DecidingBlockCase(cov, means, lambda, lower, upper, x, status, stalled, tested, side);
        }

        private static void Diagonal(Rational[,] cov, params Rational[] d)
        {
            for (var i = 0; i < d.Length; i++)
            {
                cov[i, i] = d[i];
            }
        }

        public DecidingBlockCase WithVariance(Rational s)
        {
            var cov = (Rational[,])Cov.Clone();
            cov[Tested, Tested] = s;
            return this with { Cov = cov };
        }

        // μ − (min μ + max μ)/2, or zeros for minimum variance.
        private Rational[] Centred()
        {
            if (Means is null)
            {
                return Enumerable.Repeat(Rational.Of(0, 1), N).ToArray();
            }

            var low = Means.Min()!;
            var high = Means.Max()!;
            var mid = (low + high) * Rational.Of(1, 2);
            return Means.Select(m => m - mid).ToArray();
        }

        public Rational Gradient(int i)
        {
            var sum = Rational.Of(0, 1);
            for (var j = 0; j < N; j++)
            {
                sum += Cov[i, j] * X[j];
            }

            return (Lambda * sum) - Centred()[i];
        }

        public Rational RowScale(int i, Rational nu)
        {
            var sum = Rational.Of(0, 1);
            for (var j = 0; j < N; j++)
            {
                sum += Cov[i, j].Abs() * X[j].Abs();
            }

            return nu.Abs() + (Lambda * sum) + Centred()[i].Abs();
        }

        private bool AtBound(int i) => X[i].CompareTo(Lower) == 0 || X[i].CompareTo(Upper) == 0;

        private bool TestedVariable(int i) => Status[i] == 0 ? AtBound(i) : Stalled[i];

        private int SideOf(int i) => Status[i] != 0 ? Status[i] : X[i].CompareTo(Lower) == 0 ? -1 : 1;

        private int[] Interior() => Enumerable.Range(0, N).Where(i => Status[i] == 0 && X[i].CompareTo(Lower) > 0 && X[i].CompareTo(Upper) < 0).ToArray();

        // The interior free variable of smallest variance, the smallest index among ties.
        private int Reference(int[] interior) => interior.OrderBy(i => Cov[i, i]).ThenBy(i => i).First();

        // The end of the multiplier interval a vertex variable is compared with: the largest gradient at the upper bound
        // (for a variable at the lower bound), or the smallest at the lower bound; the smallest index among ties.
        private int End(bool upperEnd)
        {
            var end = -1;
            for (var i = 0; i < N; i++)
            {
                if (X[i].CompareTo(upperEnd ? Upper : Lower) == 0 && (end < 0 || (upperEnd ? Gradient(i) > Gradient(end) : Gradient(i) < Gradient(end))))
                {
                    end = i;
                }
            }

            return end;
        }

        // The amount by which variable i's multiplier is materially negative, and the scale it is measured against.
        private (Rational Excess, Rational Scale) Measure(int i)
        {
            var interior = Interior();
            if (interior.Length > 0)
            {
                var nu = Gradient(Reference(interior));
                var multiplier = SideOf(i) < 0 ? Gradient(i) - nu : nu - Gradient(i);
                return (Rational.Of(0, 1) - multiplier, RowScale(i, nu));
            }

            var zero = Rational.Of(0, 1);
            var end = End(upperEnd: X[i].CompareTo(Lower) == 0);
            return X[i].CompareTo(Lower) == 0
                ? (Gradient(end) - Gradient(i), RowScale(i, zero) + RowScale(end, zero))
                : (Gradient(i) - Gradient(end), RowScale(i, zero) + RowScale(end, zero));
        }

        /// <summary>The tested variable's excess over its scale, the quantity the clause compares with <c>64·u</c>.</summary>
        public static Rational Ratio(DecidingBlockCase c)
        {
            var (excess, scale) = c.Measure(c.Tested);
            return excess / scale;
        }

        /// <summary>The documented deciding block: the interior free variables plus every tested variable beyond 64·u.</summary>
        public static int[] Block(DecidingBlockCase c, Rational u)
        {
            var material = Rational.Of(64, 1) * u;
            var block = new SortedSet<int>(Enumerable.Range(0, c.N).Where(i => c.Status[i] == 0 && !c.AtBound(i)));
            for (var i = 0; i < c.N; i++)
            {
                if (!c.TestedVariable(i))
                {
                    continue;
                }

                var (excess, scale) = c.Measure(i);
                if (excess > material * scale)
                {
                    block.Add(i);
                }
            }

            return [.. block];
        }

        /// <summary>
        /// The tested variable's variance in <typeparamref name="T"/> that puts its ratio nearest <paramref name="target"/>:
        /// the excess and the scale are affine in that variance, so the exact solution is rational; it is rounded into
        /// <typeparamref name="T"/> and the two neighbours on each side are compared.
        /// </summary>
        public static DecidingBlockCase Tune<T>(DecidingBlockCase c, Rational target, string label)
            where T : IFloatingPoint<T>
        {
            var zero = c.WithVariance(Rational.Of(0, 1)).Measure(c.Tested);
            var one = c.WithVariance(Rational.Of(1, 1)).Measure(c.Tested);
            var slope = (one.Excess - zero.Excess) - (target * (one.Scale - zero.Scale));
            Assert.True(slope.Sign != 0, label + ": the ratio does not depend on the tested variance");
            var exact = ((target * zero.Scale) - zero.Excess) / slope;
            var best = Rounding.ToNearest<T>(exact);
            var candidates = new List<T> { best };
            var step = best;
            var back = best;
            for (var k = 0; k < 2; k++)
            {
                step = Next(step, 1);
                back = Next(back, -1);
                candidates.Add(step);
                candidates.Add(back);
            }

            var chosen = candidates
                .Select(v => c.WithVariance(Rational.From(v)))
                .OrderBy(t => (Ratio(t) - target).Abs())
                .First();
            var variance = chosen.Cov[c.Tested, c.Tested];
            var offDiagonal = Rational.Of(0, 1);
            for (var j = 0; j < c.N; j++)
            {
                offDiagonal += j == c.Tested ? Rational.Of(0, 1) : c.Cov[c.Tested, j].Abs();
            }

            Assert.True(variance > offDiagonal, label + ": the tuned variance " + Show(variance) + " leaves the covariance without diagonal dominance");
            return chosen;
        }

        private static T Next<T>(T x, int direction)
            where T : IFloatingPoint<T>
        {
            if (x is double d)
            {
                return (T)(object)(direction > 0 ? Math.BitIncrement(d) : Math.BitDecrement(d));
            }

            var unit = 0.0000000000000000000000000001m;
            return (T)(object)((decimal)(object)x! + (direction > 0 ? unit : -unit));
        }

        public Observed.ExitState<T> ToT<T>()
            where T : IFloatingPoint<T>
        {
            var cov = new T[N, N];
            for (var i = 0; i < N; i++)
            {
                for (var j = 0; j < N; j++)
                {
                    cov[i, j] = Rounding.ToNearest<T>(Cov[i, j]);
                }
            }

            var x = X.Select(Rounding.ToNearest<T>).ToArray();
            return new Observed.ExitState<T>(
                cov,
                Means?.Select(Rounding.ToNearest<T>).ToArray(),
                Rounding.ToNearest<T>(Lambda),
                Rounding.ToNearest<T>(Lower),
                Rounding.ToNearest<T>(Upper),
                x,
                (T[])x.Clone(),
                (int[])Status.Clone(),
                (bool[])Stalled.Clone());
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------------------

    internal static T[] Solve<T>(T[,] cov, T[]? means, T lower, T upper, ActiveSetQpSolver<T>.RouteTrace trace)
        where T : IFloatingPoint<T> => means is null
            ? ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(cov, lower, upper, trace: trace)
            : ActiveSetQpSolver<T>.SolveMeanVarianceUsingCore(cov, means, T.CreateChecked(4), lower, upper, trace: trace);

    // The starts are recorded before any decision of the solve; a documented refusal later in the solve leaves them.
    internal static T[]? TrySolve<T>(T[,] cov, T[]? means, T lower, T upper, ActiveSetQpSolver<T>.RouteTrace trace)
        where T : IFloatingPoint<T>
    {
        try
        {
            return Solve(cov, means, lower, upper, trace);
        }
        catch (InvalidOperationException)
        {
            return null;
        }
    }

    internal static Rational Sum(IEnumerable<Rational> values) => values.Aggregate(Rational.Of(0, 1), (a, b) => a + b);

    internal static string Show(Rational r) => r.Den.IsOne
        ? r.Num.ToString(CultureInfo.InvariantCulture)
        : string.Create(CultureInfo.InvariantCulture, $"{r.Num}/{r.Den}");

    /// <summary>The solver's observation seams (internal test diagnostics on <c>RouteTrace</c>), in the test's terms.</summary>
    internal static class Observed
    {
        internal sealed record ReducedFactor(int[] Free, int Reference, int[]? PreviousFree, int PreviousReference, int ReusedColumns);

        internal sealed record ExitState<T>(T[,] Cov, T[]? Means, T Lambda, T Lower, T Upper, T[] Weights, T[] Point, int[] Status, bool[] Stalled);

        public static List<ReducedFactor> ReducedFactors<T>(ActiveSetQpSolver<T>.RouteTrace trace)
            where T : IFloatingPoint<T> =>
            [.. trace.ReducedFactors.Select(r => new ReducedFactor(r.Free, r.Reference, r.PreviousFree, r.PreviousReference, r.ReusedColumns))];

        // A recorded start as exact rationals: weight i is Mantissas[i] · Radix^Exponents[i].
        public static (Rational[] Weights, bool ReusedIterate)? Start<T>(ActiveSetQpSolver<T>.StartAllocation? start)
            where T : IFloatingPoint<T>
        {
            if (start is null)
            {
                return null;
            }

            var radix = new BigInteger(start.Radix);
            var weights = new Rational[start.Mantissas.Length];
            for (var i = 0; i < weights.Length; i++)
            {
                weights[i] = start.Exponents[i] >= 0
                    ? Rational.Of(start.Mantissas[i] * BigInteger.Pow(radix, start.Exponents[i]), 1)
                    : Rational.Of(start.Mantissas[i], BigInteger.Pow(radix, -start.Exponents[i]));
            }

            return (weights, start.ReusedIterate);
        }

        public static int[]? DecidingBlock<T>(ActiveSetQpSolver<T>.RouteTrace trace, ExitState<T> state)
            where T : IFloatingPoint<T>
        {
            trace.RequireDeterminedBlock(state.Cov, state.Means, state.Lambda, state.Lower, state.Upper, state.Weights, state.Point, state.Status, state.Stalled);
            return trace.DecidingBlock;
        }
    }
}
