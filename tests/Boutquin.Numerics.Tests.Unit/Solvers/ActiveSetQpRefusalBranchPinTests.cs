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

using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// Pins the documented refusal and budget branches of <see cref="ActiveSetQpSolver{T}"/> that no ordinary input pins on
/// its own, each through both public objectives at both guaranteed types and each with its exact exception type and
/// message: the release-gradient range refusal (reached by public inputs at the edge of the type's range), the LP
/// budget-or-capacity refusal one half at a time, the translation of an <see cref="OverflowException"/> raised inside the
/// solve, the exact route's working-set budget, the exit certificate's refusal of the exact route's result and of the
/// equal-weight frontier point, and the rounded route's hand-off when it spends its own budget. The branches that no
/// known public input reaches are driven through the internal <see cref="ActiveSetQpSolver{T}.RouteTrace"/> diagnostics
/// (a budget override, or a hook on the exact route's result), which change nothing for a solve that does not set them.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpRefusalBranchPinTests
{
    private const string GradientRefusal = "QP release gradient must be finite.";

    private const string ExceededRange = "QP arithmetic exceeded the numeric type's range; rescale the problem.";

    private const string LpRefusal = "LP allocation produced a non-finite budget or capacity; rescale the bounds.";

    private static readonly double[,] s_covariance =
    {
        { 4.0, 1.0, 0.5, 0.2 },
        { 1.0, 3.0, 0.2, 0.1 },
        { 0.5, 0.2, 2.0, 0.3 },
        { 0.2, 0.1, 0.3, 1.0 },
    };

    private static readonly double[] s_means = [0.05, 0.10, 0.02, 0.08];

    // ─── Release-gradient range refusal (RequireRepresentableGradientTerms) ───

    /// <summary>
    /// A two-asset long-short problem at the edge of the type's range: its optimum (about <c>(84.1, −83.1)</c> at
    /// <c>double</c>, <c>(11.6, −10.6)</c> at <c>decimal</c>) is representable, but a product <c>Σᵢⱼ·wⱼ</c> of its gradient
    /// is not. Both objectives refuse it at both types with the documented message instead of returning the vector.
    /// </summary>
    [Fact]
    public void ReleaseGradientRangeRefusal_IsRaisedByBothObjectivesAtBothTypes()
    {
        var (doubleCov, decimalCov) = EdgeOfRangeLongShort(doubleScale: 8e306, decimalScale: 1.5e28m);

        AssertRefused(GradientRefusal, () => ActiveSetQpSolver<double>.SolveMinVariance(doubleCov, -1e6, 1e6), "double min-variance");
        AssertRefused(GradientRefusal, () => ActiveSetQpSolver<double>.SolveMeanVariance(doubleCov, [0.0, 0.0], 1.0, -1e6, 1e6), "double mean-variance");
        AssertRefused(GradientRefusal, () => ActiveSetQpSolver<decimal>.SolveMinVariance(decimalCov, -1e6m, 1e6m), "decimal min-variance");
        AssertRefused(GradientRefusal, () => ActiveSetQpSolver<decimal>.SolveMeanVariance(decimalCov, [0m, 0m], 1m, -1e6m, 1e6m), "decimal mean-variance");
    }

    /// <summary>
    /// The counterpart of the range refusal: the same correlation structure at a unit scale is solved by both objectives at
    /// both types, with the same long-short optimum (the weights are invariant to a positive rescaling of the covariance),
    /// so the refusal is the range's alone.
    /// </summary>
    [Fact]
    public void ReleaseGradientRangeRefusal_DoesNotReachTheSameProblemAtAUnitScale()
    {
        var (doubleCov, decimalCov) = EdgeOfRangeLongShort(doubleScale: 1.0, decimalScale: 1m);

        foreach (var weights in new[]
        {
            ActiveSetQpSolver<double>.SolveMinVariance(doubleCov, -1e6, 1e6),
            ActiveSetQpSolver<double>.SolveMeanVariance(doubleCov, [0.0, 0.0], 1.0, -1e6, 1e6),
        })
        {
            Assert.True(Math.Abs(weights[0] - 84.11064891856147) <= 1e-6 && Math.Abs(weights[1] + 83.11064891856147) <= 1e-6, string.Join(", ", weights));
        }

        foreach (var weights in new[]
        {
            ActiveSetQpSolver<decimal>.SolveMinVariance(decimalCov, -1e6m, 1e6m),
            ActiveSetQpSolver<decimal>.SolveMeanVariance(decimalCov, [0m, 0m], 1m, -1e6m, 1e6m),
        })
        {
            Assert.True(Math.Abs(weights[0] - 11.641304347826086956521739m) <= 1e-20m && Math.Abs(weights[1] + 10.641304347826086956521739m) <= 1e-20m, string.Join(", ", weights));
        }
    }

    // ─── λ = 0 LP: the budget and the capacity, one at a time ───

    /// <summary>
    /// The λ = 0 LP refuses a budget <c>1 − n·minWeight</c> or a capacity <c>maxWeight − minWeight</c> that is not finite,
    /// each half on its own: at <c>double</c> the non-finite value is refused by the LP's own check; at <c>decimal</c>,
    /// which throws where IEEE types produce ±∞, the overflow forming it is reported as the range refusal.
    /// </summary>
    [Fact]
    public void LpBudgetOrCapacity_EachHalfAloneIsRefusedAtBothTypes()
    {
        // Capacity only: n = 1, budget 1 + 1e308 (finite), capacity 2e308 (infinite).
        AssertRefused(LpRefusal, () => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1.0 } }, [2.0], 0.0, -1e308, 1e308), "double capacity only");

        // Budget only: n = 2, budget 1 + 2e308 (infinite), capacity 1 + 1e308 (finite).
        AssertRefused(LpRefusal, () => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }, [2.0, 1.0], 0.0, -1e308, 1.0), "double budget only");

        // Capacity only at decimal: n·minWeight and the budget 1 + 5e28 are representable, the capacity 1e29 is not.
        var capacity = AssertRefused(ExceededRange, () => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[,] { { 1m } }, [2m], 0m, -5e28m, 5e28m), "decimal capacity only");
        Assert.IsType<OverflowException>(capacity.InnerException);

        // Budget only at decimal: n·minWeight = −decimal.MaxValue is representable, the budget 1 + MaxValue is not, the
        // capacity 1 + MaxValue/3 is.
        var third = -decimal.MaxValue / 3m;
        Assert.Equal(-decimal.MaxValue, 3m * third);
        var budget = AssertRefused(ExceededRange, () => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[,] { { 1m, 0m, 0m }, { 0m, 1m, 0m }, { 0m, 0m, 1m } }, [2m, 1m, 0m], 0m, third, 1m), "decimal budget only");
        Assert.IsType<OverflowException>(budget.InnerException);
    }

    /// <summary>The counterpart: finite LP budgets and capacities at the same scales are allocated, not refused.</summary>
    [Fact]
    public void LpBudgetOrCapacity_FiniteAtTheSameScales_IsAllocated()
    {
        Assert.Equal([1.0], ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1.0 } }, [2.0], 0.0, -1e307, 1e307));
        var two = ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }, [2.0, 1.0], 0.0, -1e307, 1.0);
        Assert.Equal(1.0, two[0]);
        Assert.Equal([1m], ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[,] { { 1m } }, [2m], 0m, -3e28m, 3e28m));
    }

    // ─── Translation of an OverflowException raised inside the solve ───

    /// <summary>
    /// The documented translation: an <see cref="OverflowException"/> raised inside the solve leaves both public
    /// objectives as <see cref="InvalidOperationException"/> with the range message and the overflow as the inner
    /// exception, at both types. No known public input makes a <c>λ &gt; 0</c> solve overflow, so the overflow is raised
    /// by the trace hook on the exact route's result, deep inside the solve.
    /// </summary>
    [Fact]
    public void OverflowRaisedInsideTheSolve_IsReportedAsTheRangeRefusalByBothObjectivesAtBothTypes()
    {
        AssertOverflowTranslated<double>();
        AssertOverflowTranslated<decimal>();
    }

    // ─── The exact route's budget of 12N + 12 working-set changes ───

    /// <summary>
    /// Spending the exact route's working-set budget is the documented refusal naming that budget: with the budget one
    /// below the changes the forced exact route makes, both objectives refuse at both types; with the budget equal to
    /// them, the same solve returns the same vector. The budget the solver applies otherwise is <c>12N + 12</c>, and no
    /// solve of this problem comes near it.
    /// </summary>
    [Fact]
    public void ExactRouteBudget_SpentBeforeTheOptimum_IsTheDocumentedRefusal_AndOneMoreChangeSolves()
    {
        AssertExactRouteBudget<double>(withMeans: false);
        AssertExactRouteBudget<double>(withMeans: true);
        AssertExactRouteBudget<decimal>(withMeans: false);
        AssertExactRouteBudget<decimal>(withMeans: true);
    }

    // ─── The exit certificate on the exact route's result ───

    /// <summary>
    /// A result of the exact route that the exit certificate rejects is refused, naming the failed component, never
    /// returned: the trace hook halves every weight of the exact route's rounded optimum, so the budget row fails, and
    /// both objectives refuse at both types with the certificate's budget detail.
    /// </summary>
    [Fact]
    public void ExactRouteResult_RejectedByTheCertificate_IsRefusedNamingTheComponent()
    {
        AssertCertificateRefusal<double>(withMeans: false, frontier: false);
        AssertCertificateRefusal<double>(withMeans: true, frontier: false);
        AssertCertificateRefusal<decimal>(withMeans: false, frontier: false);
        AssertCertificateRefusal<decimal>(withMeans: true, frontier: false);
    }

    /// <summary>
    /// The same refusal on the equal-weight frontier point that the exact route returns when the bounds clamped to
    /// <c>1/n</c> hold no exactly budget-feasible point (<c>n = 3</c>, <c>maxWeight</c> below <c>1/3</c>): the hook halves
    /// it, and both objectives refuse at both types instead of returning it.
    /// </summary>
    [Fact]
    public void EqualWeightFrontierPoint_RejectedByTheCertificate_IsRefusedNamingTheComponent()
    {
        AssertCertificateRefusal<double>(withMeans: false, frontier: true);
        AssertCertificateRefusal<double>(withMeans: true, frontier: true);
        AssertCertificateRefusal<decimal>(withMeans: false, frontier: true);
        AssertCertificateRefusal<decimal>(withMeans: true, frontier: true);
    }

    /// <summary>The counterpart: the unhooked exact route and frontier return certified vectors for the same problems.</summary>
    [Fact]
    public void ExactRouteResult_Unhooked_IsCertifiedAndReturned()
    {
        foreach (var frontier in new[] { false, true })
        {
            foreach (var withMeans in new[] { false, true })
            {
                AssertUnitBudget(SolveForced<double>(withMeans, frontier, hook: null), 1e-12);
                AssertUnitBudget(SolveForced<decimal>(withMeans, frontier, hook: null), 1e-24);
            }
        }
    }

    // ─── The rounded route's budget of 8N + 8 working-set changes ───

    /// <summary>
    /// The rounded route stops at its budget and hands the problem to the exact route: with the budget one below the
    /// changes the rounded route makes, it makes exactly one change more than the budget, records the hand-off for the
    /// budget, and the exact route returns the same optimum, through both objectives at both types.
    /// </summary>
    [Fact]
    public void RoundedRouteBudget_SpentBeforeTheOptimum_HandsTheProblemToTheExactRoute()
    {
        AssertRoundedHandOff<double>(withMeans: false, 1e-9);
        AssertRoundedHandOff<double>(withMeans: true, 1e-9);
        AssertRoundedHandOff<decimal>(withMeans: false, 1e-20);
        AssertRoundedHandOff<decimal>(withMeans: true, 1e-20);
    }

    /// <summary>
    /// The production budget <c>8N + 8</c>: on the corpus problems whose rounded route spends it (<c>n = 20</c>,
    /// boxes <c>±1e89</c> and <c>±1e150</c>, both objectives), the rounded route makes exactly <c>8N + 9</c> changes and
    /// hands the problem to the exact route for the budget.
    /// </summary>
    [Theory]
    [InlineData("z2g-mi0-146#0")]
    [InlineData("z2i-mi2-54#3")]
    [InlineData("z2g-mv2-174#2")]
    [InlineData("z2i-mv2-76#3")]
    public void RoundedRouteBudget_IsEightNPlusEight_OnTheCorpusWitnessesThatSpendIt(string name)
    {
        var problem = Witness(name);
        Assert.Equal("double", problem.Type);
        var n = problem.Cov.Length;
        var cov = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = Parse(problem.Cov[i][j]);
            }
        }

        var trace = new ActiveSetQpSolver<double>.RouteTrace();
        var weights = problem.Mu is null
            ? ActiveSetQpSolver<double>.SolveMinVarianceUsingCore(cov, Parse(problem.Lb), Parse(problem.Ub), forcedReference: -1, trace: trace)
            : ActiveSetQpSolver<double>.SolveMeanVarianceUsingCore(cov, problem.Mu.Select(Parse).ToArray(), Parse(problem.Lambda!), Parse(problem.Lb), Parse(problem.Ub), trace: trace);

        Assert.Equal(n, weights.Length);
        Assert.Equal(ActiveSetQpSolver<double>.ExactRouteReason.RoundedBudget, trace.ExactRouteReason);
        Assert.True(trace.RoundedBudgetReached, name);
        Assert.Equal((8 * n) + 9, trace.RoundedChanges);
        Assert.InRange(trace.ExactChanges, 0, ActiveSetQpSolver<double>.ExactRouteBudget(n));
    }

    // ─── Helpers ───

    private static (double[,] DoubleCov, decimal[,] DecimalCov) EdgeOfRangeLongShort(double doubleScale, decimal decimalScale)
    {
        const double rho = 1 - 1e-5, s = 1.01;
        var d = new[,] { { doubleScale, doubleScale * rho * s }, { doubleScale * rho * s, doubleScale * s * s } };

        const decimal sc = 1.05m, rhoM = 0.999m;
        var m = decimalScale / (sc * sc);
        var m01 = m * rhoM * sc;
        var dec = new[,] { { m, m01 }, { m01, m * sc * sc } };
        return (d, dec);
    }

    private static InvalidOperationException AssertRefused(string message, Func<object> solve, string label)
    {
        var error = Record.Exception(solve);
        Assert.True(error is InvalidOperationException, label + ": " + (error is null ? "returned a vector" : error.GetType().Name + ": " + error.Message));
        Assert.True(message == error!.Message, label + ": " + error.Message);
        return (InvalidOperationException)error;
    }

    private static T[,] Covariance<T>(int n)
        where T : IFloatingPoint<T>
    {
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = T.CreateChecked(s_covariance[i, j]);
            }
        }

        return cov;
    }

    private static T[] Means<T>(int n)
        where T : IFloatingPoint<T>
        => [.. s_means.Take(n).Select(T.CreateChecked)];

    private static T[] Solve<T>(bool withMeans, T lower, T upper, ActiveSetQpSolver<T>.RouteTrace trace, int n = 4)
        where T : IFloatingPoint<T>
        => withMeans
            ? ActiveSetQpSolver<T>.SolveMeanVarianceUsingCore(Covariance<T>(n), Means<T>(n), T.CreateChecked(3), lower, upper, trace)
            : ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(Covariance<T>(n), lower, upper, forcedReference: -1, trace: trace);

    private static T[] SolveForced<T>(bool withMeans, bool frontier, Func<T[], T[]>? hook)
        where T : IFloatingPoint<T>
    {
        var trace = new ActiveSetQpSolver<T>.RouteTrace { ForceExactRoute = true, ExactRouteResultHook = hook };

        // The frontier problem: three assets and maxWeight 0.1, raised to fl(1/3), with 3·fl(1/3) < 1.
        return frontier
            ? Solve(withMeans, T.Zero, T.CreateChecked(0.1), trace, n: 3)
            : Solve(withMeans, T.Zero, T.CreateChecked(0.4), trace);
    }

    private static void AssertOverflowTranslated<T>()
        where T : IFloatingPoint<T>
    {
        foreach (var withMeans in new[] { false, true })
        {
            var label = typeof(T).Name + (withMeans ? " mean-variance" : " min-variance");
            var error = AssertRefused(ExceededRange, () => SolveForced<T>(withMeans, frontier: false, _ => throw new OverflowException("raised inside the solve")), label);
            Assert.IsType<OverflowException>(error.InnerException);
        }
    }

    private static void AssertExactRouteBudget<T>(bool withMeans)
        where T : IFloatingPoint<T>
    {
        var label = typeof(T).Name + (withMeans ? " mean-variance" : " min-variance");
        var lower = T.Zero;
        var upper = T.CreateChecked(0.4);

        var free = new ActiveSetQpSolver<T>.RouteTrace { ForceExactRoute = true };
        var expected = Solve(withMeans, lower, upper, free);
        var changes = free.ExactChanges;
        Assert.True(changes >= 2, label + ": the forced exact route made " + changes + " changes");
        Assert.True(changes <= ActiveSetQpSolver<T>.ExactRouteBudget(4), label);

        var spent = new ActiveSetQpSolver<T>.RouteTrace { ForceExactRoute = true, ExactRouteBudgetOverride = changes - 1 };
        var kind = withMeans ? "mean-variance" : "min-variance";
        var error = Record.Exception(() => Solve(withMeans, lower, upper, spent));
        Assert.True(error is InvalidOperationException, label + ": " + (error is null ? "returned a vector" : error.GetType().Name + ": " + error.Message));
        var prefix = "Active-set QP solver did not converge for " + kind + " problem within its iteration budget of "
            + (changes - 1).ToString(CultureInfo.InvariantCulture) + " working-set changes (";
        Assert.True(error!.Message.StartsWith(prefix, StringComparison.Ordinal), label + ": " + error.Message);
        Assert.Equal(changes, spent.ExactChanges);

        var enough = new ActiveSetQpSolver<T>.RouteTrace { ForceExactRoute = true, ExactRouteBudgetOverride = changes };
        Assert.Equal(expected, Solve(withMeans, lower, upper, enough));
        Assert.Equal(changes, enough.ExactChanges);
    }

    private static void AssertCertificateRefusal<T>(bool withMeans, bool frontier)
        where T : IFloatingPoint<T>
    {
        var label = typeof(T).Name + (withMeans ? " mean-variance" : " min-variance") + (frontier ? " frontier" : string.Empty);
        var kind = withMeans ? "mean-variance" : "min-variance";
        var two = T.CreateChecked(2);
        var error = Record.Exception(() => SolveForced<T>(withMeans, frontier, w => [.. w.Select(x => x / two)]));
        Assert.True(error is InvalidOperationException, label + ": " + (error is null ? "returned a vector" : error.GetType().Name + ": " + error.Message));
        var prefix = "Active-set QP solver did not converge for " + kind + " problem: the result failed the KKT certificate (budget: |1'w - 1| = ";
        Assert.True(error!.Message.StartsWith(prefix, StringComparison.Ordinal), label + ": " + error.Message);
    }

    private static void AssertRoundedHandOff<T>(bool withMeans, double tolerance)
        where T : IFloatingPoint<T>
    {
        var label = typeof(T).Name + (withMeans ? " mean-variance" : " min-variance");
        var lower = T.Zero;
        var upper = T.CreateChecked(0.4);

        var free = new ActiveSetQpSolver<T>.RouteTrace();
        var expected = Solve(withMeans, lower, upper, free);
        var changes = free.RoundedChanges;
        Assert.True(changes >= 2, label + ": the rounded route made " + changes + " changes");
        Assert.Equal(ActiveSetQpSolver<T>.ExactRouteReason.None, free.ExactRouteReason);
        Assert.False(free.RoundedBudgetReached, label);

        var spent = new ActiveSetQpSolver<T>.RouteTrace { RoundedBudgetOverride = changes - 1 };
        var handedOff = Solve(withMeans, lower, upper, spent);
        Assert.Equal(ActiveSetQpSolver<T>.ExactRouteReason.RoundedBudget, spent.ExactRouteReason);
        Assert.True(spent.RoundedBudgetReached, label);
        Assert.Equal(changes, spent.RoundedChanges);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(double.CreateChecked(handedOff[i] - expected[i])) <= tolerance, label + ": weight " + i + " " + handedOff[i] + " vs " + expected[i]);
        }

        var enough = new ActiveSetQpSolver<T>.RouteTrace { RoundedBudgetOverride = changes };
        Assert.Equal(expected, Solve(withMeans, lower, upper, enough));
        Assert.Equal(ActiveSetQpSolver<T>.ExactRouteReason.None, enough.ExactRouteReason);
    }

    private static void AssertUnitBudget<T>(T[] weights, double tolerance)
        where T : IFloatingPoint<T>
    {
        var sum = T.Zero;
        foreach (var w in weights)
        {
            sum += w;
        }

        Assert.True(Math.Abs(double.CreateChecked(sum - T.One)) <= tolerance, typeof(T).Name + ": " + sum);
    }

    private static ActiveSetQpCorpusTests.HeldOutProblem Witness(string name)
    {
        // The corpus witnesses live with the corpus tests, which keep the one copy of their data.
        var method = typeof(ActiveSetQpCorpusTests).GetMethod("RepeatedBlockWitness", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new MissingMethodException(nameof(ActiveSetQpCorpusTests), "RepeatedBlockWitness");
        return (ActiveSetQpCorpusTests.HeldOutProblem)method.Invoke(null, [name])!;
    }

    private static double Parse(string s) => double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture);
}
