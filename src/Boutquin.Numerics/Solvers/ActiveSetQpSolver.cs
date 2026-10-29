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
using System.Runtime.InteropServices;

using Boutquin.Numerics.Internal;
using Boutquin.Numerics.LinearAlgebra;

namespace Boutquin.Numerics.Solvers;

/// <summary>
/// Generic active-set QP solver for two standard portfolio optimization problems:
/// <list type="bullet">
///   <item><description><b>MinVar:</b>  min w′Σw  s.t. 1′w=1, lb ≤ w ≤ ub</description></item>
///   <item><description><b>MeanVar:</b> max w′μ − (λ/2)w′Σw  s.t. 1′w=1, lb ≤ w ≤ ub</description></item>
/// </list>
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+√. Requires <typeparamref name="T"/> to implement
/// <see cref="IFloatingPoint{TSelf}"/> and represent the thresholds documented below. Uses <see cref="CholeskyDecomposition{T}"/>
/// internally which calls <see cref="Boutquin.Numerics.Internal.NumericPrecision{T}.Sqrt"/>. The conditioning guard
/// first evaluates certified bounds in <see cref="double"/> and double-double arithmetic (an upper bound on the block's
/// condition number from an approximate inverse, or a proved lower bound on the smallest eigenvalue of the whole
/// equilibrated covariance or of the block). For <see cref="double"/>, a proposed inverse column also supplies an
/// exact-arithmetic norm witness that can prove the block outside the margin. An undecided block is tested exactly.
/// The positive-definiteness test of the
/// covariance likewise first runs a Cholesky factorisation in <see cref="double"/> with an a-priori bound on its
/// rounding error, which can only prove that the exact test accepts; a covariance it cannot decide is factored in
/// double-double arithmetic with a rigorous bound on the factor's residual, which proves either that the exact test
/// accepts or, with a witness evaluated in exact integer arithmetic, the order of the first leading principal minor that
/// is not positive, the refusal the exact test would raise. What neither proves is decided exactly. Each check returns
/// only the exact test's decision, no <see cref="double"/> value enters a returned weight, and so the working type alone
/// sets the precision of the result.
/// </para>
/// <para>
/// <b>MinVar</b> runs a primal-feasible active-set method (Nocedal–Wright 2006, Algorithm 16.3, on the budget
/// row and the box): the problem is equilibrated once by radix powers of <c>√Σᵢᵢ</c>; a feasible start is built
/// by the greedy budget fill (every weight at the lower bound, the remaining budget up to the upper bounds in
/// index order); each step solves the equality-constrained subproblem of the free variables in a budget
/// null-space basis with exactly formed residuals; a step-length rule adds at most the blocking bound; a
/// variable is released by its multiplier relative to its own row scale (smallest index among ties); and the
/// result is checked by an independent KKT certificate before it is returned.
/// </para>
/// <para>
/// <b>MeanVar</b> with <c>λ &gt; 0</c> runs the same primal-feasible active set on the equivalent problem
/// <c>min (λ/2)·w′Σw − μ′w</c>: the gradient of every step is <c>λΣw − μ</c>, formed exactly; the null-space step,
/// the step-length rule, the release rule, the termination rule, the exit certificate (with the means and the risk
/// aversion) and the conditioning guard are those of MinVar. Each multiplier is compared with its own row's scale,
/// which includes that row's mean, never with a maximum over every asset. <c>λ = 0</c> is the greedy max-return LP.
/// </para>
/// <para>
/// Type-precision note: thresholds <c>1e-20</c> and <c>1e-10</c> must be positive
/// and finite in <typeparamref name="T"/>. Both solve entry points throw
/// <see cref="NotSupportedException"/> when either threshold underflows or is not
/// finite, including for <c>Half</c>. <c>double</c>, <c>float</c>, and <c>decimal</c>
/// satisfy this requirement.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type (e.g. <c>double</c>, <c>decimal</c>).</typeparam>
public static class ActiveSetQpSolver<T>
    where T : IFloatingPoint<T>
{
    // The positive finite thresholds the working type must represent (the documented NotSupportedException for
    // types that cannot, such as Half); see ValidateThresholdRange.
    private static readonly T s_epsilon20 = T.CreateChecked(1e-20);
    private static readonly T s_epsilon10 = T.CreateChecked(1e-10);

    private static void ValidateThresholdRange()
    {
        if (!T.IsFinite(s_epsilon20) || !T.IsFinite(s_epsilon10) || s_epsilon20 <= T.Zero || s_epsilon10 <= T.Zero)
        {
            throw new NotSupportedException("The numeric type cannot represent the positive finite QP thresholds 1e-20 and 1e-10.");
        }
    }

    // ─── Public QP solvers ─────────────────────────────────────────────

    /// <summary>
    /// Solves the minimum-variance QP: min w′Σw s.t. 1′w=1, minWeight ≤ w_i ≤ maxWeight.
    /// </summary>
    /// <param name="covariance">Symmetric positive-definite N×N covariance matrix.</param>
    /// <param name="minWeight">Lower weight bound (per asset).</param>
    /// <param name="maxWeight">Upper weight bound (per asset).</param>
    /// <returns>Optimal weight vector summing to 1.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="covariance"/> is <c>null</c>.</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> cannot represent the positive finite QP thresholds.</exception>
    /// <exception cref="ArgumentException">
    /// Matrix has zero dimension (0×0) or is not square.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// The covariance is not positive definite; or the active-set iteration did not converge — the result
    /// failed the independent KKT certificate (the message names the failed component: budget, bound,
    /// stationarity or multiplier sign) and no release that could lower the objective remains, the reduced Hessian of
    /// the free variables is not positive definite at the working precision with any reference variable and no
    /// working-set change is left to roll back (stationarity cannot be certified), the block of the covariance that
    /// decides the result is numerically singular at the working precision (stationarity cannot be certified:
    /// <c>48·N·u·κ ≥ 1</c> for the exact 1-norm condition number <c>κ</c> of the equilibrated block, <c>u</c> the unit
    /// roundoff, so the optimum is not determined by the rounded problem), or the exact route's defensive limit of
    /// <c>12N + 12</c> working-set changes was reached (the rounded route's limit of <c>8N + 8</c> hands the problem to the
    /// exact route instead of refusing it); or a product <c>Σᵢⱼ·wⱼ</c> of the result's gradient is not representable
    /// (release gradient non-finite; only possible for inputs outside the operating domain). For <c>T = decimal</c>, which
    /// throws where IEEE types produce ±∞, an <see cref="OverflowException"/> raised inside the solve
    /// is reported as this exception.
    /// </exception>
    /// <remarks>
    /// When the caller-supplied bounds are infeasible (<c>n · maxWeight &lt; 1</c> or
    /// <c>n · minWeight &gt; 1</c>), only the offending bound is relaxed to
    /// <c>1/n</c> before solving; the other bound is retained. Feasible calls are unaffected when
    /// <c>n · maxWeight ≥ 1</c> and <c>n · minWeight ≤ 1</c>. Every returned vector has passed the KKT
    /// certificate; results are deterministic (identical inputs give bit-identical outputs).
    /// </remarks>
    public static T[] SolveMinVariance(T[,] covariance, T minWeight, T maxWeight)
        => SolveMinVarianceUsingCore(covariance, minWeight, maxWeight);

    /// <summary>
    /// The min-variance solve behind <see cref="SolveMinVariance"/>, with two test-only diagnostics:
    /// <paramref name="forcedReference"/> (≥ 0) makes that variable, whenever it is free, the only budget-elimination
    /// reference the core tries, with no retry; <paramref name="trace"/> receives the route diagnostics of the solve.
    /// </summary>
    internal static T[] SolveMinVarianceUsingCore(
        T[,] covariance, T minWeight, T maxWeight, int forcedReference = -1, RouteTrace? trace = null)
    {
        try
        {
            return SolveMinVarianceCore(covariance, minWeight, maxWeight, forcedReference, trace);
        }
        catch (OverflowException ex)
        {
            throw ToOverflowContract(ex);
        }
    }

    private static T[] SolveMinVarianceCore(
        T[,] covariance, T minWeight, T maxWeight, int forcedReference, RouteTrace? trace)
    {
        ArgumentNullException.ThrowIfNull(covariance);
        ValidateThresholdRange();

        var n = covariance.GetLength(0);
        if (n == 0)
        {
            throw new ArgumentException(
                "Covariance matrix must have a positive dimension.", nameof(covariance));
        }

        if (covariance.GetLength(1) != n)
        {
            throw new ArgumentException("Covariance matrix must be square.", nameof(covariance));
        }

        ValidateCovariance(covariance, n);
        ValidateBounds(minWeight, maxWeight);

        if (n == 1)
        {
            if (covariance[0, 0] <= T.Zero)
            {
                throw new InvalidOperationException("Covariance matrix must be positive definite.");
            }
            return [T.One];
        }

        // Clamp infeasible bounds to the equal-weight frontier (1/n) so the
        // sum-to-one constraint remains satisfiable. The clamp is a no-op for
        // already-feasible bounds (n·maxWeight ≥ 1 and n·minWeight ≤ 1).
        var inv = T.One / T.CreateChecked(n);
        var effMax = T.Max(maxWeight, inv);
        var effMin = T.Min(minWeight, inv);
        minWeight = effMin;
        maxWeight = effMax;

        return PrimalActiveSet.SolveMinVariance(covariance, minWeight, maxWeight, forcedReference, trace);
    }

    /// <summary>
    /// Solves the mean-variance QP: max w′μ − (λ/2)w′Σw s.t. 1′w=1, minWeight ≤ w_i ≤ maxWeight.
    /// </summary>
    /// <param name="covariance">
    /// Symmetric N×N covariance matrix, positive definite for λ &gt; 0 (for <c>λ = 0</c> the LP branch reads it only for
    /// its validation: symmetry and finiteness).
    /// </param>
    /// <param name="means">Expected return vector of length N.</param>
    /// <param name="riskAversion">Risk-aversion parameter λ (λ = 0 reduces to pure LP).</param>
    /// <param name="minWeight">Lower weight bound (per asset).</param>
    /// <param name="maxWeight">Upper weight bound (per asset).</param>
    /// <returns>Optimal weight vector summing to 1.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="covariance"/> is <c>null</c>.</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> cannot represent the positive finite QP thresholds.</exception>
    /// <exception cref="ArgumentException">
    /// Matrix has zero dimension (0×0), is not square, or its dimensions mismatch the
    /// <paramref name="means"/> length.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="riskAversion"/> is negative or not finite, or a bound is not finite.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// For <c>λ &gt; 0</c>, the covariance is not positive definite (for <c>λ = 0</c> it is not inspected beyond symmetry
    /// and finiteness), or the active-set iteration did not converge — the
    /// result failed the independent KKT certificate (the message names the failed component: budget, bound,
    /// stationarity or multiplier sign) and no release that could lower the objective remains, the reduced Hessian of
    /// the free variables is not positive definite at the working precision with any reference variable and no
    /// working-set change is left to roll back, the block of the covariance that decides the result is numerically
    /// singular at the working precision (<c>48·N·u·κ ≥ 1</c> for the exact 1-norm condition number <c>κ</c> of the
    /// equilibrated block, <c>u</c> the unit roundoff), or the exact route's defensive limit of <c>12N + 12</c>
    /// working-set changes was reached (the rounded route's limit of <c>8N + 8</c> hands the problem to the exact route
    /// instead of refusing it); or a product <c>Σᵢⱼ·wⱼ</c> of the result's gradient is not representable, or a term <c>λ·Σᵢⱼ·wⱼ</c> is not
    /// representable together with its factor <c>λ·Σᵢⱼ</c> (release gradient non-finite; only possible for inputs outside
    /// the operating domain); or, for <c>λ = 0</c>, the LP budget or capacity is not finite.
    /// For <c>T = decimal</c>, which throws where IEEE types produce ±∞, an
    /// <see cref="OverflowException"/> raised inside the solve is reported as this exception.
    /// </exception>
    /// <remarks>
    /// When the caller-supplied bounds are infeasible (<c>n · maxWeight &lt; 1</c> or
    /// <c>n · minWeight &gt; 1</c>), only the offending bound is relaxed to
    /// <c>1/n</c> before solving; the other bound is retained. Feasible calls are unaffected when
    /// <c>n · maxWeight ≥ 1</c> and <c>n · minWeight ≤ 1</c>.
    /// Note: the <c>riskAversion = 0</c> LP path delegates to
    /// <see cref="SolveMaxReturnLP"/> which retains its own feasibility check and does not read the covariance, so
    /// positive definiteness is required only for <c>riskAversion &gt; 0</c>.
    /// LP feasibility products are evaluated in <typeparamref name="T"/>. A bound product that rounds to one
    /// is accepted; sum-to-one is subject to working-type rounding, while each returned weight stays in its box.
    /// For <c>riskAversion &gt; 0</c> every returned vector has passed the KKT certificate. Results are deterministic
    /// (identical inputs give bit-identical outputs).
    /// </remarks>
    public static T[] SolveMeanVariance(
        T[,] covariance, ReadOnlySpan<T> means, T riskAversion, T minWeight, T maxWeight)
        => SolveMeanVarianceUsingCore(covariance, means, riskAversion, minWeight, maxWeight);

    /// <summary>
    /// The mean-variance solve behind <see cref="SolveMeanVariance"/>; <paramref name="trace"/> (test use only)
    /// receives the route diagnostics of the primal core's solve for <c>λ &gt; 0</c>.
    /// </summary>
    internal static T[] SolveMeanVarianceUsingCore(
        T[,] covariance, ReadOnlySpan<T> means, T riskAversion, T minWeight, T maxWeight, RouteTrace? trace = null)
    {
        try
        {
            return SolveMeanVarianceCore(covariance, means, riskAversion, minWeight, maxWeight, trace);
        }
        catch (OverflowException ex)
        {
            throw ToOverflowContract(ex);
        }
    }

    private static T[] SolveMeanVarianceCore(
        T[,] covariance, ReadOnlySpan<T> means, T riskAversion, T minWeight, T maxWeight, RouteTrace? trace)
    {
        ArgumentNullException.ThrowIfNull(covariance);
        ValidateThresholdRange();

        var n = covariance.GetLength(0);
        if (n == 0)
        {
            throw new ArgumentException(
                "Covariance matrix must have a positive dimension.", nameof(covariance));
        }

        if (covariance.GetLength(1) != n)
        {
            throw new ArgumentException("Covariance matrix must be square.", nameof(covariance));
        }

        if (means.Length != n)
        {
            throw new ArgumentException("Means vector length must match covariance dimension.", nameof(means));
        }

        ValidateCovariance(covariance, n);
        ValidateMeans(means);
        ValidateBounds(minWeight, maxWeight);

        if (!T.IsFinite(riskAversion) || riskAversion < T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(riskAversion), riskAversion, "Risk aversion must be finite and non-negative.");
        }

        // Copy to array once for internal methods that require T[]:
        var meansArray = means.ToArray();

        // λ = 0 reduces to the pure max-return LP. Delegate before the clamp below:
        // SolveMaxReturnLP performs its own feasibility check (review S10).
        if (riskAversion == T.Zero)
        {
            return SolveMaxReturnLP(meansArray, n, minWeight, maxWeight);
        }

        if (n == 1)
        {
            if (covariance[0, 0] <= T.Zero)
            {
                throw new InvalidOperationException("Covariance matrix must be positive definite.");
            }
            return [T.One];
        }

        // Clamp infeasible bounds to the equal-weight frontier (1/n) so the sum-to-one constraint remains
        // satisfiable. The clamp is a no-op for already-feasible bounds (n·maxWeight ≥ 1 and n·minWeight ≤ 1).
        var frontier = T.One / T.CreateChecked(n);
        return PrimalActiveSet.Solve(
            covariance, meansArray, riskAversion, T.Min(minWeight, frontier), T.Max(maxWeight, frontier), forcedReference: -1, trace);
    }

    // ─── Private helpers ───────────────────────────────────────────────

    /// <summary>
    /// Translates a range overflow raised by <c>decimal</c> arithmetic (which throws where
    /// IEEE types produce ±∞) into the documented reject-overflow contract that the IEEE
    /// instantiations already honour: <see cref="InvalidOperationException"/>.
    /// </summary>
    private static InvalidOperationException ToOverflowContract(OverflowException inner) =>
        new("QP arithmetic exceeded the numeric type's range; rescale the problem.", inner);

    /// <summary>
    /// Solves the pure max-return LP: maximize w′μ s.t. 1′w=1, minWeight ≤ w_i ≤ maxWeight.
    /// Greedy: assign maxWeight to assets in descending return order until budget exhausted.
    /// </summary>
    private static T[] SolveMaxReturnLP(T[] means, int n, T minWeight, T maxWeight)
    {
        if (T.CreateChecked(n) * minWeight > T.One)
        {
            throw new ArgumentException(
                $"Infeasible bounds: n * minWeight ({T.CreateChecked(n) * minWeight}) > 1 (sum-to-one constraint cannot be satisfied).",
                nameof(minWeight));
        }

        if (T.CreateChecked(n) * maxWeight < T.One)
        {
            throw new ArgumentException(
                $"Infeasible bounds: n * maxWeight ({T.CreateChecked(n) * maxWeight}) < 1 (sum-to-one constraint cannot be satisfied).",
                nameof(maxWeight));
        }

        if (!T.IsFinite(T.One - T.CreateChecked(n) * minWeight) || !T.IsFinite(maxWeight - minWeight))
        {
            throw new InvalidOperationException(
                "LP allocation produced a non-finite budget or capacity; rescale the bounds.");
        }

        var order = Enumerable.Range(0, n).ToArray();
        Array.Sort(order, (a, b) =>
        {
            var comparison = means[b].CompareTo(means[a]);
            return comparison != 0 ? comparison : a.CompareTo(b);
        });

        return ExactGreedyAllocation(order, minWeight, maxWeight);
    }

    /// <summary>
    /// The greedy budget fill carried in exact arithmetic, as <c>PrimalActiveSet.FeasibleStart</c> carries it: the
    /// budget <c>1 − n·minWeight</c> and the capacity <c>maxWeight − minWeight</c> are formed without rounding, so every
    /// weight is <paramref name="minWeight"/>, <paramref name="maxWeight"/> or the one partially filled weight, the only
    /// value rounded into <typeparamref name="T"/> (once, to nearest), which therefore stays in the box.
    /// </summary>
    private static T[] ExactGreedyAllocation(int[] order, T minWeight, T maxWeight)
    {
        var weights = new T[order.Length];
        Array.Fill(weights, minWeight);
        var lowerX = Exact.From(minWeight);
        var capacity = Exact.From(maxWeight) - lowerX;
        var remaining = Exact.One - (Exact.FromInt(order.Length) * lowerX);
        foreach (var i in order)
        {
            if (remaining.Sign <= 0)
            {
                break;
            }

            if (remaining >= capacity)
            {
                weights[i] = maxWeight;
                remaining -= capacity;
            }
            else
            {
                weights[i] = (lowerX + remaining).ToT();
                remaining = Exact.Zero;
            }
        }

        return weights;
    }

    private static void ValidateCovariance(T[,] covariance, int n)
    {
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                if (j < i && covariance[i, j] != covariance[j, i])
                {
                    throw new ArgumentException("Covariance matrix must be symmetric.", nameof(covariance));
                }

                if (!T.IsFinite(covariance[i, j]))
                {
                    throw new ArgumentException(
                        $"Covariance entry at [{i}, {j}] must be finite.", nameof(covariance));
                }
            }
        }
    }

    private static void ValidateMeans(ReadOnlySpan<T> means)
    {
        for (var i = 0; i < means.Length; i++)
        {
            if (!T.IsFinite(means[i]))
            {
                throw new ArgumentException($"Mean at index {i} must be finite.", nameof(means));
            }
        }
    }

    private static void ValidateBounds(T minWeight, T maxWeight)
    {
        if (!T.IsFinite(minWeight))
        {
            throw new ArgumentOutOfRangeException(
                nameof(minWeight), minWeight, "Minimum weight must be finite.");
        }

        if (!T.IsFinite(maxWeight))
        {
            throw new ArgumentOutOfRangeException(
                nameof(maxWeight), maxWeight, "Maximum weight must be finite.");
        }

        if (minWeight > maxWeight)
        {
            throw new ArgumentException("Minimum weight must not exceed maximum weight.", nameof(minWeight));
        }
    }

    private static T[] FilledArray(int length, T value)
    {
        var arr = new T[length];
        Array.Fill(arr, value);
        return arr;
    }

    // ─── Primal-feasible active set (min-variance and mean-variance) ───

    /// <summary>
    /// Tolerance scale <c>κ</c> of the exit certificate (<see cref="QpKktCertificate{T}"/>): every component
    /// tolerance of the certificate is multiplied by <c>κ</c> before the returned vector is accepted.
    /// </summary>
    internal static T CertificateToleranceScale { get; } = T.CreateChecked(8);

    /// <summary>
    /// Defensive limit on the working-set changes (bounds added plus bounds released) of the rounded route of a solve,
    /// <c>8n + 8</c>: the rounded route hands a problem that spends it to the exact route, which has a budget of its own
    /// (<see cref="ExactRouteBudget"/>).
    /// </summary>
    internal static int IterationBudget(int n) => (8 * n) + 8;

    /// <summary>
    /// Defensive limit on the working-set changes of the exact route, <c>12n + 12</c>, fresh when the route is entered.
    /// The exact route follows the rules of the rounded route but, entered far from the optimum on a degenerate problem,
    /// can need more changes than it (its released variables may cross the box from bound to bound); spending this
    /// budget is the documented refusal that names the iteration budget.
    /// </summary>
    internal static int ExactRouteBudget(int n) => (12 * n) + 12;

    /// <summary>Why a solve took the exact route (test diagnostics).</summary>
    internal enum ExactRouteReason
    {
        None,
        Forced,
        UnusableWorkingSet,
        RefinementNotConverged,
        CertificateRejected,
        RoundedBudget,
    }

    /// <summary>
    /// Test diagnostics of one primal active-set solve, filled by the solve it is passed to (never shared between
    /// solves): the working-set changes each route made, whether the rounded route spent its budget, and why the exact
    /// route was taken. <see cref="ForceExactRoute"/> makes the solve run the exact route from the exact feasible start.
    /// </summary>
    internal sealed class RouteTrace
    {
        public bool ForceExactRoute { get; init; }

        public int RoundedChanges { get; set; }

        public bool RoundedBudgetReached { get; set; }

        public ExactRouteReason ExactRouteReason { get; set; }

        public int ExactChanges { get; set; }

        /// <summary>Releases the exact route undid because the next subspace minimiser did not lower the exact objective.</summary>
        public int ExactRollbacks { get; set; }

        /// <summary>Which test decided the positive definiteness of the covariance (the solve's first decision).</summary>
        public PositiveDefinitenessRoute PositiveDefinitenessRoute { get; set; }

        /// <summary>
        /// The certified <see cref="double"/> test of positive definiteness, the solve's first decision (once per solve;
        /// its factorisation and residual bound about <c>n³/3</c> loop bodies).
        /// </summary>
        public StageWork DoubleCertificate { get; } = new();

        /// <summary>
        /// The certified double-double test of positive definiteness, with its refusal witness (once per solve the double
        /// certificate leaves undecided; about <c>n³/3</c> loop bodies when it proves the covariance).
        /// </summary>
        public StageWork LeadingMinors { get; } = new();

        /// <summary>
        /// The conditioning guard's eigenvalue bound on the whole equilibrated covariance (at most once per solve, and only
        /// when a block's own bound in <see cref="double"/> leaves it undecided; about <c>n³/3</c> loop bodies).
        /// </summary>
        public StageWork ConditioningPreparation { get; } = new();

        /// <summary>
        /// The conditioning guard's bound in <see cref="double"/> on a queried block, with its exact refusal witness (at
        /// most once per queried block; about <c>13·m³/6</c> loop bodies on a block of order <c>m</c>).
        /// </summary>
        public StageWork BlockMarginBound { get; } = new();

        /// <summary>
        /// The conditioning guard's double-double eigenvalue bound of a queried block itself (at most once per queried
        /// block; about <c>m³/3</c> loop bodies on a block of order <c>m</c>).
        /// </summary>
        public StageWork BlockEigenvalueBound { get; } = new();

        /// <summary>Blocks queried by the conditioning guard (each new working set's free block and the deciding block).</summary>
        public int ConditioningQueries { get; set; }

        /// <summary>
        /// Makes the conditioning guard decide every block by its exact computation, without its certified bounds (the
        /// reference against which those bounds are checked).
        /// </summary>
        public bool ExactConditioningOnly { get; init; }

        /// <summary>Blocks the conditioning guard decided by its exact computation, no certified bound having settled them.</summary>
        public int ExactConditioningTests { get; set; }

        /// <summary>Uses the original fraction-free solve for differential verification of the exact route.</summary>
        public bool FractionFreeOnly { get; init; }

        /// <summary>Exact systems solved with a reused fraction-free factor.</summary>
        public int CachedExactSolves { get; set; }

        /// <summary>
        /// Replaces the rounded route's budget of <see cref="IterationBudget"/> working-set changes (test use only: the
        /// hand-off when that budget is spent, exercised on a problem that needs fewer changes).
        /// </summary>
        public int? RoundedBudgetOverride { get; init; }

        /// <summary>
        /// Replaces the exact route's budget of <see cref="ExactRouteBudget"/> working-set changes (test use only: the
        /// refusal when that budget is spent, which no known input reaches).
        /// </summary>
        public int? ExactRouteBudgetOverride { get; init; }

        /// <summary>
        /// Receives the exact route's rounded result, or its equal-weight frontier point, before the exit certificate checks
        /// it, and returns the vector the certificate checks instead (test use only: the certificate's refusal of the exact
        /// route, and an exception raised inside the solve, which no known input reaches).
        /// </summary>
        public Func<T[], T[]>? ExactRouteResultHook { get; init; }

        /// <summary>
        /// Every reduced Hessian the rounded route factored, in order: the working set, its reference, the previous factor
        /// handed to it and the leading Cholesky columns copied from that factor (see <see cref="ReducedFactorRecord"/>).
        /// </summary>
        public List<ReducedFactorRecord> ReducedFactors { get; } = [];

        /// <summary>
        /// The rounded route's feasible start carried exactly, before each weight is rounded once into
        /// <typeparamref name="T"/> (<see langword="null"/> when the solve did not build one).
        /// </summary>
        public StartAllocation? FeasibleStart { get; set; }

        /// <summary>The exact route's starting point (<see langword="null"/> when the solve did not take the exact route).</summary>
        public StartAllocation? ExactStart { get; set; }

        /// <summary>
        /// The last block the conditioning guard of a certified result measured: the free variables strictly inside the
        /// box plus every variable on a bound whose exact multiplier is materially negative (<see langword="null"/> before
        /// any).
        /// </summary>
        public int[]? DecidingBlock { get; set; }

        /// <summary>
        /// Runs the conditioning guard of a certified result on the supplied exit state (the iterate's weights, the subspace
        /// minimiser, the working set and the variables the exclusion rule excluded) of the problem, recording its deciding
        /// block on this trace: the guard's own membership rule and margin test, on a problem prepared as a solve prepares
        /// it. Throws the solve's refusal when the block fails the margin.
        /// </summary>
        public void RequireDeterminedBlock(
            T[,] covariance, T[]? means, T riskAversion, T lower, T upper, T[] weights, T[] point, int[] status, bool[] stalled) =>
            PrimalActiveSet.RequireDeterminedBlock(this, covariance, means, riskAversion, lower, upper, weights, point, status, stalled);
    }

    /// <summary>
    /// One reduced Hessian of the rounded route (test diagnostics): its working set's free variables, the reference that
    /// eliminated the budget row, the free variables and reference of the previous factor handed to it
    /// (<see langword="null"/> and −1 for none), and the leading Cholesky columns it copied from that factor.
    /// </summary>
    internal sealed record ReducedFactorRecord(int[] Free, int Reference, int[]? PreviousFree, int PreviousReference, int ReusedColumns);

    /// <summary>
    /// A starting point carried exactly (test diagnostics): weight <c>i</c> is <c>Mantissas[i] · Radix^Exponents[i]</c>,
    /// and <see cref="ReusedIterate"/> tells whether it is the rounded route's iterate (its budget residual moved onto a
    /// free variable) rather than the greedy feasible fill.
    /// </summary>
    internal sealed record StartAllocation(BigInteger[] Mantissas, int[] Exponents, int Radix, bool ReusedIterate);

    /// <summary>
    /// How often one certified stage of a solve ran and how much work it did (test diagnostics): every loop of the stage
    /// counts its body's executions as they happen, at every nesting level, so a stage whose loops ran more often than
    /// before reports more <see cref="Iterations"/>. The cost each stage documents is a bound on those iterations in terms
    /// of the orders it ran on (<see cref="OrderCubes"/>, <see cref="OrderSquares"/>).
    /// </summary>
    internal sealed class StageWork
    {
        private int _scope = -1;

        /// <summary>Runs of the stage.</summary>
        public int Runs { get; private set; }

        /// <summary>
        /// Runs in the same scope as the run before: every run after the first for a stage of the whole solve, a second
        /// run on one queried block for a stage of the conditioning guard's blocks.
        /// </summary>
        public int RepeatedRuns { get; private set; }

        /// <summary>Loop-body executions of every run.</summary>
        public long Iterations { get; private set; }

        /// <summary>The sum over the runs of the cube of the order each ran on.</summary>
        public long OrderCubes { get; private set; }

        /// <summary>The sum over the runs of the square of the order each ran on.</summary>
        public long OrderSquares { get; private set; }

        internal void Record(int scope, int order, long iterations)
        {
            RepeatedRuns += Runs > 0 && scope == _scope ? 1 : 0;
            _scope = scope;
            Runs++;
            Iterations += iterations;
            OrderCubes += (long)order * order * order;
            OrderSquares += (long)order * order;
        }
    }

    /// <summary>The test that decided the positive definiteness of a solve's covariance (test diagnostics).</summary>
    internal enum PositiveDefinitenessRoute
    {
        None,

        /// <summary>The certified <see cref="double"/> Cholesky proved the covariance positive definite.</summary>
        DoubleCertificate,

        /// <summary>The certified double-double factorisation proved it positive definite, or proved a refusal and its order.</summary>
        DoubleDoubleCertificate,

        /// <summary>Exact fraction-free elimination decided it.</summary>
        ExactElimination,
    }

    /// <summary>
    /// Primal-feasible active-set method for <c>min w′Σw</c> subject to <c>1′w = 1</c> and <c>lb ≤ w ≤ ub</c>
    /// (Nocedal and Wright 2006, Algorithm 16.3, on the budget row and the box), and for
    /// <c>min (λ/2)·w′Σw − μ′w</c> on the same set through <see cref="Solve"/>, with the gradient <c>λΣw − μ</c> in
    /// place of <c>Σw</c> below.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The covariance is equilibrated once by <see cref="QpScaling{T}.EquilibrateCovariance"/> (<c>Σ̃ = D⁻¹ΣD⁻¹</c>,
    /// <c>D</c> a diagonal of radix powers, so every entry of <c>Σ̃</c> is exact; the bounds and the means stay in the
    /// original units). A feasible start is built first: every weight at the lower
    /// bound, then the remaining budget distributed up to the upper bounds in index order by the greedy fill
    /// that the max-return LP uses. The working set holds the variables at a bound; at least one variable is
    /// always free, and the free variable of smallest variance is the reference <c>r</c> that eliminates the
    /// budget row. When the reduced Hessian in that basis is not positive definite at the working precision, every
    /// other free variable is tried as the reference in ascending order of <c>Σᵢᵢ</c> (smallest index among ties); a
    /// retry changes the basis of the same subspace minimiser, never a tolerance.
    /// </para>
    /// <para>
    /// <b>Unusable working sets.</b> When no reference variable yields a positive-definite reduced Hessian at the
    /// working precision, the working set's free block fails the margin <c>48·n·u·κ_eq ≥ 1</c> (<c>κ_eq</c> the exact
    /// 1-norm condition number of the equilibrated block, <c>u</c> the unit roundoff), or the exact-residual refinement
    /// on the rounded factor does not converge (the iterate still moves at the refinement limit, or the refinement that
    /// decides a blocking bound does not reach half a unit of the largest weight while each correction at least halves:
    /// the rounded factor then resolves a near-null direction of the block too coarsely, as it can near the margin at
    /// <see cref="decimal"/>, whose fixed absolute grid coarsens the relative precision of the factor), the problem is
    /// handed to the exact route described under <b>Exact route</b>, from that working set and iterate. The exact route
    /// runs only after the rounded route failed, so it is off the path of every problem the rounded route certifies.
    /// </para>
    /// <para>
    /// <b>Step.</b> Each iteration forms the gradient <c>g = Σw</c> and the budget residual <c>ρ = 1′w − 1</c>
    /// exactly (integer arithmetic on the radix fractions of the inputs and the iterate) and computes the step
    /// <c>p</c> of the equality-constrained subproblem on the free variables in the budget null-space basis
    /// <c>z_j = e_j − (D_r/D_j)·e_r</c> of the scaled problem: <c>(Z′Σ̃Z)·y = −Z′(g − ρ·Σe_r)</c>, the reduced
    /// Hessian formed exactly, equilibrated by radix powers of its own diagonal and factored by
    /// <see cref="CholeskyDecomposition{T}"/>; <c>p_j = y_j</c> and <c>p_r = −ρ − Σy_j</c>. The budget-orthogonal
    /// part of the step is therefore solved for directly and never formed as a difference of two solves. Because
    /// the right-hand side is the exact residual at the current iterate, repeating the step on an unchanged
    /// working set refines the subspace minimiser to the working precision; <c>p = 0</c> means that the rounded
    /// step no longer changes any weight (a working set whose iterate still moves at the refinement limit is solved
    /// exactly instead).
    /// </para>
    /// <para>
    /// <b>Step length.</b> <c>α = min(1, min over the free variables moving toward a bound of distance/|p_j|)</c>,
    /// compared exactly; when <c>α &lt; 1</c> only the blocking variable (the smallest index among ties) is set to
    /// its bound and added to the working set. When the first step on a working set would block, the blocking bound
    /// is decided on the refined subspace minimiser (the remaining refinement steps of the working set taken on the
    /// minimiser itself, which must converge as described under unusable working sets), so that a weight the minimiser
    /// keeps barely inside the box is not put on the wrong side of its bound by the first solve's rounding.
    /// </para>
    /// <para>
    /// <b>Release.</b> When <c>p = 0</c>, the multipliers are evaluated exactly at the subspace minimiser
    /// <c>w + p</c> (the iterate plus its unrounded refinement step), so the rounding of the stored weights does not
    /// decide a release. With an interior free variable the budget multiplier is <c>ν = g_r</c> of the interior
    /// reference, and each variable at a bound has the multiplier <c>λ_i = g_i − ν</c> (lower) or <c>ν − g_i</c>
    /// (upper); it counts as negative below <c>√u</c> times the rounding bound of its own row and of the reference
    /// row (<c>Σ_j |Σ_ij|·ρ(w_j)</c>, <c>ρ</c> half the weight's grid spacing, <c>u</c> the unit roundoff). The
    /// variable with the most negative multiplier relative to its own row scale <c>Σ_j |Σ_ij||w_j| + |ν|</c> is
    /// released, ties broken by the smallest index (Bland's rule). When every variable sits on a bound, the
    /// lower-bound variable of smallest gradient and the upper-bound variable of largest gradient are released
    /// together if moving budget between them lowers the objective, and every other variable joins the working set
    /// at its bound, so the pair alone spans the next step. With no negative multiplier the iterate is the solver's
    /// result.
    /// </para>
    /// <para>
    /// <b>Termination.</b> A release must lower the exact objective <c>x′Σx</c> of the subspace minimiser
    /// <c>x = w + p</c> before the next release is considered; otherwise the iterate and the working set return to
    /// their state before that release, the released variables are excluded until the objective next decreases, and
    /// the next candidate is tried. The objective is measured at <c>x</c> rather than at the rounded iterate because
    /// a release can lower the minimum by less than the rounding of <c>w′Σw</c>, as when the weight it moves is far
    /// below the resolution of a dominant weight. <c>x</c> is determined by the rounded iterate and the working set,
    /// which are finitely many, so the method terminates; <see cref="IterationBudget"/> remains a defensive limit on
    /// the working-set changes, not a convergence mechanism, and a rounded route that spends it hands the problem to
    /// the exact route instead of refusing it.
    /// </para>
    /// <para>
    /// <b>Exact route.</b> The same primal-feasible active-set method carried out on exact rational iterates, taken when
    /// a working set is unusable or its refinement does not converge (above), when the exit certificate rejects the
    /// rounded result and no release remains (below), and when the rounded route spends its budget. It starts from the
    /// rounded route's working set and iterate (the iterate's budget residual moved exactly onto a free variable that
    /// stays in the box, otherwise the exact feasible start) with a fresh budget of <see cref="ExactRouteBudget"/>
    /// (<c>12n + 12</c>) working-set changes. Each working set's subspace minimiser is computed exactly (the budget row eliminated, the
    /// reduced system on the equilibrated block solved by fraction-free elimination); the step-length rule compares and
    /// moves exactly, and a step of length zero (a free variable already on its bound blocks) adds that bound to the
    /// working set as a working-set change. Multipliers are judged with every variable on a bound treated as at that
    /// bound: with a free variable strictly inside the box the most negative exact multiplier relative to its row scale
    /// is released (smallest index among ties); at a vertex the lower-bound variable of smallest gradient and the
    /// upper-bound variable of largest gradient are released together when the first gradient is below the second. A
    /// release on a strictly negative exact multiplier strictly lowers the exact objective, so the exact route
    /// terminates on degenerate problems (identical assets, repeated blocks, tied means, coinciding bounds); the
    /// termination rule above and the budget remain as defensive limits, the budget's exhaustion being a refusal. The
    /// exact optimum is rounded once into <typeparamref name="T"/> and checked like any other result.
    /// </para>
    /// <para>
    /// <b>Exit.</b> The result is checked by the independent <see cref="QpKktCertificate{T}"/> at the tolerance
    /// scale <see cref="CertificateToleranceScale"/>; a result is never returned unchecked. The release rule and the
    /// certificate agree: when the certificate finds a variable at a bound with a multiplier of the wrong sign, that
    /// variable (or the one that set the other end of the multiplier interval) is released and the iteration
    /// continues. When the certificate rejects the result and no such release remains, the rounded trajectory is not
    /// that of the exact method (at weights far above 1 the rounded iterate cannot carry the budget to the unit, and the
    /// objective at its refined minimiser cannot resolve a release), so the exact route takes the problem over. A result
    /// of the exact route that the certificate still rejects raises <see cref="InvalidOperationException"/> naming the
    /// failed component (budget, bound, stationarity or multiplier sign). A certified result is then still
    /// refused (naming stationarity) when <c>48·n·u·κ_eq(B) ≥ 1</c> for the block <c>B</c> that decides it: the free
    /// variables strictly inside the box plus every variable on a bound (excluded by the termination rule, or free but
    /// sitting exactly on its bound) whose exact multiplier is materially negative (beyond <c>64·u</c> of its row
    /// scale), <c>κ_eq</c> the exact 1-norm condition number of the equilibrated block
    /// computed by fraction-free elimination. On such a block a move along the near-null direction is invisible to any
    /// residual certificate, so the alternative to the refusal is an unverifiable vector; a problem whose exact free
    /// block meets the margin is not refused by this test whenever the deciding block is that free block. Range safety
    /// comes from the equilibration, the exact residuals and the single rounding of each exact quantity into
    /// <typeparamref name="T"/>; there is no per-site rescue path.
    /// </para>
    /// </remarks>
    private static class PrimalActiveSet
    {
        // Refinement steps on one working set before its subspace minimiser is accepted as converged.
        private const int MaxRefinements = 6;

        // The corpus-membership margin c of 48·n·u·κ_eq < 1, applied by the conditioning guard to the deciding block.
        private const int ConditioningMargin = 48;

        // A multiplier counts as materially negative for the conditioning guard beyond 64·u of its own row scale.
        private const int MaterialMultiplierUnits = 64;
        private enum StepOutcome
        {
            Stationary,
            Moved,
            Blocked,
        }

        internal static T[] SolveMinVariance(T[,] covariance, T lower, T upper, int forcedReference = -1, RouteTrace? trace = null)
            => Solve(covariance, means: null, T.One, lower, upper, forcedReference, trace);

        /// <summary>
        /// Minimises <c>(λ/2)·w′Σw − μ′w</c> (with <paramref name="means"/> = <see langword="null"/>, <c>w′Σw</c>: the
        /// gradient is then <c>Σw</c>, as λ = 1 and μ = 0 give it) subject to <c>1′w = 1</c> and the box: the gradient of
        /// every step is <c>g = λΣw − μ</c>, the reduced Hessian <c>λ·Z′Σ̃Z</c>, and the objective compared by the
        /// termination rule <c>x′(λΣx) − 2μ′x</c>, all formed exactly. The means are centred exactly on the midpoint of
        /// their range first, which changes no comparison (a common shift adds only a constant under the budget row) and
        /// keeps the row scales of the multipliers free of an irrelevant offset.
        /// </summary>
        internal static T[] Solve(
            T[,] covariance, T[]? means, T riskAversion, T lower, T upper, int forcedReference, RouteTrace? trace = null)
        {
            var n = covariance.GetLength(0);
            // The core reads only the exponents and the scaled covariance, so only the covariance's scaling has to be
            // exact: a bound or mean whose scaling would round must not hold an exponent at 0.
            var scaling = QpScaling<T>.EquilibrateCovariance(covariance);
            var sigma = new Exact[n, n];
            var scaled = new Exact[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    sigma[i, j] = Exact.From(covariance[i, j]);
                    scaled[i, j] = Exact.From(scaling.Covariance[i, j]);
                }
            }

            // Σ̃ = D⁻¹ΣD⁻¹ is congruent to Σ, so it is positive definite exactly when Σ is, and every leading principal
            // minor has the sign of Σ's; certified tests decide it in O(n³) operations, and the exact elimination decides
            // whatever they cannot.
            RequirePositiveDefinite(scaled, n, trace);
            var problem = new Problem(covariance, sigma, scaled, scaling.Exponents, lower, upper)
            {
                Means = means,
                RiskAversion = riskAversion,
                Lambda = Exact.From(riskAversion),
                Linear = CentredLinear(means),
                Trace = trace,
            };
            if (trace is { ForceExactRoute: true })
            {
                return ExactRoute(problem, null, null, ExactRouteReason.Forced, trace);
            }

            var weights = FeasibleStart(n, lower, upper, out var status, trace);
            var exact = new Exact[n];
            for (var i = 0; i < n; i++)
            {
                exact[i] = Exact.From(weights[i]);
            }

            var budget = trace?.RoundedBudgetOverride ?? IterationBudget(n);
            var changes = 0;
            var refinements = 0;
            var stalled = new bool[n];
            Snapshot? beforeRelease = null;
            ReducedFactor? factor = null;

            // One working-set change of the rounded route; false once its budget is spent, when the exact route takes
            // the problem over instead of the problem being refused.
            bool Change()
            {
                changes++;
                if (trace is not null)
                {
                    trace.RoundedChanges = changes;
                    trace.RoundedBudgetReached |= changes > budget;
                }

                return changes <= budget;
            }

            // The exact route continues from the rounded route's current working set and iterate.
            T[] HandOff(ExactRouteReason reason) => ExactRoute(problem, weights, status, reason, trace);

            while (true)
            {
                var free = FreeIndices(status);
                if (free.Length < 2)
                {
                    factor = null;
                }
                else if (factor is null || !factor.Matches(free))
                {
                    // A working set is unusable when no reference variable factors its reduced Hessian at the working
                    // precision, or when its free block fails the margin 48·n·u·κ_eq ≥ 1: the exact route takes over.
                    factor = ReducedFactor.TryCreate(problem, free, forcedReference, factor);
                    if (factor is null || ExceedsConditioningMargin(problem, free, n))
                    {
                        return HandOff(ExactRouteReason.UnusableWorkingSet);
                    }
                }

                // A direct step forms its own gradient at the budget-feasible base point.
                var gradient = refinements == 0 && factor is not null ? [] : Gradient(problem, exact);
                var residual = Exact.Sum(exact) - Exact.One;
                Exact[] refinedPoint;
                try
                {
                    if (refinements < MaxRefinements)
                    {
                        var outcome = Step(problem, free, factor, gradient, residual, weights, exact, status, refinements == 0);
                        if (outcome != StepOutcome.Stationary)
                        {
                            refinements = outcome == StepOutcome.Moved ? refinements + 1 : 0;
                            if (outcome == StepOutcome.Blocked && !Change())
                            {
                                return HandOff(ExactRouteReason.RoundedBudget);
                            }

                            // The refinement has not converged when the iterate still moves at the refinement limit:
                            // the rounded reduced Hessian resolves a near-null direction of the block too coarsely for
                            // the exact-residual refinement to contract to the working precision in the steps allowed
                            // (each step can shrink the error along that direction by as little as a half), so the
                            // stopped iterate can lie several sensitivity units from the minimiser along a direction no
                            // residual certificate sees. The exact route takes over.
                            if (outcome == StepOutcome.Moved && refinements >= MaxRefinements)
                            {
                                return HandOff(ExactRouteReason.RefinementNotConverged);
                            }

                            continue;
                        }
                    }

                    if (gradient.Length == 0)
                    {
                        gradient = Gradient(problem, exact);
                    }

                    refinedPoint = RefinedPoint(problem, free, factor, gradient, residual, exact);
                }
                catch (UnusableWorkingSetException)
                {
                    // The rounded solve of this working set failed at the working precision.
                    return HandOff(ExactRouteReason.RefinementNotConverged);
                }

                var refined = Gradient(problem, refinedPoint);
                var objective = Objective(problem, refinedPoint, refined);

                // Anti-cycling: a release must lower the exact objective x′Σx of the subspace minimiser x = w + p (the
                // iterate plus its unrounded refinement step) by the time the next release is considered. Otherwise
                // the iterate and working set return to the state before that release, the released variables are
                // excluded, and the next candidate is tried; any decrease readmits every variable. The objective is
                // measured at x, not at the rounded iterate, because a release can lower the minimum by less than the
                // rounding of w′Σw (a weight change below the resolution of a dominant weight).
                if (beforeRelease is not null)
                {
                    if (objective < beforeRelease.Objective)
                    {
                        Array.Clear(stalled);
                    }
                    else
                    {
                        Restore(beforeRelease.Weights, beforeRelease.Status, weights, exact, status);
                        foreach (var i in beforeRelease.Released)
                        {
                            stalled[i] = true;
                        }

                        beforeRelease = null;
                        refinements = MaxRefinements;
                        continue;
                    }
                }

                var (release, pair) = ReleaseCandidates(problem, free, refined, weights, exact, status, stalled);
                if (release.Length == 0)
                {
                    // The release rule and the exit certificate must agree: a variable at a bound that the certificate
                    // finds with a multiplier of the wrong sign is released (the anti-cycling rule then keeps the release
                    // only if it lowers the objective).
                    var certificate = ExitCertificate(problem, weights);
                    if (!certificate.Accepted)
                    {
                        var forced = CertificateRelease(certificate, status, stalled);
                        if (forced < 0)
                        {
                            // No release is left on the rounded trajectory, although the certificate rejects its
                            // result: that trajectory is not the exact method's (at weights far above 1 the rounded
                            // iterate cannot carry the budget, and the refined objective cannot resolve a release), so
                            // the exact route takes over instead of the problem being refused.
                            return HandOff(ExactRouteReason.CertificateRejected);
                        }

                        release = [forced];
                        pair = false;
                    }
                    else if (BoundPairViolated(problem, refined, weights, status))
                    {
                        // Two variables at opposite bounds violate max g(upper) ≤ ν ≤ min g(lower) by more than their own
                        // rows' rounding, a violation that the multiplier tests measured against ν missed because ν carries
                        // the far larger rounding of the reference row: the rounded trajectory stopped on a wrong active
                        // set the certificate cannot see, so the exact route takes over.
                        return HandOff(ExactRouteReason.CertificateRejected);
                    }
                    else
                    {
                        RequireDeterminedBlock(problem, free, refined, refinedPoint, weights, status, stalled);
                        return weights;
                    }
                }

                beforeRelease = new Snapshot((T[])weights.Clone(), (int[])status.Clone(), objective, release);
                if (pair)
                {
                    // Every variable sits at a bound: the free ones join the working set at their bounds, so the
                    // released pair alone spans the step. A free variable left at its bound would make the subspace
                    // minimiser leave the box through it and block the step at α = 0.
                    foreach (var i in free)
                    {
                        if (Array.IndexOf(release, i) < 0)
                        {
                            status[i] = weights[i] == lower ? -1 : 1;
                            if (!Change())
                            {
                                return HandOff(ExactRouteReason.RoundedBudget);
                            }
                        }
                    }
                }

                foreach (var i in release)
                {
                    if (status[i] == 0)
                    {
                        continue;
                    }

                    status[i] = 0;
                    if (!Change())
                    {
                        return HandOff(ExactRouteReason.RoundedBudget);
                    }
                }

                refinements = 0;
            }
        }

        private sealed record Snapshot(T[] Weights, int[] Status, Fraction Objective, int[] Released);

        /// <summary>A working set whose reduced Hessian cannot be used at the working precision (raised by its solve).</summary>
        private sealed class UnusableWorkingSetException : Exception
        {
            public UnusableWorkingSetException(Exception inner)
                : base("The reduced Hessian of the working set cannot be solved at the working precision.", inner)
            {
            }

            public UnusableWorkingSetException()
                : base("The refinement of the working set's subspace minimiser does not converge at the working precision.")
            {
            }
        }

        private static void Restore(T[] savedWeights, int[] savedStatus, T[] weights, Exact[] exact, int[] status)
        {
            Array.Copy(savedWeights, weights, weights.Length);
            Array.Copy(savedStatus, status, status.Length);
            for (var i = 0; i < weights.Length; i++)
            {
                exact[i] = Exact.From(weights[i]);
            }
        }

        private static Exact Dot(Exact[] left, Exact[] right)
        {
            var sum = Exact.Zero;
            for (var i = 0; i < left.Length; i++)
            {
                sum += left[i] * right[i];
            }

            return sum;
        }

        private sealed record Problem(T[,] Covariance, Exact[,] Sigma, Exact[,] Scaled, int[] Exponents, T Lower, T Upper)
        {
            public Exact LowerX { get; } = Exact.From(Lower);

            public Exact UpperX { get; } = Exact.From(Upper);

            /// <summary>The caller's means (the certificate's linear term), or <see langword="null"/> for minimum variance.</summary>
            public T[]? Means { get; init; }

            /// <summary>The caller's risk aversion λ (1 for minimum variance).</summary>
            public T RiskAversion { get; init; } = T.One;

            /// <summary>λ exactly.</summary>
            public Exact Lambda { get; init; } = Exact.One;

            /// <summary>The means centred exactly on the midpoint of their range, or <see langword="null"/> for minimum variance.</summary>
            public Exact[]? Linear { get; init; }

            /// <summary>The problem's name in a refusal message.</summary>
            public string Kind => Means is null ? "min-variance" : "mean-variance";

            /// <summary>The solve's diagnostics, or <see langword="null"/>.</summary>
            public RouteTrace? Trace { get; init; }

            /// <summary>The conditioning guard's certified bounds, built on first use (<see cref="ExceedsConditioningMargin"/>).</summary>
            public ConditioningBounds Conditioning { get; } = new();

            public ExactFactorCache? ExactFactor { get; set; }

            /// <summary><see cref="Sigma"/> prepared for exact products (<see cref="Gradient(Problem, Exact[])"/>), built on first use.</summary>
            public Exact.ProductMatrix SigmaProduct => _sigmaProduct ??= new Exact.ProductMatrix(Sigma);

            private Exact.ProductMatrix? _sigmaProduct;
        }

        /// <summary>
        /// The double-double rounding of the equilibrated covariance and a proved lower bound on its smallest eigenvalue,
        /// built once per solve by <see cref="PrepareConditioningBounds"/> on the first block the double bound of
        /// <see cref="CertifiedWithinMargin"/> does not settle.
        /// </summary>
        private sealed class ConditioningBounds
        {
            public bool Prepared { get; set; }

            /// <summary><c>Ĥ</c> of the rounding <c>Ĥ + Ĝ</c> of every entry (<see cref="TryDoubleDouble"/>), or <see langword="null"/> when an entry has none.</summary>
            public double[,]? High { get; set; }

            /// <summary><c>Ĝ</c> of that rounding.</summary>
            public double[,]? Low { get; set; }

            /// <summary>A proved lower bound on the smallest eigenvalue of the whole equilibrated covariance, or 0 when none was proved.</summary>
            public double Smallest { get; set; }
        }

        // μ − (min μ + max μ)/2, exactly: the midpoint of two radix fractions is one (halving is exact in either radix).
        private static Exact[]? CentredLinear(T[]? means)
        {
            if (means is null)
            {
                return null;
            }

            var exact = means.Select(Exact.From).ToArray();
            var low = exact[0];
            var high = exact[0];
            foreach (var value in exact)
            {
                low = value < low ? value : low;
                high = value > high ? value : high;
            }

            var midpoint = (low + high).Half();
            return exact.Select(value => value - midpoint).ToArray();
        }

        // The gradient g = λΣx − μ of the objective (λ/2)·x′Σx − μ′x (minimum variance: Σx), formed exactly.
        private static Exact[] Gradient(Problem problem, Exact[] x)
        {
            var product = Gradient(problem.SigmaProduct, x);
            if (problem.Linear is null)
            {
                return product;
            }

            for (var i = 0; i < product.Length; i++)
            {
                product[i] = (problem.Lambda * product[i]) - problem.Linear[i];
            }

            return product;
        }

        // Twice the objective, x′(λΣx) − 2μ′x = x′g − μ′x for the gradient g at x (minimum variance: x′Σx), exactly.
        private static Fraction Objective(Problem problem, Exact[] x, Exact[] gradient)
        {
            var value = Dot(x, gradient);
            if (problem.Linear is not null)
            {
                value -= Dot(problem.Linear, x);
            }

            return new Fraction(value, BigInteger.One);
        }

        // λ·Σ_ij exactly (minimum variance: Σ_ij).
        private static Exact Curvature(Problem problem, int i, int j) =>
            problem.Linear is null ? problem.Sigma[i, j] : problem.Lambda * problem.Sigma[i, j];

        // |μ_i| of the centred means (0 for minimum variance): the linear term's part of row i's scale.
        private static Exact LinearMagnitude(Problem problem, int i) =>
            problem.Linear is null ? Exact.Zero : problem.Linear[i].Abs();

        // Every weight at the lower bound, the remaining budget distributed up to the upper bounds in index order
        // (the max-return LP's greedy rule); the variables at a bound form the working set, keeping one free. The
        // allocation is carried exactly and each weight rounded once into T: every weight lies in [lb, ub], so the
        // start is representable whenever the bounds are, although the budget 1 − n·lb and the capacity ub − lb need not be.
        private static T[] FeasibleStart(int n, T lower, T upper, out int[] status, RouteTrace? trace)
        {
            var weights = new T[n];
            var allocation = trace is null ? null : new Exact[n];
            var lowerX = Exact.From(lower);
            var capacity = Exact.From(upper) - lowerX;
            var remaining = Exact.One - (Exact.FromInt(n) * lowerX);
            for (var i = 0; i < n; i++)
            {
                var weight = lowerX;
                if (remaining.Sign > 0)
                {
                    var add = remaining < capacity ? remaining : capacity;
                    weight += add;
                    remaining -= add;
                }

                allocation?[i] = weight;
                weights[i] = weight.ToT();
            }

            trace?.FeasibleStart = Allocation(allocation!, reusedIterate: false);

            status = new int[n];
            var lastFilled = n - 1;
            for (var i = 0; i < n; i++)
            {
                weights[i] = T.Clamp(weights[i], lower, upper);
                status[i] = weights[i] == lower ? -1 : weights[i] == upper ? 1 : 0;
                if (weights[i] != lower)
                {
                    lastFilled = i;
                }
            }

            if (Array.IndexOf(status, 0) < 0)
            {
                status[lastFilled] = 0;
            }

            return weights;
        }

        // A starting point recorded exactly on the route trace (test diagnostics).
        private static StartAllocation Allocation(Exact[] point, bool reusedIterate) => new(
            [.. point.Select(x => x.Mantissa)], [.. point.Select(x => x.Exponent)], typeof(T) == typeof(decimal) ? 10 : 2, reusedIterate);

        private static int[] FreeIndices(int[] status)
        {
            var free = new List<int>(status.Length);
            for (var i = 0; i < status.Length; i++)
            {
                if (status[i] == 0)
                {
                    free.Add(i);
                }
            }

            return [.. free];
        }

        // The free variable of smallest variance (smallest index among ties) eliminates the budget row.
        private static int Reference(T[,] covariance, int[] free)
        {
            var r = free[0];
            foreach (var i in free)
            {
                if (covariance[i, i] < covariance[r, r])
                {
                    r = i;
                }
            }

            return r;
        }

        // Σx, every row formed exactly in one integer accumulator (Exact.ProductMatrix): the same Exact, mantissa and
        // exponent, as adding the products σ_ij·x_j to zero in index order.
        private static Exact[] Gradient(Exact.ProductMatrix sigma, Exact[] weights) => sigma.Times(weights);

        private static StepOutcome Step(
            Problem problem, int[] free, ReducedFactor? factor, Exact[] gradient, Exact residual, T[] weights, Exact[] exact,
            int[] status, bool direct)
        {
            var n = weights.Length;
            var step = new Exact[n];
            if (factor is null)
            {
                // A single free variable: no null space, only the budget residual can be removed, inside the box.
                var single = free[0];
                var corrected = T.Clamp((exact[single] - residual).ToT(), problem.Lower, problem.Upper);
                if (corrected == weights[single])
                {
                    return StepOutcome.Stationary;
                }

                weights[single] = corrected;
                exact[single] = Exact.From(corrected);
                return StepOutcome.Moved;
            }

            // The first step on a working set solves for the subspace minimiser from the budget-feasible base point
            // (fixed variables at their bounds, the remaining budget on the reference) so that its rounding is
            // relative to the minimiser, not to the distance travelled; later steps refine from the iterate.
            var r = factor.Reference;
            var others = factor.Others;
            var basePoint = exact;
            if (direct)
            {
                basePoint = (Exact[])exact.Clone();
                var remaining = Exact.One;
                for (var i = 0; i < n; i++)
                {
                    if (status[i] != 0)
                    {
                        remaining -= exact[i];
                    }
                    else
                    {
                        basePoint[i] = Exact.Zero;
                    }
                }

                basePoint[r] = remaining;
                gradient = Gradient(problem, basePoint);
                residual = Exact.Zero;
            }

            var y = NullSpaceStep(problem, factor, gradient, residual);
            var sum = Exact.Zero;
            for (var a = 0; a < others.Length; a++)
            {
                step[others[a]] = basePoint[others[a]] + y[a] - exact[others[a]];
                sum += y[a];
            }

            step[r] = basePoint[r] - residual - sum - exact[r];

            // A blocking bound is decided on the refined subspace minimiser: the first solve of a working set is accurate
            // only to about κ·u of the minimiser, which can put a weight that the minimiser keeps barely inside the box
            // (a degenerate-at-bound variable) on the wrong side of its bound. The remaining refinement steps of the
            // working set are taken on the minimiser itself, with exact residuals, before the step length is decided.
            if (direct && Blocks(problem, free, step, exact))
            {
                RefineMinimiser(problem, factor, step, exact, MaxMagnitude(y));
            }

            return ApplyStep(problem, free, step, weights, exact, status);
        }

        /// <summary>
        /// The step-length rule on the step <paramref name="step"/>: <c>α = min(1, min over the free variables moving
        /// toward a bound of distance/|p_j|)</c>, compared exactly; a full step moves the free weights to the subspace
        /// minimiser, a blocked step sets the blocking variable (the smallest index among ties) on its bound and adds it
        /// to the working set.
        /// </summary>
        private static StepOutcome ApplyStep(Problem problem, int[] free, Exact[] step, T[] weights, Exact[] exact, int[] status)
        {
            var blocking = -1;
            Exact bestDistance = default, bestMagnitude = default;
            foreach (var j in free)
            {
                if (step[j].Sign == 0)
                {
                    continue;
                }

                var distance = step[j].Sign < 0 ? exact[j] - problem.LowerX : problem.UpperX - exact[j];
                var magnitude = step[j].Abs();
                if (distance >= magnitude)
                {
                    continue;
                }

                if (blocking < 0 || (distance * bestMagnitude) < (bestDistance * magnitude))
                {
                    blocking = j;
                    bestDistance = distance;
                    bestMagnitude = magnitude;
                }
            }

            if (blocking < 0)
            {
                var changed = false;
                foreach (var j in free)
                {
                    if (step[j].Sign == 0)
                    {
                        continue;
                    }

                    var moved = T.Clamp((exact[j] + step[j]).ToT(), problem.Lower, problem.Upper);
                    if (moved != weights[j])
                    {
                        changed = true;
                        weights[j] = moved;
                        exact[j] = Exact.From(moved);
                    }
                }

                return changed ? StepOutcome.Moved : StepOutcome.Stationary;
            }

            // α = distance/|p_blocking| < 1: the blocking variable lands on its bound and joins the working set.
            var progress = bestDistance.Sign > 0;
            foreach (var j in free)
            {
                if (j == blocking)
                {
                    var atLower = step[j].Sign < 0;
                    weights[j] = atLower ? problem.Lower : problem.Upper;
                    exact[j] = atLower ? problem.LowerX : problem.UpperX;
                    status[j] = atLower ? -1 : 1;
                }
                else if (progress && step[j].Sign != 0)
                {
                    var moved = T.Clamp(
                        (exact[j] + Exact.Quotient(bestDistance * step[j], bestMagnitude)).ToT(), problem.Lower, problem.Upper);
                    weights[j] = moved;
                    exact[j] = Exact.From(moved);
                }
            }

            return StepOutcome.Blocked;
        }

        // Whether some free variable would reach a bound before the full step p.
        private static bool Blocks(Problem problem, int[] free, Exact[] step, Exact[] exact)
        {
            foreach (var j in free)
            {
                if (step[j].Sign != 0 && (step[j].Sign < 0 ? exact[j] - problem.LowerX : problem.UpperX - exact[j]) < step[j].Abs())
                {
                    return true;
                }
            }

            return false;
        }

        // Refines the subspace minimiser x = w + p of the current working set (p is rewritten): up to MaxRefinements − 1
        // further null-space solves on the exact residual at x, each added to x without rounding. Refinement contracts
        // only when the reduced Hessian resolves the minimiser at the working precision. The refined minimiser decides a
        // blocking bound only once it has converged, that is, once a correction is no larger than half a unit of the
        // largest weight; a correction that does not shrink to at most half the previous one above that resolution, or a
        // refinement still correcting by more than it after the steps allowed, leaves the minimiser undetermined along a
        // near-null direction of the block (an unconverged estimate can put the blocking bound on the wrong variable or on
        // a variable that the minimiser keeps inside the box), and the working set is unusable for the rounded solve.
        private static void RefineMinimiser(Problem problem, ReducedFactor factor, Exact[] step, Exact[] exact, Exact first)
        {
            var n = exact.Length;
            var x = new Exact[n];
            for (var i = 0; i < n; i++)
            {
                x[i] = exact[i] + step[i];
            }

            var r = factor.Reference;
            var others = factor.Others;
            // Half a unit of the largest weight, capped at the magnitude of the box (the estimate may lie far outside
            // it, beyond the range of the working type, when the step blocks long before the minimiser).
            var scale = MaxMagnitude(x);
            var box = problem.LowerX.Abs() > problem.UpperX.Abs() ? problem.LowerX.Abs() : problem.UpperX.Abs();
            var resolution = Exact.HalfUnit((scale > box ? box : scale).ToT());
            var previous = first;
            var last = first;
            for (var k = 1; k < MaxRefinements; k++)
            {
                var residual = Exact.Sum(x) - Exact.One;
                var y = NullSpaceStep(problem, factor, Gradient(problem, x), residual);
                var size = MaxMagnitude(y);
                last = size;
                if (size.Sign == 0 && residual.Sign == 0)
                {
                    break;
                }

                if (size + size > previous)
                {
                    if (size <= resolution)
                    {
                        break;
                    }

                    throw new UnusableWorkingSetException();
                }

                var sum = Exact.Zero;
                for (var a = 0; a < others.Length; a++)
                {
                    x[others[a]] += y[a];
                    sum += y[a];
                }

                x[r] -= residual + sum;
                previous = size;
            }

            if (last > resolution)
            {
                throw new UnusableWorkingSetException();
            }

            for (var i = 0; i < n; i++)
            {
                step[i] = x[i] - exact[i];
            }
        }

        private static Exact MaxMagnitude(Exact[] values)
        {
            var best = Exact.Zero;
            foreach (var value in values)
            {
                if (value.Abs() > best)
                {
                    best = value.Abs();
                }
            }

            return best;
        }

        /// <summary>
        /// Solves <c>(Z′Σ̃Z)·ỹ = −Z′(g̃ − ρ·Σ̃e_r)</c> on the scaled problem and returns <c>y</c> in the
        /// original variables. Entry <c>(j, l)</c> of <c>Z′Σ̃Z</c> is
        /// <c>Σ̃_jl − c_l·Σ̃_jr − c_j·Σ̃_rl + c_j·c_l·Σ̃_rr</c> with the exact radix powers <c>c_j = D_r/D_j</c>.
        /// </summary>
        private static Exact[] NullSpaceStep(Problem problem, ReducedFactor factor, Exact[] gradient, Exact residual)
        {
            var r = factor.Reference;
            var others = factor.Others;
            var shift = factor.Shift;
            var m = others.Length;
            var e = problem.Exponents;

            // Right-hand side −Z′(g − ρ·Σe_r) in scaled units (g̃_j = g_j/D_j), then normalised by one radix power.
            var rhs = new Exact[m];
            var exponent = int.MinValue;
            var referenceRow = gradient[r] - (residual * Curvature(problem, r, r));
            for (var a = 0; a < m; a++)
            {
                var j = others[a];
                var row = gradient[j] - (residual * Curvature(problem, j, r));
                rhs[a] = (referenceRow - row).Shift(-e[j] - shift[a]);
                if (rhs[a].Sign != 0)
                {
                    exponent = Math.Max(exponent, rhs[a].MagnitudeExponent());
                }
            }

            var y = new Exact[m];
            if (exponent == int.MinValue)
            {
                return y;
            }

            var vector = new T[m];
            for (var a = 0; a < m; a++)
            {
                vector[a] = rhs[a].Shift(-exponent).ToT();
            }

            T[] solution;
            try
            {
                solution = CholeskyDecomposition<T>.Solve(factor.Lower, vector);
            }
            catch (InvalidOperationException ex)
            {
                throw new UnusableWorkingSetException(ex);
            }

            for (var a = 0; a < m; a++)
            {
                y[a] = Exact.From(solution[a]).Shift(exponent - e[others[a]] - shift[a]);
            }

            return y;
        }

        /// <summary>
        /// The Cholesky factor of the equilibrated reduced Hessian <c>Z′Σ̃Z</c> of one working set, kept while the
        /// working set is unchanged. Entry <c>(j, l)</c> is <c>Σ̃_jl − c_l·Σ̃_jr − c_j·Σ̃_rl + c_j·c_l·Σ̃_rr</c>
        /// with the exact radix powers <c>c_j = D_r/D_j</c>, formed exactly; it is then equilibrated by radix powers
        /// <c>f_a</c> of its own diagonal and rounded once into <typeparamref name="T"/>.
        /// </summary>
        private sealed class ReducedFactor
        {
            private ReducedFactor(int[] free, int reference, int[] others, int[] shift, T[,] lower)
            {
                Free = free;
                Reference = reference;
                Others = others;
                Shift = shift;
                Lower = lower;
            }

            public int[] Free { get; }

            public int Reference { get; }

            public int[] Others { get; }

            public int[] Shift { get; }

            public T[,] Lower { get; }

            public bool Matches(int[] free) => Free.AsSpan().SequenceEqual(free);

            /// <summary>
            /// The factor of the working set, or <see langword="null"/> when it is unusable. The reference that
            /// eliminates the budget row is the free variable of smallest <c>Σᵢᵢ</c> (smallest index among ties); when
            /// the reduced Hessian in that basis is not positive definite at the working precision, each other free
            /// variable is tried in the same order. A retry changes the basis of the same subspace minimiser, never a
            /// tolerance. <paramref name="forcedReference"/> (≥ 0, test use) makes that variable, when it is free, the
            /// only reference tried, with no retry.
            /// </summary>
            public static ReducedFactor? TryCreate(Problem problem, int[] free, int forcedReference, ReducedFactor? previous)
            {
                var order = free.OrderBy(i => problem.Covariance[i, i]).ThenBy(i => i).ToArray();
                IEnumerable<int> candidates = forcedReference < 0
                    ? order
                    : [Array.IndexOf(free, forcedReference) >= 0 ? forcedReference : order[0]];
                foreach (var r in candidates)
                {
                    var (others, shift, matrix) = Equilibrated(problem, free, r);
                    if (matrix is null)
                    {
                        continue;
                    }

                    try
                    {
                        var prefix = 0;
                        if (previous is not null && previous.Reference == r)
                        {
                            while (prefix < others.Length && prefix < previous.Others.Length
                                && others[prefix] == previous.Others[prefix] && shift[prefix] == previous.Shift[prefix])
                            {
                                prefix++;
                            }
                        }

                        // The columns copied are those handed to the call that copies them (test diagnostics).
                        var reused = 0;
                        var lower = prefix > 0
                            ? CholeskyDecomposition<T>.DecomposeWithPrefix(matrix, previous!.Lower, reused = prefix)
                            : CholeskyDecomposition<T>.Decompose(matrix);
                        problem.Trace?.ReducedFactors.Add(new(free, r, previous?.Free, previous?.Reference ?? -1, reused));
                        return new ReducedFactor(free, r, others, shift, lower);
                    }
                    catch (InvalidOperationException)
                    {
                        // Not positive definite at the working precision in this basis: try the next reference.
                    }
                }

                return null;
            }

            /// <summary>
            /// The other free variables, the radix-power shifts <c>f_a</c> and the equilibrated reduced Hessian in the
            /// basis of reference <paramref name="r"/>, rounded once into <typeparamref name="T"/> (<see langword="null"/>
            /// when a diagonal entry is not positive, which exact arithmetic on a positive-definite covariance rules out).
            /// </summary>
            private static (int[] Others, int[] Shift, T[,]? Matrix) Equilibrated(Problem problem, int[] free, int r)
            {
                var others = free.Where(i => i != r).ToArray();
                var m = others.Length;
                var hessian = ReducedHessian(problem, others, r);
                var shift = new int[m];
                for (var a = 0; a < m; a++)
                {
                    if (hessian[a, a].Sign <= 0)
                    {
                        return (others, shift, null);
                    }

                    shift[a] = (hessian[a, a].MagnitudeExponent() + 1) >> 1;
                }

                var matrix = new T[m, m];
                for (var a = 0; a < m; a++)
                {
                    for (var b = 0; b <= a; b++)
                    {
                        matrix[a, b] = hessian[a, b].Shift(-(shift[a] + shift[b])).ToT();
                        matrix[b, a] = matrix[a, b];
                    }
                }

                return (others, shift, matrix);
            }
        }

        /// <summary>
        /// The reduced Hessian <c>Z′Σ̃Z</c> of the scaled problem in the budget null-space basis of reference
        /// <paramref name="r"/>, formed exactly: entry <c>(a, b)</c> is <c>Σ̃_jl − c_l·Σ̃_jr − c_j·Σ̃_rl + c_j·c_l·Σ̃_rr</c>
        /// for <c>j = others[a]</c>, <c>l = others[b]</c> and the exact radix powers <c>c_j = D_r/D_j</c>.
        /// </summary>
        private static Exact[,] ReducedHessian(Problem problem, int[] others, int r)
        {
            var m = others.Length;
            var e = problem.Exponents;
            var s = problem.Scaled;
            var hessian = new Exact[m, m];
            for (var a = 0; a < m; a++)
            {
                var j = others[a];
                for (var b = 0; b <= a; b++)
                {
                    var l = others[b];
                    hessian[a, b] = s[j, l] - s[j, r].Shift(e[r] - e[l]) - s[r, l].Shift(e[r] - e[j])
                        + s[r, r].Shift((2 * e[r]) - e[j] - e[l]);
                    if (problem.Linear is not null)
                    {
                        hessian[a, b] = problem.Lambda * hessian[a, b];
                    }

                    hessian[b, a] = hessian[a, b];
                }
            }

            return hessian;
        }

        // One radix grid for all reduced principal blocks sharing a reference. The exact triangle can then be extended
        // when a variable is released, even if its index precedes an existing free variable in canonical order.
        private sealed class ExactFactorCache
        {
            public ExactFactorCache(Problem problem, int reference)
            {
                Reference = reference;
                var all = Enumerable.Range(0, problem.Exponents.Length).Where(i => i != reference).ToArray();
                var hessian = ReducedHessian(problem, all, reference);
                Finest = int.MaxValue;
                foreach (var entry in hessian)
                {
                    if (entry.Sign != 0)
                    {
                        Finest = Math.Min(Finest, entry.Exponent);
                    }
                }

                var matrix = new BigInteger[all.Length, all.Length];
                for (var i = 0; i < all.Length; i++)
                {
                    for (var j = 0; j < all.Length; j++)
                    {
                        var entry = hessian[i, j];
                        matrix[i, j] = entry.Sign == 0 ? BigInteger.Zero : entry.Mantissa * Exact.RadixPower(entry.Exponent - Finest);
                    }
                }

                Solver = new ExactLinearSolve(matrix);
            }

            public int Reference { get; }

            public int Finest { get; }

            public ExactLinearSolve Solver { get; }
        }

        /// <summary>An objective value <c>Numerator / Denominator</c> with a positive integer denominator, compared exactly.</summary>
        private readonly record struct Fraction(Exact Numerator, BigInteger Denominator)
        {
            public static bool operator <(Fraction a, Fraction b) =>
                (a.Numerator * Exact.Of(b.Denominator, 0)) < (b.Numerator * Exact.Of(a.Denominator, 0));

            public static bool operator >(Fraction a, Fraction b) => b < a;
        }

        /// <summary>
        /// The subspace minimiser of one working set computed exactly, for a working set whose rounded reduced Hessian
        /// is unusable (no reference variable factors it at the working precision, or its free block fails the margin
        /// <c>48·n·u·κ_eq ≥ 1</c>). The budget row is eliminated with the free variable of smallest <c>Σᵢᵢ</c> as the
        /// reference, the reduced system <c>(Z′Σ̃Z)·ỹ = −Z′g̃</c> at the budget-feasible base point (fixed variables at
        /// their bounds, the remaining budget on the reference) is brought onto its finest radix grid as an integer
        /// system and solved by fraction-free (Bareiss) elimination, so the minimiser is the exact rational
        /// <c>Numerators[i] / Denominator</c>. It depends on the working set and the bounds only, never on the iterate.
        /// </summary>
        private sealed class ExactMinimiser
        {
            private ExactMinimiser(int[] status, Exact[] numerators, BigInteger denominator, Problem problem)
            {
                denominator = Reduce(numerators, denominator);
                Status = status;
                Numerators = numerators;
                Denominator = denominator;
                var n = numerators.Length;

                // The gradient at x = N/d is G/d with G = λΣN − d·μ (minimum variance: ΣN); twice the objective is
                // (N′G − d·μ′N)/d² (minimum variance: N′ΣN/d²).
                var gradient = new Exact[n];
                var det = Exact.Of(denominator, 0);
                var reference = Reference(problem.Covariance, FreeIndices(status));
                for (var i = 0; i < n; i++)
                {
                    if (status[i] == 0 && i != reference)
                    {
                        continue;
                    }

                    for (var j = 0; j < n; j++)
                    {
                        gradient[i] += problem.Sigma[i, j] * numerators[j];
                    }

                    if (problem.Linear is not null)
                    {
                        gradient[i] = (problem.Lambda * gradient[i]) - (det * problem.Linear[i]);
                    }
                }

                // The exact reduced solve proves a common objective gradient on every free variable.
                for (var i = 0; i < n; i++)
                {
                    if (status[i] == 0)
                    {
                        gradient[i] = gradient[reference];
                    }
                }

                GradientNumerators = gradient;
                Point = new Exact[n];
                Gradient = new Exact[n];
                var freeGradient = Exact.Precise(gradient[reference], denominator);
                var objective = Exact.Zero;
                for (var i = 0; i < n; i++)
                {
                    Point[i] = Exact.Precise(numerators[i], denominator);
                    Gradient[i] = status[i] == 0 ? freeGradient : Exact.Precise(gradient[i], denominator);
                    objective += numerators[i] * gradient[i];
                    if (problem.Linear is not null)
                    {
                        objective -= det * problem.Linear[i] * numerators[i];
                    }
                }

                Objective = new Fraction(objective, denominator * denominator);
            }

            public int[] Status { get; }

            /// <summary>The numerators of the exact minimiser over <see cref="Denominator"/>.</summary>
            public Exact[] Numerators { get; }

            /// <summary>The positive common denominator of the exact minimiser.</summary>
            public BigInteger Denominator { get; }

            /// <summary>The minimiser to about four times the working precision (multipliers and row scales are judged on it).</summary>
            public Exact[] Point { get; }

            /// <summary>The gradient <c>Σx</c> at the minimiser to about four times the working precision.</summary>
            public Exact[] Gradient { get; }

            public Exact[] GradientNumerators { get; }

            /// <summary>The exact objective <c>x′Σx</c> at the minimiser.</summary>
            public Fraction Objective { get; }

            /// <summary>
            /// The exact minimiser of the working set <paramref name="status"/> (one free variable takes the remaining
            /// budget), or <see langword="null"/> when no variable is free or its reduced block is singular (no exact
            /// minimiser exists; a positive-definite covariance rules this out).
            /// </summary>
            public static ExactMinimiser? Compute(Problem problem, int[] status)
            {
                var n = status.Length;
                var free = FreeIndices(status);
                if (free.Length == 0)
                {
                    return null;
                }

                var r = free.OrderBy(i => problem.Covariance[i, i]).ThenBy(i => i).First();
                var others = free.Where(i => i != r).ToArray();
                var m = others.Length;
                var e = problem.Exponents;
                var basePoint = new Exact[n];
                var remaining = Exact.One;
                for (var i = 0; i < n; i++)
                {
                    if (status[i] != 0)
                    {
                        basePoint[i] = status[i] < 0 ? problem.LowerX : problem.UpperX;
                        remaining -= basePoint[i];
                    }
                }

                basePoint[r] = remaining;
                var g = PrimalActiveSet.Gradient(problem, basePoint);
                var rhs = new Exact[m];
                var finestRhs = int.MaxValue;
                for (var a = 0; a < m; a++)
                {
                    rhs[a] = (g[r] - g[others[a]]).Shift(-e[others[a]]);
                    if (rhs[a].Sign != 0)
                    {
                        finestRhs = Math.Min(finestRhs, rhs[a].Exponent);
                    }
                }

                var status0 = (int[])status.Clone();
                if (finestRhs == int.MaxValue)
                {
                    // The base point is already the minimiser.
                    return new ExactMinimiser(status0, basePoint, BigInteger.One, problem);
                }

                var finest = int.MaxValue;
                ExactFactorCache? cache = null;
                if (m >= 12 && problem.Trace is not { FractionFreeOnly: true })
                {
                    if (problem.ExactFactor is null || problem.ExactFactor.Reference != r)
                    {
                        problem.ExactFactor = new ExactFactorCache(problem, r);
                    }

                    cache = problem.ExactFactor;
                    finest = cache.Finest;
                }

                BigInteger[,] IntegerMatrix()
                {
                    var hessian = ReducedHessian(problem, others, r);
                    if (finest == int.MaxValue)
                    {
                        foreach (var entry in hessian)
                        {
                            if (entry.Sign != 0)
                            {
                                finest = Math.Min(finest, entry.Exponent);
                            }
                        }
                    }

                    var matrix = new BigInteger[m, m];
                    for (var a = 0; a < m; a++)
                    {
                        for (var b = 0; b < m; b++)
                        {
                            var entry = hessian[a, b];
                            matrix[a, b] = entry.Sign == 0 ? BigInteger.Zero : entry.Mantissa * Exact.RadixPower(entry.Exponent - finest);
                        }
                    }

                    return matrix;
                }

                var vector = new BigInteger[m];
                for (var a = 0; a < m; a++)
                {
                    vector[a] = rhs[a].Sign == 0 ? BigInteger.Zero : rhs[a].Mantissa * Exact.RadixPower(rhs[a].Exponent - finestRhs);
                }

                BigInteger determinant;
                BigInteger[] solution;
                var indices = others.Select(i => i < r ? i : i - 1).ToArray();
                if (cache is not null && cache.Solver.TrySolve(indices, vector, out solution, out determinant))
                {
                    problem.Trace?.CachedExactSolves += 1;
                }
                else
                {
                    var matrix = IntegerMatrix();
                    // Fraction-free elimination of [A | b]: retained for small systems and a cached factor that cannot be used.
                    // Every pivot is a positive leading principal minor, so no row exchange is needed.
                    var work = new BigInteger[m, m + 1];
                    for (var i = 0; i < m; i++)
                    {
                        for (var j = 0; j < m; j++)
                        {
                            work[i, j] = matrix[i, j];
                        }

                        work[i, m] = vector[i];
                    }

                    var previous = BigInteger.One;
                    for (var k = 0; k < m; k++)
                    {
                        if (work[k, k].Sign <= 0)
                        {
                            return null;
                        }

                        for (var i = k + 1; i < m; i++)
                        {
                            for (var j = k + 1; j <= m; j++)
                            {
                                work[i, j] = ((work[k, k] * work[i, j]) - (work[i, k] * work[k, j])) / previous;
                            }

                            work[i, k] = BigInteger.Zero;
                        }

                        previous = work[k, k];
                    }

                    // det·A⁻¹b by back substitution on the fraction-free triangle; every division is exact.
                    determinant = work[m - 1, m - 1];
                    solution = new BigInteger[m];
                    for (var i = m - 1; i >= 0; i--)
                    {
                        var accumulated = determinant * work[i, m];
                        for (var j = i + 1; j < m; j++)
                        {
                            accumulated -= work[i, j] * solution[j];
                        }

                        solution[i] = accumulated / work[i, i];
                    }
                }

                // ỹ_a = solution_a/det · radix^(finestRhs − finest) in scaled units; y_j = ỹ_a / D_j in the original ones.
                var numerators = new Exact[n];
                var det = Exact.Of(determinant, 0);
                var sum = Exact.Zero;
                for (var i = 0; i < n; i++)
                {
                    if (status[i] != 0)
                    {
                        numerators[i] = basePoint[i] * det;
                    }
                }

                for (var a = 0; a < m; a++)
                {
                    numerators[others[a]] = Exact.Of(solution[a], finestRhs - finest - e[others[a]]);
                    sum += numerators[others[a]];
                }

                numerators[r] = (remaining * det) - sum;
                return new ExactMinimiser(status0, numerators, determinant, problem);
            }
        }

        /// <summary>
        /// The subspace minimiser <c>w + p</c>, the iterate plus its last (unrounded) refinement step: the
        /// multipliers and the anti-cycling objective are judged at it, so the rounding of the stored weights
        /// decides neither a release nor whether a release made progress.
        /// </summary>
        private static Exact[] RefinedPoint(
            Problem problem, int[] free, ReducedFactor? factor, Exact[] gradient, Exact residual, Exact[] exact)
        {
            var refined = (Exact[])exact.Clone();
            if (factor is null)
            {
                refined[free[0]] -= residual;
                return refined;
            }

            var r = factor.Reference;
            var others = factor.Others;
            var y = NullSpaceStep(problem, factor, gradient, residual);
            var sum = Exact.Zero;
            for (var a = 0; a < others.Length; a++)
            {
                refined[others[a]] += y[a];
                sum += y[a];
            }

            refined[r] -= residual + sum;
            return refined;
        }

        /// <summary>
        /// The variables to release at a stationary iterate (none: the iterate is the solver's result). With an
        /// interior free variable the budget multiplier is <c>ν = g_r</c> of the interior reference and one
        /// variable is released; with every variable at a bound (a free variable may sit on its bound) a
        /// multiplier exists exactly when <c>max g</c> over the variables at the upper bound does not exceed
        /// <c>min g</c> over those at the lower bound, and otherwise that pair is released together
        /// (<c>Pair</c> is then <see langword="true"/>).
        /// </summary>
        private static (int[] Release, bool Pair) ReleaseCandidates(
            Problem problem, int[] free, Exact[] gradient, T[] weights, Exact[] exact, int[] status, bool[] stalled)
        {
            if (problem.Lower == problem.Upper)
            {
                return ([], false);
            }

            var n = weights.Length;
            var resolution = new Exact[n];
            for (var j = 0; j < n; j++)
            {
                resolution[j] = Exact.HalfUnit(weights[j]);
            }

            var interior = free.Where(i => weights[i] > problem.Lower && weights[i] < problem.Upper).ToArray();
            if (interior.Length == 0)
            {
                return (BoundPairRelease(problem, gradient, weights, stalled, resolution), true);
            }

            var r = Reference(problem.Covariance, interior);
            var nu = gradient[r];
            var referenceTolerance = RowRounding(problem, r, resolution);
            var best = -1;
            Exact bestMultiplier = default, bestScale = default;
            for (var i = 0; i < n; i++)
            {
                if (status[i] == 0 || stalled[i])
                {
                    continue;
                }

                var multiplier = status[i] < 0 ? gradient[i] - nu : nu - gradient[i];
                if (multiplier.Sign >= 0)
                {
                    continue;
                }

                var threshold = (RowRounding(problem, i, resolution) + referenceTolerance) * Exact.ReleaseFraction;
                if (-multiplier <= threshold)
                {
                    continue;
                }

                var scale = RowScale(problem, i, nu, exact);

                // Most negative multiplier relative to its own row scale; the ascending scan keeps the smallest index.
                if (best < 0 || (multiplier * bestScale) < (bestMultiplier * scale))
                {
                    best = i;
                    bestMultiplier = multiplier;
                    bestScale = scale;
                }
            }

            return (best < 0 ? [] : [best], false);
        }

        // Every variable at a bound: the lower-bound variable of smallest gradient and the upper-bound variable of
        // largest gradient (smallest indices among ties, variables excluded by the anti-cycling rule skipped) are
        // released together when moving budget between them lowers the objective by more than their rows' rounding.
        private static int[] BoundPairRelease(
            Problem problem, Exact[] gradient, T[] weights, bool[] stalled, Exact[] resolution)
        {
            int low = -1, high = -1;
            for (var i = 0; i < weights.Length; i++)
            {
                // A variable excluded by the anti-cycling rule is not released again until the objective decreases.
                if (stalled[i])
                {
                    continue;
                }

                if (weights[i] == problem.Lower && (low < 0 || gradient[i] < gradient[low]))
                {
                    low = i;
                }
                else if (weights[i] == problem.Upper && (high < 0 || gradient[i] > gradient[high]))
                {
                    high = i;
                }
            }

            if (low < 0 || high < 0)
            {
                return [];
            }

            var threshold = (RowRounding(problem, low, resolution) + RowRounding(problem, high, resolution))
                * Exact.ReleaseFraction;
            if (gradient[high] - gradient[low] <= threshold)
            {
                return [];
            }

            return [low, high];
        }

        // Whether a variable at the upper bound has an exact gradient above that of a variable at the lower bound by more
        // than the two rows' rounding (the measure of the pair release): such a pair violates the KKT conditions whatever
        // the budget multiplier is, so no multiplier tolerance that includes the reference row's rounding may excuse it.
        private static bool BoundPairViolated(Problem problem, Exact[] gradient, T[] weights, int[] status)
        {
            var n = weights.Length;
            int low = -1, high = -1;
            for (var i = 0; i < n; i++)
            {
                if (status[i] < 0 && (low < 0 || gradient[i] < gradient[low]))
                {
                    low = i;
                }
                else if (status[i] > 0 && (high < 0 || gradient[i] > gradient[high]))
                {
                    high = i;
                }
            }

            // Every upper-bound gradient at or below every lower-bound gradient: no pair can violate.
            if (low < 0 || high < 0 || gradient[high] <= gradient[low])
            {
                return false;
            }

            var resolution = new Exact[n];
            for (var j = 0; j < n; j++)
            {
                resolution[j] = Exact.HalfUnit(weights[j]);
            }

            var rounding = new Exact[n];
            for (var i = 0; i < n; i++)
            {
                rounding[i] = status[i] == 0 ? Exact.Zero : RowRounding(problem, i, resolution);
            }

            for (var u = 0; u < n; u++)
            {
                if (status[u] <= 0)
                {
                    continue;
                }

                for (var l = 0; l < n; l++)
                {
                    if (status[l] < 0 && gradient[u] - gradient[l] > (rounding[u] + rounding[l]) * Exact.ReleaseFraction)
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        // The rounding bound of gradient row i, Σ_j |λΣ_ij|·ρ(w_j) (the linear term is exact and adds none).
        private static Exact RowRounding(Problem problem, int i, Exact[] resolution)
        {
            var sum = Exact.Zero;
            for (var j = 0; j < resolution.Length; j++)
            {
                sum += problem.Sigma[i, j].Abs() * resolution[j];
            }

            return problem.Linear is null ? sum : problem.Lambda * sum;
        }

        // The scale of row i's multiplier at the point x: |ν| + Σ_j |λΣ_ij|·|x_j| + |μ_i| (centred means).
        private static Exact RowScale(Problem problem, int i, Exact nu, Exact[] x)
        {
            var sum = Exact.Zero;
            for (var j = 0; j < x.Length; j++)
            {
                sum += problem.Sigma[i, j].Abs() * x[j].Abs();
            }

            return problem.Linear is null ? nu.Abs() + sum : nu.Abs() + (problem.Lambda * sum) + LinearMagnitude(problem, i);
        }

        /// <summary>
        /// The exact route: the primal-feasible active-set method carried out in exact arithmetic (<see cref="ExactActiveSet"/>),
        /// taken when a working set is unusable at the working precision, when a rounded refinement does not converge, when
        /// the exit certificate rejects the rounded trajectory's result with no release left, and when the rounded route
        /// spends its iteration budget. It starts from the rounded route's working set and iterate (the budget residual of
        /// the rounded iterate moved exactly onto a free variable), or from the exact feasible start when that point is
        /// not exactly feasible, with a fresh budget of <see cref="ExactRouteBudget"/> working-set changes. Each weight of
        /// its optimum is rounded once into <typeparamref name="T"/>, and the result passes the same exit certificate and
        /// conditioning guard as any other (a result the certificate rejects is refused, naming the failed component).
        /// </summary>
        private static T[] ExactRoute(Problem problem, T[]? weights, int[]? status, ExactRouteReason reason, RouteTrace? trace)
        {
            var n = problem.Covariance.GetLength(0);
            trace?.ExactRouteReason = reason;

            // Bounds clamped to the equal-weight frontier 1/n hold no exactly budget-feasible point when n·fl(1/n) ≠ 1:
            // the frontier point, every weight at the clamped bound, is then the only candidate, checked like any other.
            var count = Exact.FromInt(n);
            if (count * problem.LowerX > Exact.One || count * problem.UpperX < Exact.One)
            {
                var frontier = FilledArray(n, count * problem.LowerX > Exact.One ? problem.Lower : problem.Upper);
                frontier = trace?.ExactRouteResultHook?.Invoke(frontier) ?? frontier;
                var check = ExitCertificate(problem, frontier);
                return check.Accepted ? frontier : throw CertificateFailure(problem, check);
            }

            var (start, startStatus) = ExactStart(problem, weights, status);
            var (numerators, denominator, finalStatus, minimiser) = ExactActiveSet(problem, start, startStatus, trace);
            var result = RoundedPoint(problem, numerators, denominator);
            result = trace?.ExactRouteResultHook?.Invoke(result) ?? result;
            var certificate = ExitCertificate(problem, result);
            if (!certificate.Accepted)
            {
                throw CertificateFailure(problem, certificate);
            }

            RequireDeterminedBlock(problem, FreeIndices(finalStatus), minimiser.Gradient, minimiser.Point, result, finalStatus, new bool[n]);
            return result;
        }

        // Each weight X_i/d correctly rounded into T (inside the box, which the exact point never leaves).
        private static T[] RoundedPoint(Problem problem, Exact[] numerators, BigInteger denominator)
        {
            var weights = new T[numerators.Length];
            for (var i = 0; i < numerators.Length; i++)
            {
                weights[i] = T.Clamp(Exact.Precise(numerators[i], denominator).ToT(), problem.Lower, problem.Upper);
            }

            return weights;
        }

        /// <summary>
        /// The starting point of the exact route: the rounded route's iterate and working set when they are exactly
        /// feasible once the iterate's budget residual <c>1′w − 1</c> is moved onto the first free variable that stays in the
        /// box (every variable of the working set exactly on its bound, at least one free variable); otherwise the exact
        /// feasible start, every weight at the lower bound and the remaining budget filled up to the upper bounds in index
        /// order, carried exactly (with no free variable left, the last filled one is free).
        /// </summary>
        private static (Exact[] Point, int[] Status) ExactStart(Problem problem, T[]? weights, int[]? status)
        {
            var n = problem.Covariance.GetLength(0);
            if (weights is not null && status is not null && Array.IndexOf(status, 0) >= 0)
            {
                var x = new Exact[n];
                var consistent = true;
                for (var i = 0; i < n && consistent; i++)
                {
                    x[i] = Exact.From(weights[i]);
                    consistent = status[i] == 0 || x[i].CompareTo(status[i] < 0 ? problem.LowerX : problem.UpperX) == 0;
                }

                var residual = Exact.Sum(x) - Exact.One;
                if (consistent && residual.Sign != 0)
                {
                    consistent = false;
                    for (var j = 0; j < n && !consistent; j++)
                    {
                        var moved = x[j] - residual;
                        if (status[j] == 0 && moved >= problem.LowerX && moved <= problem.UpperX)
                        {
                            x[j] = moved;
                            consistent = true;
                        }
                    }
                }

                if (consistent)
                {
                    problem.Trace?.ExactStart = Allocation(x, reusedIterate: true);
                    return (x, (int[])status.Clone());
                }
            }

            var point = new Exact[n];
            var start = new int[n];
            var remaining = Exact.One - (Exact.FromInt(n) * problem.LowerX);
            var capacity = problem.UpperX - problem.LowerX;
            var lastFilled = n - 1;
            for (var i = 0; i < n; i++)
            {
                point[i] = problem.LowerX;
                if (remaining.Sign > 0)
                {
                    var add = remaining < capacity ? remaining : capacity;
                    point[i] += add;
                    remaining -= add;
                }

                start[i] = point[i].CompareTo(problem.LowerX) == 0 ? -1 : point[i].CompareTo(problem.UpperX) == 0 ? 1 : 0;
                if (point[i].CompareTo(problem.LowerX) != 0)
                {
                    lastFilled = i;
                }
            }

            if (Array.IndexOf(start, 0) < 0)
            {
                start[lastFilled] = 0;
            }

            problem.Trace?.ExactStart = Allocation(point, reusedIterate: false);
            return (point, start);
        }

        private sealed record ExactState(Exact[] Numerators, BigInteger Denominator, int[] Status, Fraction Objective, int[] Released);

        /// <summary>
        /// The primal-feasible active-set method of the core carried out in exact arithmetic, from the exactly feasible
        /// point <paramref name="start"/> with working set <paramref name="status"/>: the iterate is the exact rational
        /// <c>X/d</c> (radix-fraction numerators over a positive integer denominator) and is never rounded. Each working
        /// set's subspace minimiser is <see cref="ExactMinimiser"/>'s. The step-length rule compares exactly and moves the
        /// iterate exactly: the blocking variable (the smallest index among ties) lands on its bound and joins the working
        /// set, and a step of length zero (a free variable that already sits on its bound blocks) is such a working-set
        /// change, never a rollback. At a subspace minimiser the multipliers are judged with every variable that sits on a
        /// bound treated as at that bound: with a free variable strictly inside the box the budget multiplier is the common
        /// gradient of those variables, exact because the minimiser is, and the variable of the working set with the most
        /// negative multiplier relative to its row scale is released (smallest index among ties); with every variable on a
        /// bound (a vertex) the lower-bound variable of smallest gradient and the upper-bound variable of largest gradient
        /// are released together when the first exceeds the second, every other variable joining the working set at its
        /// bound. A release on a strictly negative exact multiplier strictly lowers the exact objective by the next subspace
        /// minimiser, so the objective strictly decreases between releases and the method terminates at the exact optimum
        /// on degenerate problems too; the exclusion rule (a release that does not lower the objective is undone and its
        /// variables excluded until the objective next decreases) and <see cref="ExactRouteBudget"/> remain defensive limits.
        /// </summary>
        private static (Exact[] Numerators, BigInteger Denominator, int[] Status, ExactMinimiser Minimiser) ExactActiveSet(
            Problem problem, Exact[] start, int[] status, RouteTrace? trace)
        {
            var n = problem.Covariance.GetLength(0);
            var x = (Exact[])start.Clone();
            var d = BigInteger.One;
            var budget = trace?.ExactRouteBudgetOverride ?? ExactRouteBudget(n);
            var changes = 0;
            var stalled = new bool[n];
            ExactState? beforeRelease = null;
            var vertexOnly = problem.LowerX.CompareTo(problem.UpperX) == 0;

            void Change()
            {
                changes++;
                trace?.ExactChanges = changes;

                if (changes > budget)
                {
                    throw BudgetExhausted(problem, RoundedPoint(problem, x, d), budget);
                }
            }

            while (true)
            {
                var minimiser = ExactMinimiser.Compute(problem, status) ?? throw NotCertifiable(problem);
                var big = Exact.Of(minimiser.Denominator, 0);
                var common = Exact.Of(minimiser.Denominator * d, 0);
                var free = FreeIndices(status);

                // The step p = N/D − X/d over the common denominator D·d: numerators q = N·d − X·D, compared exactly.
                var q = new Exact[n];
                var blocking = -1;
                Exact bestDistance = default, bestMagnitude = default;
                foreach (var j in free)
                {
                    q[j] = (minimiser.Numerators[j] * Exact.Of(d, 0)) - (x[j] * big);
                    if (q[j].Sign == 0)
                    {
                        continue;
                    }

                    var distance = q[j].Sign < 0 ? (x[j] * big) - (problem.LowerX * common) : (problem.UpperX * common) - (x[j] * big);
                    var magnitude = q[j].Abs();
                    if (distance >= magnitude)
                    {
                        continue;
                    }

                    if (blocking < 0 || (distance * bestMagnitude) < (bestDistance * magnitude))
                    {
                        blocking = j;
                        bestDistance = distance;
                        bestMagnitude = magnitude;
                    }
                }

                if (blocking >= 0)
                {
                    // x + α·p with α = distance/|q_b| < 1 (zero when the blocking variable already sits on its bound) is
                    // (X·D·|q_b| + distance·q)/(D·d·|q_b|); the radix exponent of |q_b| is folded into the numerators so
                    // that the denominator stays an integer.
                    var denominator = minimiser.Denominator * d * BigInteger.Abs(bestMagnitude.Mantissa);
                    for (var i = 0; i < n; i++)
                    {
                        var moved = x[i] * big * bestMagnitude;
                        if (status[i] == 0 && q[i].Sign != 0)
                        {
                            moved += bestDistance * q[i];
                        }

                        x[i] = moved.Shift(-bestMagnitude.Exponent);
                    }

                    var atLower = q[blocking].Sign < 0;
                    x[blocking] = (atLower ? problem.LowerX : problem.UpperX) * Exact.Of(denominator, 0);
                    status[blocking] = atLower ? -1 : 1;
                    d = Reduce(x, denominator);
                    Change();
                    continue;
                }

                Array.Copy(minimiser.Numerators, x, n);
                d = minimiser.Denominator;
                if (beforeRelease is not null)
                {
                    if (minimiser.Objective < beforeRelease.Objective)
                    {
                        Array.Clear(stalled);
                    }
                    else
                    {
                        trace?.ExactRollbacks += 1;
                        Array.Copy(beforeRelease.Numerators, x, n);
                        Array.Copy(beforeRelease.Status, status, n);
                        d = beforeRelease.Denominator;
                        foreach (var i in beforeRelease.Released)
                        {
                            stalled[i] = true;
                        }

                        beforeRelease = null;
                        continue;
                    }
                }

                if (vertexOnly)
                {
                    // lb = ub: the box holds one point, so no release can move it.
                    return (x, d, status, minimiser);
                }

                // The gradient at N/D is G/D with G = λΣN − D·μ (minimum variance: ΣN). A variable sits on a bound when its
                // numerator is the bound times D; every free variable has the same G at the exact minimiser.
                var gradient = minimiser.GradientNumerators;

                var lowerN = problem.LowerX * big;
                var upperN = problem.UpperX * big;
                bool AtLower(int i) => minimiser.Numerators[i].CompareTo(lowerN) == 0;
                bool AtUpper(int i) => minimiser.Numerators[i].CompareTo(upperN) == 0;
                var interior = Array.FindIndex(free, j => !AtLower(j) && !AtUpper(j));
                int[] release;
                if (interior >= 0)
                {
                    var r = free[interior];
                    var nu = gradient[r];
                    var best = -1;
                    Exact bestMultiplier = default, bestScale = default;
                    for (var i = 0; i < n; i++)
                    {
                        if (status[i] == 0 || stalled[i])
                        {
                            continue;
                        }

                        var multiplier = status[i] < 0 ? gradient[i] - nu : nu - gradient[i];
                        if (multiplier.Sign >= 0)
                        {
                            continue;
                        }

                        var scale = RowScale(problem, i, minimiser.Gradient[r], minimiser.Point);
                        if (best < 0 || (multiplier * bestScale) < (bestMultiplier * scale))
                        {
                            best = i;
                            bestMultiplier = multiplier;
                            bestScale = scale;
                        }
                    }

                    if (best < 0)
                    {
                        return (x, d, status, minimiser);
                    }

                    release = [best];
                }
                else
                {
                    // A vertex: a budget multiplier ν exists exactly when no upper-bound gradient exceeds a lower-bound
                    // one; otherwise moving budget from the upper-bound variable of largest gradient to the lower-bound
                    // variable of smallest gradient (smallest indices among ties) lowers the objective.
                    int low = -1, high = -1;
                    for (var i = 0; i < n; i++)
                    {
                        if (stalled[i])
                        {
                            continue;
                        }

                        if (AtLower(i) && (low < 0 || gradient[i] < gradient[low]))
                        {
                            low = i;
                        }
                        else if (AtUpper(i) && (high < 0 || gradient[i] > gradient[high]))
                        {
                            high = i;
                        }
                    }

                    if (low < 0 || high < 0 || gradient[high] <= gradient[low])
                    {
                        return (x, d, status, minimiser);
                    }

                    release = [low, high];
                }

                beforeRelease = new ExactState((Exact[])x.Clone(), d, (int[])status.Clone(), minimiser.Objective, release);
                if (release.Length == 2)
                {
                    // Every other variable joins the working set at its bound, so the released pair alone spans the step.
                    for (var i = 0; i < n; i++)
                    {
                        if (status[i] == 0 && i != release[0] && i != release[1])
                        {
                            status[i] = AtLower(i) ? -1 : 1;
                            Change();
                        }
                    }
                }

                foreach (var i in release)
                {
                    if (status[i] != 0)
                    {
                        status[i] = 0;
                        Change();
                    }
                }
            }
        }

        // Divides the denominator and every numerator mantissa by their greatest common divisor (the value is unchanged).
        private static BigInteger Reduce(Exact[] numerators, BigInteger denominator)
        {
            var divisor = denominator;
            foreach (var value in numerators)
            {
                if (value.Sign != 0)
                {
                    divisor = BigInteger.GreatestCommonDivisor(divisor, value.Mantissa);
                }
            }

            if (divisor.IsOne)
            {
                return denominator;
            }

            for (var i = 0; i < numerators.Length; i++)
            {
                numerators[i] = Exact.Of(numerators[i].Mantissa / divisor, numerators[i].Exponent);
            }

            return denominator / divisor;
        }

        // The independent certificate at the exit tolerance scale, after the documented range contract on the gradient.
        private static QpKktCertificateResult ExitCertificate(Problem problem, T[] weights)
        {
            RequireRepresentableGradientTerms(problem, weights);
            return Certificate(problem, weights);
        }

        private static QpKktCertificateResult Certificate(Problem problem, T[] weights) => QpKktCertificate<T>.Check(
            problem.Covariance, problem.Means, problem.RiskAversion, problem.Lower, problem.Upper, weights, CertificateToleranceScale);

        // For a multiplier-sign failure, the variable to release: the variable the certificate names or the one that set
        // the other end of the budget multiplier interval, whichever sits in the working set and is not excluded by the
        // anti-cycling rule (first the named one). -1 for any other component, or when neither can be released, that is,
        // when no release is left that could lower the objective.
        private static int CertificateRelease(QpKktCertificateResult certificate, int[] status, bool[] stalled)
        {
            if (certificate.FailedComponent != QpKktComponent.MultiplierSign)
            {
                return -1;
            }

            foreach (var i in new[] { certificate.FailedIndex, certificate.CounterpartIndex })
            {
                if (i >= 0 && status[i] != 0 && !stalled[i])
                {
                    return i;
                }
            }

            return -1;
        }

        private static InvalidOperationException CertificateFailure(Problem problem, QpKktCertificateResult certificate) => new(
            "Active-set QP solver did not converge for " + problem.Kind + " problem: the result failed the KKT certificate ("
            + certificate.Detail + ").");

        /// <summary>
        /// The conditioning guard on a certified result. The block that decides the result is the free variables plus
        /// every variable the anti-cycling rule excluded whose exact multiplier at the subspace minimiser is materially
        /// negative (below <c>−64·u</c> times its row scale <c>Σⱼ|Σᵢⱼ||xⱼ| + |ν|</c>): at such an exit the solver has a
        /// negative multiplier it could not act on, so that variable belongs to the block as much as a free one. A free
        /// variable whose weight sits exactly on a bound is at that bound for the KKT conditions and is tested the same
        /// way (it belongs to the block only with a materially negative multiplier at that bound), so a degenerate
        /// optimum whose interior block meets the margin is not refused. The
        /// result is refused (naming stationarity) when <c>48·n·u·κ_eq(B) ≥ 1</c>, where <c>κ_eq(B)</c> is the exact
        /// 1-norm condition number of the equilibrated block <c>Σ̃_BB</c>, <c>n</c> the number of assets and <c>u</c> the
        /// unit roundoff: the corpus-membership margin applied to the solver's own block. Along a near-null direction of
        /// such a block a move is invisible to any residual certificate, so the alternative to the refusal is an
        /// unverifiable vector. A problem whose exact free block meets the margin is not refused whenever the deciding
        /// block is that free block.
        /// </summary>
        private static void RequireDeterminedBlock(
            Problem problem, int[] free, Exact[] gradient, Exact[] point, T[] weights, int[] status, bool[] stalled)
        {
            var n = weights.Length;
            var material = Exact.UnitRoundoff * Exact.FromInt(MaterialMultiplierUnits);

            // A free variable whose weight sits exactly on a bound is at that bound for the KKT conditions, as the
            // certificate and the corpus definition of the free block treat it: it joins the block only when its
            // multiplier at that bound is materially negative, like an excluded variable.
            bool AtBound(int i) => weights[i] == problem.Lower || weights[i] == problem.Upper;
            bool Tested(int i) => status[i] == 0 ? AtBound(i) : stalled[i];
            int Side(int i) => status[i] != 0 ? status[i] : weights[i] == problem.Lower ? -1 : 1;
            var block = new SortedSet<int>(free.Where(i => !AtBound(i)));

            Exact RowScale(int i, Exact nu) => PrimalActiveSet.RowScale(problem, i, nu, point);

            var interior = free.Where(i => weights[i] > problem.Lower && weights[i] < problem.Upper).ToArray();
            if (interior.Length > 0)
            {
                var nu = gradient[Reference(problem.Covariance, interior)];
                for (var i = 0; i < n; i++)
                {
                    if (!Tested(i))
                    {
                        continue;
                    }

                    var multiplier = Side(i) < 0 ? gradient[i] - nu : nu - gradient[i];
                    if (-multiplier > material * RowScale(i, nu))
                    {
                        block.Add(i);
                    }
                }
            }
            else
            {
                // Every variable at a bound: a multiplier ν must lie between the largest gradient at the upper bound and
                // the smallest at the lower bound; an excluded variable is materially on the wrong side of that interval
                // when it passes the other end by more than 64·u of the two rows' scales.
                int upperEnd = -1, lowerEnd = -1;
                for (var i = 0; i < n; i++)
                {
                    if (weights[i] == problem.Upper && (upperEnd < 0 || gradient[i] > gradient[upperEnd]))
                    {
                        upperEnd = i;
                    }
                    else if (weights[i] == problem.Lower && (lowerEnd < 0 || gradient[i] < gradient[lowerEnd]))
                    {
                        lowerEnd = i;
                    }
                }

                for (var i = 0; i < n; i++)
                {
                    if (!Tested(i))
                    {
                        continue;
                    }

                    if (weights[i] == problem.Lower && upperEnd >= 0
                        && gradient[upperEnd] - gradient[i] > material * (RowScale(i, Exact.Zero) + RowScale(upperEnd, Exact.Zero)))
                    {
                        block.Add(i);
                    }
                    else if (weights[i] == problem.Upper && lowerEnd >= 0
                        && gradient[i] - gradient[lowerEnd] > material * (RowScale(i, Exact.Zero) + RowScale(lowerEnd, Exact.Zero)))
                    {
                        block.Add(i);
                    }
                }
            }

            problem.Trace?.DecidingBlock = [.. block];
            if (block.Count >= 2 && ExceedsConditioningMargin(problem, [.. block], n))
            {
                throw new InvalidOperationException(
                    "Active-set QP solver did not converge for " + problem.Kind + " problem: stationarity cannot be certified because the "
                    + "block of the covariance that decides the result is numerically singular at the working precision "
                    + "(48·n·u·κ ≥ 1 for the exact 1-norm condition number κ of the equilibrated block, u the unit roundoff), so "
                    + "the subspace minimiser is not determined by the rounded problem.");
            }
        }

        /// <summary>
        /// <see cref="RequireDeterminedBlock(Problem, int[], Exact[], Exact[], T[], int[], bool[])"/> on a supplied exit
        /// state (test diagnostics, <see cref="RouteTrace.RequireDeterminedBlock"/>): the problem prepared as
        /// <see cref="Solve"/> prepares it, the gradient formed exactly at <paramref name="point"/>, the free variables
        /// those of <paramref name="status"/>.
        /// </summary>
        internal static void RequireDeterminedBlock(
            RouteTrace trace, T[,] covariance, T[]? means, T riskAversion, T lower, T upper, T[] weights, T[] point, int[] status, bool[] stalled)
        {
            var n = covariance.GetLength(0);
            var scaling = QpScaling<T>.EquilibrateCovariance(covariance);
            var sigma = new Exact[n, n];
            var scaled = new Exact[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    sigma[i, j] = Exact.From(covariance[i, j]);
                    scaled[i, j] = Exact.From(scaling.Covariance[i, j]);
                }
            }

            var problem = new Problem(covariance, sigma, scaled, scaling.Exponents, lower, upper)
            {
                Means = means,
                RiskAversion = riskAversion,
                Lambda = Exact.From(riskAversion),
                Linear = CentredLinear(means),
                Trace = trace,
            };
            var exact = point.Select(Exact.From).ToArray();
            RequireDeterminedBlock(problem, FreeIndices(status), Gradient(problem, exact), exact, weights, status, stalled);
        }

        /// <summary>
        /// Whether <c>48·n·u·κ₁(A) ≥ 1</c> for the principal block <c>A = Σ̃_BB</c> (order <c>m</c>) of the equilibrated
        /// covariance, <c>κ₁(A) = ‖A‖₁·‖A⁻¹‖₁</c>, as <see cref="ExactlyExceedsConditioningMargin"/> computes it exactly
        /// (about m⁴ operations on integers of about m digits). A certified bound decides the block without that
        /// computation when it proves <c>48·n·u·κ₁(A) &lt; 1</c>; for <see cref="double"/> an exact norm witness can
        /// instead prove the opposite inequality, so the decision is the exact test's on every block.
        /// The bounds, in order: a proved lower bound <c>s</c> on the smallest eigenvalue of
        /// the whole equilibrated covariance (<see cref="PrepareConditioningBounds"/>, one double-double factorisation per
        /// solve), which bounds every principal block by Cauchy's interlacing theorem, <c>λ_min(A) ≥ s</c>; the bound of
        /// <see cref="CertifiedWithinMargin"/> on <c>κ₁(A)</c> in <see cref="double"/>; and a proved lower bound on
        /// <c>λ_min(A)</c> from a double-double factor of the block itself (<see cref="EigenvalueLowerBound"/>). The two
        /// eigenvalue bounds decide through <see cref="WithinMarginGiven"/>. Each costs O(m³) operations in
        /// <see cref="double"/>; a block none of them settles (one with <c>48·n·u·κ₁(A)</c> above about <c>1/(2√m)</c>,
        /// near or past the margin) and no refusal witness settles is decided by the exact adjugate.
        /// </summary>
        private static bool ExceedsConditioningMargin(Problem problem, int[] block, int n)
        {
            var trace = problem.Trace;
            var query = trace is null ? 0 : ++trace.ConditioningQueries;
            if (trace is not { ExactConditioningOnly: true })
            {
                var bounds = problem.Conditioning;
                if (bounds.Smallest > 0.0 && WithinMarginGiven(bounds, block, n, bounds.Smallest))
                {
                    return false;
                }

                if (CertifiedWithinMargin(problem.Scaled, block, n, out var outside, trace, query))
                {
                    return false;
                }

                if (outside)
                {
                    return true;
                }

                if (!bounds.Prepared)
                {
                    PrepareConditioningBounds(problem.Scaled, bounds, n, trace);
                    if (bounds.Smallest > 0.0 && WithinMarginGiven(bounds, block, n, bounds.Smallest))
                    {
                        return false;
                    }
                }

                if (bounds.High is not null)
                {
                    var smallest = BlockEigenvalueLowerBound(bounds, block, n, trace, query);
                    if (smallest > 0.0 && WithinMarginGiven(bounds, block, n, smallest))
                    {
                        return false;
                    }
                }
            }

            problem.Trace?.ExactConditioningTests += 1;
            return ExactlyExceedsConditioningMargin(problem.Scaled, block, n);
        }

        /// <summary>
        /// Rounds the whole equilibrated covariance <c>Σ̃</c> into double-double (<see cref="TryDoubleDouble"/>; an entry
        /// without one leaves the eigenvalue bounds unavailable) and proves, when it can, a lower bound on
        /// <c>λ_min(Σ̃)</c> with <see cref="EigenvalueLowerBound"/> at the shift twice the margin eigenvalue of the whole
        /// matrix (<see cref="MarginEigenvalue"/>): every principal block's own margin eigenvalue is at most that one
        /// (<c>√m ≤ √n</c> and a principal block's 1-norm is at most the whole matrix's), so a bound proved at that shift
        /// settles every block of the solve.
        /// </summary>
        private static void PrepareConditioningBounds(Exact[,] scaled, ConditioningBounds bounds, int n, RouteTrace? trace)
        {
            var stage = trace?.ConditioningPreparation;
            var work = 0L;
            var inner = 0L; // the loop bodies of the eigenvalue bound it calls
            bounds.Prepared = true;
            var high = new double[n, n];
            var low = new double[n, n];
            for (var i = 0; i < n; i++)
            {
                work++;
                for (var j = 0; j <= i; j++)
                {
                    work++;
                    if (!TryDoubleDouble(scaled[i, j], out var h, out var g))
                    {
                        stage?.Record(0, n, work);
                        return;
                    }

                    high[i, j] = h;
                    high[j, i] = h;
                    low[i, j] = g;
                    low[j, i] = g;
                }
            }

            bounds.High = high;
            bounds.Low = low;
            var all = new int[n];
            for (var i = 0; i < n; i++)
            {
                work++;
                all[i] = i;
            }

            bounds.Smallest = EigenvalueLowerBound(bounds, all, 2.0 * MarginEigenvalue(bounds, all, n), ref inner);
            stage?.Record(0, n, work + inner);
        }

        // The double-double eigenvalue bound of a queried block itself (EigenvalueLowerBound at twice the block's margin
        // eigenvalue), recorded as one run on that block.
        private static double BlockEigenvalueLowerBound(ConditioningBounds bounds, int[] block, int n, RouteTrace? trace, int query)
        {
            var work = 0L;
            var smallest = EigenvalueLowerBound(bounds, block, 2.0 * MarginEigenvalue(bounds, block, n), ref work);
            return Recorded(trace?.BlockEigenvalueBound, query, block.Length, work, smallest);
        }

        /// <summary>
        /// The margin eigenvalue of the block <c>A = Σ̃_BB</c>: a computed upper bound on <c>48·n·u·√m·‖A‖₁</c>. For a
        /// symmetric positive definite <c>A</c>, <c>‖A⁻¹‖₁ ≤ √m·‖A⁻¹‖₂ = √m/λ_min(A)</c>: Cauchy-Schwarz bounds
        /// each column's 1-norm by √m times its 2-norm, which is at most the matrix 2-norm. Hence
        /// <c>48·n·u·κ₁(A) ≤ 48·n·u·√m·‖A‖₁/λ_min(A)</c>, below 1 when <c>λ_min(A)</c> exceeds this value. <c>‖A‖₁</c> is
        /// bounded by the largest column sum of <c>(|ĥ_ij| + |ĝ_ij|)·(1 + c)</c> (<c>|a_ij − ĥ_ij − ĝ_ij| ≤ c·|ĝ_ij|</c>,
        /// <c>c = 4·u_d</c> for <c>decimal</c> and 0 otherwise, <see cref="TryDoubleDouble"/>), <c>u</c> by
        /// <see cref="WorkingUnitRoundoff"/> (at least the unit roundoff the exact test uses). The column sums of at most
        /// <c>m</c> non-negative terms and the six products and square root change the computed value by a factor of at
        /// least <c>1 − (m + 8)·u_d ≥ 1 − 10⁻⁶</c>, and the factor 1.001 it carries covers that with room for the
        /// comparison of <see cref="WithinMarginGiven"/>.
        /// </summary>
        private static double MarginEigenvalue(ConditioningBounds bounds, int[] block, int n)
        {
            const double unit = 1.0 / 9007199254740992.0; // 2⁻⁵³
            const double safety = 1.001;
            var high = bounds.High!;
            var low = bounds.Low!;
            var norm = 0.0;
            foreach (var c in block)
            {
                var column = 0.0;
                foreach (var r in block)
                {
                    column += Math.Abs(high[r, c]) + Math.Abs(low[r, c]);
                }

                norm = Math.Max(norm, column);
            }

            var conversion = typeof(T) == typeof(decimal) ? 4.0 * unit : 0.0;
            return safety * ConditioningMargin * (double)n * WorkingUnitRoundoff * Math.Sqrt(block.Length) * norm * (1.0 + conversion);
        }

        /// <summary>
        /// Whether a proved lower bound <paramref name="smallest"/> on <c>λ_min(Σ̃_BB)</c> exceeds the block's
        /// <see cref="MarginEigenvalue"/>, which proves <c>48·n·u·κ₁(Σ̃_BB) &lt; 1</c>: <paramref name="smallest"/> comes
        /// from <see cref="EigenvalueLowerBound"/> as <c>fl(s − r̂)</c> with <c>λ_min ≥ s − r̂</c>, so
        /// <c>λ_min ≥ smallest/(1 + u_d)</c>, and the margin eigenvalue is at least <c>1.001·(1 − 10⁻⁶)</c> times
        /// <c>48·n·u·√m·‖Σ̃_BB‖₁</c>, more than <c>(1 + u_d)</c> times it.
        /// </summary>
        private static bool WithinMarginGiven(ConditioningBounds bounds, int[] block, int n, double smallest)
        {
            var requirement = MarginEigenvalue(bounds, block, n);
            return double.IsFinite(requirement) && requirement < smallest;
        }

        /// <summary>
        /// A proved lower bound on <c>λ_min(A)</c> for the principal block <c>A = Σ̃_BB</c>, or 0 when none is proved. As in
        /// <see cref="CertifiedLeadingMinors"/>, <c>Ĥ + Ĝ</c> is the double-double rounding of <c>A</c>, exactly symmetric,
        /// <c>L = L_h + L_l</c> any lower-triangular pair of finite double matrices from <see cref="DoubleDoubleCholesky"/> of
        /// <c>Ĥ + Ĝ − s·I</c> with every column computed, and <c>F = Ĥ + Ĝ − s·I − L·L′</c>, so
        /// <c>A = L·L′ + s·I + F − (Ĥ + Ĝ − A)</c> with <c>L·L′</c> positive semidefinite and, by Weyl's inequality,
        /// <c>λ_min(A) ≥ s − ‖F‖_∞ − ‖Ĥ + Ĝ − A‖_∞</c>. <see cref="ResidualBound"/> bounds each <c>|F_ij|</c>, the
        /// conversion bounds each <c>|(Ĥ + Ĝ − A)_ij|</c>, and the computed <c>r̂ = fl(1.001·max_i Σ_j (b_ij + c_ij))</c> is at
        /// least their largest row sum (the argument of <see cref="CertifiedLeadingMinors"/>). The result is
        /// <c>fl(s − r̂)</c> when positive: <c>λ_min(A) ≥ s − r̂ ≥ fl(s − r̂)/(1 + u_d)</c>. Any failure (a column not
        /// computed, a non-finite value, a block too large for the bound) returns 0.
        /// </summary>
        private static double EigenvalueLowerBound(ConditioningBounds bounds, int[] block, double shift, ref long work)
        {
            const double unit = 1.0 / 9007199254740992.0; // 2⁻⁵³
            const double safety = 1.001;
            var count = 0L;
            var m = block.Length;
            if (!double.IsFinite(shift) || !(shift > 0.0) || ((6.0 * m) + 20.0) * unit > 1e-6)
            {
                return 0.0;
            }

            var high = new double[m, m];
            var low = new double[m, m];
            for (var i = 0; i < m; i++)
            {
                count++;
                for (var j = 0; j < m; j++)
                {
                    count++;
                    high[i, j] = bounds.High![block[i], block[j]];
                    low[i, j] = bounds.Low![block[i], block[j]];
                }
            }

            var lh = new double[m, m];
            var ll = new double[m, m];
            if (DoubleDoubleCholesky(high, low, shift, m, lh, ll, ref work) < m)
            {
                work += count;
                return 0.0;
            }

            var k = (3.0 * m) + 6.0;
            var gamma = k * unit / (1.0 - (k * unit));
            var square = 2.0 * gamma * gamma;
            var conversion = typeof(T) == typeof(decimal) ? 4.0 * unit : 0.0;
            var rows = new double[m];
            for (var t = 0; t < m; t++)
            {
                count++;
                for (var j = 0; j <= t; j++)
                {
                    count++;
                    var entry = ResidualBound(high, low, lh, ll, t, j, t == j ? shift : 0.0, square, m, ref work) + (conversion * Math.Abs(low[t, j]));
                    rows[t] += entry;
                    if (j != t)
                    {
                        rows[j] += entry;
                    }
                }
            }

            var worst = 0.0;
            foreach (var row in rows)
            {
                count++;
                worst = Math.Max(worst, row);
            }

            work += count;
            var smallest = shift - (safety * worst);
            return double.IsFinite(smallest) && smallest > 0.0 ? smallest : 0.0;
        }

        /// <summary>
        /// The exact test of <see cref="ExceedsConditioningMargin"/>: <c>A</c> is brought onto its finest radix grid as an
        /// integer matrix, fraction-free (Bareiss) elimination of <c>[A | I]</c> gives <c>det A</c> (the last pivot) and
        /// back substitution the integer adjugate <c>det A · A⁻¹</c>, so <c>κ₁ = ‖A‖₁·‖adj A‖₁ / det A</c> and the
        /// comparison is one of integers.
        /// </summary>
        private static bool ExactlyExceedsConditioningMargin(Exact[,] scaled, int[] block, int n)
        {
            var m = block.Length;
            var finest = int.MaxValue;
            for (var a = 0; a < m; a++)
            {
                for (var b = 0; b < m; b++)
                {
                    var entry = scaled[block[a], block[b]];
                    if (entry.Sign != 0)
                    {
                        finest = Math.Min(finest, entry.Exponent);
                    }
                }
            }

            var original = new BigInteger[m, m];
            var work = new BigInteger[m, 2 * m];
            for (var a = 0; a < m; a++)
            {
                for (var b = 0; b < m; b++)
                {
                    var entry = scaled[block[a], block[b]];
                    original[a, b] = entry.Sign == 0 ? BigInteger.Zero : entry.Mantissa * Exact.RadixPower(entry.Exponent - finest);
                    work[a, b] = original[a, b];
                }

                work[a, m + a] = BigInteger.One;
            }

            // Fraction-free elimination: every leading principal minor of a positive-definite block is positive, so the
            // pivots never vanish and no row exchange is needed.
            var previous = BigInteger.One;
            for (var k = 0; k < m; k++)
            {
                if (work[k, k].Sign <= 0)
                {
                    return true;
                }

                for (var i = k + 1; i < m; i++)
                {
                    for (var j = k + 1; j < 2 * m; j++)
                    {
                        work[i, j] = ((work[k, k] * work[i, j]) - (work[i, k] * work[k, j])) / previous;
                    }

                    work[i, k] = BigInteger.Zero;
                }

                previous = work[k, k];
            }

            var determinant = work[m - 1, m - 1];
            var adjugate = new BigInteger[m, m];
            for (var c = 0; c < m; c++)
            {
                for (var i = m - 1; i >= 0; i--)
                {
                    var accumulated = determinant * work[i, m + c];
                    for (var j = i + 1; j < m; j++)
                    {
                        accumulated -= work[i, j] * adjugate[j, c];
                    }

                    adjugate[i, c] = accumulated / work[i, i];
                }
            }

            BigInteger ColumnNorm(BigInteger[,] matrix)
            {
                var best = BigInteger.Zero;
                for (var c = 0; c < m; c++)
                {
                    var sum = BigInteger.Zero;
                    for (var r = 0; r < m; r++)
                    {
                        sum += BigInteger.Abs(matrix[r, c]);
                    }

                    best = BigInteger.Max(best, sum);
                }

                return best;
            }

            // 48·n·u·‖A‖₁·‖adj A‖₁/det A ≥ 1  ⇔  48·n·‖A‖₁·‖adj A‖₁ ≥ det A / u.
            return ConditioningMargin * n * ColumnNorm(original) * ColumnNorm(adjugate) >= determinant * Exact.InverseUnitRoundoff;
        }

        /// <summary>
        /// Whether a rigorous upper bound on <c>48·n·u·κ₁(A)</c> for the principal block <c>A = Σ̃_BB</c> is below 1, so
        /// that the exact test of <see cref="ExceedsConditioningMargin"/> is certainly negative. <c>Â</c> is <c>A</c> rounded
        /// into <see cref="double"/> (relative error at most <c>16·u_d</c> per entry, <c>u_d = 2⁻⁵³</c>), <c>X</c> an
        /// approximate inverse from the Cholesky factor of <c>Â</c>. With <c>δ</c> a rigorous bound on <c>‖I − X·A‖₁</c>
        /// (the computed residual plus the rounding bound <c>γ·‖|X|·|Â|‖₁</c> of its products and the conversion error),
        /// <c>δ &lt; 1/2</c> gives <c>‖A⁻¹‖₁ ≤ ‖X‖₁/(1 − δ)</c> and so <c>κ₁(A) ≤ ‖A‖₁·‖X‖₁/(1 − δ)</c>; every rounding of
        /// the bound itself is absorbed by a safety factor of 1.001. For <see cref="double"/>, a column of X can also
        /// supply the exact norm witness of <see cref="CertifiedOutsideMargin"/>; <paramref name="outside"/> reports
        /// that proof. Otherwise a failed factorisation, a non-finite value, <c>δ ≥ 1/2</c> or an inconclusive bound
        /// returns <see langword="false"/> and the remaining tests decide. The block is a principal block of the equilibrated
        /// covariance, which <see cref="RequirePositiveDefinite"/> has verified, so it is positive definite and the exact
        /// test's non-positive-pivot exit cannot occur on it.
        /// </summary>
        private static bool CertifiedWithinMargin(Exact[,] scaled, int[] block, int n, out bool outside, RouteTrace? trace, int query)
        {
            outside = false;
            const double unit = 1.0 / 9007199254740992.0; // 2⁻⁵³
            const double safety = 1.001;
            var stage = trace?.BlockMarginBound;
            var work = 0L;
            var m = block.Length;
            if (m * unit > 1e-6)
            {
                return Recorded(stage, query, m, work, false);
            }

            var a = new double[m, m];
            for (var i = 0; i < m; i++)
            {
                work++;
                for (var j = 0; j < m; j++)
                {
                    work++;
                    a[i, j] = ToDouble(scaled[block[i], block[j]]);
                    if (!double.IsFinite(a[i, j]))
                    {
                        return Recorded(stage, query, m, work, false);
                    }
                }
            }

            // Cholesky factor L of Â (lower triangle).
            var l = new double[m, m];
            for (var j = 0; j < m; j++)
            {
                work++;
                var diagonal = a[j, j];
                for (var k = 0; k < j; k++)
                {
                    work++;
                    diagonal -= l[j, k] * l[j, k];
                }

                if (!(diagonal > 0.0) || !double.IsFinite(diagonal))
                {
                    return Recorded(stage, query, m, work, false);
                }

                l[j, j] = Math.Sqrt(diagonal);
                for (var i = j + 1; i < m; i++)
                {
                    work++;
                    var sum = a[i, j];
                    for (var k = 0; k < j; k++)
                    {
                        work++;
                        sum -= l[i, k] * l[j, k];
                    }

                    l[i, j] = sum / l[j, j];
                }
            }

            // X ≈ Â⁻¹, column by column: L·y = e_c, then Lᵀ·x = y.
            var x = new double[m, m];
            var y = new double[m];
            for (var c = 0; c < m; c++)
            {
                work++;
                for (var i = 0; i < m; i++)
                {
                    work++;
                    var sum = i == c ? 1.0 : 0.0;
                    for (var k = 0; k < i; k++)
                    {
                        work++;
                        sum -= l[i, k] * y[k];
                    }

                    y[i] = sum / l[i, i];
                }

                for (var i = m - 1; i >= 0; i--)
                {
                    work++;
                    var sum = y[i];
                    for (var k = i + 1; k < m; k++)
                    {
                        work++;
                        sum -= l[k, i] * x[k, c];
                    }

                    x[i, c] = sum / l[i, i];
                }
            }

            // ‖Â‖₁, ‖X‖₁, the computed residual ‖I − fl(X·Â)‖₁ and ‖|X|·|Â|‖₁, each the largest column sum.
            double normA = 0.0, normX = 0.0, residual = 0.0, magnitude = 0.0;
            var largestColumn = 0;
            for (var c = 0; c < m; c++)
            {
                work++;
                double columnA = 0.0, columnX = 0.0, columnResidual = 0.0, columnMagnitude = 0.0;
                for (var r = 0; r < m; r++)
                {
                    work++;
                    columnA += Math.Abs(a[r, c]);
                    columnX += Math.Abs(x[r, c]);
                    double dot = 0.0, absolute = 0.0;
                    for (var k = 0; k < m; k++)
                    {
                        work++;
                        dot += x[r, k] * a[k, c];
                        absolute += Math.Abs(x[r, k]) * Math.Abs(a[k, c]);
                    }

                    columnResidual += Math.Abs((r == c ? 1.0 : 0.0) - dot);
                    columnMagnitude += absolute;
                }

                normA = Math.Max(normA, columnA);
                if (columnX > normX)
                {
                    normX = columnX;
                    largestColumn = c;
                }
                residual = Math.Max(residual, columnResidual);
                magnitude = Math.Max(magnitude, columnMagnitude);
            }

            if (typeof(T) == typeof(double))
            {
                var witness = 0L; // a separate counter, so that the loop counter above stays a register
                outside = CertifiedOutsideMargin(scaled, block, n, x, largestColumn, ref witness);
                work += witness;
                if (outside)
                {
                    return Recorded(stage, query, m, work, false);
                }
            }

            // ‖I − X·A‖₁ ≤ ‖I − fl(X·Â)‖₁ + γ_{m+2}·‖|X|·|Â|‖₁ + ‖X·(Â − A)‖₁, the last at most 2·16·u_d·‖|X|·|Â|‖₁; a term
            // of 1e-300 per column covers gradual underflow in the products.
            const double conversion = 16.0 * unit;
            var gamma = (m + 2) * unit / (1.0 - ((m + 2) * unit));
            var delta = safety * ((residual * (1.0 + (2.0 * unit))) + ((gamma + (2.0 * conversion)) * magnitude) + (m * 1e-300));
            if (!double.IsFinite(delta) || !(delta < 0.5))
            {
                return Recorded(stage, query, m, work, false);
            }

            var kappa = safety * normA * (1.0 + (2.0 * conversion)) * normX / (1.0 - delta);
            var bound = safety * ConditioningMargin * (double)n * WorkingUnitRoundoff * kappa;
            return Recorded(stage, query, m, work, double.IsFinite(bound) && bound < 1.0);
        }

        // For any nonzero vector x, ||A^-1||_1 >= ||x||_1 / ||Ax||_1. A column of the approximate inverse is only a
        // proposed witness: all three norms and the final comparison are evaluated exactly on the original block.
        // Thus this can prove the exact guard's refusal without constructing the full adjugate; it cannot accept.
        private static bool CertifiedOutsideMargin(Exact[,] scaled, int[] block, int n, double[,] inverse, int column, ref long work)
        {
            var count = 0L;
            var m = block.Length;
            var witness = new Exact[m];
            var normX = Exact.Zero;
            for (var i = 0; i < m; i++)
            {
                count++;
                if (!double.IsFinite(inverse[i, column]))
                {
                    work += count;
                    return false;
                }

                witness[i] = Exact.From(T.CreateChecked(inverse[i, column]));
                normX += witness[i].Abs();
            }

            if (normX.Sign == 0)
            {
                work += count;
                return false;
            }

            var normA = Exact.Zero;
            var normProduct = Exact.Zero;
            for (var i = 0; i < m; i++)
            {
                count++;
                var row = Exact.Zero;
                var product = Exact.Zero;
                for (var j = 0; j < m; j++)
                {
                    count++;
                    var entry = scaled[block[i], block[j]];
                    row += entry.Abs();
                    product += entry * witness[j];
                }

                normA = row > normA ? row : normA; // symmetric: infinity norm equals 1-norm
                normProduct += product.Abs();
            }

            work += count;
            return Exact.FromInt(ConditioningMargin) * Exact.FromInt(n) * Exact.UnitRoundoff * normA * normX >= normProduct;
        }

        // The unit roundoff u of T as a double, rounded up (decimal 1e-28, float 2⁻²⁴, double 2⁻⁵³).
        private static double WorkingUnitRoundoff =>
            typeof(T) == typeof(decimal) ? 1.0000001e-28 : typeof(T) == typeof(float) ? Math.ScaleB(1.0, -24) : Math.ScaleB(1.0, -53);

        // An exact radix fraction rounded into double: exact for binary radix fractions of T = double or float, within a
        // few units of the last place for decimal (the mantissa's conversion, the power of ten and their product).
        private static double ToDouble(Exact value)
        {
            if (value.Sign == 0)
            {
                return 0.0;
            }

            var mantissa = (double)value.Mantissa;
            if (typeof(T) != typeof(decimal))
            {
                return Math.ScaleB(mantissa, value.Exponent);
            }

            return value.Exponent >= 0 ? mantissa * Math.Pow(10.0, value.Exponent) : mantissa / Math.Pow(10.0, -value.Exponent);
        }

        private static InvalidOperationException BudgetExhausted(Problem problem, T[] weights, int budget)
        {
            var certificate = Certificate(problem, weights);
            return new InvalidOperationException(
                "Active-set QP solver did not converge for " + problem.Kind + " problem within its iteration budget of " + budget
                + " working-set changes (" + (certificate.Accepted ? "the last iterate is stationary" : certificate.Detail) + ").");
        }

        private static InvalidOperationException NotCertifiable(Problem problem) => new(
            "Active-set QP solver did not converge for " + problem.Kind + " problem: stationarity cannot be certified because the "
            + "reduced Hessian of the free variables is not positive definite at the working precision.");

        // The documented range contract on the returned point's gradient, whose rows the core and the certificate form
        // exactly: every product Σ_ij·w_j must be representable in T, and for mean variance a quadratic term λ·Σ_ij·w_j
        // must not overflow T together with its factor λ·Σ_ij. Both factors are pairwise products of inputs, which
        // the operating domain keeps representable (decimal: |x| ≤ 1e14), so the refusal is reachable only from an
        // input outside it. A term λ·Σ_ij·w_j beyond the range whose factors are representable is a triple product of
        // in-domain magnitudes (a near-null block can leverage w_j far from 1/n while each gradient row, a difference of
        // such terms, stays representable); the exact rows decide such a result, so it is never refused for its terms.
        private static void RequireRepresentableGradientTerms(Problem problem, T[] weights)
        {
            var covariance = problem.Covariance;
            for (var i = 0; i < weights.Length; i++)
            {
                for (var j = 0; j < weights.Length; j++)
                {
                    if (!TryProduct(covariance[i, j], weights[j], out var term)
                        || (problem.Means is not null
                            && !TryProduct(problem.RiskAversion, covariance[i, j], out _)
                            && !TryProduct(problem.RiskAversion, term, out _)))
                    {
                        throw new InvalidOperationException("QP release gradient must be finite.");
                    }
                }
            }
        }

        // The working-type product a·b when it is finite (an overflow is ±∞ for IEEE types, OverflowException for decimal).
        private static bool TryProduct(T a, T b, out T product)
        {
            try
            {
                product = a * b;
                return T.IsFinite(product);
            }
            catch (OverflowException)
            {
                product = T.Zero;
                return false;
            }
        }

        // Sylvester's criterion on the exact covariance: every leading principal minor positive, by fraction-free
        // (Bareiss) elimination on the integer matrix of the inputs over their finest common radix grid. That
        // elimination costs about n⁵ on integers of about n·150 bits, so it runs only for a matrix that neither
        // certified decision settles: CertifiedPositiveDefinite proves positive definiteness in double, and
        // CertifiedLeadingMinors proves, in double-double arithmetic with an exact witness, either positive definiteness
        // or the order of the first leading principal minor that is not positive. Each decides only what the elimination
        // would decide, so the refused covariances, the exception and its message are the elimination's.
        private static void RequirePositiveDefinite(Exact[,] sigma, int n, RouteTrace? trace)
        {
            if (CertifiedPositiveDefinite(sigma, n, trace))
            {
                RecordPositiveDefiniteness(trace, PositiveDefinitenessRoute.DoubleCertificate);
                return;
            }

            var order = CertifiedLeadingMinors(sigma, n, trace);
            if (order >= 0)
            {
                RecordPositiveDefiniteness(trace, PositiveDefinitenessRoute.DoubleDoubleCertificate);
                if (order == 0)
                {
                    return;
                }

                throw NotPositiveDefinite(order);
            }

            RecordPositiveDefiniteness(trace, PositiveDefinitenessRoute.ExactElimination);
            var finest = int.MaxValue;
            foreach (var entry in sigma)
            {
                if (entry.Sign != 0)
                {
                    finest = Math.Min(finest, entry.Exponent);
                }
            }

            var a = new BigInteger[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    a[i, j] = sigma[i, j].Sign == 0 ? BigInteger.Zero : sigma[i, j].Mantissa * Exact.RadixPower(sigma[i, j].Exponent - finest);
                }
            }

            var previous = BigInteger.One;
            for (var k = 0; k < n; k++)
            {
                if (a[k, k].Sign <= 0)
                {
                    throw NotPositiveDefinite(k + 1);
                }

                for (var i = k + 1; i < n; i++)
                {
                    for (var j = k + 1; j < n; j++)
                    {
                        a[i, j] = ((a[k, k] * a[i, j]) - (a[i, k] * a[k, j])) / previous;
                    }
                }

                previous = a[k, k];
            }
        }

        // 10^k for 0 ≤ k ≤ 22: every one is a double exactly (5²² < 2⁵³), and each literal is parsed correctly rounded.
        private static readonly double[] s_exactPowersOfTen =
        [
            1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
            1e20, 1e21, 1e22,
        ];

        /// <summary>
        /// Whether the equilibrated covariance <c>A = Σ̃</c> is proved positive definite in <see cref="double"/>, so that
        /// the exact test of <see cref="RequirePositiveDefinite"/> certainly accepts it. <c>Â</c> is the lower triangle of
        /// <c>A</c> rounded into <see cref="double"/> and mirrored (<see cref="CertifiedDouble"/>: exact for
        /// <c>T = double</c> and <c>float</c>, <c>|â_ij − a_ij| ≤ 4·u_d·|â_ij|</c> for <c>decimal</c>, <c>u_d = 2⁻⁵³</c>),
        /// so <c>Â</c> and <c>A − Â</c> are exactly symmetric. <c>L</c> is any lower-triangular factor computed by
        /// Cholesky in <see cref="double"/> from <c>Â − s·I</c> for a shift <c>s &gt; 0</c> (no property of <c>L</c> is
        /// assumed beyond its entries being finite doubles), and <c>F = Â − s·I − L·L′</c>. Then
        /// <c>A = L·L′ + s·I + (A − Â) + F</c> with <c>L·L′</c> positive semidefinite, so by Weyl's inequality
        /// <c>λ_min(A) ≥ s − ‖A − Â‖₂ − ‖F‖₂</c>, and for a symmetric matrix <c>‖M‖₂ = ρ(M) ≤ ‖M‖_∞</c>, the largest row
        /// sum of <c>|M|</c>. Each <c>F_ij</c> (<c>j ≤ i</c>) is formed in <see cref="double"/> as
        /// <c>â_ij − s·δ_ij − Σ_{k≤j} l_ik·l_jk</c> by recursive summation, together with
        /// <c>m_ij = |â_ij| + s·δ_ij + Σ_{k≤j} |l_ik|·|l_jk|</c>; with at most <c>n</c> products and one leading
        /// subtraction, <c>|F_ij − f̂_ij| ≤ γ_{n+2}·m_ij</c> (Higham 2002, §3.1), <c>γ_k = k·u_d/(1 − k·u_d)</c>, and
        /// <c>m_ij ≤ m̂_ij/(1 − γ_{n+2})</c>. Gradual underflow adds at most <c>2⁻¹⁰⁷⁵</c> per product carried through at
        /// most <c>n + 2</c> roundings, under <c>n·2⁻¹⁰⁷²</c> per entry and <c>n²·10⁻³⁰⁰</c> per row; an overflow leaves a
        /// non-finite sum, which is refused. Every remaining rounding of the bound (the row sums of at most <c>n</c>
        /// non-negative terms, <c>γ</c>, the products and the sums of its three terms) changes it by a factor of at most
        /// <c>1 + (2n + 8)·u_d ≤ 1 + 10⁻⁶</c>, which the safety factor of 1.001 absorbs. So the computed bound below
        /// <c>s</c> proves <c>λ_min(A) &gt; 0</c>: <c>A</c> is symmetric and positive definite, every leading principal
        /// minor is positive (Sylvester), and every Bareiss pivot of the exact test, a positive multiple of one, is
        /// positive. The <see cref="double"/> side only ever decides this positive outcome and returns nothing else: a
        /// failed factorisation, a non-finite value or a bound at or above <c>s</c> returns <see langword="false"/>, and
        /// <see cref="CertifiedLeadingMinors"/> or the exact elimination decides, so the accepted and the refused
        /// covariances, and the refusal, are the exact test's.
        /// </summary>
        private static bool CertifiedPositiveDefinite(Exact[,] sigma, int n, RouteTrace? trace)
        {
            const double unit = 1.0 / 9007199254740992.0; // 2⁻⁵³
            const double safety = 1.001;
            var stage = trace?.DoubleCertificate;
            var work = 0L;
            if ((2.0 * n + 8.0) * unit > 1e-6)
            {
                return Recorded(stage, 0, n, work, false);
            }

            var a = new double[n, n];
            for (var i = 0; i < n; i++)
            {
                work++;
                for (var j = 0; j <= i; j++)
                {
                    work++;
                    var value = CertifiedDouble(sigma[i, j]);
                    if (!double.IsFinite(value))
                    {
                        return Recorded(stage, 0, n, work, false);
                    }

                    a[i, j] = value;
                    a[j, i] = value;
                }
            }

            double maxRow = 0.0, maxDiagonal = 0.0;
            for (var i = 0; i < n; i++)
            {
                work++;
                var row = 0.0;
                for (var j = 0; j < n; j++)
                {
                    work++;
                    row += Math.Abs(a[i, j]);
                }

                maxRow = Math.Max(maxRow, row);
                maxDiagonal = Math.Max(maxDiagonal, a[i, i]);
            }

            if (!(maxDiagonal > 0.0) || !double.IsFinite(maxRow))
            {
                return Recorded(stage, 0, n, work, false);
            }

            // The shift is a few times the a-priori size of the bound: Σ_j m_ij ≤ Σ_j |â_ij| + s + n·max_i â_ii by
            // Cauchy–Schwarz on the rows of L, whose squared norms are about the diagonal of Â − s·I.
            var conversion = typeof(T) == typeof(decimal) ? 4.0 * unit : 0.0;
            var tiny = n * (double)n * 1e-300;
            var gamma = (n + 2) * unit / (1.0 - ((n + 2) * unit));
            var shift = (4.0 * ((gamma * (maxRow + (n * maxDiagonal))) + (conversion * maxRow))) + (4.0 * tiny);

            // L: the Cholesky factor of Â − s·I (lower triangle).
            var l = new double[n, n];
            for (var j = 0; j < n; j++)
            {
                work++;
                var diagonal = a[j, j] - shift;
                for (var k = 0; k < j; k++)
                {
                    work++;
                    diagonal -= l[j, k] * l[j, k];
                }

                if (!(diagonal > 0.0) || !double.IsFinite(diagonal))
                {
                    return Recorded(stage, 0, n, work, false);
                }

                l[j, j] = Math.Sqrt(diagonal);
                for (var i = j + 1; i < n; i++)
                {
                    work++;
                    var sum = a[i, j];
                    for (var k = 0; k < j; k++)
                    {
                        work++;
                        sum -= l[i, k] * l[j, k];
                    }

                    l[i, j] = sum / l[j, j];
                }
            }

            // Row sums of |f̂_ij| + γ·m̂_ij, the bound on |F_ij|, over the symmetric F (lower triangle formed).
            var rows = new double[n];
            for (var i = 0; i < n; i++)
            {
                work++;
                for (var j = 0; j <= i; j++)
                {
                    work++;
                    var f = i == j ? a[i, i] - shift : a[i, j];
                    var m = i == j ? Math.Abs(a[i, i]) + shift : Math.Abs(a[i, j]);
                    for (var k = 0; k <= j; k++)
                    {
                        work++;
                        f -= l[i, k] * l[j, k];
                        m += Math.Abs(l[i, k]) * Math.Abs(l[j, k]);
                    }

                    var entry = Math.Abs(f) + (gamma * m);
                    if (!double.IsFinite(entry))
                    {
                        return Recorded(stage, 0, n, work, false);
                    }

                    rows[i] += entry;
                    if (j != i)
                    {
                        rows[j] += entry;
                    }
                }
            }

            var residual = 0.0;
            foreach (var row in rows)
            {
                work++;
                residual = Math.Max(residual, row);
            }

            var bound = safety * (residual + (conversion * maxRow) + tiny);
            return Recorded(stage, 0, n, work, double.IsFinite(bound) && bound < shift);
        }

        // A stage's run recorded on the solve's trace (scope 0 for a stage of the whole solve, the query for a block's),
        // its result passed through: every exit of a stage returns through this, after its loops have counted their work.
        private static TResult Recorded<TResult>(StageWork? stage, int scope, int order, long work, TResult result)
        {
            stage?.Record(scope, order, work);
            return result;
        }

        // An exact radix fraction m·radix^e of the scaled covariance rounded into double. For T = double or float it is
        // a value of T, whose mantissa has at most 53 bits and which is a double, so the conversion and the power-of-two
        // scaling are exact. For decimal (|m| < 2⁹⁶, −28 ≤ e ≤ 0) m is split as h·2⁴⁸ + l with h, l < 2⁴⁸, both exact
        // doubles, summed with one rounding and divided by at most two exact powers of ten: three correctly rounded
        // operations on values of at least 10⁻²⁸ in magnitude, so |â − a| ≤ ((1 + u_d)³ − 1)·|a| ≤ 4·u_d·|â|.
        // Anything else returns NaN, which the caller treats as undecided.
        private static double CertifiedDouble(Exact value)
        {
            if (value.Sign == 0)
            {
                return 0.0;
            }

            if (typeof(T) != typeof(decimal))
            {
                return Math.ScaleB((double)value.Mantissa, value.Exponent);
            }

            var magnitude = BigInteger.Abs(value.Mantissa);
            if (value.Exponent > 0 || value.Exponent < -28 || magnitude.GetBitLength() > 96)
            {
                return double.NaN;
            }

            var result = Math.ScaleB((double)(magnitude >> 48), 48) + (double)(magnitude & ((BigInteger.One << 48) - 1));
            var scale = -value.Exponent;
            if (scale > 22)
            {
                result /= 1e22;
                scale -= 22;
            }

            result /= s_exactPowersOfTen[scale];
            return value.Sign < 0 ? -result : result;
        }

        private static InvalidOperationException NotPositiveDefinite(int order) => new(
            "Covariance matrix must be positive definite (leading principal minor of order " + order + " is not positive).");

        private static void RecordPositiveDefiniteness(RouteTrace? trace, PositiveDefinitenessRoute route)
        {
            trace?.PositiveDefinitenessRoute = route;
        }

        // The outcome of CertifiedLeadingMinors that leaves the decision to the exact elimination.
        private const int Undecided = -1;

        // The radix grids, coarsest first, onto which the computed minimiser is rounded to form further integer
        // witnesses: an exact linear dependence with short coefficients (a repeated asset, an asset that is a short
        // combination of others) has a Schur complement of exactly zero, which only an exact null vector exhibits.
        private static readonly double[] s_witnessGrids =
            typeof(T) == typeof(decimal) ? [1.0, 10.0, 100.0, 1e4, 1e6] : [1.0, 4.0, 16.0, 256.0, 65536.0];

        /// <summary>
        /// Decides Sylvester's criterion for the equilibrated covariance <c>A = Σ̃</c> exactly as the elimination of
        /// <see cref="RequirePositiveDefinite"/> would, or returns <see cref="Undecided"/>: 0 when <c>A</c> is proved
        /// positive definite, <c>k ≥ 1</c> when the first leading principal minor that is not positive is proved to have
        /// order <c>k</c>. Cost O(n³) operations in <see cref="double"/> plus O(k²) exact integer products for a refusal.
        /// <para>
        /// <b>Positive leading blocks.</b> <c>Â = H + G</c> is a double-double rounding of <c>A</c>, exactly symmetric
        /// (<see cref="TryDoubleDouble"/>: exact for <c>T = double</c> and <c>float</c>, <c>|a_ij − ĥ_ij − ĝ_ij| ≤ 4·u_d·|ĝ_ij|</c>
        /// for <c>decimal</c>, <c>u_d = 2⁻⁵³</c>). <c>L = L_h + L_l</c> is any lower-triangular pair of finite double
        /// matrices computed from <c>Â − s·I</c> (by <see cref="DoubleDoubleCholesky"/>, whose accuracy the proof does not
        /// use), and <c>F = Â − s·I − L·L′</c>, exactly symmetric. For a leading block of order <c>m</c> whose columns were
        /// all computed, <c>A_m = L_m·L_m′ + s·I + F_m − (Â − A)_m</c> with <c>L_m·L_m′</c> positive semidefinite (the
        /// leading block of <c>L·L′</c> is <c>L_m·L_m′</c>), so by Weyl's inequality
        /// <c>λ_min(A_m) ≥ s − ‖F_m‖_∞ − ‖(Â − A)_m‖_∞</c>. Each <c>F_ij</c> (<c>j ≤ i</c>) is bounded by
        /// <see cref="ResidualBound"/> without any assumption on <c>L</c>: every product <c>l_h,ik·l_h,jk</c> is split
        /// exactly by <see cref="Math.FusedMultiplyAdd"/> into <c>p + e</c>, every subtraction of <c>p</c> from the running
        /// sum <c>S</c> is split exactly by Knuth's TwoSum into a new sum and its error <c>q</c>, so <c>F_ij</c> equals the
        /// final <c>S</c> plus the exact sum <c>Z</c> of the small terms <c>z</c> (<c>ĝ_ij</c>, the <c>q</c>, <c>−e</c> and
        /// <c>−x</c>, <c>x</c> the computed cross product <c>l_h·l_l + l_l·l_h + l_l·l_l</c>) minus the rounding of the
        /// <c>x</c>, at most <c>γ_3</c> times the sum <c>X</c> of their magnitudes (Higham 2002, §3.1),
        /// <c>γ_k = k·u_d/(1 − k·u_d)</c>. The at most <c>3n + 2</c> small terms are summed by cascaded TwoSum with their
        /// errors accumulated in double (Ogita, Rump and Oishi 2005, Algorithm 4.4, Sum2): the result <c>ẑ</c> satisfies
        /// <c>|ẑ − Z| ≤ u_d·|Z| + γ_{3n+6}²·Σ|z|</c> (their Proposition 4.5), so
        /// <c>|F_ij| ≤ |S + ẑ| + |Z − ẑ| + γ_3·X ≤ (1 + 2u_d)·|fl(S + ẑ)| + 2u_d·|ẑ| + 2γ_{3n+6}²·Σ|z| + γ_3·X</c>, which the
        /// computed bound exceeds. The bound is therefore the factor's actual residual plus a few <c>u_d²</c> of
        /// <c>Σ_k |l_ik|·|l_jk|</c>, not a multiple of <c>n·u_d</c> of the small terms' magnitudes. Gradual underflow adds at
        /// most <c>2⁻¹⁰⁷⁵</c> per product (four per <c>k</c>; TwoSum and every other sum stay exact or relatively bounded),
        /// under <c>n·10⁻³⁰⁰</c> per entry; an overflow leaves a non-finite bound, which stops the certificate. The row
        /// sums of the entry bounds and of the conversion bounds are formed with fewer than
        /// <c>n + 8</c> roundings of non-negative terms, a relative change below <c>10⁻⁶</c> that the safety factor 1.001
        /// absorbs. So a computed <c>1.001·max_i Σ_j (b_ij + c_ij) &lt; s</c> over the block proves
        /// <c>λ_min(A_m) &gt; 0</c>: every leading principal minor of order up to <c>m</c> is positive. The blocks grow one
        /// order at a time and their bounds only grow, so the first failure fixes the largest certified order <c>m</c>.
        /// </para>
        /// <para>
        /// <b>The first non-positive minor.</b> When <c>m &lt; n</c>, a witness <c>x = (−y, 1)</c> is formed from the
        /// minimiser <c>y ≈ A_m⁻¹·a</c> (<c>a</c> the column of order <c>m + 1</c> above the diagonal, solved with an
        /// unshifted double-double factor of <c>Â_m</c>), then from the roundings of <c>y</c> to a few coarse radix grids.
        /// Each <c>x</c> is an integer vector up to a positive power of two (or of the grid), and <c>x′·A_{m+1}·x</c> is
        /// evaluated exactly on the integer matrix of <c>A_{m+1}</c> over its finest radix grid. A value <c>≤ 0</c>, with
        /// <c>x ≠ 0</c>, proves that <c>A_{m+1}</c> is not positive definite; since <c>A_m</c> is proved positive definite,
        /// the Schur complement <c>det A_{m+1} / det A_m</c> is not positive, so the leading principal minors of orders 1
        /// to <c>m</c> are positive and the one of order <c>m + 1</c> is not: the elimination's first non-positive pivot,
        /// a positive multiple of that minor, is the one of order <c>m + 1</c>, and the refusal and its message are the
        /// elimination's. Any other outcome (a factorisation that fails, a non-finite value, no witness at or below zero)
        /// is <see cref="Undecided"/>.
        /// </para>
        /// <para>
        /// What it cannot decide, so the elimination still does: a leading block whose smallest eigenvalue is below the
        /// shift <c>s = 16·(n + 3)·u_d²·(max row sum + n·max diagonal)</c> (at most <c>32·(n + 3)·n·u_d²</c> of the
        /// largest diagonal entry: about 4·10⁻²⁷ of it at n = 100, 4·10⁻²⁵ at n = 1000), and a first non-positive minor that
        /// is exactly zero unless one of the integer witnesses is an exact null vector. Since <c>‖A⁻¹‖₂ ≤ ‖A⁻¹‖₁</c> for a
        /// symmetric matrix, a <c>decimal</c> covariance inside the conditioning guard's margin (<c>48·n·u·κ₁ &lt; 1</c>,
        /// <c>u = 10⁻²⁸</c>) has <c>λ_min &gt; 48·n·u·‖A‖₁</c>, above <c>s</c> for every n below about 24,000.
        /// </para>
        /// </summary>
        private static int CertifiedLeadingMinors(Exact[,] sigma, int n, RouteTrace? trace)
        {
            const double unit = 1.0 / 9007199254740992.0; // u_d = 2⁻⁵³
            const double safety = 1.001;
            var stage = trace?.LeadingMinors;
            var work = 0L;
            var inner = 0L; // the loop bodies of the factorisations, the entry bounds and the refusal witness it calls
            if (((6.0 * n) + 20.0) * unit > 1e-6)
            {
                return Recorded(stage, 0, n, work, Undecided);
            }

            var high = new double[n, n];
            var low = new double[n, n];
            for (var i = 0; i < n; i++)
            {
                work++;
                for (var j = 0; j <= i; j++)
                {
                    work++;
                    if (!TryDoubleDouble(sigma[i, j], out var h, out var g))
                    {
                        return Recorded(stage, 0, n, work, Undecided);
                    }

                    high[i, j] = h;
                    high[j, i] = h;
                    low[i, j] = g;
                    low[j, i] = g;
                }
            }

            double maxRow = 0.0, maxDiagonal = 0.0, maxLow = 0.0;
            for (var i = 0; i < n; i++)
            {
                work++;
                double row = 0.0, lowRow = 0.0;
                for (var j = 0; j < n; j++)
                {
                    work++;
                    row += Math.Abs(high[i, j]);
                    lowRow += Math.Abs(low[i, j]);
                }

                maxRow = Math.Max(maxRow, row);
                maxLow = Math.Max(maxLow, lowRow);
                maxDiagonal = Math.Max(maxDiagonal, high[i, i]);
            }

            // The shift is several times the a-priori size of the bound: the double-double factor's own residual is at most
            // about (n + 3)·u_d²·(|â_ij| + Σ_k |l_ik|·|l_jk|) per entry (each of at most n + 3 double-double operations
            // errs by a few u_d² of its operands), the bound adds a few u_d² of the same per entry, and
            // Σ_j Σ_k |l_ik|·|l_jk| ≤ n·max_i â_ii by Cauchy–Schwarz on the rows of L.
            var conversion = typeof(T) == typeof(decimal) ? 4.0 * unit : 0.0;
            var tiny = n * (double)n * 1e-300;
            var shift = (16.0 * (n + 3.0) * unit * unit * (maxRow + (n * maxDiagonal))) + (4.0 * conversion * maxLow) + (4.0 * tiny);
            if (!double.IsFinite(shift))
            {
                return Recorded(stage, 0, n, work, Undecided);
            }

            var lh = new double[n, n];
            var ll = new double[n, n];
            var columns = DoubleDoubleCholesky(high, low, shift, n, lh, ll, ref inner);

            var k = (3.0 * n) + 6.0;
            var gamma = k * unit / (1.0 - (k * unit));
            var square = 2.0 * gamma * gamma;
            var rows = new double[n];
            var worst = 0.0;
            var certified = 0;
            for (var t = 0; t < columns; t++)
            {
                work++;
                for (var j = 0; j <= t; j++)
                {
                    work++;
                    var entry = ResidualBound(high, low, lh, ll, t, j, t == j ? shift : 0.0, square, n, ref inner) + (conversion * Math.Abs(low[t, j]));
                    rows[t] += entry;
                    if (j != t)
                    {
                        rows[j] += entry;
                    }
                }

                for (var j = 0; j <= t; j++)
                {
                    work++;
                    worst = Math.Max(worst, rows[j]);
                }

                if (!double.IsFinite(worst) || !(safety * worst < shift))
                {
                    break;
                }

                certified = t + 1;
            }

            var order = certified == n ? 0 : RefutedOrder(sigma, high, low, certified, ref inner);
            return Recorded(stage, 0, n, work + inner, order);
        }

        // The order m + 1 when an integer witness x proves x′·A_{m+1}·x ≤ 0 for the certified positive-definite leading
        // block of order m (see CertifiedLeadingMinors), otherwise Undecided.
        private static int RefutedOrder(Exact[,] sigma, double[,] high, double[,] low, int m, ref long work)
        {
            var count = 0L;
            var yh = new double[m];
            var yl = new double[m];
            if (m > 0)
            {
                var lh = new double[m, m];
                var ll = new double[m, m];
                if (DoubleDoubleCholesky(high, low, 0.0, m, lh, ll, ref work) < m)
                {
                    work += count;
                    return Undecided;
                }

                // L·z = a, then L′·y = z, in double-double.
                for (var i = 0; i < m; i++)
                {
                    count++;
                    var (sh, sl) = (high[i, m], low[i, m]);
                    for (var k = 0; k < i; k++)
                    {
                        count++;
                        (sh, sl) = SubtractProduct(sh, sl, lh[i, k], ll[i, k], yh[k], yl[k]);
                    }

                    (yh[i], yl[i]) = DivideDoubleDouble(sh, sl, lh[i, i], ll[i, i]);
                }

                for (var i = m - 1; i >= 0; i--)
                {
                    count++;
                    var (sh, sl) = (yh[i], yl[i]);
                    for (var k = i + 1; k < m; k++)
                    {
                        count++;
                        (sh, sl) = SubtractProduct(sh, sl, lh[k, i], ll[k, i], yh[k], yl[k]);
                    }

                    (yh[i], yl[i]) = DivideDoubleDouble(sh, sl, lh[i, i], ll[i, i]);
                }

                for (var i = 0; i < m; i++)
                {
                    count++;
                    if (!double.IsFinite(yh[i]) || !double.IsFinite(yl[i]))
                    {
                        work += count;
                        return Undecided;
                    }
                }
            }

            var block = IntegerBlock(sigma, m + 1, ref work);
            if (QuadraticFormSign(block, BinaryWitness(yh, yl, ref work), ref work) <= 0)
            {
                work += count;
                return m + 1;
            }

            BigInteger[]? previous = null;
            foreach (var grid in s_witnessGrids)
            {
                count++;
                var x = new BigInteger[m + 1];
                var representable = true;
                for (var i = 0; i < m && representable; i++)
                {
                    count++;
                    var rounded = Math.Round((yh[i] * grid) + (yl[i] * grid));
                    representable = Math.Abs(rounded) < 4503599627370496.0; // 2⁵²
                    x[i] = representable ? -new BigInteger(rounded) : BigInteger.Zero;
                }

                x[m] = new BigInteger(grid);
                if (!representable || (previous is not null && SameDirection(previous, x, ref work)))
                {
                    continue;
                }

                if (QuadraticFormSign(block, x, ref work) <= 0)
                {
                    work += count;
                    return m + 1;
                }

                previous = x;
            }

            work += count;
            return Undecided;
        }

        // Whether two integer witnesses are positive multiples of each other's last component scaling (the same vector
        // up to the grid), so that evaluating the second would repeat the first.
        private static bool SameDirection(BigInteger[] first, BigInteger[] second, ref long work)
        {
            var last = first.Length - 1;
            for (var i = 0; i < last; i++)
            {
                work++;
                if (first[i] * second[last] != second[i] * first[last])
                {
                    return false;
                }
            }

            return true;
        }

        // The lower triangle of the leading block of order m of A as integers over its finest radix grid: a positive
        // multiple of the exact block.
        private static BigInteger[,] IntegerBlock(Exact[,] sigma, int m, ref long work)
        {
            var count = 0L;
            var finest = int.MaxValue;
            for (var i = 0; i < m; i++)
            {
                count++;
                for (var j = 0; j <= i; j++)
                {
                    count++;
                    if (sigma[i, j].Sign != 0)
                    {
                        finest = Math.Min(finest, sigma[i, j].Exponent);
                    }
                }
            }

            var block = new BigInteger[m, m];
            for (var i = 0; i < m; i++)
            {
                count++;
                for (var j = 0; j <= i; j++)
                {
                    count++;
                    var entry = sigma[i, j];
                    block[i, j] = entry.Sign == 0 ? BigInteger.Zero : entry.Mantissa * Exact.RadixPower(entry.Exponent - finest);
                }
            }

            work += count;
            return block;
        }

        // The sign of x′·B·x for the symmetric integer matrix B given by its lower triangle, exactly.
        private static int QuadraticFormSign(BigInteger[,] block, BigInteger[] x, ref long work)
        {
            var count = 0L;
            var total = BigInteger.Zero;
            for (var i = 0; i < x.Length; i++)
            {
                count++;
                if (x[i].IsZero)
                {
                    continue;
                }

                var off = BigInteger.Zero;
                for (var j = 0; j < i; j++)
                {
                    count++;
                    if (!x[j].IsZero)
                    {
                        off += block[i, j] * x[j];
                    }
                }

                total += x[i] * ((block[i, i] * x[i]) + (off << 1));
            }

            work += count;
            return total.Sign;
        }

        // x = (−(y_h + y_l), 1) as integers times one common power of two: every finite double is an integer times a
        // power of two, so the vector is exact.
        private static BigInteger[] BinaryWitness(double[] yh, double[] yl, ref long work)
        {
            var m = yh.Length;
            var finest = 0;
            for (var i = 0; i < m; i++)
            {
                work++;
                finest = Math.Min(finest, Math.Min(Lsb(yh[i]), Lsb(yl[i])));
            }

            var x = new BigInteger[m + 1];
            for (var i = 0; i < m; i++)
            {
                work++;
                x[i] = -(Scaled(yh[i], finest) + Scaled(yl[i], finest));
            }

            x[m] = BigInteger.One << -finest;
            return x;

            static int Lsb(double value)
            {
                if (value == 0.0)
                {
                    return 0;
                }

                var raw = BitConverter.DoubleToInt64Bits(value);
                var biased = (int)((raw >> 52) & 0x7FF);
                return (biased == 0 ? 1 : biased) - 1075;
            }

            static BigInteger Scaled(double value, int exponent)
            {
                if (value == 0.0)
                {
                    return BigInteger.Zero;
                }

                var raw = BitConverter.DoubleToInt64Bits(value);
                var biased = (int)((raw >> 52) & 0x7FF);
                var fraction = raw & 0xFFFFFFFFFFFFFL;
                var mantissa = biased == 0 ? fraction : fraction | (1L << 52);
                var magnitude = new BigInteger(mantissa) << ((biased == 0 ? 1 : biased) - 1075 - exponent);
                return raw < 0 ? -magnitude : magnitude;
            }
        }

        /// <summary>
        /// A double-double rounding <c>ĥ + ĝ</c> of an exact entry. For <c>T = double</c> or <c>float</c> the entry is a
        /// double and <c>ĝ = 0</c>. For <c>decimal</c> (<c>m·10^e</c>, <c>|m| &lt; 2⁹⁶</c>, <c>−28 ≤ e ≤ 0</c>)
        /// <c>ĥ = </c><see cref="CertifiedDouble"/>, the remainder <c>r = a − ĥ</c> is formed exactly as an integer
        /// <c>R</c> times <c>2^q / 10^s</c> (<c>|R| &lt; 2⁹⁶</c>), and <c>ĝ</c> is <c>R</c> split into two exact 48-bit halves,
        /// summed with one rounding, scaled exactly by <c>2^q</c> and divided by at most two exact powers of ten: three
        /// correctly rounded operations, so <c>|r − ĝ| ≤ ((1 + u_d)³ − 1)·|r| ≤ 4·u_d·|ĝ|</c>. Returns
        /// <see langword="false"/> for anything else.
        /// </summary>
        private static bool TryDoubleDouble(Exact value, out double high, out double low)
        {
            low = 0.0;
            high = CertifiedDouble(value);
            if (!double.IsFinite(high))
            {
                return false;
            }

            if (typeof(T) != typeof(decimal) || value.Sign == 0)
            {
                return true;
            }

            var raw = BitConverter.DoubleToInt64Bits(high);
            var biased = (int)((raw >> 52) & 0x7FF);
            var fraction = raw & 0xFFFFFFFFFFFFFL;
            var hm = new BigInteger(biased == 0 ? fraction : fraction | (1L << 52));
            if (raw < 0)
            {
                hm = -hm;
            }

            var q = (biased == 0 ? 1 : biased) - 1075;
            var s = -value.Exponent;
            var tenToS = Exact.RadixPower(s);

            // a − ĥ = R·2^min(q, 0) / 10^s with R an integer.
            var remainder = q >= 0
                ? value.Mantissa - ((hm << q) * tenToS)
                : (value.Mantissa << -q) - (hm * tenToS);
            if (remainder.IsZero)
            {
                return true;
            }

            var magnitude = BigInteger.Abs(remainder);
            if (magnitude.GetBitLength() > 96)
            {
                return false;
            }

            var result = Math.ScaleB((double)(magnitude >> 48), 48) + (double)(magnitude & ((BigInteger.One << 48) - 1));
            result = Math.ScaleB(result, Math.Min(q, 0));
            if (s > 22)
            {
                result /= 1e22;
                s -= 22;
            }

            result /= s_exactPowersOfTen[s];
            low = remainder.Sign < 0 ? -result : result;
            return double.IsFinite(low);
        }

        // A lower-triangular double-double factor L = L_h + L_l of the leading block of order size of Ĥ + Ĝ − shift·I by
        // Cholesky with every inner product accumulated in double-double; returns the number of columns computed (the
        // first column whose pivot is not positive and finite stops it). Its accuracy matters only to how often the
        // certificates succeed, never to their soundness.
        private static int DoubleDoubleCholesky(double[,] high, double[,] low, double shift, int size, double[,] lh, double[,] ll, ref long work)
        {
            var count = 0L;
            for (var j = 0; j < size; j++)
            {
                count++;
                var (dh, dl) = TwoSum(high[j, j], -shift);
                dl += low[j, j];
                for (var k = 0; k < j; k++)
                {
                    count++;
                    (dh, dl) = SubtractProduct(dh, dl, lh[j, k], ll[j, k], lh[j, k], ll[j, k]);
                }

                (dh, dl) = TwoSum(dh, dl);
                if (!(dh > 0.0) || !double.IsFinite(dh) || !double.IsFinite(dl))
                {
                    work += count;
                    return j;
                }

                var root = Math.Sqrt(dh);
                var square = root * root;
                var correction = (((dh - square) - Math.FusedMultiplyAdd(root, root, -square)) + dl) / (2.0 * root);
                var (rh, rl) = TwoSum(root, correction);
                lh[j, j] = rh;
                ll[j, j] = rl;
                for (var i = j + 1; i < size; i++)
                {
                    count++;
                    var (sh, sl) = (high[i, j], low[i, j]);
                    for (var k = 0; k < j; k++)
                    {
                        count++;
                        (sh, sl) = SubtractProduct(sh, sl, lh[i, k], ll[i, k], lh[j, k], ll[j, k]);
                    }

                    var (qh, ql) = DivideDoubleDouble(sh, sl, rh, rl);
                    if (!double.IsFinite(qh) || !double.IsFinite(ql))
                    {
                        work += count;
                        return j;
                    }

                    lh[i, j] = qh;
                    ll[i, j] = ql;
                }
            }

            work += count;
            return size;
        }

        // (sh + sl) − (ah + al)·(bh + bl) to about double-double accuracy, the high part carried by TwoSum.
        private static (double High, double Low) SubtractProduct(double sh, double sl, double ah, double al, double bh, double bl)
        {
            var p = ah * bh;
            var e = Math.FusedMultiplyAdd(ah, bh, -p) + (ah * bl) + (al * bh);
            var (s, q) = TwoSum(sh, -p);
            return (s, sl + (q - e));
        }

        private static (double High, double Low) DivideDoubleDouble(double ah, double al, double bh, double bl)
        {
            var (nh, nl) = TwoSum(ah, al);
            var quotient = nh / bh;
            var product = quotient * bh;
            var correction = (((nh - product) - Math.FusedMultiplyAdd(quotient, bh, -product)) + nl - (quotient * bl)) / bh;
            return TwoSum(quotient, correction);
        }

        // Knuth's TwoSum: s = fl(a + b) and the exact error a + b − s (round to nearest, no overflow).
        private static (double Sum, double Error) TwoSum(double a, double b)
        {
            var s = a + b;
            var bb = s - a;
            return (s, (a - (s - bb)) + (b - bb));
        }

        // A rigorous bound on |F_tj| for F = Ĥ + Ĝ − shift·I − L·L′ (j ≤ t), L = L_h + L_l: see CertifiedLeadingMinors.
        // square is 2·γ_k² for k at least the number of small terms.
        private static double ResidualBound(double[,] high, double[,] low, double[,] lh, double[,] ll, int t, int j, double shift, double square, int n, ref long work)
        {
            const double unit = 1.0 / 9007199254740992.0;
            var count = 0L;
            var sum = high[t, j];
            var small = low[t, j];
            var error = 0.0;
            var magnitude = Math.Abs(small);
            var cross = 0.0;
            if (shift != 0.0)
            {
                var (s, q) = TwoSum(sum, -shift);
                sum = s;
                (small, var r) = TwoSum(small, q);
                error += r;
                magnitude += Math.Abs(q);
            }

            for (var k = 0; k <= j; k++)
            {
                count++;
                double a = lh[t, k], b = lh[j, k], c = ll[t, k], d = ll[j, k];
                var p = a * b;
                var e = Math.FusedMultiplyAdd(a, b, -p);
                var x = (a * d) + (c * b) + (c * d);
                var (s, q) = TwoSum(sum, -p);
                sum = s;
                (small, var r1) = TwoSum(small, q);
                (small, var r2) = TwoSum(small, -e);
                (small, var r3) = TwoSum(small, -x);
                error += r1 + r2 + r3;
                magnitude += Math.Abs(q) + Math.Abs(e) + Math.Abs(x);
                cross += (Math.Abs(a) * Math.Abs(d)) + (Math.Abs(c) * Math.Abs(b)) + (Math.Abs(c) * Math.Abs(d));
            }

            work += count;
            var total = small + error;
            var bound = ((Math.Abs(sum + total) + (2.0 * unit * Math.Abs(total))) * (1.0 + (4.0 * unit))) + (square * magnitude) + (4.0 * unit * cross) + (n * 1e-300);
            return double.IsFinite(bound) ? bound : double.PositiveInfinity;
        }
    }

    /// <summary>
    /// An exact radix fraction <c>Mantissa · radix^Exponent</c> (radix 2 for IEEE types, 10 for
    /// <see cref="decimal"/>): every finite value of <typeparamref name="T"/> is one, and sums, differences and
    /// products of them are formed without rounding. <see cref="ToT"/> rounds once to nearest (ties to even).
    /// </summary>
    private readonly struct Exact : IComparable<Exact>
    {
        private static readonly bool s_decimal = typeof(T) == typeof(decimal);
        private static readonly bool s_float = typeof(T) == typeof(float);
        private static readonly BigInteger s_decimalLimit = (BigInteger.One << 96) - 1;
        private static readonly BigInteger[] s_powersOfTen = BuildPowersOfTen(160);

        private Exact(BigInteger mantissa, int exponent)
        {
            Mantissa = mantissa;
            Exponent = mantissa.IsZero ? 0 : exponent;
        }

        public static Exact Zero => default;

        public static Exact One => new(BigInteger.One, 0);

        public BigInteger Mantissa { get; }

        public int Exponent { get; }

        public int Sign => Mantissa.Sign;

        /// <summary>The unit roundoff <c>u</c> (<c>double</c> <c>2⁻⁵³</c>, <c>float</c> <c>2⁻²⁴</c>, <c>decimal</c> <c>1e-28</c>).</summary>
        public static Exact UnitRoundoff => s_decimal ? new Exact(1, -28) : new Exact(1, s_float ? -24 : -53);

        /// <summary><c>1/u</c> as an integer (<c>double</c> <c>2⁵³</c>, <c>float</c> <c>2²⁴</c>, <c>decimal</c> <c>10²⁸</c>).</summary>
        public static BigInteger InverseUnitRoundoff => s_decimal ? RadixPower(28) : RadixPower(s_float ? 24 : 53);

        public static Exact FromInt(int value) => new(value, 0);

        /// <summary>
        /// About the square root of the unit roundoff (<c>double</c> <c>2⁻²⁶</c>, <c>float</c> <c>2⁻¹²</c>,
        /// <c>decimal</c> <c>1e-14</c>): the fraction of a row's rounding bound below which a refined multiplier
        /// counts as negative.
        /// </summary>
        public static Exact ReleaseFraction => s_decimal ? new Exact(1, -14) : new Exact(1, s_float ? -12 : -26);

        public static Exact From(T value)
        {
            if (s_decimal)
            {
                var d = (decimal)(object)value;
                Span<int> bits = stackalloc int[4];
                decimal.GetBits(d, bits);
                var magnitude = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | new BigInteger((uint)bits[0]);
                var scale = (bits[3] >> 16) & 0xFF;
                return new Exact(bits[3] < 0 ? -magnitude : magnitude, -scale);
            }

            var x = s_float ? (float)(object)value : (double)(object)value;
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

            return new Exact(raw < 0 ? -(BigInteger)fraction : fraction, biased - 1075);
        }

        /// <summary>Half the spacing of <typeparamref name="T"/>'s grid at <c>|value|</c>: the rounding bound of one weight.</summary>
        public static Exact HalfUnit(T value)
        {
            if (s_decimal)
            {
                var d = (decimal)(object)value;

                // Below about 7.92 the finest grid (scale 28) holds the value with room for ten more units.
                if (Math.Abs(d) <= 7.9228162514264337593543950m)
                {
                    return new Exact(5, -29);
                }

                Span<int> bits = stackalloc int[4];
                decimal.GetBits(d, bits);
                var magnitude = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | new BigInteger((uint)bits[0]);
                var scale = (bits[3] >> 16) & 0xFF;

                // Finest scale s <= 28 with |value|·10^s + 10 within the 96-bit mantissa.
                var s = 28;
                while (true)
                {
                    var t = s - scale;
                    var fits = t >= 0
                        ? (magnitude * RadixPower(t)) + 10 <= s_decimalLimit
                        : magnitude + (10 * RadixPower(-t)) <= s_decimalLimit * RadixPower(-t);
                    if (fits)
                    {
                        break;
                    }

                    s--;
                }

                return new Exact(5, -(s + 1));
            }

            if (s_float)
            {
                var biasedSingle = (BitConverter.SingleToInt32Bits((float)(object)value) >> 23) & 0xFF;
                return new Exact(BigInteger.One, Math.Max(biasedSingle, 1) - 151);
            }

            var biased = (int)((BitConverter.DoubleToInt64Bits((double)(object)value) >> 52) & 0x7FF);
            return new Exact(BigInteger.One, Math.Max(biased, 1) - 1076);
        }

        public static Exact Sum(Exact[] values)
        {
            var sum = Zero;
            foreach (var value in values)
            {
                sum += value;
            }

            return sum;
        }

        /// <summary><c>radix^k</c> for <c>k ≥ 0</c>.</summary>
        public static BigInteger RadixPower(int k) =>
            s_decimal ? (k < s_powersOfTen.Length ? s_powersOfTen[k] : BigInteger.Pow(10, k)) : BigInteger.One << k;

        /// <summary><c>mantissa · radix^exponent</c>, exactly.</summary>
        public static Exact Of(BigInteger mantissa, int exponent) => new(mantissa, exponent);

        /// <summary><c>numerator / denominator</c> to at least 66 bits (31 digits), the discarded remainder kept as a sticky unit.</summary>
        public static Exact Quotient(Exact numerator, Exact denominator) => Quotient(numerator, denominator, s_decimal ? 31 : 66);

        /// <summary>
        /// <c>numerator / denominator</c> for a positive integer denominator to at least 256 bits (120 digits), about
        /// four times the working precision, the discarded remainder kept as a sticky unit; rounding the result once into
        /// <typeparamref name="T"/> with <see cref="ToT"/> gives the correctly rounded quotient.
        /// </summary>
        public static Exact Precise(Exact numerator, BigInteger denominator) =>
            Quotient(numerator, new Exact(denominator, 0), s_decimal ? 120 : 256);

        private static Exact Quotient(Exact numerator, Exact denominator, int precision)
        {
            if (numerator.Sign == 0)
            {
                return Zero;
            }

            var top = BigInteger.Abs(numerator.Mantissa);
            var bottom = BigInteger.Abs(denominator.Mantissa);
            int extra;
            BigInteger scaled;
            if (s_decimal)
            {
                extra = Math.Max(0, DigitCount(bottom) - DigitCount(top) + precision);
                scaled = top * RadixPower(extra);
            }
            else
            {
                extra = (int)Math.Max(0, bottom.GetBitLength() - top.GetBitLength() + precision);
                scaled = top << extra;
            }

            var quotient = BigInteger.DivRem(scaled, bottom, out var remainder);
            var sticky = remainder.IsZero ? BigInteger.Zero : BigInteger.One;
            var mantissa = (quotient * (s_decimal ? 10 : 2)) + sticky;
            var negative = numerator.Sign != denominator.Sign;
            return new Exact(negative ? -mantissa : mantissa, numerator.Exponent - denominator.Exponent - extra - 1);
        }

        /// <summary>This value times <c>radix^k</c>, exactly.</summary>
        public Exact Shift(int k) => new(Mantissa, Exponent + k);

        /// <summary>For a nonzero value, the <c>k</c> with <c>radix^(k−1) ≤ |value| &lt; radix^k</c>.</summary>
        public int MagnitudeExponent()
        {
            var m = BigInteger.Abs(Mantissa);
            return Exponent + (s_decimal ? DigitCount(m) : (int)m.GetBitLength());
        }

        /// <summary>
        /// The nearest <typeparamref name="T"/> (ties to even). An unrepresentable magnitude follows the type's
        /// convention: <c>±∞</c> for IEEE types, <see cref="OverflowException"/> for <see cref="decimal"/>.
        /// </summary>
        public T ToT()
        {
            if (Mantissa.IsZero)
            {
                return T.Zero;
            }

            var negative = Mantissa.Sign < 0;
            var m = BigInteger.Abs(Mantissa);
            if (s_decimal)
            {
                return (T)(object)ToDecimal(m, Exponent, negative);
            }

            var precision = s_float ? 24 : 53;
            var smallest = s_float ? -149 : -1074;
            var length = (int)m.GetBitLength();
            var lead = Exponent + length - 1;
            var bits = Math.Min(precision, lead - smallest + 1);
            BigInteger q;
            int exponent;
            if (bits <= 0)
            {
                // Below the smallest subnormal: round to zero or to it (an exact half ties to zero).
                q = lead == smallest - 1 && !m.IsPowerOfTwo ? BigInteger.One : BigInteger.Zero;
                exponent = smallest;
            }
            else
            {
                var drop = length - bits;
                if (drop > 0)
                {
                    q = m >> drop;
                    var remainder = m - (q << drop);
                    var half = BigInteger.One << (drop - 1);
                    if (remainder > half || (remainder == half && !q.IsEven))
                    {
                        q++;
                    }
                }
                else
                {
                    q = m << -drop;
                }

                exponent = Exponent + drop;
            }

            var magnitude = s_float
                ? (T)(object)MathF.ScaleB((long)q, exponent)
                : (T)(object)Math.ScaleB((long)q, exponent);
            return negative ? -magnitude : magnitude;
        }

        public static Exact operator +(Exact a, Exact b)
        {
            if (a.Mantissa.IsZero)
            {
                return b;
            }

            if (b.Mantissa.IsZero)
            {
                return a;
            }

            var e = Math.Min(a.Exponent, b.Exponent);
            return new Exact(Align(a, e) + Align(b, e), e);
        }

        public static Exact operator -(Exact a) => new(-a.Mantissa, a.Exponent);

        public static Exact operator -(Exact a, Exact b) => a + (-b);

        public static Exact operator *(Exact a, Exact b) =>
            a.Mantissa.IsZero || b.Mantissa.IsZero ? Zero : new Exact(a.Mantissa * b.Mantissa, a.Exponent + b.Exponent);

        public static bool operator <(Exact a, Exact b) => a.CompareTo(b) < 0;

        public static bool operator >(Exact a, Exact b) => a.CompareTo(b) > 0;

        public static bool operator <=(Exact a, Exact b) => a.CompareTo(b) <= 0;

        public static bool operator >=(Exact a, Exact b) => a.CompareTo(b) >= 0;

        public Exact Abs() => Mantissa.Sign < 0 ? -this : this;

        /// <summary>Half this value, exactly (radix 2: one exponent step; radix 10: five tenths).</summary>
        public Exact Half() => s_decimal ? new Exact(Mantissa * 5, Exponent - 1) : new Exact(Mantissa, Exponent - 1);

        public int CompareTo(Exact other) => (this - other).Mantissa.Sign;

        private static BigInteger Align(Exact x, int exponent)
        {
            var shift = x.Exponent - exponent;
            return shift == 0 ? x.Mantissa : s_decimal ? x.Mantissa * RadixPower(shift) : x.Mantissa << shift;
        }

        private static decimal ToDecimal(BigInteger m, int exponent, bool negative)
        {
            BigInteger q;
            int scale;
            if (exponent >= 0)
            {
                q = m * RadixPower(exponent);
                scale = 0;
                if (q > s_decimalLimit)
                {
                    throw new OverflowException("Value is beyond the decimal range.");
                }
            }
            else
            {
                scale = Math.Min(28, -exponent);
                while (true)
                {
                    var drop = -exponent - scale;
                    q = drop == 0 ? m : RoundedQuotient(m, RadixPower(drop));
                    if (q <= s_decimalLimit)
                    {
                        break;
                    }

                    if (scale == 0)
                    {
                        throw new OverflowException("Value is beyond the decimal range.");
                    }

                    scale--;
                }
            }

            return new decimal(
                (int)(uint)(q & uint.MaxValue), (int)(uint)((q >> 32) & uint.MaxValue), (int)(uint)(q >> 64), negative && !q.IsZero, (byte)scale);
        }

        private static BigInteger RoundedQuotient(BigInteger value, BigInteger divisor)
        {
            var q = BigInteger.DivRem(value, divisor, out var remainder);
            var twice = remainder << 1;
            if (twice > divisor || (twice == divisor && !q.IsEven))
            {
                q++;
            }

            return q;
        }

        // Number of decimal digits of m > 0.
        private static int DigitCount(BigInteger m)
        {
            var d = (int)(m.GetBitLength() * 30103 / 100000);
            while (d > 0 && m < RadixPower(d - 1))
            {
                d--;
            }

            while (m >= RadixPower(d))
            {
                d++;
            }

            return d;
        }

        private static BigInteger[] BuildPowersOfTen(int count)
        {
            var powers = new BigInteger[count];
            powers[0] = BigInteger.One;
            for (var k = 1; k < count; k++)
            {
                powers[k] = powers[k - 1] * 10;
            }

            return powers;
        }

        /// <summary>
        /// A matrix of exact values prepared for <see cref="Times"/>: the magnitude of every entry split once into
        /// 64-bit limbs beside its sign and exponent, so that a product <c>A·x</c> forms each row in one fixed-width
        /// two's-complement integer aligned at the row's smallest product exponent (radix 2: a shift; radix 10: a
        /// multiplication by a power of ten) instead of one <see cref="BigInteger"/> product and one sum per term. The
        /// accumulator holds the largest partial sum (below <c>count·2^top</c>) with its sign, so no addition overflows;
        /// a row whose width would exceed <see cref="MaxWidthBits"/> is summed term by term instead.
        /// </summary>
        public sealed class ProductMatrix
        {
            // Rows wider than this are summed term by term (an exponent spread no solve's data reaches).
            private const long MaxWidthBits = 1L << 20;

            // 10^k for k = 0 … 19, the radix-10 alignment factors that fit in one limb.
            private static readonly ulong[] s_limbPowersOfTen = BuildLimbPowersOfTen();

            private readonly Exact[,] _matrix;
            private readonly Limbs _entries;
            private readonly int _columns;

            public ProductMatrix(Exact[,] matrix)
            {
                _matrix = matrix;
                _columns = matrix.GetLength(1);
                var values = new Exact[matrix.Length];
                var k = 0;
                for (var i = 0; i < matrix.GetLength(0); i++)
                {
                    for (var j = 0; j < _columns; j++)
                    {
                        values[k++] = matrix[i, j];
                    }
                }

                _entries = new Limbs(values);
            }

            /// <summary>
            /// <c>A·x</c>: each row the same <see cref="Exact"/>, mantissa and exponent, as adding the products
            /// <c>a_ij·x_j</c> to <see cref="Zero"/> with <c>+</c> in index order. That sum carries the smallest exponent
            /// among the nonzero products added since its last partial sum that was exactly zero (a zero sum takes the
            /// exponent of the next product), and each row returns that exponent.
            /// </summary>
            public Exact[] Times(Exact[] vector)
            {
                var rows = _matrix.GetLength(0);
                var product = new Exact[rows];
                var x = new Limbs(vector);
                var scratch = new Scratch();
                for (var i = 0; i < rows; i++)
                {
                    product[i] = Row(i, vector, x, scratch);
                }

                return product;
            }

            private Exact Row(int i, Exact[] vector, Limbs x, Scratch scratch)
            {
                var a = _entries;
                var n = _columns;
                var start = i * n;
                var lowest = int.MaxValue;
                var count = 0;
                var highest = long.MinValue;
                for (var j = 0; j < n; j++)
                {
                    if (a.Length[start + j] > 0 && x.Length[j] > 0)
                    {
                        var exponent = a.Exponent[start + j] + x.Exponent[j];
                        lowest = Math.Min(lowest, exponent);
                        highest = Math.Max(highest, exponent + a.Bits[start + j] + x.Bits[j]);
                        count++;
                    }
                }

                if (count == 0)
                {
                    return Zero;
                }

                // Every aligned product lies below 2^top, so the sum of count of them, with its sign, fits in widthBits.
                var top = highest - lowest;
                if (s_decimal)
                {
                    top = 0;
                    for (var j = 0; j < n; j++)
                    {
                        if (a.Length[start + j] > 0 && x.Length[j] > 0)
                        {
                            var shift = (long)(a.Exponent[start + j] + x.Exponent[j]) - lowest;
                            top = Math.Max(top, a.Bits[start + j] + x.Bits[j] + (shift * 3322 / 1000) + 2);
                        }
                    }
                }

                var widthBits = top + BitOperations.Log2((uint)count) + 2;
                if (widthBits > MaxWidthBits || !BitConverter.IsLittleEndian)
                {
                    return TermByTerm(i, vector);
                }

                // Two spare limbs: a two-limb product is added three limbs at a time.
                var width = (int)(widthBits / 64) + 3;
                Ensure(ref scratch.Sum, width);
                Ensure(ref scratch.Term, width + 1);
                var sum = scratch.Sum.AsSpan(0, width);
                sum.Clear();

                // The smallest exponent among the products added since the last partial sum that was exactly zero.
                var restart = int.MaxValue;
                for (var j = 0; j < n; j++)
                {
                    var entry = start + j;
                    if (a.Length[entry] == 0 || x.Length[j] == 0)
                    {
                        continue;
                    }

                    var exponent = a.Exponent[entry] + x.Exponent[j];
                    var shift = exponent - lowest;
                    var negative = a.Negative[entry] != x.Negative[j];
                    if (!s_decimal && a.Length[entry] == 1 && x.Length[j] == 1)
                    {
                        // One limb by one limb, the common case of binary data.
                        var high = Math.BigMul(a.Values[a.Offset[entry]], x.Values[x.Offset[j]], out var low);
                        AccumulateTwo(sum, low, high, shift, negative);
                    }
                    else
                    {
                        var length = Multiply(a, entry, x, j, scratch);
                        if (s_decimal)
                        {
                            length = ScaleByPowerOfTen(scratch.Term, length, shift);
                            shift = 0;
                        }

                        Accumulate(sum, scratch.Term.AsSpan(0, length), shift, negative);
                    }

                    restart = Math.Min(restart, exponent);
                    if (!sum.ContainsAnyExcept(0UL))
                    {
                        restart = int.MaxValue;
                    }
                }

                if (restart == int.MaxValue)
                {
                    return Zero;
                }

                var value = new BigInteger(MemoryMarshal.AsBytes(sum), isUnsigned: false, isBigEndian: false);
                var drop = restart - lowest;
                if (drop > 0)
                {
                    value = s_decimal ? value / RadixPower(drop) : value >> drop;
                }

                return new Exact(value, restart);
            }

            private static ulong[] BuildLimbPowersOfTen()
            {
                var powers = new ulong[20];
                powers[0] = 1;
                for (var k = 1; k < powers.Length; k++)
                {
                    powers[k] = powers[k - 1] * 10;
                }

                return powers;
            }

            private static void Ensure(ref ulong[] buffer, int length)
            {
                if (buffer.Length < length)
                {
                    buffer = new ulong[Math.Max(length, 2 * buffer.Length)];
                }
            }

            // Adds (or subtracts) term·2^shift to the two's-complement sum, carrying (borrowing) as far as needed.
            private static void Accumulate(Span<ulong> sum, ReadOnlySpan<ulong> term, int shift, bool negative)
            {
                var k = shift >> 6;
                var bitShift = shift & 63;
                ulong carry = 0;
                ulong previous = 0;
                for (var t = 0; t <= term.Length; t++)
                {
                    var current = t < term.Length ? term[t] : 0UL;
                    var limb = bitShift == 0 ? current : (current << bitShift) | (previous >> (64 - bitShift));
                    previous = current;
                    if (t == term.Length && limb == 0)
                    {
                        break;
                    }

                    carry = negative ? SubtractLimb(sum, k, limb, carry) : AddLimb(sum, k, limb, carry);
                    k++;
                }

                while (carry != 0 && k < sum.Length)
                {
                    carry = negative ? SubtractLimb(sum, k, 0, carry) : AddLimb(sum, k, 0, carry);
                    k++;
                }
            }

            // Adds (or subtracts) (high·2^64 + low)·2^shift to the two's-complement sum.
            private static void AccumulateTwo(Span<ulong> sum, ulong low, ulong high, int shift, bool negative)
            {
                var k = shift >> 6;
                var bitShift = shift & 63;
                ulong first = low, second = high, third = 0;
                if (bitShift != 0)
                {
                    first = low << bitShift;
                    second = (high << bitShift) | (low >> (64 - bitShift));
                    third = high >> (64 - bitShift);
                }

                ulong carry;
                if (negative)
                {
                    carry = SubtractLimb(sum, k, first, 0);
                    carry = SubtractLimb(sum, k + 1, second, carry);
                    carry = SubtractLimb(sum, k + 2, third, carry);
                }
                else
                {
                    carry = AddLimb(sum, k, first, 0);
                    carry = AddLimb(sum, k + 1, second, carry);
                    carry = AddLimb(sum, k + 2, third, carry);
                }

                for (k += 3; carry != 0 && k < sum.Length; k++)
                {
                    carry = negative ? SubtractLimb(sum, k, 0, carry) : AddLimb(sum, k, 0, carry);
                }
            }

            private static ulong AddLimb(Span<ulong> sum, int k, ulong value, ulong carry)
            {
                var partial = sum[k] + value;
                var next = partial < value ? 1UL : 0UL;
                var total = partial + carry;
                next += total < partial ? 1UL : 0UL;
                sum[k] = total;
                return next;
            }

            private static ulong SubtractLimb(Span<ulong> sum, int k, ulong value, ulong borrow)
            {
                var current = sum[k];
                var partial = current - value;
                var next = current < value ? 1UL : 0UL;
                var total = partial - borrow;
                next += partial < borrow ? 1UL : 0UL;
                sum[k] = total;
                return next;
            }

            // |a_entry|·|x_j| into the scratch term; returns its length in limbs without leading zero limbs.
            private static int Multiply(Limbs a, int entry, Limbs x, int j, Scratch scratch)
            {
                var left = a.Values.AsSpan(a.Offset[entry], a.Length[entry]);
                var right = x.Values.AsSpan(x.Offset[j], x.Length[j]);
                var length = left.Length + right.Length;
                Ensure(ref scratch.Term, length);
                var term = scratch.Term.AsSpan(0, length);
                if (length == 2)
                {
                    term[1] = Math.BigMul(left[0], right[0], out term[0]);
                }
                else
                {
                    term.Clear();
                    for (var p = 0; p < left.Length; p++)
                    {
                        ulong carry = 0;
                        for (var q = 0; q < right.Length; q++)
                        {
                            var high = Math.BigMul(left[p], right[q], out var low);
                            var partial = low + term[p + q];
                            high += partial < low ? 1UL : 0UL;
                            var total = partial + carry;
                            high += total < partial ? 1UL : 0UL;
                            term[p + q] = total;
                            carry = high;
                        }

                        term[p + right.Length] = carry;
                    }
                }

                while (length > 0 && term[length - 1] == 0)
                {
                    length--;
                }

                return length;
            }

            // The first length limbs of term times 10^power, in place; returns the new length.
            private static int ScaleByPowerOfTen(ulong[] term, int length, int power)
            {
                while (power > 0)
                {
                    var step = Math.Min(power, 19);
                    var factor = s_limbPowersOfTen[step];
                    ulong carry = 0;
                    for (var t = 0; t < length; t++)
                    {
                        var high = Math.BigMul(term[t], factor, out var low);
                        low += carry;
                        high += low < carry ? 1UL : 0UL;
                        term[t] = low;
                        carry = high;
                    }

                    if (carry != 0)
                    {
                        term[length++] = carry;
                    }

                    power -= step;
                }

                return length;
            }

            // The reference evaluation: one Exact product and one sum per term.
            private Exact TermByTerm(int i, Exact[] vector)
            {
                var sum = Zero;
                for (var j = 0; j < vector.Length; j++)
                {
                    sum += _matrix[i, j] * vector[j];
                }

                return sum;
            }

            // The working buffers of one product, grown as a row needs.
            private sealed class Scratch
            {
                public ulong[] Sum = new ulong[4];

                public ulong[] Term = new ulong[4];
            }

            // Exact values as little-endian 64-bit limbs of their magnitudes (none for zero), signs and exponents.
            private sealed class Limbs
            {
                public Limbs(Exact[] values)
                {
                    var count = values.Length;
                    Offset = new int[count];
                    Length = new int[count];
                    Bits = new long[count];
                    Exponent = new int[count];
                    Negative = new bool[count];
                    var total = 0;
                    for (var k = 0; k < count; k++)
                    {
                        Bits[k] = BigInteger.Abs(values[k].Mantissa).GetBitLength();
                        Length[k] = (int)((Bits[k] + 63) / 64);
                        Offset[k] = total;
                        Exponent[k] = values[k].Exponent;
                        Negative[k] = values[k].Sign < 0;
                        total += Length[k];
                    }

                    Values = new ulong[total];
                    for (var k = 0; k < count; k++)
                    {
                        var magnitude = BigInteger.Abs(values[k].Mantissa);
                        if (Bits[k] > 64)
                        {
                            _ = magnitude.TryWriteBytes(MemoryMarshal.AsBytes(Values.AsSpan(Offset[k], Length[k])), out _, isUnsigned: true, isBigEndian: false);
                        }
                        else if (Length[k] > 0)
                        {
                            Values[Offset[k]] = (ulong)magnitude;
                        }
                    }
                }

                public ulong[] Values { get; }

                public int[] Offset { get; }

                public int[] Length { get; }

                public long[] Bits { get; }

                public int[] Exponent { get; }

                public bool[] Negative { get; }
            }
        }
    }
}

/// <summary>
/// Decimal facade for <see cref="ActiveSetQpSolver{T}"/>, forwarding to
/// <see cref="ActiveSetQpSolver{T}"/> instantiated at <c>T = decimal</c>.
/// </summary>
[Obsolete(
    "Use ActiveSetQpSolver<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class ActiveSetQpSolver
{
    /// <inheritdoc cref="ActiveSetQpSolver{T}.SolveMinVariance"/>
    public static decimal[] SolveMinVariance(
        decimal[,] covariance, decimal minWeight, decimal maxWeight)
        => ActiveSetQpSolver<decimal>.SolveMinVariance(covariance, minWeight, maxWeight);

    /// <inheritdoc cref="ActiveSetQpSolver{T}.SolveMeanVariance"/>
    public static decimal[] SolveMeanVariance(
        decimal[,] covariance, ReadOnlySpan<decimal> means, decimal riskAversion,
        decimal minWeight, decimal maxWeight)
        => ActiveSetQpSolver<decimal>.SolveMeanVariance(
            covariance, means, riskAversion, minWeight, maxWeight);
}
