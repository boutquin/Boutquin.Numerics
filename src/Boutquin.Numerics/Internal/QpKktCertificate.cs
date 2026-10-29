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

namespace Boutquin.Numerics.Internal;

/// <summary>The KKT condition a candidate portfolio failed, in the order the certificate checks them.</summary>
internal enum QpKktComponent
{
    /// <summary>Every condition holds; the candidate is certified.</summary>
    None = 0,

    /// <summary>A weight lies outside <c>[lb, ub]</c> by more than the bound's resolution, or is not finite.</summary>
    Bound,

    /// <summary><c>|1′w − 1|</c> exceeds the budget row's tolerance.</summary>
    Budget,

    /// <summary>No single budget multiplier makes every free variable stationary within its row tolerance.</summary>
    Stationarity,

    /// <summary>
    /// The free variables are stationary, but a variable at a bound has a multiplier of the wrong sign: a variable at
    /// its upper bound has a larger gradient than a variable at its lower bound (a bound pair), or a variable at a bound
    /// lies on the wrong side of the budget multiplier interval.
    /// </summary>
    MultiplierSign,
}

/// <summary>Outcome of <see cref="QpKktCertificate{T}"/>: the failed component (or <see cref="QpKktComponent.None"/>) and a description.</summary>
/// <param name="FailedComponent">The first condition that failed, or <see cref="QpKktComponent.None"/>.</param>
/// <param name="Detail">A human-readable description naming the component, the offending index and the residual.</param>
internal readonly record struct QpKktCertificateResult(QpKktComponent FailedComponent, string Detail)
{
    /// <summary>Whether every KKT condition holds.</summary>
    public bool Accepted => FailedComponent == QpKktComponent.None;

    /// <summary>
    /// The variable the failed condition names (the weight outside its bound, a free variable of the stationarity
    /// conflict, or the variable at a bound whose multiplier has the wrong sign), or −1.
    /// </summary>
    public int FailedIndex { get; init; } = -1;

    /// <summary>
    /// For a stationarity or multiplier-sign failure, the variable that set the other end of the budget multiplier
    /// interval the named variable conflicts with; otherwise −1.
    /// </summary>
    public int CounterpartIndex { get; init; } = -1;
}

/// <summary>
/// Independent KKT certificate for the budget-constrained box QP solved by <c>ActiveSetQpSolver&lt;T&gt;</c>:
/// <c>max μ′w − (λ/2)·w′Σw</c> (or <c>min w′Σw</c>) subject to <c>1′w = 1</c> and <c>lb ≤ w ≤ ub</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A (arithmetic and comparison only, no transcendental). Supported for
/// <see cref="double"/> and <see cref="decimal"/>; <see cref="float"/> is best-effort; <see cref="Half"/> is
/// rejected with <see cref="NotSupportedException"/>.
/// </para>
/// <para>
/// <b>Exact residuals.</b> Every operand is an exact radix fraction (a binary fraction for IEEE types, a
/// decimal fraction for <see cref="decimal"/>), so the certificate forms every residual and every tolerance
/// exactly in integer arithmetic. The certificate has no evaluation error; its only tolerance is the
/// resolution of the candidate weights themselves.
/// </para>
/// <para>
/// <b>Tolerances.</b> The resolution of a weight is <c>ρ(w) = ulp(w)/2</c>, the largest error of a correctly
/// rounded weight: for IEEE types <c>ulp(w)</c> is the spacing of the binade of <c>|w|</c> (relative,
/// <c>2⁻⁵²·2^⌊log₂|w|⌋</c>, down to the subnormal spacing <c>2⁻¹⁰⁷⁴</c>); for <see cref="decimal"/> it is the
/// spacing of the finest grid that holds <c>|w|</c> with its 96-bit mantissa (the absolute <c>1e-28</c> for
/// <c>|w|</c> below about 7.9, coarser above, one decade coarser within ten units of a grid change, where a
/// correctly rounded value may come from the coarser grid). This half-unit is the type's absolute resolution
/// term, one weight's rounding bound (decimal <c>5e-29</c> below about 7.9), accumulated over a row as that
/// rounding bound: a row with coefficients <c>c_j</c> has tolerance <c>κ·Σ_j |c_j|·ρ(w_j)</c>, relative to the
/// row's own scale and never a global maximum. The budget row's residual <c>1′w − 1</c> is formed exactly from
/// weights that each lie on their own radix grid, so it is a whole multiple of the finest of those grids
/// <c>g</c>; its tolerance is <c>κ·Σ_j ρ(w_j)</c> rounded down to a multiple of <c>g</c>. The rounding changes no
/// verdict (a multiple of <c>g</c> is within a bound exactly when it is within that bound rounded down to
/// <c>g</c>); it states the tolerance on the residual's own grid, and loses no correctly rounded optimum, whose
/// residual is such a multiple and at most <c>Σ_j ρ(w_j)</c> in magnitude.
/// </para>
/// <para>
/// <b>Tolerance scale.</b> <c>κ ≥ 1</c> multiplies every component tolerance: the two bound tolerances
/// <c>κ·ρ(lb)</c> and <c>κ·ρ(ub)</c>, the budget tolerance (before its grid rounding), and every stationarity and
/// multiplier-sign row tolerance. <c>κ = 1</c> is nominal (the overload without a scale): the correctly rounded
/// exact optimum of any problem is accepted, and the perturbation below is rejected. A solver's exit check may
/// pass a larger <c>κ</c>, because a solver result within its accuracy bound need not be correctly rounded, and
/// no residual certificate can both reject a few-unit move and accept every such result. <c>κ</c> below 1 is
/// refused: it would reject correctly rounded optima.
/// </para>
/// <para>
/// <b>Multiplier signs.</b> After stationarity, every pair of variables at opposite bounds is checked first:
/// a variable at its upper bound must not have a larger gradient than a variable at its lower bound,
/// <c>max g(upper) ≤ min g(lower)</c>, each pair measured against the tolerances of its own two rows, whatever
/// the budget multiplier. A violated pair is reported as a multiplier-sign failure naming both variables. The
/// remaining variables at a bound are then checked against the budget multiplier interval that the free
/// variables and the other bound variables leave.
/// </para>
/// <para>
/// <b>Rejection floor at <c>κ = 1</c>.</b> A single weight moved by <c>δ</c> from the correctly rounded optimum
/// changes the budget residual by <c>δ</c>. A correctly rounded optimum's residual is at most <c>⌊n/2⌋</c> grid
/// units and the budget tolerance is <c>⌊n/2⌋</c> units when every weight is on the grid <c>g</c>, so the budget
/// row alone rejects every move of <c>δ ≥ (n+1)·g</c>; this is the <c>max(4, n+1)</c>-ulp floor of the
/// mutation check. A smaller move is rejected only where the stationarity rows also see it: when the last
/// asset's covariance column is constant, for example, a 4-unit move at <c>n = 5</c> shifts every stationarity
/// row equally (the budget multiplier absorbs it) and can leave a residual of the same magnitude as the
/// optimum's, so no residual test that accepts every correctly rounded optimum separates the two vectors.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type of the problem and the candidate.</typeparam>
internal static class QpKktCertificate<T>
    where T : IFloatingPoint<T>
{
    [ThreadStatic]
    private static ActiveSetQpSolver<T>.StageWork? s_threadWork;

    /// <summary>
    /// The stage-work record of the checks made on the current thread, or <see langword="null"/> (test diagnostics, as the
    /// solver's certified stages record theirs on its route trace): each check records one run of order <c>n</c> with the
    /// loop bodies of its own loops, <c>2n² + 6n</c> when every component passes. Set by a test before a solve; a solve
    /// checks its candidates on the thread it runs on.
    /// </summary>
    internal static ActiveSetQpSolver<T>.StageWork? ThreadWork
    {
        get => s_threadWork;
        set => s_threadWork = value;
    }

    /// <summary>
    /// Certifies <paramref name="weights"/> at the nominal tolerance scale <c>κ = 1</c>, the scale at which the
    /// certificate accepts every correctly rounded exact optimum and rejects a copy of it with one weight moved by
    /// <c>max(4, n+1)</c> units of its grid or more (see the rejection floor in the remarks). Identical
    /// to <see cref="Check(T[,], T[], T, T, T, T[], T)"/> with <c>toleranceScale = 1</c>.
    /// </summary>
    /// <param name="covariance">Symmetric <c>n × n</c> covariance <c>Σ</c>.</param>
    /// <param name="means">Linear term <c>μ</c> of length <c>n</c>, or <see langword="null"/> for minimum variance.</param>
    /// <param name="riskAversion">Curvature <c>λ ≥ 0</c> of the mean-variance objective.</param>
    /// <param name="lowerBound">Common lower bound <c>lb</c>.</param>
    /// <param name="upperBound">Common upper bound <c>ub</c>.</param>
    /// <param name="weights">Candidate portfolio of length <c>n</c>.</param>
    /// <returns>The first failed component, or <see cref="QpKktComponent.None"/>.</returns>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is <see cref="Half"/> or has no exact representation.</exception>
    /// <exception cref="ArgumentException">The dimensions disagree or a problem input is not finite.</exception>
    internal static QpKktCertificateResult Check(
        T[,] covariance,
        T[]? means,
        T riskAversion,
        T lowerBound,
        T upperBound,
        T[] weights)
        => Check(covariance, means, riskAversion, lowerBound, upperBound, weights, T.One);

    /// <summary>
    /// Certifies <paramref name="weights"/> against the KKT conditions of the problem. Pass
    /// <paramref name="means"/> = <see langword="null"/> for minimum variance (then <paramref name="riskAversion"/>
    /// is ignored).
    /// </summary>
    /// <param name="covariance">Symmetric <c>n × n</c> covariance <c>Σ</c>.</param>
    /// <param name="means">Linear term <c>μ</c> of length <c>n</c>, or <see langword="null"/> for minimum variance.</param>
    /// <param name="riskAversion">Curvature <c>λ ≥ 0</c> of the mean-variance objective.</param>
    /// <param name="lowerBound">Common lower bound <c>lb</c>.</param>
    /// <param name="upperBound">Common upper bound <c>ub</c>.</param>
    /// <param name="weights">Candidate portfolio of length <c>n</c>.</param>
    /// <param name="toleranceScale">
    /// Tolerance scale <c>κ ≥ 1</c> multiplying every component tolerance (both bound tolerances, the budget
    /// tolerance before its grid rounding, and every stationarity and multiplier-sign row tolerance). <c>κ = 1</c>
    /// is nominal; a solver's exit check may pass a larger <c>κ</c> to accept a result that is within its own
    /// accuracy bound but not correctly rounded.
    /// </param>
    /// <returns>The first failed component, or <see cref="QpKktComponent.None"/>.</returns>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is <see cref="Half"/> or has no exact representation.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="toleranceScale"/> is below 1 or not finite.</exception>
    /// <exception cref="ArgumentException">The dimensions disagree or a problem input is not finite.</exception>
    internal static QpKktCertificateResult Check(
        T[,] covariance,
        T[]? means,
        T riskAversion,
        T lowerBound,
        T upperBound,
        T[] weights,
        T toleranceScale)
    {
        ArgumentNullException.ThrowIfNull(covariance);
        ArgumentNullException.ThrowIfNull(weights);
        Exact.EnsureSupported();
        var n = weights.Length;
        if (covariance.GetLength(0) != n || covariance.GetLength(1) != n || (means is not null && means.Length != n))
        {
            throw new ArgumentException("The covariance, means and weights dimensions must agree.", nameof(weights));
        }

        if (!T.IsFinite(toleranceScale) || toleranceScale < T.One)
        {
            throw new ArgumentOutOfRangeException(nameof(toleranceScale), "The tolerance scale must be finite and at least 1.");
        }

        if (!T.IsFinite(lowerBound) || !T.IsFinite(upperBound))
        {
            throw new ArgumentException("The bounds must be finite.", nameof(lowerBound));
        }

        if (means is not null && (!T.IsFinite(riskAversion) || riskAversion < T.Zero || means.Any(m => !T.IsFinite(m))))
        {
            throw new ArgumentException("The means must be finite and the risk aversion finite and non-negative.", nameof(means));
        }

        // Loop bodies executed by this check (its stage work).
        var work = 0L;
        foreach (var entry in covariance)
        {
            work++;
            if (!T.IsFinite(entry))
            {
                throw new ArgumentException("The covariance must be finite.", nameof(covariance));
            }
        }

        var kappa = Exact.From(toleranceScale);
        var lb = Exact.From(lowerBound);
        var ub = Exact.From(upperBound);
        var lbTolerance = kappa * Exact.HalfUnit(lowerBound);
        var ubTolerance = kappa * Exact.HalfUnit(upperBound);

        // Bound feasibility; at-bound classification for the multiplier step.
        var w = new Exact[n];
        var resolution = new Exact[n];
        for (var i = 0; i < n; i++)
        {
            work++;
            if (!T.IsFinite(weights[i]))
            {
                return Recorded(Fail(QpKktComponent.Bound, "bound: weight " + i + " is not finite", i), n, work);
            }

            w[i] = Exact.From(weights[i]);
            resolution[i] = Exact.HalfUnit(weights[i]);
            if (w[i] < lb - lbTolerance)
            {
                return Recorded(Fail(QpKktComponent.Bound, "bound: weight " + i + " = " + w[i] + " is below the lower bound " + lb + " by more than " + lbTolerance, i), n, work);
            }

            if (w[i] > ub + ubTolerance)
            {
                return Recorded(Fail(QpKktComponent.Bound, "bound: weight " + i + " = " + w[i] + " is above the upper bound " + ub + " by more than " + ubTolerance, i), n, work);
            }
        }

        // Budget: |1'w - 1| against kappa * sum_j rho(w_j), rounded down to the residual's own grid.
        var sum = Exact.Zero;
        var budgetScale = Exact.Zero;
        var grid = 0;
        for (var i = 0; i < n; i++)
        {
            work++;
            sum += w[i];
            budgetScale += resolution[i];
            if (!w[i].Mantissa.IsZero)
            {
                grid = Math.Min(grid, w[i].GridExponent());
            }
        }

        var budgetError = (sum - Exact.One).Abs();
        var budgetTolerance = (kappa * budgetScale).FloorToGrid(grid);
        if (budgetError > budgetTolerance)
        {
            return Recorded(Fail(QpKktComponent.Budget, "budget: |1'w - 1| = " + budgetError + " exceeds the budget row tolerance " + budgetTolerance), n, work);
        }

        // Ascent direction h = μ − λΣw (minimum variance: h = −Σw) and each row's tolerance κ·λ·Σ_j |Σ_ij|·ρ(w_j).
        var lambda = means is null ? Exact.One : Exact.From(riskAversion);
        var h = new Exact[n];
        var t = new Exact[n];
        for (var i = 0; i < n; i++)
        {
            work++;
            var product = Exact.Zero;
            var scale = Exact.Zero;
            for (var j = 0; j < n; j++)
            {
                work++;
                var sigma = Exact.From(covariance[i, j]);
                product += sigma * w[j];
                scale += sigma.Abs() * resolution[j];
            }

            h[i] = (means is null ? Exact.Zero : Exact.From(means[i])) - (lambda * product);
            t[i] = kappa * lambda * scale;
        }

        // KKT with a free budget multiplier ν: free i needs |h_i − ν| ≤ t_i; at lb, h_i − t_i ≤ ν; at ub,
        // ν ≤ h_i + t_i; a variable at both bounds (lb = ub) constrains nothing. Feasible ν form an interval.
        Exact? low = null;
        Exact? high = null;
        int lowIndex = -1, highIndex = -1;
        for (var i = 0; i < n; i++)
        {
            work++;
            if (IsAtBound(w[i], lb, lbTolerance) || IsAtBound(w[i], ub, ubTolerance))
            {
                continue;
            }

            Narrow(ref low, ref lowIndex, ref high, ref highIndex, h[i] - t[i], h[i] + t[i], i);
        }

        if (low is not null && high is not null && low.Value > high.Value)
        {
            return Recorded(
                Fail(
                    QpKktComponent.Stationarity,
                    "stationarity: free variables " + lowIndex + " and " + highIndex + " have ascent components " + h[lowIndex] + " and " + h[highIndex]
                    + ", farther apart than their row tolerances " + t[lowIndex] + " and " + t[highIndex],
                    lowIndex,
                    highIndex),
                n,
                work);
        }

        // Bound pairs: every variable at its upper bound needs a gradient no larger than every variable at its lower
        // bound, max g(upper) ≤ min g(lower), whatever the budget multiplier; in ascent components h = −g, each pair
        // (u at ub, l at lb) needs h_l − t_l ≤ h_u + t_u with the two rows' own tolerances. The pair with the largest
        // h_l − t_l and the smallest h_u + t_u decides every pair.
        int pairLower = -1, pairUpper = -1;
        for (var i = 0; i < n; i++)
        {
            work++;
            var atLower = IsAtBound(w[i], lb, lbTolerance);
            var atUpper = IsAtBound(w[i], ub, ubTolerance);
            if (atLower == atUpper)
            {
                continue;
            }

            if (atLower && (pairLower < 0 || h[i] - t[i] > h[pairLower] - t[pairLower]))
            {
                pairLower = i;
            }
            else if (atUpper && (pairUpper < 0 || h[i] + t[i] < h[pairUpper] + t[pairUpper]))
            {
                pairUpper = i;
            }
        }

        if (pairLower >= 0 && pairUpper >= 0 && h[pairLower] - t[pairLower] > h[pairUpper] + t[pairUpper])
        {
            return Recorded(
                Fail(
                    QpKktComponent.MultiplierSign,
                    "multiplier sign: bound pair: variable " + pairUpper + " at its upper bound has ascent component " + h[pairUpper] + ", below the ascent component "
                    + h[pairLower] + " of variable " + pairLower + " at its lower bound by more than the two rows' tolerances " + t[pairUpper] + " and " + t[pairLower],
                    Math.Max(pairUpper, pairLower),
                    Math.Min(pairUpper, pairLower)),
                n,
                work);
        }

        for (var i = 0; i < n; i++)
        {
            work++;
            var atLower = IsAtBound(w[i], lb, lbTolerance);
            var atUpper = IsAtBound(w[i], ub, ubTolerance);
            if (atLower == atUpper)
            {
                continue;
            }

            Narrow(ref low, ref lowIndex, ref high, ref highIndex, atLower ? h[i] - t[i] : (Exact?)null, atUpper ? h[i] + t[i] : (Exact?)null, i);
            if (low is not null && high is not null && low.Value > high.Value)
            {
                return Recorded(
                    Fail(
                        QpKktComponent.MultiplierSign,
                        "multiplier sign: variable " + i + " at its " + (atLower ? "lower" : "upper") + " bound has ascent component " + h[i]
                        + " on the wrong side of the budget multiplier interval set by variable " + (atLower ? highIndex : lowIndex) + " (tolerance " + t[i] + ")",
                        i,
                        atLower ? highIndex : lowIndex),
                    n,
                    work);
            }
        }

        return Recorded(new QpKktCertificateResult(QpKktComponent.None, "certified"), n, work);
    }

    // Records one check of order n and the loop bodies it executed on this thread's stage-work record, if one is set.
    private static QpKktCertificateResult Recorded(QpKktCertificateResult result, int n, long work)
    {
        s_threadWork?.Record(0, n, work);
        return result;
    }

    private static QpKktCertificateResult Fail(QpKktComponent component, string detail, int index = -1, int counterpart = -1) =>
        new(component, detail) { FailedIndex = index, CounterpartIndex = counterpart };

    private static bool IsAtBound(Exact weight, Exact bound, Exact tolerance) => (weight - bound).Abs() <= tolerance;

    // Intersects the multiplier interval [low, high] with [lower, upper] (a null end is unbounded), remembering
    // which variable set each end.
    private static void Narrow(ref Exact? low, ref int lowIndex, ref Exact? high, ref int highIndex, Exact? lower, Exact? upper, int index)
    {
        if (lower is not null && (low is null || lower.Value > low.Value))
        {
            low = lower;
            lowIndex = index;
        }

        if (upper is not null && (high is null || upper.Value < high.Value))
        {
            high = upper;
            highIndex = index;
        }
    }

    /// <summary>An exact radix fraction <c>Mantissa · radix^Exponent</c> (radix 2 for IEEE types, 10 for decimal).</summary>
    private readonly struct Exact : IComparable<Exact>
    {
        private static readonly bool s_decimal = typeof(T) == typeof(decimal);
        private static readonly BigInteger s_decimalLimit = (BigInteger.One << 96) - 1;

        private Exact(BigInteger mantissa, int exponent)
        {
            Mantissa = mantissa;
            Exponent = exponent;
        }

        public static Exact Zero => default;

        public static Exact One => new(BigInteger.One, 0);

        public BigInteger Mantissa { get; }

        public int Exponent { get; }

        public static void EnsureSupported()
        {
            if (typeof(T) != typeof(double) && typeof(T) != typeof(float) && typeof(T) != typeof(decimal))
            {
                throw new NotSupportedException($"QpKktCertificate supports double, float and decimal, not {typeof(T)}.");
            }
        }

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

            var x = typeof(T) == typeof(float) ? (float)(object)value : (double)(object)value;
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

        // Half the spacing of the type's grid at |value|: the largest error of a correctly rounded value.
        public static Exact HalfUnit(T value)
        {
            if (s_decimal)
            {
                var d = (decimal)(object)value;
                Span<int> bits = stackalloc int[4];
                decimal.GetBits(d, bits);
                var magnitude = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | new BigInteger((uint)bits[0]);
                var scale = (bits[3] >> 16) & 0xFF;

                // Finest scale s <= 28 with |value|·10^s + 10 <= the mantissa limit; |value|·10^s = magnitude·10^(s - scale).
                var s = 28;
                while (true)
                {
                    var t = s - scale;
                    var fits = t >= 0
                        ? (magnitude * BigInteger.Pow(10, t)) + 10 <= s_decimalLimit
                        : magnitude + (10 * BigInteger.Pow(10, -t)) <= s_decimalLimit * BigInteger.Pow(10, -t);
                    if (fits)
                    {
                        break;
                    }

                    s--;
                }

                return new Exact(5, -(s + 1));
            }

            if (typeof(T) == typeof(float))
            {
                var biasedSingle = (BitConverter.SingleToInt32Bits((float)(object)value) >> 23) & 0xFF;
                return new Exact(BigInteger.One, Math.Max(biasedSingle, 1) - 151);
            }

            var biased = (int)((BitConverter.DoubleToInt64Bits((double)(object)value) >> 52) & 0x7FF);
            return new Exact(BigInteger.One, Math.Max(biased, 1) - 1076);
        }

        // The exponent of the coarsest radix grid this nonzero value lies on (trailing radix digits stripped).
        public int GridExponent()
        {
            var radix = s_decimal ? 10 : 2;
            var m = BigInteger.Abs(Mantissa);
            var e = Exponent;
            while (m % radix == 0)
            {
                m /= radix;
                e++;
            }

            return e;
        }

        // The largest multiple of radix^gridExponent not above this non-negative value.
        public Exact FloorToGrid(int gridExponent)
        {
            if (Mantissa.IsZero || Exponent >= gridExponent)
            {
                return this;
            }

            var shift = gridExponent - Exponent;
            var divisor = s_decimal ? BigInteger.Pow(10, shift) : BigInteger.One << shift;
            return new Exact(BigInteger.Divide(Mantissa, divisor), gridExponent);
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

        public int CompareTo(Exact other) => (this - other).Mantissa.Sign;

        public override string ToString()
        {
            if (Mantissa.IsZero)
            {
                return "0";
            }

            if (s_decimal)
            {
                return double.Parse(Mantissa.ToString(CultureInfo.InvariantCulture) + "E" + Exponent.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture)
                    .ToString("G6", CultureInfo.InvariantCulture);
            }

            var shift = (int)Math.Max(0, BigInteger.Abs(Mantissa).GetBitLength() - 62);
            return Math.ScaleB((double)(Mantissa >> shift), Exponent + shift).ToString("G6", CultureInfo.InvariantCulture);
        }

        private static BigInteger Align(Exact x, int exponent)
        {
            var shift = x.Exponent - exponent;
            if (shift == 0)
            {
                return x.Mantissa;
            }

            return s_decimal ? x.Mantissa * BigInteger.Pow(10, shift) : x.Mantissa << shift;
        }
    }
}
