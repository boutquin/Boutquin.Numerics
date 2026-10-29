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

namespace Boutquin.Numerics.Solvers;

/// <summary>
/// Options controlling the Nelder-Mead simplex minimisation.
/// Null fields use the documented defaults.
/// </summary>
/// <typeparam name="T">
/// Arithmetic floating-point type satisfying <see cref="IFloatingPoint{TSelf}"/>.
/// </typeparam>
/// <param name="MaxIterations">
/// Iteration counter threshold. The counter starts at zero and increments once per simplex update;
/// resource termination occurs when it exceeds this value, so up to MaxIterations + 1 updates run.
/// The resource stop reports <c>Converged = false</c>. Default: 2 000.
/// </param>
/// <param name="XTolerance">
/// Convergence threshold on the simplex diameter (largest vertex-to-best distance).
/// When the diameter falls below this value the search is declared converged.
/// Must be non-negative. Default: 1e-8.
/// </param>
/// <param name="FTolerance">
/// Convergence threshold on the spread of objective values across the simplex
/// vertices. When the spread falls below this value the search is declared converged.
/// Must be non-negative. Default: 0 (only an exact objective-value tie triggers this criterion unless supplied).
/// </param>
/// <param name="InitialStep">
/// Edge length of the initial simplex. Each vertex is formed by perturbing one
/// coordinate of <c>initialGuess</c> by this amount. A positive supplied step
/// must change every perturbed coordinate at the selected floating-point precision.
/// Default: 0.05 × max(|x₀ⁱ|, 1).
/// A supplied zero or negative step uses that default.
/// </param>
/// <param name="Reflection">α, positive reflection coefficient. Default: 1.</param>
/// <param name="Expansion">γ, expansion coefficient greater than both 1 and α. Default: 2.</param>
/// <param name="Contraction">ρ, contraction coefficient strictly between 0 and 1. Default: ½.</param>
/// <param name="Shrink">σ, shrink coefficient strictly between 0 and 1. Default: ½.</param>
public sealed record SimplexOptions<T>(
    int MaxIterations = 2000,
    T? XTolerance = default,
    T? FTolerance = default,
    T? InitialStep = default,
    T? Reflection = default,
    T? Expansion = default,
    T? Contraction = default,
    T? Shrink = default)
    where T : struct, IFloatingPoint<T>;

/// <summary>
/// The result returned by <see cref="NelderMeadMinimizer{T}.Minimize"/>.
/// </summary>
/// <typeparam name="T">
/// Arithmetic floating-point type satisfying <see cref="IFloatingPoint{TSelf}"/>.
/// </typeparam>
/// <param name="Argmin">The approximate minimiser.</param>
/// <param name="MinValue">The objective value at <paramref name="Argmin"/>.</param>
/// <param name="Iterations">Total simplex iterations performed.</param>
/// <param name="Converged">
/// <c>true</c> when a convergence criterion was satisfied before the iteration cap.
/// </param>
public sealed record SimplexResult<T>(T[] Argmin, T MinValue, int Iterations, bool Converged)
    where T : struct, IFloatingPoint<T>
{
    private T[] _argmin = (T[])Argmin.Clone();

    /// <summary>The approximate minimiser, copied on input and access.</summary>
    public T[] Argmin
    {
        get => (T[])_argmin.Clone();
        init => _argmin = (T[])value.Clone();
    }
}

/// <summary>
/// Generic derivative-free Nelder-Mead simplex minimiser.
/// </summary>
/// <remarks>
/// <para><b>Generic-math tier: A</b> — arithmetic-only (<c>+</c>, <c>−</c>,
/// <c>*</c>, <c>/</c>, comparison). The minimiser uses no transcendental functions and
/// is therefore usable at <c>T = decimal</c> as well as <c>double</c>, <c>float</c>,
/// and <c>Half</c>. The <em>objective</em> <c>Func&lt;T[], T&gt;</c> may itself use
/// transcendentals; the minimiser simply evaluates it as a black box.</para>
///
/// <para>
/// The algorithm is the Nelder-Mead simplex method with the standard coefficients of
/// Nelder &amp; Mead (1965) and the decision tree of Lagarias, Reeds, Wright &amp;
/// Wright (1998), as restated by Gao &amp; Han (2012):
/// </para>
/// <list type="number">
///   <item>Build an initial simplex of <c>n+1</c> vertices from <c>initialGuess</c>.</item>
///   <item>Sort vertices so vertex 0 is the best (lowest <c>f</c>).</item>
///   <item>Compute the centroid of the best <c>n</c> vertices.</item>
///   <item>Reflect the worst vertex through the centroid.</item>
///   <item>If the reflection is the new best, try an expansion; otherwise
///       try an outside or inside contraction; if neither contraction improves,
///       shrink the whole simplex toward the best vertex.</item>
///   <item>Repeat until the simplex diameter or value spread is below tolerance,
///       or until the iteration cap is reached.</item>
/// </list>
///
/// <para>
/// Equal-valued accepted points follow existing equal vertices. After shrink,
/// the old best remains first among equally best vertices (Lagarias et al., §2).
/// Default coefficients: α = 1 (reflection), γ = 2 (expansion),
/// ρ = ½ (contraction), σ = ½ (shrink).
/// </para>
///
/// <para>
/// <b>Allocation strategy:</b> all working arrays are allocated once before the
/// iteration loop. No per-iteration heap allocation occurs. The caller-supplied
/// <c>objective</c> delegate is the only allocation source.
/// </para>
/// <para>
/// <b>Non-finite objective values (design decision):</b> the reference vertices
/// of the <em>initial</em> simplex must be finite — a non-finite value there
/// throws <see cref="ArgumentException"/> immediately (unchanged). Once iteration
/// begins, any subsequently evaluated vertex (reflection, expansion, contraction,
/// or shrink) whose objective value is non-finite (NaN or ±∞; reachable only for
/// IEEE <c>T</c>, never for <c>T = decimal</c>) is instead ranked as if pinned to
/// <c>+∞</c> — strictly worse than every finite value, and equal-ranked against
/// any other non-finite value. This is a deliberate ordering rule (the reference
/// algorithm is silent on non-finite mid-iteration values): it preserves every
/// ordering guarantee for finite values unchanged while guaranteeing a non-finite
/// evaluation can never be sorted or accepted as the simplex's best vertex, which
/// would otherwise corrupt every subsequent centroid, reflection, and convergence
/// check for the remainder of the search.
/// </para>
/// </remarks>
/// <typeparam name="T">
/// A floating-point type implementing <see cref="IFloatingPoint{TSelf}"/>.
/// At <c>T = double</c> or <c>T = float</c> this gives IEEE 754 arithmetic;
/// at <c>T = decimal</c> it gives 28-digit fixed-base-10 arithmetic.
/// </typeparam>
public static class NelderMeadMinimizer<T>
    where T : struct, IFloatingPoint<T>
{
    /// <summary>
    /// Ranks two objective values for simplex ordering purposes, treating any
    /// non-finite value (NaN or ±∞ — reachable only for IEEE <c>T</c>; never for
    /// <c>T = decimal</c>, which has no such representation) as worst-ranked, as if
    /// pinned to <c>+∞</c>. A raw <c>T.CompareTo</c>/<c>Comparer&lt;T&gt;.Default</c>
    /// ordering treats <c>NaN</c> as LESS than every other value (the
    /// <see cref="IComparable{T}"/> convention), which would sort a NaN-tagged
    /// vertex as the simplex's <em>best</em> vertex — corrupting every subsequent
    /// centroid, reflection, and convergence check. This method is the single
    /// point of truth for objective-value ordering throughout <see cref="Minimize"/>;
    /// for two finite operands it is exactly the ordinary <c>&lt;</c>/<c>&gt;</c>
    /// ordering (no behaviour change vs. the historical comparer for finite inputs).
    /// </summary>
    private static int CompareRanked(T x, T y)
    {
        var xFinite = T.IsFinite(x);
        var yFinite = T.IsFinite(y);
        if (xFinite && yFinite)
        {
            return x < y ? -1 : (x > y ? 1 : 0);
        }

        if (!xFinite && !yFinite)
        {
            return 0;
        }

        return xFinite ? -1 : 1;
    }

    /// <summary>
    /// <see langword="true"/> when <paramref name="candidate"/> ranks strictly
    /// better (lower) than <paramref name="reference"/> under the same
    /// non-finite-is-worst rule as <see cref="CompareRanked"/>.
    /// </summary>
    private static bool IsBetter(T candidate, T reference) => CompareRanked(candidate, reference) < 0;

    /// <summary>
    /// <see langword="true"/> when <paramref name="candidate"/> ranks better than or
    /// equal to <paramref name="reference"/> under the same non-finite-is-worst rule
    /// as <see cref="CompareRanked"/>.
    /// </summary>
    private static bool IsBetterOrEqual(T candidate, T reference) => CompareRanked(candidate, reference) <= 0;

    private static readonly T s_half = T.CreateChecked(0.5);
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_defaultTol = T.CreateChecked(1e-8);
    private static readonly T s_defaultFTol = T.Zero;
    private static readonly T s_defaultInitialStepFactor = T.CreateChecked(0.05);
    private static readonly T s_one = T.One;
    private static readonly T s_zero = T.Zero;

    /// <summary>
    /// Minimises <paramref name="objective"/> starting from <paramref name="initialGuess"/>.
    /// </summary>
    /// <param name="objective">
    /// The function to minimise. Called with a temporary array — do not cache the
    /// reference across calls.
    /// </param>
    /// <param name="initialGuess">
    /// Starting point. Must be non-empty. A copy is taken; the original is not modified.
    /// </param>
    /// <param name="options">
    /// Simplex options. Null fields use defaults. Pass <c>null</c> for all defaults.
    /// </param>
    /// <returns>
    /// A <see cref="SimplexResult{T}"/> with the best vertex found, its objective
    /// value, the iteration count, and a convergence flag.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="initialGuess"/> is empty or contains a non-finite
    /// coordinate, when a constructed initial simplex coordinate is non-finite, or
    /// when the objective is non-finite at an initial simplex vertex.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when a numeric option is non-finite, a tolerance is negative, a coefficient
    /// violates its documented range, or when
    /// a positive <c>InitialStep</c> leaves a perturbed coordinate unchanged.
    /// </exception>
    public static SimplexResult<T> Minimize(
        Func<T[], T> objective,
        T[] initialGuess,
        SimplexOptions<T>? options = null)
    {
        ArgumentNullException.ThrowIfNull(objective);
        ArgumentNullException.ThrowIfNull(initialGuess);

        var n = initialGuess.Length;
        if (n == 0)
        {
            throw new ArgumentException("initialGuess must be non-empty.", nameof(initialGuess));
        }

        for (var i = 0; i < n; i++)
        {
            if (!T.IsFinite(initialGuess[i]))
            {
                throw new ArgumentException(
                    $"initialGuess entry at index {i} must be finite (value {initialGuess[i]}).",
                    nameof(initialGuess));
            }
        }

        options ??= new SimplexOptions<T>();
        var maxIter = options.MaxIterations;
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxIter, nameof(options.MaxIterations));
        var xTol = options.XTolerance ?? s_defaultTol;
        var fTol = options.FTolerance ?? s_defaultFTol;
        var alpha = options.Reflection ?? s_one;
        var gamma = options.Expansion ?? s_two;
        var rho = options.Contraction ?? s_half;
        var sigma = options.Shrink ?? s_half;
        ValidateFiniteOption(xTol, nameof(options.XTolerance));
        ValidateFiniteOption(fTol, nameof(options.FTolerance));
        ValidateFiniteOption(alpha, nameof(options.Reflection));
        ValidateFiniteOption(gamma, nameof(options.Expansion));
        ValidateFiniteOption(rho, nameof(options.Contraction));
        ValidateFiniteOption(sigma, nameof(options.Shrink));
        ArgumentOutOfRangeException.ThrowIfNegative(xTol, nameof(options.XTolerance));
        ArgumentOutOfRangeException.ThrowIfNegative(fTol, nameof(options.FTolerance));
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(alpha, s_zero, nameof(options.Reflection));
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(gamma, s_one, nameof(options.Expansion));
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(gamma, alpha, nameof(options.Expansion));
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(rho, s_zero, nameof(options.Contraction));
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(rho, s_one, nameof(options.Contraction));
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(sigma, s_zero, nameof(options.Shrink));
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(sigma, s_one, nameof(options.Shrink));
        // A supplied InitialStep is honoured only when it is strictly positive. A
        // non-positive value (≤ 0) would build a degenerate zero-diameter simplex that
        // satisfies the first-iteration convergence test immediately, falsely reporting
        // Converged = true at the initial guess; treat it as "not supplied" and fall
        // back to the non-degenerate default step instead.
        var initialStep = options.InitialStep.GetValueOrDefault();
        if (options.InitialStep.HasValue)
        {
            ValidateFiniteOption(initialStep, nameof(options.InitialStep));
        }

        var hasInitialStep = options.InitialStep.HasValue && initialStep > s_zero;

        // -----------------------------------------------------------------
        //  Allocate all working buffers up front (no per-iteration alloc).
        // -----------------------------------------------------------------
        var m = n + 1; // simplex vertex count
        var simplex = new T[m][];
        for (var i = 0; i < m; i++)
        {
            simplex[i] = new T[n];
        }

        var fVals = new T[m];
        var centroid = new T[n];
        var reflected = new T[n];
        var expanded = new T[n];
        var contracted = new T[n];
        var order = new int[m]; // index sort

        for (var i = 0; i < m; i++)
        {
            order[i] = i;
        }

        // -----------------------------------------------------------------
        //  Build initial simplex.
        //  Vertex 0 = initialGuess.
        //  Vertex i+1 = initialGuess with coordinate i perturbed by step[i].
        // -----------------------------------------------------------------
        initialGuess.CopyTo(simplex[0], 0);
        for (var i = 0; i < n; i++)
        {
            initialGuess.CopyTo(simplex[i + 1], 0);
            var xi = initialGuess[i];
            T step;
            if (hasInitialStep)
            {
                step = initialStep;
            }
            else
            {
                // 0.05 * max(|x₀ⁱ|, 1)
                var absXi = T.Abs(xi);
                step = s_defaultInitialStepFactor * (absXi > s_one ? absXi : s_one);
            }
            var perturbed = xi + step;
            if (perturbed == xi)
            {
                if (hasInitialStep)
                {
                    throw new ArgumentOutOfRangeException(
                        nameof(options),
                        $"InitialStep does not change initialGuess coordinate {i} at the selected precision.");
                }

                throw new ArgumentException(
                    $"The default initial step does not change initialGuess coordinate {i} at the selected precision.",
                    nameof(initialGuess));
            }

            if (!T.IsFinite(perturbed))
            {
                throw new ArgumentException(
                    $"The initial simplex coordinate {i} must be finite.",
                    nameof(initialGuess));
            }

            simplex[i + 1][i] = perturbed;
        }

        T Evaluate(T[] point)
        {
            foreach (var coordinate in point)
            {
                if (!T.IsFinite(coordinate))
                {
                    throw new ArithmeticException("Generated simplex coordinates must be finite before objective evaluation.");
                }
            }

            return objective(point);
        }

        // Evaluate at each initial vertex.
        for (var i = 0; i < m; i++)
        {
            fVals[i] = Evaluate(simplex[i]);
            if (!T.IsFinite(fVals[i]))
            {
                throw new ArgumentException(
                    $"Objective returned non-finite value at initial simplex vertex {i}.",
                    nameof(objective));
            }
        }

        // -----------------------------------------------------------------
        //  Main iteration loop.
        // -----------------------------------------------------------------
        var iter = 0;
        while (iter <= maxIter)
        {
            // Preserve prior ranks. The replacement occupies the old worst's last
            // position, so stable sorting places it after existing equal vertices.
            // After shrink the old best is still first among equally best points.
            SortStable(order, fVals);

            var iBest = order[0];
            var iWorst = order[n];
            var iSecWorst = order[n - 1];

            // --- Convergence checks --------------------------------------
            //  X-diameter: max distance from best to any other vertex.
            //  A diameter beyond the type's range is larger than every tolerance: IEEE
            //  types already yield +∞ there, and decimal's OverflowException is the same
            //  fact, so it must not escape for vertices that are themselves representable.
            var xDiam = s_zero;
            var xDiamOverflowed = false;
            try
            {
                for (var j = 1; j < m; j++)
                {
                    var vj = simplex[order[j]];
                    var vBest = simplex[iBest];
                    for (var k = 0; k < n; k++)
                    {
                        var diff = T.Abs(vj[k] - vBest[k]);
                        if (diff > xDiam)
                        {
                            xDiam = diff;
                        }
                    }
                }
            }
            catch (OverflowException)
            {
                xDiamOverflowed = true;
            }

            //  F-spread: worst - best. Every objective value is finite and representable, so a
            //  spread beyond the type's range is larger than every tolerance: IEEE types already
            //  yield +∞ there (so the test is false), and decimal's OverflowException is the same
            //  fact — "not converged" — and must not escape as a raw exception.
            var fSpreadConverged = false;
            try
            {
                fSpreadConverged = fVals[iWorst] - fVals[iBest] <= fTol;
            }
            catch (OverflowException)
            {
                // fSpreadConverged stays false.
            }

            iter++;

            if (T.IsFinite(fVals[iWorst]) && ((!xDiamOverflowed && xDiam <= xTol) || fSpreadConverged))
            {
                // Converged — return best vertex.
                return new SimplexResult<T>(
                    (T[])simplex[iBest].Clone(),
                    fVals[iBest],
                    iter,
                    true);
            }

            // --- Centroid of best n vertices (exclude worst) --------------
            for (var k = 0; k < n; k++)
            {
                centroid[k] = s_zero;
            }

            var nT = T.CreateChecked(n);
            for (var k = 0; k < n; k++)
            {
                var overflowed = false;
                try
                {
                    for (var j = 0; j < n; j++) // skip order[n] (worst)
                    {
                        centroid[k] += simplex[order[j]][k];
                    }
                }
                catch (OverflowException)
                {
                    // Decimal throws where IEEE arithmetic produces infinity.
                    overflowed = true;
                }

                if (!overflowed && T.IsFinite(centroid[k]))
                {
                    // Preserve ordinary-scale rounding and solver trajectories.
                    centroid[k] /= nT;
                    continue;
                }

                centroid[k] = s_zero;
                for (var j = 0; j < n; j++)
                {
                    var value = simplex[order[j]][k];
                    var count = T.CreateChecked(j + 1);
                    var mean = centroid[k];
                    // Same-sign subtraction cannot overflow. For opposite signs,
                    // weight each operand first: their difference could overflow.
                    centroid[k] = T.IsNegative(mean) == T.IsNegative(value)
                        ? mean + (value - mean) / count
                        : mean - mean / count + value / count;
                }
            }

            // --- Reflection: xr = centroid + α*(centroid - xWorst) --------
            var xWorst = simplex[iWorst];
            for (var k = 0; k < n; k++)
            {
                reflected[k] = TrialCoordinate(centroid[k], xWorst[k], centroid[k], alpha);
            }
            var fRefl = Evaluate(reflected);

            if (IsBetter(fRefl, fVals[iBest]))
            {
                // Reflection is better than best → try expansion.
                for (var k = 0; k < n; k++)
                {
                    expanded[k] = TrialCoordinate(centroid[k], centroid[k], reflected[k], gamma);
                }
                var fExp = Evaluate(expanded);
                if (IsBetter(fExp, fRefl))
                {
                    // Accept expansion.
                    expanded.CopyTo(simplex[iWorst], 0);
                    fVals[iWorst] = fExp;
                }
                else
                {
                    // Accept reflection.
                    reflected.CopyTo(simplex[iWorst], 0);
                    fVals[iWorst] = fRefl;
                }
            }
            else if (IsBetter(fRefl, fVals[iSecWorst]))
            {
                // Reflection is better than second-worst → accept it.
                reflected.CopyTo(simplex[iWorst], 0);
                fVals[iWorst] = fRefl;
            }
            else
            {
                // Contraction.
                if (IsBetter(fRefl, fVals[iWorst]))
                {
                    // Outside contraction: xoc = centroid + ρ*(xr - centroid)
                    for (var k = 0; k < n; k++)
                    {
                        contracted[k] = TrialCoordinate(centroid[k], centroid[k], reflected[k], rho);
                    }
                    var fOc = Evaluate(contracted);
                    if (IsBetterOrEqual(fOc, fRefl))
                    {
                        contracted.CopyTo(simplex[iWorst], 0);
                        fVals[iWorst] = fOc;
                        continue;
                    }
                }
                else
                {
                    // Inside contraction: xic = centroid + ρ*(xWorst - centroid)
                    for (var k = 0; k < n; k++)
                    {
                        contracted[k] = TrialCoordinate(centroid[k], centroid[k], xWorst[k], rho);
                    }
                    var fIc = Evaluate(contracted);
                    if (IsBetter(fIc, fVals[iWorst]))
                    {
                        contracted.CopyTo(simplex[iWorst], 0);
                        fVals[iWorst] = fIc;
                        continue;
                    }
                }

                // Shrink: all vertices toward best.
                var xBest = simplex[iBest];
                for (var j = 1; j < m; j++)
                {
                    var vj = simplex[order[j]];
                    for (var k = 0; k < n; k++)
                    {
                        vj[k] = TrialCoordinate(xBest[k], xBest[k], vj[k], sigma);
                    }
                    fVals[order[j]] = Evaluate(vj);
                }
            }
        }

        // --- Iteration cap hit: find best and return with Converged=false -
        SortStable(order, fVals);
        return new SimplexResult<T>(
            (T[])simplex[order[0]].Clone(),
            fVals[order[0]],
            iter,
            false);
    }

    /// <summary>
    /// Evaluates the simplex trial coordinate <c>origin + coefficient·(to − from)</c>.
    /// </summary>
    /// <remarks>
    /// The direct expression is tried first, so every trajectory whose intermediates stay
    /// in range is bit-for-bit unchanged. When it overflows — to ±∞/NaN for IEEE types,
    /// or an <see cref="OverflowException"/> for <c>decimal</c> — the difference or the
    /// scaled product exceeded the range although the trial coordinate itself may not.
    /// The fallback halves both operands and the origin before combining them, so no
    /// intermediate exceeds the magnitude of the trial it produces unless that trial is
    /// itself unrepresentable; the final doubling then overflows exactly when it is, and
    /// the caller's finite-coordinate guard rejects it as before.
    /// </remarks>
    private static T TrialCoordinate(T origin, T from, T to, T coefficient)
    {
        try
        {
            var direct = origin + (coefficient * (to - from));
            if (T.IsFinite(direct))
            {
                return direct;
            }
        }
        catch (OverflowException)
        {
            // Decimal throws where IEEE arithmetic produces infinity; use the scaled form.
        }

        var halfDifference = (to * s_half) - (from * s_half);
        return ((origin * s_half) + (coefficient * halfDifference)) * s_two;
    }

    private static void SortStable(int[] order, T[] values)
    {
        // Insertion sorting allocates nothing and preserves temporal order on ties.
        for (var i = 1; i < order.Length; i++)
        {
            var index = order[i];
            var j = i;
            while (j > 0 && CompareRanked(values[index], values[order[j - 1]]) < 0)
            {
                order[j] = order[j - 1];
                j--;
            }

            order[j] = index;
        }
    }

    private static void ValidateFiniteOption(T value, string paramName)
    {
        if (!T.IsFinite(value))
        {
            throw new ArgumentOutOfRangeException(paramName, value, "Option must be finite.");
        }
    }
}
