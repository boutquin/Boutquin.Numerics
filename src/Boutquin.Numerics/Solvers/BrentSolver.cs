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
/// Generic Brent's method root solver — interpolation with bracketing safety
/// and a forced-bisection safeguard that halves the bracket at least every two iterations.
/// </summary>
/// <remarks>
/// <para><b>Generic-math tier:</b> B (IEEE 754 transcendental). Works for any <c>T</c>
/// implementing <see cref="IFloatingPointIeee754{TSelf}"/> — <c>double</c>, <c>float</c>,
/// <c>Half</c>. The constraint ensures the caller can construct objective functions
/// using transcendentals.</para>
/// <para>
/// Combines inverse quadratic interpolation, secant steps, and bisection. Interpolation
/// can accelerate smooth problems, but the forced-bisection safeguard can also increase
/// their evaluation count compared with classic Brent. No universal superlinear rate
/// or typical iteration count is promised; the bracket-reduction bound below applies.
/// </para>
/// <para>
/// Forced-bisection safeguard: when an iteration fails to shrink the bracket to at most half
/// its previous width, the next step is a bisection. The bracket therefore halves at least
/// every two iterations, bounding the worst case at <c>2⌈log₂((b − a) / tolerance)⌉ + 2</c>
/// iterations, twice bisection's count. The safeguard is this library's own. Steffen,
/// Della Pasqua, de Oliveira &amp; da Silva (2025), "Halving interval guaranteed for Dekker and
/// Brent root finding methods" (Examples and Counterexamples 7, 100173), take a different
/// route, modifying Brent's acceptance test; they demonstrate per-iteration halving on two
/// examples and state no iteration bound.
/// </para>
/// </remarks>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
public sealed class BrentSolver<T> : IBracketedRootSolver<T>
    where T : IFloatingPointIeee754<T>
{
    private static readonly T s_half = T.CreateChecked(0.5);
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_three = T.CreateChecked(3);

    // Machine epsilon for T: the smallest positive value eps such that
    // (T.One + eps) != T.One. Brent's algorithm uses this (not T.Epsilon,
    // which is the smallest positive denormal — ~5e-324 for double) to scale
    // tolerance relative to |b|.
    private static readonly T s_machineEpsilon = ComputeMachineEpsilon();

    private readonly T _tolerance;
    private readonly int _maxIterations;

    private static T ComputeMachineEpsilon()
    {
        var eps = T.One;
        while (T.One + eps / s_two != T.One)
        {
            eps /= s_two;
        }

        return eps;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="BrentSolver{T}"/> type.
    /// </summary>
    /// <remarks>
    /// <para><b>Tolerance-API note.</b> Brent takes a <em>single</em> <c>tolerance</c>,
    /// using Numerical Recipes (3rd edition), section 9.3's <c>zbrent</c> convention:
    /// iteration stops when the bracket half-width is within
    /// <c>2·ε·|b| + tolerance/2</c> or <c>f(b) = 0</c> exactly.
    /// This corresponds to <c>t = tolerance/2</c> in Brent's (1973) <c>zero</c>.
    /// A small residual alone never stops it, because on a flat function a point far from the
    /// root can have <c>|f|</c> below any tolerance. The bracket-shrinking solvers
    /// (<see cref="BisectionSolver{T}"/>, <see cref="NewtonRaphsonSolver{T}"/>) instead expose
    /// separate <c>functionTolerance</c>/<c>bracketTolerance</c>(/<c>stepTolerance</c>)
    /// controls.</para>
    /// </remarks>
    /// <param name="tolerance">Positive finite absolute tolerance in the <c>zbrent</c>
    /// convention. Brent's rounded-arithmetic bound becomes approximately
    /// <c>tolerance + 6·ε·|root|</c>, under his arithmetic and sign-bracket assumptions,
    /// for a zero of the computed objective; inaccurate function evaluation can move that zero.</param>
    /// <param name="maxIterations">Maximum number of iterations.</param>
    public BrentSolver(T tolerance, int maxIterations = 100)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxIterations);
        if (!T.IsFinite(tolerance) || tolerance <= T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(tolerance), tolerance, "Tolerance must be a positive value.");
        }

        _tolerance = tolerance;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc/>
    public RootSolverResult<T> Solve(Func<T, T> objective, T lowerBound, T upperBound)
    {
        ArgumentNullException.ThrowIfNull(objective);

        if (!T.IsFinite(lowerBound))
        {
            throw new ArgumentOutOfRangeException(nameof(lowerBound), lowerBound, "Bound must be finite.");
        }

        if (!T.IsFinite(upperBound))
        {
            throw new ArgumentOutOfRangeException(nameof(upperBound), upperBound, "Bound must be finite.");
        }

        if (lowerBound >= upperBound)
        {
            throw new ArgumentException("Lower bound must be less than upper bound.", nameof(lowerBound));
        }

        var a = lowerBound;
        var b = upperBound;
        var fa = EvaluateObjective(objective, a);
        var fb = EvaluateObjective(objective, b);

        // Brent's `zero` accepts an endpoint only when f vanishes there exactly; a small
        // residual on a flat function says nothing about the distance to the root.
        if (T.IsZero(fa))
        {
            return new RootSolverResult<T>(a, true, 0, fa, EstimatedError: T.Zero);
        }

        if (T.IsZero(fb))
        {
            return new RootSolverResult<T>(b, true, 0, fb, EstimatedError: T.Zero);
        }

        if (SameSign(fa, fb))
        {
            throw new InvalidOperationException(
                $"Brent's method requires a sign change: f({a}) = {fa}, f({b}) = {fb}.");
        }

        var c = a;
        var fc = fa;
        var d = b - a;
        if (!T.IsFinite(d))
        {
            d = HalfDifference(a, b);
        }

        var e = d;

        var prevHalfWidth = T.Abs(HalfDifference(a, b));

        for (var i = 0; i < _maxIterations; i++)
        {
            if (SameSign(fb, fc))
            {
                c = a;
                fc = fa;
                d = b - a;
                if (!T.IsFinite(d))
                {
                    d = HalfDifference(a, b);
                }

                e = d;
            }

            if (T.Abs(fc) < T.Abs(fb))
            {
                a = b;
                b = c;
                c = a;
                fa = fb;
                fb = fc;
                fc = fa;
            }

            var tol = s_two * s_machineEpsilon * T.Abs(b) + s_half * _tolerance;
            var m = HalfDifference(b, c);

            // Brent (1973) `zero`: stop when the bracket half-width is within tol or f(b) = 0.
            if (T.Abs(m) <= tol || T.IsZero(fb))
            {
                return new RootSolverResult<T>(b, true, i + 1, fb, EstimatedError: T.Abs(m));
            }

            var currentHalfWidth = T.Abs(m);
            var forceHalving = currentHalfWidth > s_half * prevHalfWidth && i > 0;
            prevHalfWidth = currentHalfWidth;

            // Interpolation uses full differences; bisect while any of them is unrepresentable.
            if (!forceHalving && T.IsFinite(c - b) && T.IsFinite(b - a) && T.IsFinite(a - c)
                && T.Abs(e) >= tol && T.Abs(fa) > T.Abs(fb))
            {
                T s;
                if (a == c)
                {
                    // Secant method
                    s = fb / fa;
                    var p = s_two * m * s;
                    var q = T.One - s;
                    if (p > T.Zero)
                    {
                        q = -q;
                    }
                    else
                    {
                        p = -p;
                    }

                    if (s_two * p < T.Min(s_three * m * q - T.Abs(tol * q), T.Abs(e * q)))
                    {
                        e = d;
                        d = p / q;
                    }
                    else
                    {
                        d = m;
                        e = m;
                    }
                }
                else
                {
                    // Inverse quadratic interpolation
                    var q2 = fa / fc;
                    var r = fb / fc;
                    s = fb / fa;
                    var p = s * (s_two * m * q2 * (q2 - r) - (b - a) * (r - T.One));
                    var q1 = (q2 - T.One) * (r - T.One) * (s - T.One);
                    if (p > T.Zero)
                    {
                        q1 = -q1;
                    }
                    else
                    {
                        p = -p;
                    }

                    if (s_two * p < T.Min(s_three * m * q1 - T.Abs(tol * q1), T.Abs(e * q1)))
                    {
                        e = d;
                        d = p / q1;
                    }
                    else
                    {
                        d = m;
                        e = m;
                    }
                }
            }
            else
            {
                d = m;
                e = m;
            }

            a = b;
            fa = fb;

            var next = b + (T.Abs(d) > tol ? d : (m > T.Zero ? tol : -tol));
            if (!T.IsFinite(next) || next < T.Min(b, c) || next > T.Max(b, c))
            {
                next = b + m;
            }

            b = next;
            fb = EvaluateObjective(objective, b);
        }

        if (SameSign(fb, fc))
        {
            c = a;
        }

        var finalHalfWidth = T.Abs(HalfDifference(b, c));
        var finalTolerance = s_two * s_machineEpsilon * T.Abs(b) + s_half * _tolerance;
        return new RootSolverResult<T>(b, T.IsZero(fb) || finalHalfWidth <= finalTolerance,
            _maxIterations, fb, EstimatedError: finalHalfWidth);
    }

    private static T HalfDifference(T from, T to)
    {
        var difference = to - from;
        return T.IsFinite(difference) ? s_half * difference : s_half * to - s_half * from;
    }

    private static T EvaluateObjective(Func<T, T> objective, T point)
    {
        var value = objective(point);
        if (!T.IsFinite(value))
        {
            throw new InvalidOperationException($"Objective returned a non-finite value at x = {point}.");
        }

        return value;
    }

    private static bool SameSign(T a, T b)
    {
        if (T.IsZero(a) || T.IsZero(b))
        {
            return T.IsZero(a) && T.IsZero(b);
        }

        return T.IsNegative(a) == T.IsNegative(b);
    }
}

/// <summary>
/// Brent's method root solver — interpolation with bracketing safety
/// and a forced-bisection safeguard. Legacy concrete-typed facade forwarding to
/// <see cref="BrentSolver{T}"/> at <c>T = double</c>.
/// </summary>
[Obsolete(
    "Use BrentSolver<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class BrentSolver : IBracketedRootSolver
{
    private readonly BrentSolver<double> _inner;

    /// <summary>
    /// Initializes a new instance of the <see cref="BrentSolver"/> type.
    /// </summary>
    /// <param name="tolerance">Absolute tolerance for convergence.</param>
    /// <param name="maxIterations">Maximum number of iterations.</param>
    public BrentSolver(double tolerance = 1e-12, int maxIterations = 100)
        => _inner = new BrentSolver<double>(tolerance, maxIterations);

    /// <inheritdoc/>
    public RootSolverResult Solve(Func<double, double> objective, double lowerBound, double upperBound)
    {
        var r = _inner.Solve(objective, lowerBound, upperBound);
        return new RootSolverResult(r.Root, r.Converged, r.Iterations, r.FinalResidual, r.EstimatedError);
    }
}
