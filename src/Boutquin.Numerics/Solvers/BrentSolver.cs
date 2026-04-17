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
/// Generic Brent's method root solver — superlinear convergence with bracketing safety
/// and guaranteed halving interval.
/// </summary>
/// <remarks>
/// <para><b>Generic-math tier:</b> B (IEEE 754 transcendental). Works for any <c>T</c>
/// implementing <see cref="IFloatingPointIeee754{TSelf}"/> — <c>double</c>, <c>float</c>,
/// <c>Half</c>. The constraint ensures the caller can construct objective functions
/// using transcendentals.</para>
/// <para>
/// Production default for curve bootstrap. Combines inverse quadratic interpolation
/// (order ~1.84), the secant method (order φ ≈ 1.618), and bisection (linear) to
/// obtain superlinear convergence on smooth functions while retaining bisection's
/// guaranteed convergence on pathological ones. Typical cost is 5–15 iterations for
/// well-behaved objectives.
/// </para>
/// <para>
/// Incorporates the halving-interval guarantee from Oliveira, Della Pasqua &amp; Steffen
/// (2024) — "Halving Interval Guaranteed for Dekker and Brent Root Finding Methods".
/// </para>
/// </remarks>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
public sealed class BrentSolver<T> : IBracketedRootSolver<T>
    where T : IFloatingPointIeee754<T>
{
    private static readonly T s_half = T.CreateChecked(0.5);
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_three = T.CreateChecked(3);

    private readonly T _tolerance;
    private readonly int _maxIterations;

    /// <summary>
    /// Initializes a new instance of the <see cref="BrentSolver{T}"/> type.
    /// </summary>
    /// <param name="tolerance">Absolute tolerance for convergence.</param>
    /// <param name="maxIterations">Maximum number of iterations.</param>
    public BrentSolver(T tolerance, int maxIterations = 100)
    {
        _tolerance = tolerance;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc/>
    public RootSolverResult<T> Solve(Func<T, T> objective, T lowerBound, T upperBound)
    {
        var a = lowerBound;
        var b = upperBound;
        var fa = objective(a);
        var fb = objective(b);

        if (T.Abs(fa) < _tolerance)
        {
            return new RootSolverResult<T>(a, true, 0, fa, EstimatedError: T.Zero);
        }

        if (T.Abs(fb) < _tolerance)
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
        var e = d;

        var prevBracketWidth = T.Abs(b - a);

        for (var i = 0; i < _maxIterations; i++)
        {
            if (SameSign(fb, fc))
            {
                c = a;
                fc = fa;
                d = b - a;
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

            var tol = s_two * T.Epsilon * T.Abs(b) + s_half * _tolerance;
            var m = s_half * (c - b);

            if (T.Abs(m) <= tol || T.Abs(fb) < _tolerance)
            {
                return new RootSolverResult<T>(b, true, i + 1, fb, EstimatedError: T.Abs(m));
            }

            var currentBracketWidth = T.Abs(c - b);
            var forceHalving = currentBracketWidth > s_half * prevBracketWidth && i > 0;
            prevBracketWidth = currentBracketWidth;

            if (!forceHalving && T.Abs(e) >= tol && T.Abs(fa) > T.Abs(fb))
            {
                T s;
                if (T.Abs(a - c) < T.Epsilon)
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

            b += T.Abs(d) > tol ? d : (m > T.Zero ? tol : -tol);
            fb = objective(b);
        }

        return new RootSolverResult<T>(b, false, _maxIterations, fb, EstimatedError: T.Abs(s_half * (c - b)));
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
/// Brent's method root solver — superlinear convergence with bracketing safety
/// and guaranteed halving interval. Legacy concrete-typed facade forwarding to
/// <see cref="BrentSolver{T}"/> at <c>T = double</c>.
/// </summary>
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
