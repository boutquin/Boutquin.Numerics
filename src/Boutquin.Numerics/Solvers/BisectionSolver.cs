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
/// Generic bisection root solver — guaranteed convergence for bracketed continuous functions.
/// </summary>
/// <remarks>
/// <para><b>Generic-math tier:</b> B (IEEE 754 transcendental). Works for any <c>T</c>
/// implementing <see cref="IFloatingPointIeee754{TSelf}"/>.</para>
/// <para>
/// Linear convergence: gains one bit of precision per iteration. For faster convergence
/// on smooth functions, prefer <see cref="BrentSolver{T}"/> or <see cref="NewtonRaphsonSolver{T}"/>.
/// </para>
/// </remarks>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
public sealed class BisectionSolver<T> : IBracketedRootSolver<T>
    where T : IFloatingPointIeee754<T>
{
    private static readonly T s_half = T.CreateChecked(0.5);

    private readonly T _functionTolerance;
    private readonly T _bracketTolerance;
    private readonly int _maxIterations;

    /// <summary>
    /// Initializes a new instance of the <see cref="BisectionSolver{T}"/> type.
    /// </summary>
    /// <remarks>
    /// <para><c>functionTolerance</c> is the library's residual stopping extension.
    /// <c>bracketTolerance</c> is the strict bound on the halved signed step in
    /// Numerical Recipes' <c>rtbis</c>; equality does not stop. This step criterion
    /// returns the endpoint with nonpositive function value, with the step magnitude
    /// as its error estimate. It does not return the final interval midpoint.</para>
    /// <para>Exact endpoint roots are accepted before iteration. At machine resolution
    /// the signed step can continue halving even when adding it no longer changes a
    /// coordinate. A convergence flag records the selected stopping rule, not a
    /// guarantee that the residual is small.</para>
    /// </remarks>
    public BisectionSolver(T functionTolerance, T bracketTolerance, int maxIterations = 200)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxIterations);
        if (!T.IsFinite(functionTolerance) || functionTolerance <= T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(functionTolerance), functionTolerance, "Function tolerance must be a positive value.");
        }

        if (!T.IsFinite(bracketTolerance) || bracketTolerance <= T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(bracketTolerance), bracketTolerance, "Bracket tolerance must be a positive value.");
        }

        _functionTolerance = functionTolerance;
        _bracketTolerance = bracketTolerance;
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

        if (T.IsZero(fa))
        {
            return new RootSolverResult<T>(a, Converged: true, Iterations: 0, FinalResidual: fa, EstimatedError: T.Zero);
        }

        if (T.IsZero(fb))
        {
            return new RootSolverResult<T>(b, Converged: true, Iterations: 0, FinalResidual: fb, EstimatedError: T.Zero);
        }

        if (SameSign(fa, fb))
        {
            throw new InvalidOperationException(
                $"Bisection requires a sign change: f({a}) = {fa}, f({b}) = {fb}.");
        }

        // NR rtbis orients the search from the nonpositive-function endpoint.
        // Its signed step is halved independently of rounded endpoint spacing.
        var negativePoint = T.IsNegative(fa) ? a : b;
        var negativeResidual = T.IsNegative(fa) ? fa : fb;
        var step = T.IsNegative(fa) ? HalfWidth(a, b) : -HalfWidth(a, b);
        var estimatedError = T.Abs(step);

        for (var iterations = 0; iterations < _maxIterations; iterations++)
        {
            var mid = negativePoint + step;
            var fmid = EvaluateObjective(objective, mid);
            estimatedError = T.Abs(step);

            if (fmid <= T.Zero)
            {
                negativePoint = mid;
                negativeResidual = fmid;
            }

            // The residual tolerance is an explicit library extension. The listing's
            // exact-zero case is included and returns the evaluated midpoint.
            if (T.Abs(fmid) < _functionTolerance)
            {
                return new RootSolverResult<T>(mid, Converged: true, iterations + 1, FinalResidual: fmid,
                    EstimatedError: estimatedError);
            }

            // Strict halved-step stopping returns rtb, not an unevaluated midpoint.
            if (estimatedError < _bracketTolerance)
            {
                return new RootSolverResult<T>(negativePoint, Converged: true, iterations + 1,
                    FinalResidual: negativeResidual, EstimatedError: estimatedError);
            }

            step *= s_half;
        }

        // Preserve the result-based budget extension: expose the midpoint of the
        // remaining interval, without treating the next unperformed step as convergence.
        var finalMid = negativePoint + step;
        var finalResidual = EvaluateObjective(objective, finalMid);
        return new RootSolverResult<T>(finalMid, Converged: T.Abs(finalResidual) < _functionTolerance,
            _maxIterations, FinalResidual: finalResidual, EstimatedError: T.Abs(step));
    }

    private static T HalfWidth(T a, T b)
    {
        var width = b - a;
        return T.IsFinite(width) ? s_half * width : s_half * b - s_half * a;
    }

    private static bool SameSign(T a, T b)
    {
        if (T.IsZero(a) || T.IsZero(b))
        {
            return T.IsZero(a) && T.IsZero(b);
        }

        return T.IsNegative(a) == T.IsNegative(b);
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
}

/// <summary>
/// Bisection root solver — legacy concrete-typed facade forwarding to
/// <see cref="BisectionSolver{T}"/> at <c>T = double</c>.
/// </summary>
[Obsolete(
    "Use BisectionSolver<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class BisectionSolver : IBracketedRootSolver
{
    private readonly BisectionSolver<double> _inner;

    /// <summary>
    /// Initializes a new instance of the <see cref="BisectionSolver"/> type.
    /// </summary>
    public BisectionSolver(
        double functionTolerance = 1e-12,
        double bracketTolerance = 1e-12,
        int maxIterations = 200)
        => _inner = new BisectionSolver<double>(functionTolerance, bracketTolerance, maxIterations);

    /// <inheritdoc/>
    public RootSolverResult Solve(Func<double, double> objective, double lowerBound, double upperBound)
    {
        var r = _inner.Solve(objective, lowerBound, upperBound);
        return new RootSolverResult(r.Root, r.Converged, r.Iterations, r.FinalResidual, r.EstimatedError);
    }
}
