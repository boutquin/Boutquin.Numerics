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
/// Generic secant method root solver — superlinear convergence without derivatives.
/// </summary>
/// <remarks>
/// <para><b>Generic-math tier:</b> B (IEEE 754 transcendental). Works for any <c>T</c>
/// implementing <see cref="IFloatingPointIeee754{TSelf}"/>.</para>
/// <para>
/// Convergence order φ ≈ 1.618 (golden ratio). Uses two initial points to approximate
/// the derivative via a secant line.
/// </para>
/// </remarks>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
public sealed class SecantSolver<T> : IUnbracketedRootSolver<T>
    where T : IFloatingPointIeee754<T>
{
    private static readonly T s_half = T.CreateChecked(0.5);
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_perturbation = T.CreateChecked(1e-4);

    private readonly T _tolerance;
    private readonly int _maxIterations;

    /// <summary>
    /// Initializes a new instance of the <see cref="SecantSolver{T}"/> type.
    /// </summary>
    /// <remarks>
    /// <para><b>Tolerance-API note.</b> Like <see cref="MullerSolver{T}"/>, Secant takes a
    /// <em>single</em> <c>tolerance</c> serving both the residual and step-size roles:
    /// it converges when <c>|f(x)|</c> is below the tolerance, or when the computed secant step is strictly below the
    /// tolerance. A small step does not by itself certify a root; the returned residual
    /// remains available to distinguish numerical stopping from root accuracy.
    /// The bracket-shrinking solvers (<see cref="BisectionSolver{T}"/>,
    /// <see cref="NewtonRaphsonSolver{T}"/>) instead expose separate
    /// <c>functionTolerance</c>/<c>bracketTolerance</c>(/<c>stepTolerance</c>) controls.
    /// This asymmetry is intentional.</para>
    /// </remarks>
    public SecantSolver(T tolerance, int maxIterations = 50)
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
    public RootSolverResult<T> Solve(Func<T, T> objective, T initialGuess)
    {
        ArgumentNullException.ThrowIfNull(objective);
        if (!T.IsFinite(initialGuess))
        {
            throw new ArgumentOutOfRangeException(nameof(initialGuess), initialGuess, "Initial point must be finite.");
        }

        var initialResidual = EvaluateObjective(objective, initialGuess);
        if (initialResidual == T.Zero)
        {
            return new RootSolverResult<T>(initialGuess, Converged: true, Iterations: 0,
                FinalResidual: initialResidual, EstimatedError: T.Zero);
        }

        var delta = T.Max(s_perturbation, T.Abs(initialGuess) * s_perturbation);
        var second = initialGuess + delta;
        if (!T.IsFinite(second) || second == initialGuess)
        {
            second = initialGuess - delta;
            if (!T.IsFinite(second) || second == initialGuess)
            {
                second = initialGuess >= T.Zero
                    ? T.BitDecrement(initialGuess)
                    : T.BitIncrement(initialGuess);
            }
        }

        return Solve(objective, initialGuess, second);
    }

    /// <summary>
    /// Solves <c>f(x) = 0</c> using the Secant method with two initial points.
    /// </summary>
    public RootSolverResult<T> Solve(Func<T, T> objective, T p0, T p1)
    {
        ArgumentNullException.ThrowIfNull(objective);
        if (!T.IsFinite(p0))
        {
            throw new ArgumentOutOfRangeException(nameof(p0), p0, "Initial point must be finite.");
        }

        if (!T.IsFinite(p1))
        {
            throw new ArgumentOutOfRangeException(nameof(p1), p1, "Initial point must be finite.");
        }

        var f0 = EvaluateObjective(objective, p0);
        var f1 = EvaluateObjective(objective, p1);

        if (T.Abs(f0) < _tolerance)
        {
            return new RootSolverResult<T>(p0, Converged: true, Iterations: 0, FinalResidual: f0,
                EstimatedError: T.Zero);
        }

        if (T.Abs(f1) < _tolerance)
        {
            return new RootSolverResult<T>(p1, Converged: true, Iterations: 0, FinalResidual: f1,
                EstimatedError: T.Zero);
        }

        // NR rtsec starts from the smaller residual; equality keeps the second seed current.
        if (T.Abs(f0) < T.Abs(f1))
        {
            (p0, p1) = (p1, p0);
            (f0, f1) = (f1, f0);
        }

        for (var i = 0; i < _maxIterations; i++)
        {
            var denominator = f1 - f0;

            // The secant weight f1/(f1 - f0) is invariant under a common residual scale, so only
            // exactly equal residuals leave the secant line without a root to extrapolate to;
            // a small nonzero difference is a scale. A weight too large to represent is caught
            // by the finite check on the next iterate below.
            if (T.IsZero(denominator))
            {
                return new RootSolverResult<T>(p1, Converged: false, i + 1, FinalResidual: f1,
                    EstimatedError: T.Abs(p1 - p0));
            }

            var weight = f1 / denominator;
            if (!T.IsFinite(denominator))
            {
                // Dividing both residuals by two preserves their ratio without overflow.
                weight = (f1 / T.CreateChecked(2)) / (f1 / T.CreateChecked(2) - f0 / T.CreateChecked(2));
            }

            var width = p1 - p0;
            T p2;
            T stepSize;
            if (!T.IsFinite(width))
            {
                p2 = (T.One - weight) * p1 + weight * p0;
                stepSize = T.Abs(p2 - p1);
            }
            else
            {
                var move = weight * width;
                if (T.IsFinite(move))
                {
                    p2 = p1 - move;
                    stepSize = T.Abs(move);
                }
                else
                {
                    // The product weight*width overflowed, but p1 - weight*width is representable
                    // when p1 offsets it. Halve the width before the product (a power-of-two
                    // scaling, exact here since the overflow forces |p1| to be large) and double
                    // the result; an iterate that is itself out of range still fails the finite
                    // check below.
                    var halfMove = weight * (width * s_half);
                    p2 = (p1 * s_half - halfMove) * s_two;
                    stepSize = T.Abs(halfMove) * s_two;
                }
            }

            if (!T.IsFinite(p2))
            {
                return new RootSolverResult<T>(p1, Converged: false, i + 1, FinalResidual: f1,
                    EstimatedError: T.NaN);
            }

            var f2 = EvaluateObjective(objective, p2);

            // NR rtsec tests the computed secant increment, not merely the rounded
            // displacement, and does not require a residual sign change.
            if (T.Abs(f2) < _tolerance || stepSize < _tolerance)
            {
                return new RootSolverResult<T>(p2, Converged: true, i + 1, FinalResidual: f2,
                    EstimatedError: stepSize);
            }

            p0 = p1;
            f0 = f1;
            p1 = p2;
            f1 = f2;
        }

        return new RootSolverResult<T>(p1, Converged: false, _maxIterations, FinalResidual: f1,
            EstimatedError: T.Abs(p1 - p0));
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
/// Secant method root solver — legacy concrete-typed facade forwarding to
/// <see cref="SecantSolver{T}"/> at <c>T = double</c>.
/// </summary>
[Obsolete(
    "Use SecantSolver<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class SecantSolver : IUnbracketedRootSolver
{
    private readonly SecantSolver<double> _inner;

    /// <summary>
    /// Initializes a new instance of the <see cref="SecantSolver"/> type.
    /// </summary>
    public SecantSolver(double tolerance = 1e-12, int maxIterations = 50)
        => _inner = new SecantSolver<double>(tolerance, maxIterations);

    /// <inheritdoc/>
    public RootSolverResult Solve(Func<double, double> objective, double initialGuess)
    {
        var r = _inner.Solve(objective, initialGuess);
        return new RootSolverResult(r.Root, r.Converged, r.Iterations, r.FinalResidual, r.EstimatedError);
    }

    /// <summary>
    /// Solves <c>f(x) = 0</c> using the Secant method with two initial points.
    /// </summary>
    public RootSolverResult Solve(Func<double, double> objective, double p0, double p1)
    {
        var r = _inner.Solve(objective, p0, p1);
        return new RootSolverResult(r.Root, r.Converged, r.Iterations, r.FinalResidual, r.EstimatedError);
    }
}
