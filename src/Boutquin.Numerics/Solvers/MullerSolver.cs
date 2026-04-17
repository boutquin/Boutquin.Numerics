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
/// Generic Muller's method root solver — quadratic interpolation through three points.
/// </summary>
/// <remarks>
/// <para><b>Generic-math tier:</b> B (IEEE 754 transcendental). Works for any <c>T</c>
/// implementing <see cref="IFloatingPointIeee754{TSelf}"/>. Uses <c>T.Sqrt</c> for the
/// discriminant (available via <see cref="IRootFunctions{TSelf}"/> which
/// <see cref="IFloatingPointIeee754{TSelf}"/> inherits).</para>
/// <para>
/// Convergence order ≈ 1.84, faster than the Secant method (φ ≈ 1.618).
/// </para>
/// </remarks>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
public sealed class MullerSolver<T> : IUnbracketedRootSolver<T>
    where T : IFloatingPointIeee754<T>
{
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_four = T.CreateChecked(4);
    private static readonly T s_nearZero = T.CreateChecked(1e-30);
    private static readonly T s_perturbation = T.CreateChecked(1e-4);

    private readonly T _tolerance;
    private readonly int _maxIterations;

    /// <summary>
    /// Initializes a new instance of the <see cref="MullerSolver{T}"/> type.
    /// </summary>
    public MullerSolver(T tolerance, int maxIterations = 50)
    {
        _tolerance = tolerance;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc/>
    public RootSolverResult<T> Solve(Func<T, T> objective, T initialGuess)
    {
        var delta = T.Max(s_perturbation, T.Abs(initialGuess) * s_perturbation);
        return Solve(objective, initialGuess - delta, initialGuess, initialGuess + delta);
    }

    /// <summary>
    /// Solves <c>f(x) = 0</c> using Muller's method with three initial points.
    /// </summary>
    public RootSolverResult<T> Solve(Func<T, T> objective, T x0, T x1, T x2)
    {
        var f0 = objective(x0);
        var f1 = objective(x1);
        var f2 = objective(x2);

        if (T.Abs(f0) < _tolerance)
        {
            return new RootSolverResult<T>(x0, Converged: true, Iterations: 0, FinalResidual: f0, EstimatedError: T.Zero);
        }

        if (T.Abs(f1) < _tolerance)
        {
            return new RootSolverResult<T>(x1, Converged: true, Iterations: 0, FinalResidual: f1, EstimatedError: T.Zero);
        }

        if (T.Abs(f2) < _tolerance)
        {
            return new RootSolverResult<T>(x2, Converged: true, Iterations: 0, FinalResidual: f2, EstimatedError: T.Zero);
        }

        for (var i = 0; i < _maxIterations; i++)
        {
            var h0 = x1 - x0;
            var h1 = x2 - x1;
            var delta0 = (f1 - f0) / h0;
            var delta1 = (f2 - f1) / h1;

            var a = (delta1 - delta0) / (h1 + h0);
            var b = a * h1 + delta1;
            var c = f2;

            var disc = b * b - s_four * a * c;

            // Use |disc| under sqrt when negative (near complex root).
            var sqrtDisc = T.Sqrt(T.Abs(disc));

            T x3;

            var denomPlus = b + sqrtDisc;
            var denomMinus = b - sqrtDisc;
            var denom = T.Abs(denomPlus) >= T.Abs(denomMinus) ? denomPlus : denomMinus;

            if (T.Abs(denom) < s_nearZero)
            {
                // Degenerate quadratic — fall back to secant step.
                var secantDenom = f2 - f1;

                if (T.Abs(secantDenom) < s_nearZero)
                {
                    return new RootSolverResult<T>(x2, Converged: false, i + 1, FinalResidual: f2,
                        EstimatedError: T.Abs(x2 - x1));
                }

                x3 = x2 - f2 * (x2 - x1) / secantDenom;
            }
            else
            {
                x3 = x2 - s_two * c / denom;
            }

            var f3 = objective(x3);
            var stepSize = T.Abs(x3 - x2);

            if (T.Abs(f3) < _tolerance || stepSize < _tolerance)
            {
                return new RootSolverResult<T>(x3, Converged: true, i + 1, FinalResidual: f3,
                    EstimatedError: stepSize);
            }

            x0 = x1;
            f0 = f1;
            x1 = x2;
            f1 = f2;
            x2 = x3;
            f2 = f3;
        }

        return new RootSolverResult<T>(x2, Converged: false, _maxIterations, FinalResidual: f2,
            EstimatedError: T.Abs(x2 - x1));
    }
}

/// <summary>
/// Muller's method root solver — legacy concrete-typed facade forwarding to
/// <see cref="MullerSolver{T}"/> at <c>T = double</c>.
/// </summary>
public sealed class MullerSolver : IUnbracketedRootSolver
{
    private readonly MullerSolver<double> _inner;

    /// <summary>
    /// Initializes a new instance of the <see cref="MullerSolver"/> type.
    /// </summary>
    public MullerSolver(double tolerance = 1e-12, int maxIterations = 50)
        => _inner = new MullerSolver<double>(tolerance, maxIterations);

    /// <inheritdoc/>
    public RootSolverResult Solve(Func<double, double> objective, double initialGuess)
    {
        var r = _inner.Solve(objective, initialGuess);
        return new RootSolverResult(r.Root, r.Converged, r.Iterations, r.FinalResidual, r.EstimatedError);
    }

    /// <summary>
    /// Solves <c>f(x) = 0</c> using Muller's method with three initial points.
    /// </summary>
    public RootSolverResult Solve(Func<double, double> objective, double x0, double x1, double x2)
    {
        var r = _inner.Solve(objective, x0, x1, x2);
        return new RootSolverResult(r.Root, r.Converged, r.Iterations, r.FinalResidual, r.EstimatedError);
    }
}
