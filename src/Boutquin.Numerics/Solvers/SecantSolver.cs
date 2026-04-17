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
    private static readonly T s_nearZero = T.CreateChecked(1e-30);
    private static readonly T s_perturbation = T.CreateChecked(1e-4);

    private readonly T _tolerance;
    private readonly int _maxIterations;

    /// <summary>
    /// Initializes a new instance of the <see cref="SecantSolver{T}"/> type.
    /// </summary>
    public SecantSolver(T tolerance, int maxIterations = 50)
    {
        _tolerance = tolerance;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc/>
    public RootSolverResult<T> Solve(Func<T, T> objective, T initialGuess)
    {
        var delta = T.Max(s_perturbation, T.Abs(initialGuess) * s_perturbation);
        return Solve(objective, initialGuess, initialGuess + delta);
    }

    /// <summary>
    /// Solves <c>f(x) = 0</c> using the Secant method with two initial points.
    /// </summary>
    public RootSolverResult<T> Solve(Func<T, T> objective, T p0, T p1)
    {
        var f0 = objective(p0);
        var f1 = objective(p1);

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

        for (var i = 0; i < _maxIterations; i++)
        {
            var denominator = f1 - f0;

            if (T.Abs(denominator) < s_nearZero)
            {
                return new RootSolverResult<T>(p1, Converged: false, i + 1, FinalResidual: f1,
                    EstimatedError: T.Abs(p1 - p0));
            }

            var p2 = p1 - f1 * (p1 - p0) / denominator;
            var f2 = objective(p2);
            var stepSize = T.Abs(p2 - p1);

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
}

/// <summary>
/// Secant method root solver — legacy concrete-typed facade forwarding to
/// <see cref="SecantSolver{T}"/> at <c>T = double</c>.
/// </summary>
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
