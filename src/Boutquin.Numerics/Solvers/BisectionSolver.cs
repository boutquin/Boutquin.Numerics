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
    private static readonly T s_two = T.CreateChecked(2);

    private readonly T _functionTolerance;
    private readonly T _bracketTolerance;
    private readonly int _maxIterations;

    /// <summary>
    /// Initializes a new instance of the <see cref="BisectionSolver{T}"/> type.
    /// </summary>
    public BisectionSolver(T functionTolerance, T bracketTolerance, int maxIterations = 200)
    {
        _functionTolerance = functionTolerance;
        _bracketTolerance = bracketTolerance;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc/>
    public RootSolverResult<T> Solve(Func<T, T> objective, T lowerBound, T upperBound)
    {
        var a = lowerBound;
        var b = upperBound;
        var fa = objective(a);
        var fb = objective(b);

        if (T.Abs(fa) < _functionTolerance)
        {
            return new RootSolverResult<T>(a, Converged: true, Iterations: 0, FinalResidual: fa, EstimatedError: T.Zero);
        }

        if (T.Abs(fb) < _functionTolerance)
        {
            return new RootSolverResult<T>(b, Converged: true, Iterations: 0, FinalResidual: fb, EstimatedError: T.Zero);
        }

        if (SameSign(fa, fb))
        {
            throw new InvalidOperationException(
                $"Bisection requires a sign change: f({a}) = {fa}, f({b}) = {fb}.");
        }

        var iterations = 0;

        while (iterations < _maxIterations)
        {
            var mid = a + s_half * (b - a);
            var fmid = objective(mid);
            iterations++;

            if (T.Abs(fmid) < _functionTolerance || (b - a) <= _bracketTolerance)
            {
                return new RootSolverResult<T>(mid, Converged: true, iterations, FinalResidual: fmid,
                    EstimatedError: (b - a) / s_two);
            }

            if (!SameSign(fa, fmid))
            {
                b = mid;
            }
            else
            {
                a = mid;
                fa = fmid;
            }
        }

        var finalMid = a + s_half * (b - a);
        var finalResidual = objective(finalMid);
        var converged = (b - a) <= _bracketTolerance || T.Abs(finalResidual) < _functionTolerance;
        return new RootSolverResult<T>(finalMid, converged, iterations, finalResidual,
            EstimatedError: (b - a) / s_two);
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
/// Bisection root solver — legacy concrete-typed facade forwarding to
/// <see cref="BisectionSolver{T}"/> at <c>T = double</c>.
/// </summary>
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
