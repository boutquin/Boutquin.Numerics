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
/// Generic Newton-Raphson solver with safeguarded bisection fallback.
/// </summary>
/// <remarks>
/// <para><b>Generic-math tier:</b> B (IEEE 754 transcendental). Works for any <c>T</c>
/// implementing <see cref="IFloatingPointIeee754{TSelf}"/>.</para>
/// <para>
/// Quadratic convergence when the derivative is well-behaved. Falls back to bisection
/// when the Newton step would leave the bracketed interval.
/// </para>
/// </remarks>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
public sealed class NewtonRaphsonSolver<T> : IBracketedRootSolver<T>, IUnbracketedRootSolver<T>
    where T : IFloatingPointIeee754<T>
{
    private static readonly T s_half = T.CreateChecked(0.5);
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_nearZero = T.CreateChecked(1e-30);
    private static readonly T s_fdStep = T.CreateChecked(1e-8);

    private readonly T _functionTolerance;
    private readonly T _bracketTolerance;
    private readonly T _stepTolerance;
    private readonly int _maxIterations;
    private readonly Func<T, T>? _derivative;

    /// <summary>
    /// Initializes a new instance of the <see cref="NewtonRaphsonSolver{T}"/> type.
    /// </summary>
    public NewtonRaphsonSolver(
        T functionTolerance,
        T bracketTolerance,
        T stepTolerance,
        int maxIterations = 50,
        Func<T, T>? derivative = null)
    {
        _functionTolerance = functionTolerance;
        _bracketTolerance = bracketTolerance;
        _stepTolerance = stepTolerance;
        _maxIterations = maxIterations;
        _derivative = derivative;
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
                $"Newton-Raphson requires a sign change: f({a}) = {fa}, f({b}) = {fb}.");
        }

        var x = a + s_half * (b - a);
        var fx = objective(x);
        var lastStepSize = T.NaN;

        for (var i = 0; i < _maxIterations; i++)
        {
            if (T.Abs(fx) < _functionTolerance)
            {
                return new RootSolverResult<T>(x, Converged: true, i + 1, FinalResidual: fx,
                    EstimatedError: T.IsNaN(lastStepSize) ? (b - a) / s_two : lastStepSize);
            }

            var dfx = _derivative is not null ? _derivative(x) : NumericalDerivative(objective, x);
            var newtonStep = T.Abs(dfx) > s_nearZero ? x - fx / dfx : T.NaN;

            T xNew;
            if (T.IsFinite(newtonStep) && newtonStep > a && newtonStep < b)
            {
                xNew = newtonStep;
            }
            else
            {
                xNew = a + s_half * (b - a);
            }

            lastStepSize = T.Abs(xNew - x);

            if (_stepTolerance > T.Zero && lastStepSize <= _stepTolerance)
            {
                var fNew = objective(xNew);
                return new RootSolverResult<T>(xNew, Converged: true, i + 1, FinalResidual: fNew,
                    EstimatedError: lastStepSize);
            }

            x = xNew;
            fx = objective(x);

            if (!SameSign(fa, fx))
            {
                b = x;
            }
            else
            {
                a = x;
                fa = fx;
            }

            if ((b - a) <= _bracketTolerance)
            {
                return new RootSolverResult<T>(x, Converged: true, i + 1, FinalResidual: fx,
                    EstimatedError: (b - a) / s_two);
            }
        }

        return new RootSolverResult<T>(x, Converged: false, _maxIterations, FinalResidual: fx,
            EstimatedError: (b - a) / s_two);
    }

    /// <inheritdoc/>
    public RootSolverResult<T> Solve(Func<T, T> objective, T initialGuess)
    {
        var x = initialGuess;
        var fx = objective(x);

        if (T.Abs(fx) < _functionTolerance)
        {
            return new RootSolverResult<T>(x, Converged: true, Iterations: 0, FinalResidual: fx,
                EstimatedError: T.Zero);
        }

        for (var i = 0; i < _maxIterations; i++)
        {
            var dfx = _derivative is not null ? _derivative(x) : NumericalDerivative(objective, x);

            if (T.Abs(dfx) <= s_nearZero)
            {
                return new RootSolverResult<T>(x, Converged: false, i + 1, FinalResidual: fx,
                    EstimatedError: T.NaN);
            }

            var step = fx / dfx;
            var xNew = x - step;

            if (!T.IsFinite(xNew))
            {
                return new RootSolverResult<T>(x, Converged: false, i + 1, FinalResidual: fx,
                    EstimatedError: T.NaN);
            }

            var stepSize = T.Abs(step);

            x = xNew;
            fx = objective(x);

            if (T.Abs(fx) < _functionTolerance)
            {
                return new RootSolverResult<T>(x, Converged: true, i + 1, FinalResidual: fx,
                    EstimatedError: stepSize);
            }

            if (_stepTolerance > T.Zero && stepSize <= _stepTolerance)
            {
                return new RootSolverResult<T>(x, Converged: true, i + 1, FinalResidual: fx,
                    EstimatedError: stepSize);
            }
        }

        return new RootSolverResult<T>(x, Converged: false, _maxIterations, FinalResidual: fx,
            EstimatedError: T.NaN);
    }

    private static T NumericalDerivative(Func<T, T> f, T x)
    {
        var h = T.Max(s_fdStep, T.Abs(x) * s_fdStep);
        return (f(x + h) - f(x - h)) / (s_two * h);
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
/// Newton-Raphson solver — legacy concrete-typed facade forwarding to
/// <see cref="NewtonRaphsonSolver{T}"/> at <c>T = double</c>.
/// </summary>
public sealed class NewtonRaphsonSolver : IBracketedRootSolver, IUnbracketedRootSolver
{
    private readonly NewtonRaphsonSolver<double> _inner;

    /// <summary>
    /// Initializes a new instance of the <see cref="NewtonRaphsonSolver"/> type.
    /// </summary>
    public NewtonRaphsonSolver(
        double functionTolerance = 1e-12,
        double bracketTolerance = 1e-12,
        double stepTolerance = 0,
        int maxIterations = 50,
        Func<double, double>? derivative = null)
        => _inner = new NewtonRaphsonSolver<double>(
            functionTolerance, bracketTolerance, stepTolerance, maxIterations, derivative);

    /// <inheritdoc/>
    public RootSolverResult Solve(Func<double, double> objective, double lowerBound, double upperBound)
    {
        var r = _inner.Solve(objective, lowerBound, upperBound);
        return new RootSolverResult(r.Root, r.Converged, r.Iterations, r.FinalResidual, r.EstimatedError);
    }

    /// <inheritdoc/>
    public RootSolverResult Solve(Func<double, double> objective, double initialGuess)
    {
        var r = _inner.Solve(objective, initialGuess);
        return new RootSolverResult(r.Root, r.Converged, r.Iterations, r.FinalResidual, r.EstimatedError);
    }
}
