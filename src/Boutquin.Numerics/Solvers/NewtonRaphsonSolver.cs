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
/// Quadratic convergence when the derivative is well-behaved. The bracketed entry point
/// follows Numerical Recipes' <c>rtsafe</c> (Press et al., 3rd ed., §9.4): it bisects when
/// the Newton step would leave the bracketed interval or is not decreasing fast enough
/// (<c>|2f| &gt; |dxold·f′|</c>, which catches the linear convergence of Newton at a
/// multiple root).
/// </para>
/// <para>
/// Without an analytic <c>derivative</c>, a central difference with relative step
/// <c>∛ε</c> of <c>T</c> is used, so the finite-difference step resolves at <c>float</c>
/// and <c>Half</c> as well as <c>double</c>. The bracketed entry point clamps the stencil
/// to <c>[lowerBound, upperBound]</c> (one-sided at an edge), so the objective is never
/// evaluated outside the caller's bracket.
/// </para>
/// </remarks>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
public sealed class NewtonRaphsonSolver<T> : IBracketedRootSolver<T>, IUnbracketedRootSolver<T>
    where T : IFloatingPointIeee754<T>
{
    private static readonly T s_half = T.CreateChecked(0.5);
    private static readonly T s_two = T.CreateChecked(2);

    // Relative central-difference step ∛ε of the working type (ε = ulp(1)): it balances the
    // O(h²) truncation error against the O(ε/h) rounding error. A fixed double-calibrated
    // step is below float resolution and rounds to zero at Half.
    private static readonly T s_fdStep = T.Cbrt(T.BitIncrement(T.One) - T.One);

    private readonly T _functionTolerance;
    private readonly T _bracketTolerance;
    private readonly T _stepTolerance;
    private readonly int _maxIterations;
    private readonly Func<T, T>? _derivative;

    /// <summary>
    /// Initializes a new instance of the <see cref="NewtonRaphsonSolver{T}"/> type.
    /// </summary>
    /// <remarks>
    /// <para><b>Tolerance-API note.</b> This solver exposes three <em>separate</em>
    /// convergence controls — <c>functionTolerance</c> on the residual <c>|f(x)|</c>,
    /// <c>bracketTolerance</c> on the bracket width <c>(b − a)</c>, and <c>stepTolerance</c>
    /// on the iterate step <c>|xₙ − xₙ₋₁|</c> — each governing an independent stopping
    /// condition. The interpolation-based solvers instead take a single <c>tolerance</c>:
    /// a bracket tolerance for <see cref="BrentSolver{T}"/>, a residual and step tolerance for
    /// <see cref="SecantSolver{T}"/> and <see cref="MullerSolver{T}"/>. The asymmetry is intentional.
    /// The bracketed entry point follows <c>rtsafe</c>: a strictly sub-tolerance computed
    /// step or representational stagnation stops independently of the residual. The
    /// unbracketed entry point retains its separate residual-based safeguard.</para>
    /// </remarks>
    public NewtonRaphsonSolver(
        T functionTolerance,
        T bracketTolerance,
        T stepTolerance,
        int maxIterations = 50,
        Func<T, T>? derivative = null)
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

        // stepTolerance = 0 is the documented sentinel for "no step-based stopping"
        // (the step-convergence branches gate on `_stepTolerance > T.Zero`); a non-finite
        // or negative value silently corrupts that gate, so those are rejected.
        if (!T.IsFinite(stepTolerance) || stepTolerance < T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(stepTolerance), stepTolerance, "Step tolerance must be zero or a positive value.");
        }

        _functionTolerance = functionTolerance;
        _bracketTolerance = bracketTolerance;
        _stepTolerance = stepTolerance;
        _maxIterations = maxIterations;
        _derivative = derivative;
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
                $"Newton-Raphson requires a sign change: f({a}) = {fa}, f({b}) = {fb}.");
        }

        var x = a + HalfWidth(a, b);
        var fx = EvaluateObjective(objective, x);
        var lastStepSize = T.NaN;

        // Step history for rtsafe's slow-decrease test (NR 3e §9.4): dx is the last step,
        // dxOld the one before it. Both start at the bracket width, as in rtsafe.
        var dx = T.Abs(b - a);
        var dxOld = dx;

        for (var i = 0; i < _maxIterations; i++)
        {
            if (T.Abs(fx) < _functionTolerance)
            {
                return new RootSolverResult<T>(x, Converged: true, i + 1, FinalResidual: fx,
                    EstimatedError: T.IsNaN(lastStepSize) ? HalfWidth(a, b) : lastStepSize);
            }

            // The finite-difference stencil is clamped to the caller's bracket: the objective
            // need not be defined outside [lowerBound, upperBound] (e.g. √x or log x on a
            // bracket whose edge is the domain boundary).
            var dfx = _derivative is not null
                ? _derivative(x)
                : NumericalDerivative(objective, x, lowerBound, upperBound);
            var newtonStep = T.IsFinite(dfx) && dfx != T.Zero ? x - fx / dfx : T.NaN;

            // rtsafe bisects when Newton is not decreasing fast enough: the proposed step
            // |f/f′| exceeds half the step before last, i.e. |2f| > |dxOld·f′|. Without this
            // test a multiple root drives Newton through linearly shrinking steps that can
            // be several times slower than bisection.
            var slowDecrease = T.Abs(s_two * fx) > T.Abs(dxOld * dfx);
            dxOld = dx;

            T xNew;
            bool stagnated;
            // NR rtsafe's out-of-range test is strict: endpoint equality alone
            // permits a Newton step. Comparisons avoid overflowing its product test.
            if (T.IsFinite(newtonStep) && newtonStep >= a && newtonStep <= b && !slowDecrease)
            {
                xNew = newtonStep;
                dx = T.Abs(fx / dfx);
                stagnated = xNew == x;
            }
            else
            {
                // Orient the bisection from the endpoint whose function is negative,
                // as in rtsafe. Its stagnation test compares against that endpoint.
                var negativeEndpoint = T.IsNegative(fa) ? a : b;
                var positiveEndpoint = T.IsNegative(fa) ? b : a;
                dx = HalfWidth(negativeEndpoint, positiveEndpoint);
                xNew = negativeEndpoint + dx;
                stagnated = xNew == negativeEndpoint;
            }

            lastStepSize = T.Abs(xNew - x);

            // The selected listing stops on stagnation or a strictly small computed
            // step. Neither requires a small residual; this is a stopping diagnostic.
            if (stagnated || T.Abs(dx) < _stepTolerance)
            {
                var fNew = xNew == x ? fx : EvaluateObjective(objective, xNew);
                return new RootSolverResult<T>(xNew, Converged: true, i + 1, FinalResidual: fNew,
                    EstimatedError: T.Abs(dx));
            }

            x = xNew;
            fx = EvaluateObjective(objective, x);

            if (!SameSign(fa, fx))
            {
                b = x;
            }
            else
            {
                a = x;
                fa = fx;
            }

            // A bracket whose midpoint rounds onto an endpoint is at machine resolution and
            // cannot shrink further, even when bracketTolerance is below ulp(root).
            var midpoint = a + HalfWidth(a, b);
            if ((b - a) <= _bracketTolerance || midpoint <= a || midpoint >= b)
            {
                return new RootSolverResult<T>(x, Converged: true, i + 1, FinalResidual: fx,
                    EstimatedError: HalfWidth(a, b));
            }
        }

        return new RootSolverResult<T>(x, Converged: T.Abs(fx) < _functionTolerance,
            _maxIterations, FinalResidual: fx,
            EstimatedError: HalfWidth(a, b));
    }

    private static T HalfWidth(T a, T b)
    {
        var width = b - a;
        return T.IsFinite(width) ? s_half * width : s_half * b - s_half * a;
    }

    /// <inheritdoc/>
    public RootSolverResult<T> Solve(Func<T, T> objective, T initialGuess)
    {
        ArgumentNullException.ThrowIfNull(objective);
        if (!T.IsFinite(initialGuess))
        {
            throw new ArgumentOutOfRangeException(nameof(initialGuess), initialGuess, "Initial point must be finite.");
        }

        var x = initialGuess;
        var fx = EvaluateObjective(objective, x);

        if (T.Abs(fx) < _functionTolerance)
        {
            return new RootSolverResult<T>(x, Converged: true, Iterations: 0, FinalResidual: fx,
                EstimatedError: T.Zero);
        }

        for (var i = 0; i < _maxIterations; i++)
        {
            var dfx = _derivative is not null
                ? _derivative(x)
                : NumericalDerivative(objective, x, T.NegativeInfinity, T.PositiveInfinity);

            // The iteration is scale-invariant (f and f′ scale together), so only a derivative that
            // is exactly zero carries no direction; a small nonzero slope is a scale, not a
            // vanishing one. An iterate pushed outside the type's range is caught below.
            if (T.IsZero(dfx))
            {
                return new RootSolverResult<T>(x, Converged: false, i + 1, FinalResidual: fx,
                    EstimatedError: T.NaN);
            }

            var step = fx / dfx;
            T xNew;
            T stepSize;
            if (T.IsFinite(step))
            {
                xNew = x - step;
                stepSize = T.Abs(step);
            }
            else
            {
                // The quotient overflowed, but x - fx/dfx is representable when |fx/dfx| is
                // within a factor of two of the range and x offsets it. Halve the step before
                // the subtraction (a power-of-two scaling, exact here) and double the result;
                // if the iterate itself is out of range the finite check below still fails.
                // |dfx| < |fx| / MaxValue, so doubling dfx cannot overflow.
                var halfStep = fx / (dfx * s_two);
                xNew = (x * s_half - halfStep) * s_two;
                stepSize = T.Abs(halfStep) * s_two;
            }

            if (!T.IsFinite(xNew))
            {
                return new RootSolverResult<T>(x, Converged: false, i + 1, FinalResidual: fx,
                    EstimatedError: T.NaN);
            }

            x = xNew;
            fx = EvaluateObjective(objective, x);

            if (T.Abs(fx) < _functionTolerance)
            {
                return new RootSolverResult<T>(x, Converged: true, i + 1, FinalResidual: fx,
                    EstimatedError: stepSize);
            }

        }

        return new RootSolverResult<T>(x, Converged: false, _maxIterations, FinalResidual: fx,
            EstimatedError: T.NaN);
    }

    private static T NumericalDerivative(Func<T, T> f, T x, T lowerLimit, T upperLimit)
    {
        // The stencil never leaves [lowerLimit, upperLimit]; at a limit the difference
        // degrades to one-sided rather than evaluating the objective outside it.
        var h = x == T.Zero ? s_fdStep : T.Abs(x) * s_fdStep;
        var upper = T.Min(x + h, upperLimit);
        var lower = T.Max(x - h, lowerLimit);

        // Near a finite type limit, use the representable side of the stencil.
        if (!T.IsFinite(upper))
        {
            return DifferenceQuotient(EvaluateObjective(f, x), EvaluateObjective(f, lower), x - lower);
        }

        if (!T.IsFinite(lower))
        {
            return DifferenceQuotient(EvaluateObjective(f, upper), EvaluateObjective(f, x), upper - x);
        }

        return DifferenceQuotient(EvaluateObjective(f, upper), EvaluateObjective(f, lower), upper - lower);
    }

    private static T DifferenceQuotient(T upper, T lower, T interval)
    {
        var difference = upper - lower;
        // Scale finite opposite-sign callback values before their subtraction overflows.
        return T.IsFinite(difference) ? difference / interval : upper / interval - lower / interval;
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
/// Newton-Raphson solver — legacy concrete-typed facade forwarding to
/// <see cref="NewtonRaphsonSolver{T}"/> at <c>T = double</c>.
/// </summary>
[Obsolete(
    "Use NewtonRaphsonSolver<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
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
