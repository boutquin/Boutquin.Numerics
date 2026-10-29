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
/// Local convergence order is approximately 1.84 near a simple root of a sufficiently
/// smooth function (Muller1956, equations17–21). This real-valued API cannot return
/// complex roots. A negative discriminant uses its absolute value under the square
/// root as a real-valued continuation heuristic, not complex arithmetic.
/// The zero-denominator continuation and repeated residual-growth halving follow
/// Muller1956 p214. Non-finite trial residuals are also halved; non-finite seed
/// residuals are rejected. The iteration cap can return a non-converged result.
/// </para>
/// <para>
/// The interpolating parabola is formed in coordinates divided by a power of two near the larger
/// iterate spacing, so its curvature stays representable and a rescaled abscissa
/// (<c>x = s * u</c>) follows the unit-scale iteration; a power-of-two <c>s</c> reproduces it exactly.
/// The one-point <see cref="Solve(Func{T, T}, T)"/> overload seeds its starting stencil with an absolute
/// <c>1e-4</c> floor, so it is scale-covariant only for starting points of magnitude one or more.
/// </para>
/// </remarks>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
public sealed class MullerSolver<T> : IUnbracketedRootSolver<T>
    where T : IFloatingPointIeee754<T>
{
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_four = T.CreateChecked(4);
    private static readonly T s_ten = T.CreateChecked(10);
    private static readonly T s_perturbation = T.CreateChecked(1e-4);

    private readonly T _tolerance;
    private readonly int _maxIterations;

    /// <summary>
    /// Initializes a new instance of the <see cref="MullerSolver{T}"/> type.
    /// </summary>
    /// <remarks>
    /// <para><b>Tolerance-API note.</b> Like the other interpolation-based solvers
    /// (<see cref="BrentSolver{T}"/>, <see cref="SecantSolver{T}"/>), Muller takes a
    /// <em>single</em> <c>tolerance</c> serving both the residual and step-size roles:
    /// it stops when <c>|f(x)|</c> is below the tolerance, or when
    /// <c>|xNew - xOld| / |xNew|</c> is below the tolerance at a nonzero new iterate
    /// (Muller1956, p. 210), provided the interpolating quadratic has a real root.
    /// At zero or after a negative-discriminant continuation, only the residual criterion applies.
    /// A small relative step is not a universal guarantee of a small residual. The bracket-shrinking solvers (<see cref="BisectionSolver{T}"/>,
    /// <see cref="NewtonRaphsonSolver{T}"/>) instead expose separate
    /// <c>functionTolerance</c>/<c>bracketTolerance</c>(/<c>stepTolerance</c>) controls.
    /// This asymmetry is intentional.</para>
    /// </remarks>
    public MullerSolver(T tolerance, int maxIterations = 50)
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
        var lower = initialGuess - delta;
        var middle = initialGuess;
        var upper = initialGuess + delta;
        if (lower == middle)
        {
            lower = T.BitDecrement(middle);
        }

        if (upper == middle)
        {
            upper = T.BitIncrement(middle);
        }

        if (!T.IsFinite(lower))
        {
            lower = initialGuess;
            middle = T.BitIncrement(lower);
            upper = T.BitIncrement(middle);
        }
        else if (!T.IsFinite(upper))
        {
            upper = initialGuess;
            middle = T.BitDecrement(upper);
            lower = T.BitDecrement(middle);
        }

        return Solve(objective, lower, middle, upper);
    }

    /// <summary>
    /// Solves <c>f(x) = 0</c> using Muller's method with three initial points.
    /// </summary>
    public RootSolverResult<T> Solve(Func<T, T> objective, T x0, T x1, T x2)
    {
        ArgumentNullException.ThrowIfNull(objective);
        if (!T.IsFinite(x0))
        {
            throw new ArgumentOutOfRangeException(nameof(x0), x0, "Initial point must be finite.");
        }

        if (!T.IsFinite(x1))
        {
            throw new ArgumentOutOfRangeException(nameof(x1), x1, "Initial point must be finite.");
        }

        if (!T.IsFinite(x2))
        {
            throw new ArgumentOutOfRangeException(nameof(x2), x2, "Initial point must be finite.");
        }

        var f0 = EvaluateObjective(objective, x0);
        var f1 = EvaluateObjective(objective, x1);
        var f2 = EvaluateObjective(objective, x2);

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

            // Muller's parabola is invariant under an affine change of abscissa, but its curvature
            // (about delta-f / h^2) is not representable at every spacing: it overflows at tiny
            // spacing and underflows at huge spacing. The parabola is therefore built in the
            // dimensionless variable t = (x - x2) / 2^k, with 2^k the largest power of two not
            // exceeding max(|h0|, |h1|). Scaling by a power of two is exact, so the coefficients
            // (a, b, c) are those of the unscaled parabola times exact powers of two, and the
            // arithmetic is bitwise-identical to the unscaled form wherever that form is in range.
            // The scaling is exact only if neither spacing loses bits: when the two spacings differ by
            // more than the type's dynamic range, the smaller one flushes to zero or to a subnormal,
            // and the unscaled form (k = 0) is kept instead.
            var spacingExponent = SpacingExponent(h0, h1);
            var scaledH0 = T.ScaleB(h0, -spacingExponent);
            var scaledH1 = T.ScaleB(h1, -spacingExponent);
            if (spacingExponent != 0
                && (T.ScaleB(scaledH0, spacingExponent) != h0 || T.ScaleB(scaledH1, spacingExponent) != h1))
            {
                spacingExponent = 0;
                scaledH0 = h0;
                scaledH1 = h1;
            }

            BuildCoefficients(f0, f1, f2, scaledH0, scaledH1, out var a, out var b);
            var c = f2;

            if (!T.IsFinite(a) || !T.IsFinite(b))
            {
                // Finite residuals can still overflow in a difference of residuals or of slopes
                // (opposite-sign values near the type's range). Dividing every residual by one
                // positive scale leaves the interpolating roots, and so x3, unchanged, so the
                // coefficients are rebuilt from residuals normalized before any subtraction.
                var residualScale = T.Max(T.Abs(f0), T.Max(T.Abs(f1), T.Abs(f2)));
                if (residualScale > T.Zero)
                {
                    BuildCoefficients(
                        f0 / residualScale, f1 / residualScale, f2 / residualScale, scaledH0, scaledH1, out a, out b);
                    c = f2 / residualScale;
                }
            }

            var disc = b * b - s_four * a * c;
            var coefficientScale = T.Max(T.Abs(a), T.Max(T.Abs(b), T.Abs(c)));
            if (!T.IsFinite(disc) || (coefficientScale > T.Zero && !T.IsNormal(coefficientScale * coefficientScale)))
            {
                // A common coefficient scale preserves the interpolating roots while
                // avoiding overflow or underflow in discriminant products.
                a /= coefficientScale;
                b /= coefficientScale;
                c /= coefficientScale;
                disc = b * b - s_four * a * c;
            }

            // Real-valued continuation heuristic; this API cannot represent complex roots.
            var sqrtDisc = T.Sqrt(T.Abs(disc));

            T x3;

            var denomPlus = b + sqrtDisc;
            var denomMinus = b - sqrtDisc;
            var denom = T.Abs(denomPlus) >= T.Abs(denomMinus) ? denomPlus : denomMinus;

            if (denom == T.Zero)
            {
                // Muller1956 p214: lambda_next=1 when the quadratic denominator
                // vanishes, so the next increment repeats the previous increment.
                x3 = x2 + h1;
            }
            else
            {
                // The quadratic step is in units of 2^k; restoring it is an exact power-of-two scaling.
                x3 = x2 - T.ScaleB(s_two * c / denom, spacingExponent);
            }

            // Guard against a non-finite update: an overflowing curvature term or a
            // near-zero denominator can drive x3 to ±∞ or NaN. Evaluating the objective
            // there would poison the result; terminate cleanly as non-converged instead
            // (mirrors NewtonRaphson's unbracketed `!T.IsFinite(xNew)` guard).
            if (!T.IsFinite(x3))
            {
                return new RootSolverResult<T>(x2, Converged: false, i + 1, FinalResidual: f2,
                    EstimatedError: T.Abs(x2 - x1));
            }

            var f3 = objective(x3);
            // Muller1956 p214: repeatedly halve a trial whose residual grows by >10.
            while (!T.IsFinite(f3) || T.Abs(f3) / T.Abs(f2) > s_ten)
            {
                var halved = T.IsNegative(x2) == T.IsNegative(x3)
                    ? x2 + (x3 - x2) / s_two
                    : x2 / s_two + x3 / s_two;
                if (halved == x3 || halved == x2)
                {
                    return new RootSolverResult<T>(x2, Converged: false, i + 1,
                        FinalResidual: f2, EstimatedError: T.Abs(x3 - x2));
                }

                x3 = halved;
                f3 = objective(x3);
            }

            var stepSize = T.Abs(x3 - x2);

            // Muller1956 p210 stops on the relative change at a nonzero iterate.
            // At zero, the relative quotient is undefined; only the residual test applies.
            // The sqrt(abs(discriminant)) continuation is not a real quadratic root,
            // so it cannot establish the paper's relative-step convergence criterion.
            // This is an iteration stopping criterion, not a universal root certificate.
            if (T.Abs(f3) < _tolerance
                || (disc >= T.Zero && x3 != T.Zero && stepSize / T.Abs(x3) < _tolerance))
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

    // Exponent k of the power-of-two spacing scale 2^k: the binary exponent of the larger |spacing|.
    // A zero or non-finite spacing (coincident or overflowing iterates) has no exponent, so the
    // unscaled form (k = 0) is kept and its degenerate-input handling is unchanged.
    private static int SpacingExponent(T h0, T h1)
    {
        var largest = T.Max(T.Abs(h0), T.Abs(h1));
        return largest > T.Zero && T.IsFinite(largest) ? T.ILogB(largest) : 0;
    }

    // Newton divided differences of the interpolating parabola through (x0,f0), (x1,f1), (x2,f2),
    // taking spacings already divided by the common power of two 2^k: slopes delta0, delta1,
    // curvature a, and the slope b of the parabola at x2, all expressed in t = (x - x2) / 2^k.
    private static void BuildCoefficients(T f0, T f1, T f2, T h0, T h1, out T a, out T b)
    {
        var delta0 = (f1 - f0) / h0;
        var delta1 = (f2 - f1) / h1;
        a = (delta1 - delta0) / (h1 + h0);
        b = a * h1 + delta1;
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
/// Muller's method root solver — legacy concrete-typed facade forwarding to
/// <see cref="MullerSolver{T}"/> at <c>T = double</c>.
/// </summary>
[Obsolete(
    "Use MullerSolver<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
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
