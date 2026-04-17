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

using Boutquin.Numerics.Solvers.Internal;

namespace Boutquin.Numerics.Solvers;

/// <summary>
/// Levenberg–Marquardt solver for nonlinear least-squares problems, generic over any
/// IEEE 754 floating-point type. Minimizes <c>½ Σᵢ rᵢ(θ)²</c> by interpolating between
/// Gauss–Newton (fast near the optimum) and gradient descent (robust far from it) via a
/// damping parameter <c>λ</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> B (IEEE 754 transcendental). The user's residual function
/// typically involves transcendentals; the <see cref="IFloatingPointIeee754{TSelf}"/>
/// constraint ensures the caller can construct such functions.
/// </para>
/// <para>
/// References: Marquardt 1963 (original algorithm), More 1978 (robust implementation),
/// Nielsen 1999 (trust-region damping update — see Madsen/Nielsen/Tingleff "Methods
/// for Non-Linear Least Squares Problems", 2004, §3.2). On a successful step the
/// damping multiplier is adapted from the gain ratio
/// <c>ρ = actual_reduction / predicted_reduction</c> via
/// <c>λ ← λ · max(1/3, 1 − (2ρ − 1)³)</c>; on a rejected step <c>λ ← λ · ν</c>
/// where <c>ν</c> is a doubling multiplier that restarts at 2 on the next
/// acceptance. The step equation is <c>(JᵀJ + λ · diag(JᵀJ)) · δ = −Jᵀr</c>
/// (Marquardt's scaled-diagonal form), which is invariant under per-parameter
/// rescaling.
/// </para>
/// <para>
/// Bounds (optional) are handled by projection: after each accepted step, parameters
/// are clipped into <c>[lowerBounds, upperBounds]</c>. When any component lies on a bound
/// at termination the <see cref="MultivariateSolverResult{T}.BoundsActive"/> flag is set;
/// convergence is reported orthogonally based on the tolerance tests evaluated on the
/// free (non-bound-clipped) components.
/// </para>
/// <para>
/// Deterministic: no RNG dependency, no dependence on evaluation order across threads.
/// Same inputs produce bit-identical outputs.
/// </para>
/// <para>
/// Allocation profile: two public overloads share the same iteration logic. The legacy
/// overload allocates a private <see cref="LevenbergMarquardtBuffers{T}"/> instance per
/// call — convenient for one-shot calibrations. The pooled overload accepts a caller-owned
/// pool and adds no managed-heap allocation inside the iteration loop — appropriate for
/// bootstrap loops, Monte Carlo calibration, and real-time refit engines where GC jitter
/// is material. Both overloads produce bit-identical results given the same inputs.
/// </para>
/// </remarks>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
public sealed class LevenbergMarquardtSolver<T> : IMultivariateLeastSquaresSolver<T>
    where T : IFloatingPointIeee754<T>
{
    private static readonly T s_half = T.CreateChecked(0.5);
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_lambdaCeiling = T.CreateChecked(1e16);
    private static readonly T s_lambdaFloor = T.CreateChecked(1e-12);
    private static readonly T s_initialRejectionMultiplier = T.CreateChecked(2);
    private static readonly T s_oneThird = T.CreateChecked(1.0 / 3.0);

    private readonly int _maxIterations;
    private readonly T _functionTolerance;
    private readonly T _parameterTolerance;
    private readonly T _gradientTolerance;
    private readonly T _initialDamping;

    /// <summary>
    /// Initializes a new <see cref="LevenbergMarquardtSolver{T}"/>.
    /// </summary>
    /// <param name="maxIterations">Maximum outer iterations. Defaults to 200.</param>
    /// <param name="functionTolerance">Relative cost-decrease threshold (MINPACK lmder ftol). Defaults to 1e-10.</param>
    /// <param name="parameterTolerance">Step-norm threshold relative to parameter norm. Defaults to 1e-10.</param>
    /// <param name="gradientTolerance">Gradient infinity-norm threshold. Defaults to 1e-10.</param>
    /// <param name="initialDamping">Starting value of the damping parameter λ₀. Defaults to 1e-3.</param>
    public LevenbergMarquardtSolver(
        int maxIterations,
        T functionTolerance,
        T parameterTolerance,
        T gradientTolerance,
        T initialDamping)
    {
        _maxIterations = maxIterations;
        _functionTolerance = functionTolerance;
        _parameterTolerance = parameterTolerance;
        _gradientTolerance = gradientTolerance;
        _initialDamping = initialDamping;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Convenience overload for one-shot callers. Constructs a private
    /// <see cref="LevenbergMarquardtBuffers{T}"/> instance for this solve, runs the
    /// iteration, and discards the pool. Hot-path consumers should use the
    /// pooled overload to reuse a pool across many solves.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="residuals"/> or <paramref name="initialGuess"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException"><paramref name="initialGuess"/> is empty, contains non-finite values, or bounds do not match in length.</exception>
#pragma warning disable RS0026 // Do not add multiple overloads with optional parameters — suppressed
    // symmetrically with the pooled overload; see that overload's suppression
    // comment for the disambiguation rationale.
    public MultivariateSolverResult<T> Solve(
        Func<T[], T[]> residuals,
        T[] initialGuess,
        Func<T[], T[,]>? jacobian = null,
        T[]? lowerBounds = null,
        T[]? upperBounds = null)
#pragma warning restore RS0026
    {
        ArgumentNullException.ThrowIfNull(residuals);
        ArgumentNullException.ThrowIfNull(initialGuess);

        ValidateInitialGuess(initialGuess);
        ValidateBounds(initialGuess.Length, lowerBounds, upperBounds);

        // Materialize the working parameter vector up-front (applying bounds clipping)
        // so we can size the pool against the first residual evaluation.
        var theta = (T[])initialGuess.Clone();
        ClipToBounds(theta, lowerBounds, upperBounds);

        var firstResidual = residuals(theta);
        if (firstResidual.Length == 0)
        {
            throw new InvalidOperationException("Residual function returned an empty vector.");
        }

        var buffers = new LevenbergMarquardtBuffers<T>(theta.Length, firstResidual.Length);
        Array.Copy(theta, buffers.Theta, theta.Length);
        Array.Copy(firstResidual, buffers.Residual, firstResidual.Length);

        return SolveCore(residuals, buffers, jacobian, lowerBounds, upperBounds);
    }

    /// <summary>
    /// Minimizes <c>½ Σᵢ rᵢ(θ)²</c> using a caller-owned pre-allocated buffer pool.
    /// Produces bit-identical results to the pool-free overload; eliminates per-iteration
    /// managed-heap allocation from the solver's own work.
    /// </summary>
    /// <param name="residuals">
    /// Residual function. Returns a vector of length <c>buffers.ResidualCount</c> on every call.
    /// Callers targeting the zero-allocation contract should return a shared or pool-backed
    /// array (the solver copies each returned vector into pool storage before any subsequent
    /// iteration, so reusing the same array across calls is safe).
    /// </param>
    /// <param name="initialGuess">
    /// Starting parameter vector. Length must match <c>buffers.ParameterCount</c>; if it does
    /// not, an <see cref="ArgumentException"/> is thrown with a remediation message directing
    /// the caller to <see cref="LevenbergMarquardtBuffers{T}.Reset(int, int)"/>.
    /// </param>
    /// <param name="buffers">Caller-owned scratch pool. Reusable across solves of matching dimensions.</param>
    /// <param name="jacobian">Optional analytic Jacobian. Same contract as the pool-free overload.</param>
    /// <param name="lowerBounds">Optional element-wise lower bounds. Same contract as the pool-free overload.</param>
    /// <param name="upperBounds">Optional element-wise upper bounds. Same contract as the pool-free overload.</param>
    /// <returns>Final parameter estimate, residuals, cost, and convergence diagnostics.</returns>
    /// <exception cref="ArgumentNullException">Any of <paramref name="residuals"/>, <paramref name="initialGuess"/>, or <paramref name="buffers"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="initialGuess"/> is empty, contains non-finite values, or bounds do not match in length;
    /// or <c>buffers.ParameterCount</c> differs from <paramref name="initialGuess"/> length;
    /// or the residual function returns a vector whose length differs from <c>buffers.ResidualCount</c>.
    /// </exception>
#pragma warning disable RS0026 // Do not add multiple overloads with optional parameters
    //
    // The two Solve overloads have three and two required parameters respectively;
    // the third required parameter (`buffers`) of this overload is a reference-type
    // class with no implicit conversion to the legacy overload's types, so no
    // call-site ambiguity can occur. Preserving the optional tail parameters keeps
    // the pooled overload's public shape symmetric with the legacy one, which is
    // the principle of least surprise for consumers switching between the two.
    public MultivariateSolverResult<T> Solve(
        Func<T[], T[]> residuals,
        T[] initialGuess,
        LevenbergMarquardtBuffers<T> buffers,
        Func<T[], T[,]>? jacobian = null,
        T[]? lowerBounds = null,
        T[]? upperBounds = null)
#pragma warning restore RS0026
    {
        ArgumentNullException.ThrowIfNull(residuals);
        ArgumentNullException.ThrowIfNull(initialGuess);
        ArgumentNullException.ThrowIfNull(buffers);

        ValidateInitialGuess(initialGuess);
        ValidateBounds(initialGuess.Length, lowerBounds, upperBounds);

        if (buffers.ParameterCount != initialGuess.Length)
        {
            throw new ArgumentException(
                $"Buffer pool parameter count {buffers.ParameterCount} does not match " +
                $"initial-guess length {initialGuess.Length}. Call buffers.Reset({initialGuess.Length}, " +
                $"<residualCount>) before invoking Solve.",
                nameof(buffers));
        }

        // Install the initial guess into the pool, clipping bounds before the first eval
        // so the first residual/cost reflect a feasible point (matching the legacy overload).
        Array.Copy(initialGuess, buffers.Theta, initialGuess.Length);
        ClipToBounds(buffers.Theta, lowerBounds, upperBounds);

        var firstResidual = residuals(buffers.Theta);
        if (firstResidual.Length == 0)
        {
            throw new InvalidOperationException("Residual function returned an empty vector.");
        }

        if (firstResidual.Length != buffers.ResidualCount)
        {
            throw new ArgumentException(
                $"Residual function returned {firstResidual.Length} residuals but the buffer pool " +
                $"was sized for {buffers.ResidualCount}. Call buffers.Reset({buffers.ParameterCount}, " +
                $"{firstResidual.Length}) before invoking Solve.",
                nameof(buffers));
        }

        Array.Copy(firstResidual, buffers.Residual, firstResidual.Length);

        return SolveCore(residuals, buffers, jacobian, lowerBounds, upperBounds);
    }

    private MultivariateSolverResult<T> SolveCore(
        Func<T[], T[]> residuals,
        LevenbergMarquardtBuffers<T> buffers,
        Func<T[], T[,]>? jacobian,
        T[]? lowerBounds,
        T[]? upperBounds)
    {
        var parameterCount = buffers.ParameterCount;
        var residualCount = buffers.ResidualCount;
        var theta = buffers.Theta;
        var residual = buffers.Residual;
        var trialTheta = buffers.TrialTheta;
        var trialResidual = buffers.TrialResidual;
        var effectiveDelta = buffers.EffectiveDelta;
        var delta = buffers.Delta;

        var cost = s_half * SumSquares(residual, residualCount);

        // Initial Jacobian — needed for the iteration-0 gradient check and the first linear solve.
        EvaluateJacobianInto(residuals, theta, jacobian, buffers);

        var iterations = 0;
        var acceptedIterations = 0;

        // λ₀ = initialDamping (absolute; Moré 1978 convention). The scaled-diagonal
        // damping form `(JᵀJ + λ·diag(JᵀJ))·δ = −Jᵀr` makes the step itself invariant
        // under per-parameter rescaling, so a fixed initial λ suffices across scales.
        var lambda = _initialDamping;
        var nu = s_initialRejectionMultiplier;

        // Gradient check at the initial point — if already stationary, return immediately.
        if (GradientInfinityNorm(buffers.Jacobian, residual, residualCount, parameterCount) <= _gradientTolerance)
        {
            return BuildResult(
                buffers,
                cost,
                acceptedIterations,
                converged: true,
                LmTerminationReason.GradientToleranceReached,
                lowerBounds,
                upperBounds);
        }

        for (; iterations < _maxIterations; iterations++)
        {
            if (!DampedLinearSolve<T>.TrySolve(residual, lambda, buffers))
            {
                // Damped matrix not factorable — treat as a rejection with Nielsen growth.
                lambda = T.Min(lambda * nu, s_lambdaCeiling);
                nu *= s_two;
                if (lambda >= s_lambdaCeiling)
                {
                    return BuildResult(
                        buffers,
                        cost,
                        acceptedIterations,
                        converged: false,
                        LmTerminationReason.DampingOverflow,
                        lowerBounds,
                        upperBounds);
                }

                continue;
            }

            // Candidate parameters: clip to bounds before evaluating residuals.
            for (var i = 0; i < parameterCount; i++)
            {
                trialTheta[i] = theta[i] + delta[i];
            }

            ClipToBounds(trialTheta, lowerBounds, upperBounds);

            // Effective step (after clipping) — used for parameter-tolerance test.
            for (var i = 0; i < parameterCount; i++)
            {
                effectiveDelta[i] = trialTheta[i] - theta[i];
            }

            var trialResidualArray = residuals(trialTheta);
            Array.Copy(trialResidualArray, trialResidual, residualCount);
            var trialCost = s_half * SumSquares(trialResidual, residualCount);

            // Gain ratio ρ = actual_reduction / predicted_reduction; the quadratic model's
            // predicted reduction is ½·δᵀ·(λ·diag(JᵀJ)·δ − Jᵀr) for Marquardt damping.
            // Compute using the pre-clipping step so the ratio reflects the model's own prediction.
            var predictedReduction = PredictedReduction(
                buffers.Jacobian, residual, delta, lambda, residualCount, parameterCount);
            var actualReduction = cost - trialCost;
            var gainRatio = predictedReduction > T.Zero ? actualReduction / predictedReduction : -T.One;

            if (gainRatio > T.Zero)
            {
                // Step accepted.
                var costDelta = actualReduction;
                var priorCost = cost;
                var stepNorm = Norm2(effectiveDelta, parameterCount);
                var thetaNorm = Norm2(theta, parameterCount);

                Array.Copy(trialTheta, theta, parameterCount);
                Array.Copy(trialResidual, residual, residualCount);
                cost = trialCost;
                acceptedIterations++;

                // Nielsen damping update: scale λ by max(1/3, 1 − (2ρ − 1)³).
                // Use T.Pow for parity with Math.Pow in the legacy code path —
                // exp(3·log(x)) and x·x·x can differ by ULPs, and the LM convergence
                // path is sensitive to that butterfly effect.
                var nielsenFactor = T.One - T.Pow(s_two * gainRatio - T.One, T.CreateChecked(3));
                lambda = T.Max(lambda * T.Max(s_oneThird, nielsenFactor), s_lambdaFloor);
                nu = s_initialRejectionMultiplier;

                // Recompute Jacobian at the new point (needed for the next step AND for gradient check).
                EvaluateJacobianInto(residuals, theta, jacobian, buffers);

                // Tolerance tests — check in priority order: function, parameter, gradient.
                // Relative reduction `costDelta / priorCost` matches MINPACK lmder's ftol
                // semantics and stays scale-invariant as the cost shrinks toward zero.
                var relativeReduction = priorCost > T.Zero ? costDelta / priorCost : costDelta;
                if (relativeReduction <= _functionTolerance)
                {
                    return BuildResult(
                        buffers,
                        cost,
                        acceptedIterations,
                        converged: true,
                        LmTerminationReason.FunctionToleranceReached,
                        lowerBounds,
                        upperBounds);
                }

                if (stepNorm <= _parameterTolerance * (thetaNorm + _parameterTolerance))
                {
                    return BuildResult(
                        buffers,
                        cost,
                        acceptedIterations,
                        converged: true,
                        LmTerminationReason.ParameterToleranceReached,
                        lowerBounds,
                        upperBounds);
                }

                if (GradientInfinityNorm(buffers.Jacobian, residual, residualCount, parameterCount) <= _gradientTolerance)
                {
                    return BuildResult(
                        buffers,
                        cost,
                        acceptedIterations,
                        converged: true,
                        LmTerminationReason.GradientToleranceReached,
                        lowerBounds,
                        upperBounds);
                }
            }
            else
            {
                // Step rejected — Nielsen growth: λ ← λ·ν, ν ← 2ν.
                lambda = T.Min(lambda * nu, s_lambdaCeiling);
                nu *= s_two;

                if (lambda >= s_lambdaCeiling)
                {
                    return BuildResult(
                        buffers,
                        cost,
                        acceptedIterations,
                        converged: false,
                        LmTerminationReason.DampingOverflow,
                        lowerBounds,
                        upperBounds);
                }
            }
        }

        return BuildResult(
            buffers,
            cost,
            acceptedIterations,
            converged: false,
            LmTerminationReason.MaxIterationsReached,
            lowerBounds,
            upperBounds);
    }

    private static void EvaluateJacobianInto(
        Func<T[], T[]> residuals,
        T[] theta,
        Func<T[], T[,]>? analyticJacobian,
        LevenbergMarquardtBuffers<T> buffers)
    {
        if (analyticJacobian is null)
        {
            FiniteDifferenceJacobian<T>.EvaluateInto(residuals, theta, buffers);
            return;
        }

        // User-supplied analytic Jacobian: copy into the pool's Jacobian slab. The user's
        // returned array is allocated per call (outside our control); our inner work
        // (TrySolve, PredictedReduction, GradientInfinityNorm) reads exclusively from
        // the pool slab so allocation on this path is bounded to the user's callback.
        var userJacobian = analyticJacobian(theta);
        var m = buffers.ResidualCount;
        var n = buffers.ParameterCount;

        if (userJacobian.GetLength(0) != m || userJacobian.GetLength(1) != n)
        {
            throw new InvalidOperationException(
                $"Analytic Jacobian returned shape [{userJacobian.GetLength(0)}, " +
                $"{userJacobian.GetLength(1)}] but expected [{m}, {n}].");
        }

        var target = buffers.Jacobian;
        for (var i = 0; i < m; i++)
        {
            for (var j = 0; j < n; j++)
            {
                target[i, j] = userJacobian[i, j];
            }
        }
    }

    private static T PredictedReduction(
        T[,] jacobian,
        T[] residual,
        T[] delta,
        T lambda,
        int residualCount,
        int parameterCount)
    {
        // Predicted reduction for the quadratic model with Marquardt damping:
        // predicted = ½·δᵀ·(λ·diag(JᵀJ)·δ − Jᵀr).
        var sum = T.Zero;
        for (var j = 0; j < parameterCount; j++)
        {
            // Diagonal of JᵀJ and j-th component of the gradient Jᵀr.
            var diag = T.Zero;
            var gj = T.Zero;
            for (var i = 0; i < residualCount; i++)
            {
                diag += jacobian[i, j] * jacobian[i, j];
                gj += jacobian[i, j] * residual[i];
            }

            sum += delta[j] * ((lambda * diag * delta[j]) - gj);
        }

        return s_half * sum;
    }

    private static MultivariateSolverResult<T> BuildResult(
        LevenbergMarquardtBuffers<T> buffers,
        T cost,
        int iterations,
        bool converged,
        LmTerminationReason reason,
        T[]? lowerBounds,
        T[]? upperBounds)
    {
        var parameterCount = buffers.ParameterCount;
        var residualCount = buffers.ResidualCount;

        // Result owns fresh copies of parameters and residuals — the pool's arrays remain
        // usable for subsequent solves against the same buffer instance.
        var parameters = new T[parameterCount];
        Array.Copy(buffers.Theta, parameters, parameterCount);

        var finalResiduals = new T[residualCount];
        Array.Copy(buffers.Residual, finalResiduals, residualCount);

        var boundsActive = IsAnyComponentOnBound(parameters, lowerBounds, upperBounds);
        var covariance = BuildCovariance(buffers.Jacobian, buffers.Residual, residualCount, parameterCount, converged);

        return new MultivariateSolverResult<T>(
            parameters,
            cost,
            finalResiduals,
            covariance,
            iterations,
            converged,
            reason,
            boundsActive);
    }

    private static T[,]? BuildCovariance(
        T[,] jacobian,
        T[] residual,
        int residualCount,
        int parameterCount,
        bool converged)
    {
        // Only report covariance when we believe we are near an optimum.
        if (!converged)
        {
            return null;
        }

        if (residualCount <= parameterCount)
        {
            return null;
        }

        var inverse = DampedLinearSolve<T>.TryInvertNormalEquations(jacobian, residualCount, parameterCount);
        if (inverse is null)
        {
            return null;
        }

        // σ̂² = 2·cost / (m − n) = RSS / (m − n).
        var rss = SumSquares(residual, residualCount);
        var sigmaSq = rss / T.CreateChecked(residualCount - parameterCount);

        for (var i = 0; i < parameterCount; i++)
        {
            for (var j = 0; j < parameterCount; j++)
            {
                inverse[i, j] *= sigmaSq;
            }
        }

        return inverse;
    }

    private static void ValidateInitialGuess(T[] initialGuess)
    {
        if (initialGuess.Length == 0)
        {
            throw new ArgumentException("Initial guess must be non-empty.", nameof(initialGuess));
        }

        for (var i = 0; i < initialGuess.Length; i++)
        {
            if (!T.IsFinite(initialGuess[i]))
            {
                throw new ArgumentException(
                    $"Initial guess component at index {i} is not finite.", nameof(initialGuess));
            }
        }
    }

    private static void ValidateBounds(int parameterCount, T[]? lowerBounds, T[]? upperBounds)
    {
        if (lowerBounds is not null && lowerBounds.Length != parameterCount)
        {
            throw new ArgumentException(
                $"Lower bounds length {lowerBounds.Length} does not match parameter count {parameterCount}.",
                nameof(lowerBounds));
        }

        if (upperBounds is not null && upperBounds.Length != parameterCount)
        {
            throw new ArgumentException(
                $"Upper bounds length {upperBounds.Length} does not match parameter count {parameterCount}.",
                nameof(upperBounds));
        }
    }

    private static T SumSquares(T[] vector, int length)
    {
        var sum = T.Zero;
        for (var i = 0; i < length; i++)
        {
            sum += vector[i] * vector[i];
        }

        return sum;
    }

    private static T Norm2(T[] vector, int length) => T.Sqrt(SumSquares(vector, length));

    private static T GradientInfinityNorm(
        T[,] jacobian,
        T[] residual,
        int residualCount,
        int parameterCount)
    {
        var maxAbs = T.Zero;

        for (var j = 0; j < parameterCount; j++)
        {
            var gj = T.Zero;
            for (var i = 0; i < residualCount; i++)
            {
                gj += jacobian[i, j] * residual[i];
            }

            var abs = T.Abs(gj);
            if (abs > maxAbs)
            {
                maxAbs = abs;
            }
        }

        return maxAbs;
    }

    private static void ClipToBounds(T[] theta, T[]? lowerBounds, T[]? upperBounds)
    {
        if (lowerBounds is null && upperBounds is null)
        {
            return;
        }

        for (var i = 0; i < theta.Length; i++)
        {
            if (lowerBounds is not null && theta[i] < lowerBounds[i])
            {
                theta[i] = lowerBounds[i];
            }

            if (upperBounds is not null && theta[i] > upperBounds[i])
            {
                theta[i] = upperBounds[i];
            }
        }
    }

    private static bool IsAnyComponentOnBound(
        T[] theta,
        T[]? lowerBounds,
        T[]? upperBounds)
    {
        if (lowerBounds is null && upperBounds is null)
        {
            return false;
        }

        for (var i = 0; i < theta.Length; i++)
        {
            if (lowerBounds is not null && theta[i] == lowerBounds[i])
            {
                return true;
            }

            if (upperBounds is not null && theta[i] == upperBounds[i])
            {
                return true;
            }
        }

        return false;
    }
}

/// <summary>
/// Levenberg–Marquardt solver for nonlinear least-squares problems. Minimizes
/// <c>½ Σᵢ rᵢ(θ)²</c> by interpolating between Gauss–Newton (fast near the optimum)
/// and gradient descent (robust far from it) via a damping parameter <c>λ</c>.
/// </summary>
/// <remarks>
/// <para>
/// Legacy facade wrapping <see cref="LevenbergMarquardtSolver{T}"/> with
/// <c>T = <see cref="double"/></c>. All operations delegate to the underlying generic
/// instance.
/// </para>
/// <para>
/// References: Marquardt 1963 (original algorithm), More 1978 (robust implementation),
/// Nielsen 1999 (trust-region damping update — see Madsen/Nielsen/Tingleff "Methods
/// for Non-Linear Least Squares Problems", 2004, §3.2). On a successful step the
/// damping multiplier is adapted from the gain ratio
/// <c>ρ = actual_reduction / predicted_reduction</c> via
/// <c>λ ← λ · max(1/3, 1 − (2ρ − 1)³)</c>; on a rejected step <c>λ ← λ · ν</c>
/// where <c>ν</c> is a doubling multiplier that restarts at 2 on the next
/// acceptance. The step equation is <c>(JᵀJ + λ · diag(JᵀJ)) · δ = −Jᵀr</c>
/// (Marquardt's scaled-diagonal form), which is invariant under per-parameter
/// rescaling.
/// </para>
/// <para>
/// Bounds (optional) are handled by projection: after each accepted step, parameters
/// are clipped into <c>[lowerBounds, upperBounds]</c>. When any component lies on a bound
/// at termination the <see cref="MultivariateSolverResult.BoundsActive"/> flag is set;
/// convergence is reported orthogonally based on the tolerance tests evaluated on the
/// free (non-bound-clipped) components.
/// </para>
/// <para>
/// Deterministic: no RNG dependency, no dependence on evaluation order across threads.
/// Same inputs produce bit-identical outputs.
/// </para>
/// <para>
/// Allocation profile: two public overloads share the same iteration logic. The legacy
/// <see cref="Solve(System.Func{double[], double[]}, double[], System.Func{double[], double[,]}, double[], double[])"/>
/// overload allocates a private <see cref="LevenbergMarquardtBuffers"/> instance per call
/// — convenient for one-shot calibrations. The pooled
/// <see cref="Solve(System.Func{double[], double[]}, double[], LevenbergMarquardtBuffers, System.Func{double[], double[,]}, double[], double[])"/>
/// overload accepts a caller-owned pool and adds no managed-heap allocation inside the
/// iteration loop — appropriate for bootstrap loops, Monte Carlo calibration, and real-time
/// refit engines where GC jitter is material. Both overloads produce bit-identical results
/// given the same inputs.
/// </para>
/// </remarks>
public sealed class LevenbergMarquardtSolver : IMultivariateLeastSquaresSolver
{
    private readonly LevenbergMarquardtSolver<double> _inner;

    /// <summary>
    /// Initializes a new <see cref="LevenbergMarquardtSolver"/>.
    /// </summary>
    /// <param name="maxIterations">Maximum outer iterations. Defaults to 200.</param>
    /// <param name="functionTolerance">
    /// Stop when the relative cost decrease <c>(cost_prev − cost_new)/cost_prev</c>
    /// between two successive accepted steps is below this value. Matches the
    /// MINPACK lmder <c>ftol</c> convention. Defaults to 1e-10. Relative tolerance
    /// is essential for problems where the optimal cost is far from unity — an
    /// absolute threshold would trigger prematurely for very-low-cost problems
    /// (e.g., NIST Lanczos3 with optimal cost near 1e-8) and never trigger for
    /// very-high-cost problems.
    /// </param>
    /// <param name="parameterTolerance">
    /// Stop when the step norm <c>‖δ‖ ≤ parameterTolerance · (‖θ‖ + parameterTolerance)</c>.
    /// Defaults to 1e-10.
    /// </param>
    /// <param name="gradientTolerance">
    /// Stop when <c>‖Jᵀr‖∞ ≤ gradientTolerance</c>. Defaults to 1e-10.
    /// </param>
    /// <param name="initialDamping">
    /// Absolute starting value of the damping parameter <c>λ₀ = initialDamping</c>
    /// (Moré 1978 convention). Per-parameter scale invariance is already delivered by
    /// the Marquardt scaled-diagonal form inside the step equation
    /// <c>(JᵀJ + λ · diag(JᵀJ)) · δ = −Jᵀr</c>; double-scaling at initialization
    /// (the Nielsen τ·max(diag) variant) was tried and reverted — it produced
    /// zero-progress failures on ill-scaled problems like NIST Hahn1 where
    /// <c>max(diag(JᵀJ))</c> at the starting point drove λ above the step
    /// resolution of double precision. Defaults to 1e-3.
    /// </param>
    public LevenbergMarquardtSolver(
        int maxIterations = 200,
        double functionTolerance = 1e-10,
        double parameterTolerance = 1e-10,
        double gradientTolerance = 1e-10,
        double initialDamping = 1e-3)
    {
        _inner = new LevenbergMarquardtSolver<double>(
            maxIterations,
            functionTolerance,
            parameterTolerance,
            gradientTolerance,
            initialDamping);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Convenience overload for one-shot callers. Constructs a private
    /// <see cref="LevenbergMarquardtBuffers"/> instance for this solve, runs the
    /// iteration, and discards the pool. Hot-path consumers should use the
    /// <see cref="Solve(System.Func{double[], double[]}, double[], LevenbergMarquardtBuffers, System.Func{double[], double[,]}, double[], double[])"/>
    /// overload to reuse a pool across many solves.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="residuals"/> or <paramref name="initialGuess"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException"><paramref name="initialGuess"/> is empty, contains non-finite values, or bounds do not match in length.</exception>
#pragma warning disable RS0026 // Do not add multiple overloads with optional parameters — suppressed
    // symmetrically with the pooled overload; see that overload's suppression
    // comment for the disambiguation rationale.
    public MultivariateSolverResult Solve(
        Func<double[], double[]> residuals,
        double[] initialGuess,
        Func<double[], double[,]>? jacobian = null,
        double[]? lowerBounds = null,
        double[]? upperBounds = null)
#pragma warning restore RS0026
    {
        var r = _inner.Solve(residuals, initialGuess, jacobian, lowerBounds, upperBounds);
        return new MultivariateSolverResult(r.Parameters, r.FinalCost, r.FinalResiduals, r.ParameterCovariance, r.Iterations, r.Converged, r.TerminationReason, r.BoundsActive);
    }

    /// <summary>
    /// Minimizes <c>½ Σᵢ rᵢ(θ)²</c> using a caller-owned pre-allocated buffer pool.
    /// Produces bit-identical results to the pool-free overload; eliminates per-iteration
    /// managed-heap allocation from the solver's own work.
    /// </summary>
    /// <param name="residuals">
    /// Residual function. Returns a vector of length <c>buffers.ResidualCount</c> on every call.
    /// Callers targeting the zero-allocation contract should return a shared or pool-backed
    /// array (the solver copies each returned vector into pool storage before any subsequent
    /// iteration, so reusing the same array across calls is safe).
    /// </param>
    /// <param name="initialGuess">
    /// Starting parameter vector. Length must match <c>buffers.ParameterCount</c>; if it does
    /// not, an <see cref="ArgumentException"/> is thrown with a remediation message directing
    /// the caller to <see cref="LevenbergMarquardtBuffers.Reset(int, int)"/>.
    /// </param>
    /// <param name="buffers">Caller-owned scratch pool. Reusable across solves of matching dimensions.</param>
    /// <param name="jacobian">Optional analytic Jacobian. Same contract as the pool-free overload.</param>
    /// <param name="lowerBounds">Optional element-wise lower bounds. Same contract as the pool-free overload.</param>
    /// <param name="upperBounds">Optional element-wise upper bounds. Same contract as the pool-free overload.</param>
    /// <returns>Final parameter estimate, residuals, cost, and convergence diagnostics.</returns>
    /// <exception cref="ArgumentNullException">Any of <paramref name="residuals"/>, <paramref name="initialGuess"/>, or <paramref name="buffers"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="initialGuess"/> is empty, contains non-finite values, or bounds do not match in length;
    /// or <c>buffers.ParameterCount</c> differs from <paramref name="initialGuess"/> length;
    /// or the residual function returns a vector whose length differs from <c>buffers.ResidualCount</c>.
    /// </exception>
#pragma warning disable RS0026 // Do not add multiple overloads with optional parameters
    //
    // The two Solve overloads have three and two required parameters respectively;
    // the third required parameter (`buffers`) of this overload is a reference-type
    // class with no implicit conversion to the legacy overload's types, so no
    // call-site ambiguity can occur. Preserving the optional tail parameters keeps
    // the pooled overload's public shape symmetric with the legacy one, which is
    // the principle of least surprise for consumers switching between the two.
    public MultivariateSolverResult Solve(
        Func<double[], double[]> residuals,
        double[] initialGuess,
        LevenbergMarquardtBuffers buffers,
        Func<double[], double[,]>? jacobian = null,
        double[]? lowerBounds = null,
        double[]? upperBounds = null)
#pragma warning restore RS0026
    {
        ArgumentNullException.ThrowIfNull(buffers);
        var r = _inner.Solve(residuals, initialGuess, buffers.Inner, jacobian, lowerBounds, upperBounds);
        return new MultivariateSolverResult(r.Parameters, r.FinalCost, r.FinalResiduals, r.ParameterCovariance, r.Iterations, r.Converged, r.TerminationReason, r.BoundsActive);
    }
}
