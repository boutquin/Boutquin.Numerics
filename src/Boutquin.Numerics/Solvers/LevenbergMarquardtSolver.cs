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
/// Bounds (optional) use a free-coordinate damped solve: fixed coordinates and active
/// bounds with outward descent directions have zero steps. Each trial is then clipped before its residual
/// and predicted model reduction are evaluated, into <c>[lowerBounds, upperBounds]</c>. When any component lies on a bound
/// at termination the <see cref="MultivariateSolverResult{T}.BoundsActive"/> flag is set;
/// convergence is reported orthogonally based on the tolerance tests evaluated on the
/// free (non-bound-clipped) components.
/// </para>
/// <para>
/// Deterministic: no RNG dependency, no dependence on evaluation order across threads.
/// Same inputs produce bit-identical outputs.
/// </para>
/// <para>
/// Range limit of the convergence guarantee: after a trial step the representable range
/// rejects, <c>FunctionToleranceReached</c> and <c>ParameterToleranceReached</c> are
/// withheld until damping returns to its prior level, so an optimum reachable only through
/// such steps ends with <c>Converged = false</c>. When no step is range-rejected the
/// MINPACK tolerance tests apply unchanged, and either stop can still certify a
/// non-stationary point of a numerically rank-deficient problem whose exact optimum lies
/// beyond the representable range; <c>Converged = true</c> from those two stops is not a
/// proof of stationarity.
/// </para>
/// <para>
/// Allocation profile: convenience overloads allocate working buffers per call; pooled
/// overloads reuse caller-owned <see cref="LevenbergMarquardtBuffers{T}"/> instances.
/// Buffer-fill callbacks avoid residual/Jacobian array allocation inside the iteration
/// loop. Array-returning callbacks still own those allocations, even with pooled working
/// buffers. A completed solve allocates its returned result and parameter copy.
/// </para>
/// <para>
/// Type-precision note: damping thresholds <c>1e-12</c> and <c>1e16</c> must be
/// positive and finite in <typeparamref name="T"/>. The constructor throws
/// <see cref="NotSupportedException"/> when this range is unavailable, including
/// <c>Half</c>. <c>double</c> and <c>float</c> satisfy this requirement.
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
    /// <param name="maxIterations">Maximum trial steps, including rejected trials. The result's iteration count includes accepted steps only.</param>
    /// <param name="functionTolerance">
    /// Stop when both the absolute actual and nonnegative predicted reductions, relative
    /// to the current cost, are at most this value. Checked after accepted and rejected
    /// trials; a rejected trial leaves the parameters unchanged. MINPACK lmder also
    /// requires half the actual-to-predicted reduction ratio to be at most one. Like every
    /// MINPACK tolerance stop, it can still certify a non-stationary point of a numerically
    /// rank-deficient problem whose exact optimum lies beyond the representable range.
    /// </param>
    /// <param name="parameterTolerance">
    /// Stop when the step norm <c>‖δ‖ ≤ parameterTolerance · (‖θ‖ + parameterTolerance)</c>.
    /// It can still certify a non-stationary point of a numerically rank-deficient
    /// problem whose exact optimum lies beyond the representable range.
    /// </param>
    /// <param name="gradientTolerance">Stop when <c>‖Jᵀr‖∞ ≤ gradientTolerance</c>. This is an absolute tolerance in residual-times-Jacobian units; rescaling residuals by <c>s</c> requires scaling this tolerance by <c>s²</c>.</param>
    /// <param name="initialDamping">
    /// Absolute starting value of the damping parameter <c>λ₀ = initialDamping</c>
    /// (the Numerical Recipes §15.5.2 starting-value convention). Must be below 1e16.
    /// When it exceeds one, function and parameter stops wait until damping has fallen
    /// to at most one; a strongly damped initial step alone cannot certify convergence.
    /// </param>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> cannot represent the positive finite damping thresholds.</exception>
    public LevenbergMarquardtSolver(
        int maxIterations,
        T functionTolerance,
        T parameterTolerance,
        T gradientTolerance,
        T initialDamping)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxIterations);

        if (!T.IsFinite(s_lambdaCeiling) || !T.IsFinite(s_lambdaFloor) || s_lambdaFloor <= T.Zero)
        {
            throw new NotSupportedException("The numeric type cannot represent the positive finite damping thresholds 1e-12 and 1e16.");
        }

        if (!T.IsFinite(functionTolerance) || functionTolerance <= T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(functionTolerance), functionTolerance, "Function tolerance must be a positive value.");
        }

        if (!T.IsFinite(parameterTolerance) || parameterTolerance <= T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(parameterTolerance), parameterTolerance, "Parameter tolerance must be a positive value.");
        }

        // gradientTolerance = 0 is the legitimate exact-stationary-point gate
        // (‖Jᵀr‖∞ ≤ 0 fires only when the gradient is exactly zero); non-finite and
        // negative values, which silently disable or corrupt the gate, are rejected.
        if (!T.IsFinite(gradientTolerance) || gradientTolerance < T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(gradientTolerance), gradientTolerance, "Gradient tolerance must be zero or a positive value.");
        }

        // λ₀ is the absolute starting damping; a non-positive value cannot grow under the
        // multiplicative rejection update (0·ν = 0) and a negative value is not a valid
        // damping, so the initial damping must be strictly positive.
        if (!T.IsFinite(initialDamping) || initialDamping <= T.Zero || initialDamping >= s_lambdaCeiling)
        {
            throw new ArgumentOutOfRangeException(
                nameof(initialDamping), initialDamping, "Initial damping must be positive and less than the damping ceiling 1e16.");
        }

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
        ReadOnlySpan<T> initialGuess,
        Func<T[], T[,]>? jacobian = null,
        ReadOnlySpan<T> lowerBounds = default,
        ReadOnlySpan<T> upperBounds = default)
#pragma warning restore RS0026
    {
        ArgumentNullException.ThrowIfNull(residuals);

        ValidateInitialGuess(initialGuess);
        ValidateBounds(initialGuess.Length, lowerBounds, upperBounds);

        // Materialize the working parameter vector up-front (applying bounds clipping)
        // so we can size the pool against the first residual evaluation.
        var theta = initialGuess.ToArray();
        ClipToBounds(theta, theta.Length, lowerBounds, upperBounds);

        // Match pooled callback isolation before the residual count is known.
        var firstResidual = residuals((T[])theta.Clone())
            ?? throw new InvalidOperationException("Residual function returned null.");
        if (firstResidual.Length == 0)
        {
            throw new InvalidOperationException("Residual function returned an empty vector.");
        }
        ValidateFiniteResidual(firstResidual);

        var buffers = new LevenbergMarquardtBuffers<T>(theta.Length, firstResidual.Length);
        theta.CopyTo(buffers.Theta.AsSpan());
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
        ReadOnlySpan<T> initialGuess,
        LevenbergMarquardtBuffers<T> buffers,
        Func<T[], T[,]>? jacobian = null,
        ReadOnlySpan<T> lowerBounds = default,
        ReadOnlySpan<T> upperBounds = default)
#pragma warning restore RS0026
    {
        ArgumentNullException.ThrowIfNull(residuals);
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
        initialGuess.CopyTo(buffers.Theta.AsSpan());
        ClipToBounds(buffers.Theta, buffers.ParameterCount, lowerBounds, upperBounds);

        // Hand the residual callback a logical-length array: on a shrunk grow-only pool
        // buffers.Theta is a high-water physical array longer than ParameterCount with a stale
        // tail, and a contract-conforming residual that iterates over parameters.Length would
        // read that tail. On a fresh/same-size pool the array is passed through unchanged.
        var firstResidual = residuals(PrepareCallbackParameters(buffers.Theta, buffers))
            ?? throw new InvalidOperationException("Residual function returned null.");
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
        ValidateFiniteResidual(firstResidual);

        Array.Copy(firstResidual, buffers.Residual, firstResidual.Length);

        return SolveCore(residuals, buffers, jacobian, lowerBounds, upperBounds);
    }

    /// <summary>
    /// Pool-free writer overload. Allocates a private buffer pool internally.
    /// Use when the residual count is known up-front and you want zero-allocation
    /// only for the hot loop rather than for setup.
    /// </summary>
    /// <param name="residuals">Buffer-fill callback. Receives <c>(parameters, residualSpan)</c> where <c>residualSpan.Length == residualCount</c>. Must initialize every element of <c>residualSpan</c> on every call; the span is prefilled with NaN and unwritten or non-finite entries cause InvalidOperationException.</param>
    /// <param name="initialGuess">Starting parameter vector.</param>
    /// <param name="residualCount">Number of residuals the callback fills (= m).</param>
    /// <param name="jacobian">Optional analytic Jacobian writer. See <see cref="LmJacobianWriter{T}"/> for layout.</param>
    /// <param name="lowerBounds">Optional element-wise lower bounds. Empty span = no bounds.</param>
    /// <param name="upperBounds">Optional element-wise upper bounds. Empty span = no bounds.</param>
    /// <returns>Final parameter estimate, residuals, cost, and convergence diagnostics.</returns>
#pragma warning disable RS0026
    public MultivariateSolverResult<T> Solve(
        LmResidualWriter<T> residuals,
        ReadOnlySpan<T> initialGuess,
        int residualCount,
        LmJacobianWriter<T>? jacobian = null,
        ReadOnlySpan<T> lowerBounds = default,
        ReadOnlySpan<T> upperBounds = default)
#pragma warning restore RS0026
    {
        ArgumentNullException.ThrowIfNull(residuals);

        ValidateInitialGuess(initialGuess);
        ValidateBounds(initialGuess.Length, lowerBounds, upperBounds);

        if (residualCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(residualCount), residualCount,
                "Residual count must be positive.");
        }

        var theta = initialGuess.ToArray();
        ClipToBounds(theta, theta.Length, lowerBounds, upperBounds);

        var buffers = new LevenbergMarquardtBuffers<T>(theta.Length, residualCount);
        theta.CopyTo(buffers.Theta.AsSpan());
        var initialResidualSpan = buffers.Residual.AsSpan(0, residualCount);
        initialResidualSpan.Fill(T.NaN);
        residuals(buffers.Theta.AsSpan(0, theta.Length), initialResidualSpan);
        ValidateFiniteResidual(initialResidualSpan);

        return SolveCoreWriter(residuals, buffers, jacobian, lowerBounds, upperBounds);
    }

    /// <summary>
    /// Pooled writer overload. No per-iteration managed allocation after warmup when
    /// the residual and Jacobian callbacks are non-allocating. The result record,
    /// its owned arrays, and any one-shot covariance scratch still allocate per solve.
    /// </summary>
    /// <param name="residuals">Buffer-fill callback. Receives <c>(parameters, residualSpan)</c> where <c>residualSpan.Length == buffers.ResidualCount</c>. Must initialize every element of <c>residualSpan</c> on every call; the span is prefilled with NaN and unwritten or non-finite entries cause InvalidOperationException.</param>
    /// <param name="initialGuess">Starting parameter vector. Length must match <c>buffers.ParameterCount</c>.</param>
    /// <param name="buffers">Caller-owned scratch pool. Reusable across solves of matching dimensions.</param>
    /// <param name="jacobian">Optional analytic Jacobian writer.</param>
    /// <param name="lowerBounds">Optional element-wise lower bounds.</param>
    /// <param name="upperBounds">Optional element-wise upper bounds.</param>
    /// <returns>Final parameter estimate, residuals, cost, and convergence diagnostics.</returns>
#pragma warning disable RS0026
    public MultivariateSolverResult<T> Solve(
        LmResidualWriter<T> residuals,
        ReadOnlySpan<T> initialGuess,
        LevenbergMarquardtBuffers<T> buffers,
        LmJacobianWriter<T>? jacobian = null,
        ReadOnlySpan<T> lowerBounds = default,
        ReadOnlySpan<T> upperBounds = default)
#pragma warning restore RS0026
    {
        ArgumentNullException.ThrowIfNull(residuals);
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

        initialGuess.CopyTo(buffers.Theta.AsSpan());
        ClipToBounds(buffers.Theta, buffers.ParameterCount, lowerBounds, upperBounds);

        // Slice to the logical dimensions: on a shrunk grow-only pool the physical arrays
        // exceed the logical counts, and the writer contract requires
        // residualSpan.Length == buffers.ResidualCount (and a parameter span carrying no
        // stale tail). This mirrors the in-loop writer call.
        var pooledResidualSpan = buffers.Residual.AsSpan(0, buffers.ResidualCount);
        pooledResidualSpan.Fill(T.NaN);
        residuals(buffers.Theta.AsSpan(0, buffers.ParameterCount), pooledResidualSpan);
        ValidateFiniteResidual(pooledResidualSpan);

        return SolveCoreWriter(residuals, buffers, jacobian, lowerBounds, upperBounds);
    }

    private MultivariateSolverResult<T> SolveCore(
        Func<T[], T[]>? residuals,
        LevenbergMarquardtBuffers<T> buffers,
        Func<T[], T[,]>? jacobian,
        ReadOnlySpan<T> lowerBounds,
        ReadOnlySpan<T> upperBounds,
        LmResidualWriter<T>? writer = null,
        LmJacobianWriter<T>? analyticWriter = null)
    {
        var parameterCount = buffers.ParameterCount;
        var residualCount = buffers.ResidualCount;
        var theta = buffers.Theta;
        var residual = buffers.Residual;
        var trialTheta = buffers.TrialTheta;
        var trialResidual = buffers.TrialResidual;
        var effectiveDelta = buffers.EffectiveDelta;
        var delta = buffers.Delta;

        var cost = HalfSumSquares(residual, residualCount);

        // Initial Jacobian — needed for the iteration-0 gradient check and the first linear solve.
        EvaluateCallbackJacobian(residuals, writer, theta, jacobian, analyticWriter, buffers, lowerBounds, upperBounds);

        var iterations = 0;
        var acceptedIterations = 0;

        // λ₀ = initialDamping (absolute; Numerical Recipes §15.5.2 starting-value convention). The scaled-diagonal
        // damping form `(JᵀJ + λ·diag(JᵀJ))·δ = −Jᵀr` makes the step itself invariant
        // under per-parameter rescaling, so a fixed initial λ suffices across scales.
        var lambda = _initialDamping;
        var nu = s_initialRejectionMultiplier;
        var initialDampingDominates = lambda > T.One;

        // Range-limited damping: set when a trial point is rejected only because the step leaves
        // the representable range, cleared once an accepted step brings λ back to the level at
        // which that first happened. While set, λ is sized by the range rather than the model, so
        // the small reductions and steps it forces cannot certify FTOL or XTOL.
        var rangeLimited = false;
        var rangeLimitedDamping = lambda;

        // Gradient check at the initial point — if already stationary, return immediately.
        if (ProjectedGradientInfinityNormInto(
                buffers.Jacobian, residual, buffers.Gradient, theta,
                residualCount, parameterCount, lowerBounds, upperBounds) <= _gradientTolerance)
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
            // A user-supplied dominant initial damping can manufacture a tiny first
            // step far from a solution. Withhold step-based stops until it has relaxed;
            // later model-driven damping retains the usual MINPACK stopping policy.
            initialDampingDominates &= lambda > T.One;
            var solved = DampedLinearSolve<T>.TrySolve(residual, lambda, buffers, gradientAlreadyComputed: true,
                lowerBounds, upperBounds);
            if (!solved || !TryFormTrialPoint(theta, delta, trialTheta, parameterCount, lowerBounds, upperBounds))
            {
                // Damped matrix not factorable, or the clipped trial point is not representable —
                // treat as a rejection with Nielsen growth.
                if (!rangeLimited && (solved || IsStepUnrepresentable(delta, parameterCount)))
                {
                    rangeLimited = true;
                    rangeLimitedDamping = lambda;
                }

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

            var stepWasClipped = false;

            // Effective step (after clipping) — used for parameter-tolerance test.
            for (var i = 0; i < parameterCount; i++)
            {
                effectiveDelta[i] = trialTheta[i] - theta[i];
                stepWasClipped |= trialTheta[i] != theta[i] + delta[i];
            }

            EvaluateCallbackResidual(residuals, writer, trialTheta, trialResidual, buffers);
            var trialCost = HalfSumSquares(trialResidual, residualCount);

            // Projection invalidates the damped-system identity; evaluate the model directly.
            // The damped solve's Rhs/Y workspace is free until the next solve rewrites it.
            var residualExponent = DampedLinearSolve<T>.VectorScaleExponent(residual, residualCount);
            var scaledPredictedReduction = stepWasClipped
                ? ProjectedStepPredictedReduction(buffers.Jacobian, residual, effectiveDelta, residualExponent,
                    buffers.Rhs, buffers.Y, residualCount, parameterCount)
                : PredictedReduction(buffers.Jacobian, residual, effectiveDelta, lambda, residualExponent,
                    residualCount, parameterCount);
            var predictedReduction = T.ScaleB(scaledPredictedReduction, 2 * residualExponent);
            var actualReduction = cost - trialCost;

            // Gain ratio and MINPACK relative reductions (relative to the current cost) in the
            // same residual scale: ½Σr² can round to zero, or overflow, although the residuals
            // and these ratios are representable.
            var scaledCost = ScaledHalfSumSquares(residual, residualCount, residualExponent);
            var scaledActualReduction = scaledCost - ScaledHalfSumSquares(trialResidual, residualCount, residualExponent);
            var gainRatio = scaledPredictedReduction > T.Zero ? scaledActualReduction / scaledPredictedReduction : -T.One;
            var relativeActualReduction = scaledCost > T.Zero ? scaledActualReduction / scaledCost : actualReduction;
            var relativePredictedReduction = scaledCost > T.Zero ? scaledPredictedReduction / scaledCost : predictedReduction;

            if (gainRatio > T.Zero)
            {
                // Step accepted.
                var parameterStepIsSmall = ParameterStepWithinTolerance(effectiveDelta, theta, parameterCount);

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
                EvaluateCallbackJacobian(residuals, writer, theta, jacobian, analyticWriter, buffers, lowerBounds, upperBounds);

                // Tolerance tests — check in priority order: function, parameter, gradient.
                // MINPACK lmder.f 418-419 requires both relative reductions and
                // half the gain ratio <= 1; tiny reductions alone do not satisfy FTOL.
                if (!stepWasClipped && !rangeLimited && !initialDampingDominates && T.Abs(relativeActualReduction) <= _functionTolerance
                    && relativePredictedReduction <= _functionTolerance
                    && gainRatio / s_two <= T.One)
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

                if (!stepWasClipped && !rangeLimited && !initialDampingDominates && parameterStepIsSmall)
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

                if (ProjectedGradientInfinityNormInto(
                        buffers.Jacobian, residual, buffers.Gradient, theta,
                        residualCount, parameterCount, lowerBounds, upperBounds) <= _gradientTolerance)
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

                rangeLimited &= lambda > rangeLimitedDamping;
            }
            else
            {
                // Step rejected — Nielsen growth: λ ← λ·ν, ν ← 2ν.
                // MINPACK lmder tests convergence after unsuccessful iterations too.
                // Keep the current point, but recognize a rounded-away cost change
                // only when the independently predicted decrease is also small.
                // The current projected gradient already failed its tolerance check.
                // A shortened projected step cannot supply an unconstrained stop certificate,
                // and neither can one shortened by range-limited damping.
                if (!stepWasClipped && !rangeLimited && !initialDampingDominates && relativePredictedReduction >= T.Zero
                    && T.Abs(relativeActualReduction) <= _functionTolerance
                    && relativePredictedReduction <= _functionTolerance
                    && gainRatio / s_two <= T.One)
                {
                    return BuildResult(buffers, cost, acceptedIterations, converged: true,
                        LmTerminationReason.FunctionToleranceReached, lowerBounds, upperBounds);
                }

                // Madsen–Nielsen–Tingleff (3.15b) also stops on a small rejected
                // step. Preserve the current point when its trial cost did not improve.
                if (!stepWasClipped && !rangeLimited && !initialDampingDominates && ParameterStepWithinTolerance(effectiveDelta, theta, parameterCount))
                {
                    return BuildResult(buffers, cost, acceptedIterations, converged: true,
                        LmTerminationReason.ParameterToleranceReached, lowerBounds, upperBounds);
                }

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
        LevenbergMarquardtBuffers<T> buffers,
        ReadOnlySpan<T> lowerBounds,
        ReadOnlySpan<T> upperBounds)
    {
        if (analyticJacobian is null)
        {
            FiniteDifferenceJacobian<T>.EvaluateInto(residuals, theta, buffers, lowerBounds, upperBounds);
            return;
        }

        // User-supplied analytic Jacobian: copy into the pool's Jacobian slab. The user's
        // returned array is allocated per call (outside our control); our inner work
        // (TrySolve, PredictedReduction, GradientInfinityNorm) reads exclusively from
        // the pool slab so allocation on this path is bounded to the user's callback.
        var m = buffers.ResidualCount;
        var n = buffers.ParameterCount;

        // Hand the callback a logical-length parameter array: on a shrunk grow-only pool theta
        // is a high-water physical array longer than n, and a contract-conforming Jacobian that
        // sizes its column count from parameters.Length would return an [m, capacity] matrix
        // that fails the shape check below. On a fresh/same-size pool (theta.Length == n) the
        // array is passed through unchanged — no allocation, identical behavior.
        var userJacobian = analyticJacobian(PrepareCallbackParameters(theta, buffers))
            ?? throw new InvalidOperationException("Analytic Jacobian returned null.");

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
                if (!T.IsFinite(userJacobian[i, j]))
                {
                    throw new InvalidOperationException(
                        $"Analytic Jacobian returned a non-finite value at row {i}, column {j}.");
                }

                target[i, j] = userJacobian[i, j];
            }
        }
    }

    // Both model reductions below are evaluated in the power-of-two variables of the damped
    // solve: J = U·D with D = diag(2^eⱼ) from the column maxima, r = s·ρ with s = 2^residualExponent,
    // and z = D·δ/s. They return the reduction divided by s², so no square or product of
    // unscaled entries is formed and a representable reduction (or ratio of reductions) is not
    // lost to an intermediate overflow or underflow. Every scale is a power of two, so the
    // value equals the unscaled formula exactly whenever that formula stays in normal range.

    private static T ProjectedStepPredictedReduction(
        T[,] jacobian,
        T[] residual,
        T[] delta,
        int residualExponent,
        T[] columnFactors,
        T[] scaledStep,
        int residualCount,
        int parameterCount)
    {
        // Madsen, Nielsen and Tingleff (2004), p.25: -rᵀJδ - ½||Jδ||².
        // This expression holds for an arbitrary step, including a clipped one.
        // `columnFactors` and `scaledStep` are caller-owned length-n scratch.
        var residualFactor = T.ScaleB(T.One, -residualExponent);
        for (var j = 0; j < parameterCount; j++)
        {
            var columnExponent = DampedLinearSolve<T>.ColumnScaleExponent(jacobian, residualCount, j);
            columnFactors[j] = T.ScaleB(T.One, -columnExponent);
            scaledStep[j] = T.ScaleB(delta[j], columnExponent - residualExponent);
        }

        var reduction = T.Zero;
        for (var i = 0; i < residualCount; i++)
        {
            var modelStep = T.Zero;
            for (var j = 0; j < parameterCount; j++)
            {
                modelStep += (jacobian[i, j] * columnFactors[j]) * scaledStep[j];
            }

            reduction -= (residual[i] * residualFactor) * modelStep + s_half * modelStep * modelStep;
        }

        return reduction;
    }

    private static T PredictedReduction(
        T[,] jacobian,
        T[] residual,
        T[] delta,
        T lambda,
        int residualExponent,
        int residualCount,
        int parameterCount)
    {
        // Predicted reduction for the quadratic model with Marquardt damping:
        // predicted = ½·δᵀ·(λ·diag(JᵀJ)·δ − Jᵀr) = s²·½·Σⱼ zⱼ·(λ·diag(UᵀU)ⱼ·zⱼ − (Uᵀρ)ⱼ).
        var residualFactor = T.ScaleB(T.One, -residualExponent);
        var sum = T.Zero;
        for (var j = 0; j < parameterCount; j++)
        {
            // Diagonal of UᵀU and j-th component of the scaled gradient Uᵀρ.
            var columnExponent = DampedLinearSolve<T>.ColumnScaleExponent(jacobian, residualCount, j);
            var columnFactor = T.ScaleB(T.One, -columnExponent);
            var diag = T.Zero;
            var gj = T.Zero;
            for (var i = 0; i < residualCount; i++)
            {
                var u = jacobian[i, j] * columnFactor;
                diag += u * u;
                gj += u * (residual[i] * residualFactor);
            }

            var z = T.ScaleB(delta[j], columnExponent - residualExponent);
            sum += z * ((lambda * diag * z) - gj);
        }

        return s_half * sum;
    }

    private static MultivariateSolverResult<T> BuildResult(
        LevenbergMarquardtBuffers<T> buffers,
        T cost,
        int iterations,
        bool converged,
        LmTerminationReason reason,
        ReadOnlySpan<T> lowerBounds,
        ReadOnlySpan<T> upperBounds)
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
        var covariance = BuildCovariance(buffers.Jacobian, finalResiduals, cost, residualCount, parameterCount, converged);

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
        T[] residuals,
        T cost,
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

        // Scale the triangular inverse by σ̂ before taking its Gram product. This
        // avoids materializing either an overflowing inverse or an overflowing RSS.
        var degreesOfFreedom = T.CreateChecked(residualCount - parameterCount);
        var sigma = T.Sqrt(cost) * T.Sqrt(s_two / degreesOfFreedom);
        if (!T.IsNormal(cost))
        {
            // A zero or subnormal squared cost need not imply a zero residual norm.
            var scale = T.Zero;
            for (var i = 0; i < residualCount; i++)
            {
                scale = T.Max(scale, T.Abs(residuals[i]));
            }
            if (scale > T.Zero)
            {
                var sum = T.Zero;
                for (var i = 0; i < residualCount; i++)
                {
                    var normalized = residuals[i] / scale;
                    sum += normalized * normalized;
                }
                sigma = scale * T.Sqrt(sum / degreesOfFreedom);
            }
        }
        return DampedLinearSolve<T>.TryInvertNormalEquations(jacobian, residualCount, parameterCount, sigma);
    }

    private static void ValidateInitialGuess(ReadOnlySpan<T> initialGuess)
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

    private static void ValidateBounds(int parameterCount, ReadOnlySpan<T> lowerBounds, ReadOnlySpan<T> upperBounds)
    {
        if (!lowerBounds.IsEmpty && lowerBounds.Length != parameterCount)
        {
            throw new ArgumentException(
                $"Lower bounds length {lowerBounds.Length} does not match parameter count {parameterCount}.",
                nameof(lowerBounds));
        }

        if (!upperBounds.IsEmpty && upperBounds.Length != parameterCount)
        {
            throw new ArgumentException(
                $"Upper bounds length {upperBounds.Length} does not match parameter count {parameterCount}.",
                nameof(upperBounds));
        }

        for (var i = 0; i < parameterCount; i++)
        {
            if (!lowerBounds.IsEmpty &&
                (T.IsNaN(lowerBounds[i]) || T.IsPositiveInfinity(lowerBounds[i])))
            {
                throw new ArgumentException(
                    $"Lower bound at index {i} must be finite or negative infinity.", nameof(lowerBounds));
            }

            if (!upperBounds.IsEmpty &&
                (T.IsNaN(upperBounds[i]) || T.IsNegativeInfinity(upperBounds[i])))
            {
                throw new ArgumentException(
                    $"Upper bound at index {i} must be finite or positive infinity.", nameof(upperBounds));
            }

            if (!lowerBounds.IsEmpty && !upperBounds.IsEmpty && lowerBounds[i] > upperBounds[i])
            {
                throw new ArgumentException(
                    $"Lower bound at index {i} must not exceed the corresponding upper bound.",
                    nameof(lowerBounds));
            }
        }
    }

    private static void ValidateResidualLength(T[] residual, int expectedCount)
    {
        if (residual is null)
        {
            throw new InvalidOperationException("Residual function returned null.");
        }

        if (residual.Length != expectedCount)
        {
            throw new InvalidOperationException(
                $"Residual function returned {residual.Length} residuals; expected {expectedCount}.");
        }
    }

    private static void ValidateFiniteResidual(ReadOnlySpan<T> residual)
    {
        for (var i = 0; i < residual.Length; i++)
        {
            if (!T.IsFinite(residual[i]))
            {
                throw new InvalidOperationException(
                    $"Residual function returned a non-finite value at index {i}.");
            }
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

    private static T HalfSumSquares(T[] vector, int length)
    {
        var sum = SumSquares(vector, length);
        if (T.IsNormal(sum))
        {
            return s_half * sum;
        }

        // Individual squares can underflow although their aggregate is representable;
        // conversely a raw sum can overflow although half the sum fits.
        var scale = T.Zero;
        for (var i = 0; i < length; i++)
        {
            scale = T.Max(scale, T.Abs(vector[i]));
        }
        if (scale == T.Zero)
        {
            return T.Zero;
        }
        sum = T.Zero;
        for (var i = 0; i < length; i++)
        {
            var normalized = vector[i] / scale;
            sum += normalized * normalized;
        }

        return (scale * (s_half * sum)) * scale;
    }

    /// <summary>
    /// <c>½Σvᵢ² / 2^(2·exponent)</c>, evaluated as <c>½Σ(vᵢ·2^-exponent)²</c>. With the
    /// current residual's range-scaling exponent the current entries have magnitude below 2,
    /// so a cost that rounds to zero (or overflows) in <typeparamref name="T"/> keeps its
    /// relation to the other reductions measured in the same scale.
    /// </summary>
    private static T ScaledHalfSumSquares(T[] vector, int length, int exponent)
    {
        var factor = T.ScaleB(T.One, -exponent);
        var sum = T.Zero;
        for (var i = 0; i < length; i++)
        {
            var scaled = vector[i] * factor;
            sum += scaled * scaled;
        }

        return s_half * sum;
    }

    private static T Norm2(T[] vector, int length)
    {
        var sum = SumSquares(vector, length);
        if (T.IsFinite(sum) && sum > T.Zero)
        {
            return T.Sqrt(sum);
        }

        // Scale only when squaring exceeds the working range (or underflows).
        // Preserve the established rounding path for ordinary inputs.
        var scale = T.Zero;
        for (var i = 0; i < length; i++)
        {
            scale = T.Max(scale, T.Abs(vector[i]));
        }
        if (scale == T.Zero)
        {
            return T.Zero;
        }

        sum = T.Zero;
        for (var i = 0; i < length; i++)
        {
            var normalized = vector[i] / scale;
            sum += normalized * normalized;
        }
        return scale * T.Sqrt(sum);
    }

    /// <summary>
    /// The documented parameter-tolerance predicate <c>‖δ‖ ≤ tol·(‖θ‖ + tol)</c>. When the
    /// threshold is not finite (‖θ‖, or its product with the tolerance, exceeds the range),
    /// the predicate is evaluated after dividing δ, θ and the additive tolerance by one common
    /// power of two, so an unrepresentable ‖θ‖ cannot become an infinite threshold that
    /// certifies any step.
    /// </summary>
    private bool ParameterStepWithinTolerance(T[] step, T[] theta, int parameterCount)
    {
        var stepNorm = Norm2(step, parameterCount);
        var threshold = _parameterTolerance * (Norm2(theta, parameterCount) + _parameterTolerance);
        if (T.IsFinite(threshold))
        {
            return stepNorm <= threshold;
        }

        var exponent = int.Max(
            DampedLinearSolve<T>.VectorScaleExponent(step, parameterCount),
            DampedLinearSolve<T>.VectorScaleExponent(theta, parameterCount));
        return ScaledNorm2(step, parameterCount, exponent)
            <= _parameterTolerance * (ScaledNorm2(theta, parameterCount, exponent)
                + T.ScaleB(_parameterTolerance, -exponent));
    }

    /// <summary>
    /// <c>‖v‖ / 2^exponent</c> for an <paramref name="exponent"/> at least the vector's own
    /// range-scaling exponent, so every scaled entry has magnitude below 2.
    /// </summary>
    private static T ScaledNorm2(T[] vector, int length, int exponent)
    {
        var factor = T.ScaleB(T.One, -exponent);
        var sum = T.Zero;
        for (var i = 0; i < length; i++)
        {
            var scaled = vector[i] * factor;
            sum += scaled * scaled;
        }

        return T.Sqrt(sum);
    }

    /// <summary>
    /// Computes the gradient <c>Jᵀr</c> into <paramref name="gradient"/> (the pool's
    /// <see cref="LevenbergMarquardtBuffers{T}.Gradient"/> slab) and returns its ∞-norm.
    /// Persisting the vector lets the immediately following damped solve reuse it instead of
    /// recomputing <c>Jᵀr</c>. Products that would underflow or overflow are accumulated
    /// after power-of-two scaling, with the combined exponent restored after summation. A nonzero
    /// gradient that flushes to zero when restored is reported as the smallest positive value, so
    /// the returned norm is zero exactly when the exact gradient is zero. An exact gradient beyond
    /// the representable range is reported as <c>±∞</c> (norm <c>+∞</c>), which no finite tolerance
    /// accepts; only a NaN sum raises <see cref="InvalidOperationException"/>.
    /// </summary>
    private static T ProjectedGradientInfinityNormInto(
        T[,] jacobian,
        T[] residual,
        T[] gradient,
        T[] theta,
        int residualCount,
        int parameterCount,
        ReadOnlySpan<T> lowerBounds,
        ReadOnlySpan<T> upperBounds)
    {
        var maxAbs = T.Zero;

        for (var j = 0; j < parameterCount; j++)
        {
            var gj = T.Zero;
            var underflow = false;
            var jacobianScale = T.Zero;
            var residualScale = T.Zero;
            for (var i = 0; i < residualCount; i++)
            {
                var product = jacobian[i, j] * residual[i];
                gj += product;
                underflow |= jacobian[i, j] != T.Zero && residual[i] != T.Zero
                    && T.IsFinite(product) && !T.IsNormal(product);
                jacobianScale = T.Max(jacobianScale, T.Abs(jacobian[i, j]));
                residualScale = T.Max(residualScale, T.Abs(residual[i]));
            }

            // `direction` carries the gradient's sign (and exact-zero status) to the bound test
            // and the magnitude test below; `gj` is the absolute value stored for the caller.
            // They differ only on the rescale branch, where restoring the power-of-two scale can
            // flush a nonzero scaled sum to zero.
            var direction = gj;
            // Every factor is finite, so a non-finite raw sum means a product or partial sum
            // overflowed; the exact gradient may still be representable (e.g. by cancellation).
            if (underflow || !T.IsFinite(gj))
            {
                var jacobianExponent = T.ILogB(jacobianScale);
                var residualExponent = T.ILogB(residualScale);
                direction = T.Zero;
                for (var i = 0; i < residualCount; i++)
                {
                    direction += T.ScaleB(jacobian[i, j], -jacobianExponent)
                        * T.ScaleB(residual[i], -residualExponent);
                }
                gj = T.ScaleB(direction, jacobianExponent + residualExponent);
            }

            // After the rescale every term is finite, so only a NaN sum is an arithmetic failure.
            // An exact gradient beyond the representable range restores to ±∞ and is kept: it is
            // larger than every finite tolerance (never "stationary"), its sign still drives the
            // bound test, and the damped step is solved in range-scaled variables without it.
            if (T.IsNaN(gj))
            {
                throw new InvalidOperationException(
                    $"Accumulated gradient is non-finite at parameter {j}; rescale the residuals or parameters.");
            }

            gradient[j] = gj;

            var blockedAtLower = !lowerBounds.IsEmpty && theta[j] <= lowerBounds[j] && direction > T.Zero;
            var blockedAtUpper = !upperBounds.IsEmpty && theta[j] >= upperBounds[j] && direction < T.Zero;
            if (blockedAtLower || blockedAtUpper)
            {
                continue;
            }

            // The norm feeds a `<= gradientTolerance` test against a representable tolerance, so
            // a nonzero gradient that flushed to zero is rounded away from zero to the smallest
            // positive value: within every positive tolerance, as the exact value is, but not
            // within the exact-stationary tolerance 0. Magnitudes that fit are unchanged.
            var abs = T.Abs(gj);
            if (T.IsZero(abs) && !T.IsZero(direction))
            {
                abs = T.Epsilon;
            }

            if (abs > maxAbs)
            {
                maxAbs = abs;
            }
        }

        return maxAbs;
    }

    private static T[] PrepareCallbackParameters(T[] source, LevenbergMarquardtBuffers<T> buffers)
    {
        var target = buffers.CallbackParameters;
        Array.Copy(source, target, buffers.ParameterCount);
        return target;
    }

    /// <summary>
    /// After a failed damped solve, reports whether the failure was a step beyond the
    /// representable range rather than an unusable pivot. The solve writes each unscaled step
    /// component into <paramref name="delta"/> and fails on the first non-finite one; before
    /// that point the buffer holds finite power-of-two column scales.
    /// </summary>
    private static bool IsStepUnrepresentable(T[] delta, int parameterCount)
    {
        for (var i = 0; i < parameterCount; i++)
        {
            if (!T.IsFinite(delta[i]))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Forms the bound-clipped trial <c>θ + δ</c>. Returns <see langword="false"/> when a
    /// component is not finite after clipping: a finite step can still carry a finite
    /// parameter past the representable range, and such a trial is rejected rather than
    /// passed to the caller's callbacks.
    /// </summary>
    private static bool TryFormTrialPoint(
        T[] theta,
        T[] delta,
        T[] trialTheta,
        int parameterCount,
        ReadOnlySpan<T> lowerBounds,
        ReadOnlySpan<T> upperBounds)
    {
        for (var i = 0; i < parameterCount; i++)
        {
            trialTheta[i] = theta[i] + delta[i];
        }

        ClipToBounds(trialTheta, parameterCount, lowerBounds, upperBounds);

        for (var i = 0; i < parameterCount; i++)
        {
            if (!T.IsFinite(trialTheta[i]))
            {
                return false;
            }
        }

        return true;
    }

    private static void ClipToBounds(T[] theta, int count, ReadOnlySpan<T> lowerBounds, ReadOnlySpan<T> upperBounds)
    {
        if (lowerBounds.IsEmpty && upperBounds.IsEmpty)
        {
            return;
        }

        // Iterate the LOGICAL parameter count, not theta.Length: on a grow-only pool that
        // has been shrunk via Reset, theta is a high-water physical array longer than the
        // logical count, but the bound spans are sized to the logical count. Indexing
        // lowerBounds[i]/upperBounds[i] past the logical count would throw.
        for (var i = 0; i < count; i++)
        {
            if (!lowerBounds.IsEmpty && theta[i] < lowerBounds[i])
            {
                theta[i] = lowerBounds[i];
            }

            if (!upperBounds.IsEmpty && theta[i] > upperBounds[i])
            {
                theta[i] = upperBounds[i];
            }
        }
    }

    private static bool IsAnyComponentOnBound(
        T[] theta,
        ReadOnlySpan<T> lowerBounds,
        ReadOnlySpan<T> upperBounds)
    {
        if (lowerBounds.IsEmpty && upperBounds.IsEmpty)
        {
            return false;
        }

        for (var i = 0; i < theta.Length; i++)
        {
            if (!lowerBounds.IsEmpty && theta[i] == lowerBounds[i])
            {
                return true;
            }

            if (!upperBounds.IsEmpty && theta[i] == upperBounds[i])
            {
                return true;
            }
        }

        return false;
    }

    // Callback adapters share the iteration engine without allocating delegates in the loop.
    private MultivariateSolverResult<T> SolveCoreWriter(
        LmResidualWriter<T> writer,
        LevenbergMarquardtBuffers<T> buffers,
        LmJacobianWriter<T>? analyticJacobian,
        ReadOnlySpan<T> lowerBounds,
        ReadOnlySpan<T> upperBounds)
        => SolveCore(null, buffers, null, lowerBounds, upperBounds, writer, analyticJacobian);

    private static void EvaluateCallbackResidual(
        Func<T[], T[]>? residuals,
        LmResidualWriter<T>? writer,
        T[] parameters,
        T[] destination,
        LevenbergMarquardtBuffers<T> buffers)
    {
        if (writer is null)
        {
            var values = residuals!(PrepareCallbackParameters(parameters, buffers));
            ValidateResidualLength(values, buffers.ResidualCount);
            ValidateFiniteResidual(values);
            Array.Copy(values, destination, buffers.ResidualCount);
            return;
        }

        // Slice high-water buffers to the current problem's logical dimensions.
        var residualSpan = destination.AsSpan(0, buffers.ResidualCount);
        residualSpan.Fill(T.NaN);
        writer(parameters.AsSpan(0, buffers.ParameterCount), residualSpan);
        ValidateFiniteResidual(residualSpan);
    }

    private static void EvaluateCallbackJacobian(
        Func<T[], T[]>? residuals,
        LmResidualWriter<T>? writer,
        T[] parameters,
        Func<T[], T[,]>? jacobian,
        LmJacobianWriter<T>? analyticWriter,
        LevenbergMarquardtBuffers<T> buffers,
        ReadOnlySpan<T> lowerBounds,
        ReadOnlySpan<T> upperBounds)
    {
        if (writer is null)
        {
            EvaluateJacobianInto(residuals!, parameters, jacobian, buffers, lowerBounds, upperBounds);
        }
        else
        {
            EvaluateJacobianIntoWriter(writer, parameters, analyticWriter, buffers, lowerBounds, upperBounds);
        }
    }

    private static void EvaluateJacobianIntoWriter(
        LmResidualWriter<T> writer,
        T[] theta,
        LmJacobianWriter<T>? analyticJacobian,
        LevenbergMarquardtBuffers<T> buffers,
        ReadOnlySpan<T> lowerBounds,
        ReadOnlySpan<T> upperBounds)
    {
        if (analyticJacobian is null)
        {
            FiniteDifferenceJacobian<T>.EvaluateInto(writer, theta, buffers, lowerBounds, upperBounds);
            return;
        }

        // User-supplied analytic Jacobian writer: the contract is a flat row-major span
        // with stride = parameterCount. The pool's Jacobian slab is grow-only, so after a
        // shrinking Reset its physical column stride (slab width) can exceed the logical
        // parameterCount. In that case a flat span over the slab is NOT contiguous at the
        // writer's stride, so we must scatter row-by-row; only when the slab width equals
        // parameterCount is the slab already contiguous and we can write in place (no alloc).
        var m = buffers.ResidualCount;
        var n = buffers.ParameterCount;
        var slab = buffers.Jacobian;

        // Slice theta to the LOGICAL parameter count before handing it to the user callback:
        // on a shrunk grow-only pool theta is a high-water physical array longer than n, and a
        // contract-conforming writer that derives its column stride from parameters.Length
        // would otherwise overflow the m·n output span. On a fresh/same-size pool
        // (capacity == n) this slice is a no-op.
        var thetaLogical = theta.AsSpan(0, n);
        if (slab.GetLength(1) == n)
        {
            var flatJacobian = System.Runtime.InteropServices.MemoryMarshal.CreateSpan(
                ref slab[0, 0], m * n);
            flatJacobian.Fill(T.NaN);
            analyticJacobian(thetaLogical, flatJacobian);
            ValidateFiniteJacobian(flatJacobian, n);
            return;
        }

        // Shrunk pool: fill a contiguous temporary at the logical stride, then scatter into
        // the slab honoring its physical row stride. The scratch is a pool-owned buffer sized
        // to the high-water m·n capacity, sliced to the logical m·n here, so this path adds
        // no per-call allocation (honoring the pool's zero-allocation-after-warmup contract).
        var flat = buffers.JacobianFlatScratch.AsSpan(0, m * n);
        flat.Fill(T.NaN);
        analyticJacobian(thetaLogical, flat);
        ValidateFiniteJacobian(flat, n);
        for (var i = 0; i < m; i++)
        {
            for (var j = 0; j < n; j++)
            {
                slab[i, j] = flat[(i * n) + j];
            }
        }
    }

    private static void ValidateFiniteJacobian(ReadOnlySpan<T> jacobian, int parameterCount)
    {
        for (var i = 0; i < jacobian.Length; i++)
        {
            if (!T.IsFinite(jacobian[i]))
            {
                throw new InvalidOperationException(
                    $"Analytic Jacobian returned a non-finite value at row {i / parameterCount}, " +
                    $"column {i % parameterCount}.");
            }
        }
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
/// Bounds (optional) are handled by projection: each trial is clipped before its residual
/// and predicted model reduction are evaluated, into <c>[lowerBounds, upperBounds]</c>. When any component lies on a bound
/// at termination the <see cref="MultivariateSolverResult.BoundsActive"/> flag is set;
/// convergence is reported orthogonally based on the tolerance tests evaluated on the
/// free (non-bound-clipped) components.
/// </para>
/// <para>
/// Deterministic: no RNG dependency, no dependence on evaluation order across threads.
/// Same inputs produce bit-identical outputs.
/// </para>
/// <para>
/// Range limit of the convergence guarantee: after a trial step the representable range
/// rejects, <c>FunctionToleranceReached</c> and <c>ParameterToleranceReached</c> are
/// withheld until damping returns to its prior level, so an optimum reachable only through
/// such steps ends with <c>Converged = false</c>. When no step is range-rejected the
/// MINPACK tolerance tests apply unchanged, and either stop can still certify a
/// non-stationary point of a numerically rank-deficient problem whose exact optimum lies
/// beyond the representable range; <c>Converged = true</c> from those two stops is not a
/// proof of stationarity.
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
[Obsolete(
    "Use LevenbergMarquardtSolver<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class LevenbergMarquardtSolver : IMultivariateLeastSquaresSolver
{
    private readonly LevenbergMarquardtSolver<double> _inner;

    /// <summary>
    /// Initializes a new <see cref="LevenbergMarquardtSolver"/>.
    /// </summary>
    /// <param name="maxIterations">Maximum trial steps, including rejected trials. Defaults to 200. The result counts accepted steps only.</param>
    /// <param name="functionTolerance">
    /// Stop when both the absolute actual and nonnegative predicted reductions, relative
    /// to the current cost, are at most this value. Checked after accepted and rejected
    /// trials; a rejected trial leaves the parameters unchanged. MINPACK lmder also
    /// requires half the actual-to-predicted reduction ratio to be at most one. Like every
    /// MINPACK tolerance stop, it can still certify a non-stationary point of a numerically
    /// rank-deficient problem whose exact optimum lies beyond the representable range.
    /// Defaults to 1e-10.
    /// </param>
    /// <param name="parameterTolerance">
    /// Stop when the step norm <c>‖δ‖ ≤ parameterTolerance · (‖θ‖ + parameterTolerance)</c>.
    /// It can still certify a non-stationary point of a numerically rank-deficient
    /// problem whose exact optimum lies beyond the representable range.
    /// Defaults to 1e-10.
    /// </param>
    /// <param name="gradientTolerance">
    /// Stop when <c>‖Jᵀr‖∞ ≤ gradientTolerance</c>. Defaults to 1e-10. This absolute
    /// tolerance has residual-times-Jacobian units; scaling residuals by <c>s</c> requires scaling it by <c>s²</c>.
    /// </param>
    /// <param name="initialDamping">
    /// Absolute starting value of the damping parameter <c>λ₀ = initialDamping</c>
    /// (the Numerical Recipes §15.5.2 starting-value convention). Per-parameter scale invariance is already delivered by
    /// the Marquardt scaled-diagonal form inside the step equation
    /// <c>(JᵀJ + λ · diag(JᵀJ)) · δ = −Jᵀr</c>; double-scaling at initialization
    /// (the Nielsen τ·max(diag) variant) was tried and reverted — it produced
    /// zero-progress failures on ill-scaled problems like NIST Hahn1 where
    /// <c>max(diag(JᵀJ))</c> at the starting point drove λ above the step
    /// resolution of double precision. Defaults to 1e-3. Must be below 1e16.
    /// If it starts above one, function and parameter stops wait until damping has
    /// fallen to at most one, preventing initial regularization from certifying convergence.
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
        ArgumentNullException.ThrowIfNull(initialGuess);
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
        ArgumentNullException.ThrowIfNull(initialGuess);
        ArgumentNullException.ThrowIfNull(buffers);
        var r = _inner.Solve(residuals, initialGuess, buffers.Inner, jacobian, lowerBounds, upperBounds);
        return new MultivariateSolverResult(r.Parameters, r.FinalCost, r.FinalResiduals, r.ParameterCovariance, r.Iterations, r.Converged, r.TerminationReason, r.BoundsActive);
    }
}
