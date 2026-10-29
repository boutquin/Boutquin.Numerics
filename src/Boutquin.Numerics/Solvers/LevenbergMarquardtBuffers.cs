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
/// Pre-allocated scratch buffers for Levenberg–Marquardt iterations, generic over
/// any <see cref="IFloatingPointIeee754{TSelf}"/> type.
/// Reusing one <see cref="LevenbergMarquardtBuffers{T}"/> instance across many
/// solve calls — provided the problem dimensions match — eliminates per-iteration
/// managed-heap allocation for hot-path consumers (bootstrap loops, real-time refits).
/// </summary>
/// <typeparam name="T">
/// Floating-point element type. Must satisfy <see cref="IFloatingPointIeee754{TSelf}"/>
/// (Tier B generic-math constraint).
/// </typeparam>
/// <remarks>
/// <para>
/// The buffer set is sized for one specific <c>(parameterCount, residualCount)</c> pair.
/// Calling <see cref="Reset(int, int)"/> reconfigures the storage when those change
/// (e.g., fitting a different model against the same pool). The policy is grow-only:
/// repeated resets to smaller sizes retain the original work-buffer capacity and
/// update the logical <see cref="ParameterCount"/> / <see cref="ResidualCount"/>.
/// A change in parameter count also replaces the exact-length callback parameter array.
/// </para>
/// <para>
/// Thread safety: not thread-safe. One buffer instance per thread; each thread constructs
/// its own pool and reuses it across sequential solves on that thread. Concurrent solves
/// sharing a pool produce undefined results.
/// </para>
/// <para>
/// Allocation contract: construction and <see cref="Reset(int, int)"/> calls that exceed
/// the current capacity allocate the internal arrays. A reset to a different parameter
/// count also allocates an exact-length callback parameter array, even within capacity.
/// Subsequent solves using the pool do not grow managed-heap allocation inside
/// the iteration loop — the only per-solve
/// allocations come from the caller's residual (and optional analytic Jacobian) callbacks
/// and from the returned <see cref="MultivariateSolverResult"/> record plus its owned arrays.
/// </para>
/// </remarks>
public sealed class LevenbergMarquardtBuffers<T>
    where T : IFloatingPointIeee754<T>
{
    // Storage capacities — only grow, never shrink.
    private int _capacityParameterCount;
    private int _capacityResidualCount;

    // Working storage. Fields are strictly private and exposed to the solver and its
    // internal helpers via PascalCase internal properties. All entries are fully
    // overwritten on every use; no zero-initialization between iterations is required.
    private T[,] _jacobian = null!;
    private T[,] _normalEquations = null!;
    private T[] _gradient = null!;
    private T[] _delta = null!;
    private T[] _rhs = null!;
    private T[] _y = null!;
    private T[] _theta = null!;
    private T[] _trialTheta = null!;
    private T[] _effectiveDelta = null!;
    private T[] _perturbedParameters = null!;
    private T[] _plusBuffer = null!;
    private T[] _minusBuffer = null!;
    private T[] _residual = null!;
    private T[] _trialResidual = null!;
    private T[] _jacobianFlatScratch = null!;
    private T[] _callbackParameters = [];

    /// <summary>
    /// Initializes a buffer pool sized for a specific <c>(parameterCount, residualCount)</c> problem.
    /// </summary>
    /// <param name="parameterCount">Dimension <c>n</c> of the parameter vector <c>theta</c>. Must be positive.</param>
    /// <param name="residualCount">Dimension <c>m</c> of the residual vector <c>r(theta)</c>. Must be positive.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="parameterCount"/> or <paramref name="residualCount"/> is not positive.
    /// </exception>
    public LevenbergMarquardtBuffers(int parameterCount, int residualCount)
    {
        Reset(parameterCount, residualCount);
    }

    /// <summary>
    /// Logical parameter count <c>n</c> the pool is currently configured for.
    /// May be less than the underlying allocated capacity after a shrinking <see cref="Reset(int, int)"/>.
    /// </summary>
    public int ParameterCount { get; private set; }

    /// <summary>
    /// Logical residual count <c>m</c> the pool is currently configured for.
    /// May be less than the underlying allocated capacity after a shrinking <see cref="Reset(int, int)"/>.
    /// </summary>
    public int ResidualCount { get; private set; }

    // Internal accessor properties — inlined by the JIT, so the indirection is free
    // relative to direct field access from callers in this assembly (solver + helpers).

    /// <summary>Jacobian matrix <c>J</c>, dimensioned <c>m x n</c>.</summary>
    internal T[,] Jacobian => _jacobian;

    /// <summary>Normal equations matrix <c>JtJ</c>, dimensioned <c>n x n</c>.</summary>
    internal T[,] NormalEquations => _normalEquations;

    /// <summary>Gradient vector <c>Jt * r</c>, length <c>n</c>.</summary>
    internal T[] Gradient => _gradient;

    /// <summary>Step vector <c>delta</c>, length <c>n</c>.</summary>
    internal T[] Delta => _delta;

    /// <summary>Right-hand side vector for Cholesky solve, length <c>n</c>.</summary>
    internal T[] Rhs => _rhs;

    /// <summary>Forward-substitution workspace, length <c>n</c>.</summary>
    internal T[] Y => _y;

    /// <summary>Current parameter vector, length <c>n</c>.</summary>
    internal T[] Theta => _theta;

    /// <summary>Trial parameter vector, length <c>n</c>.</summary>
    internal T[] TrialTheta => _trialTheta;

    /// <summary>Effective (post-bounds-clipping) step vector, length <c>n</c>.</summary>
    internal T[] EffectiveDelta => _effectiveDelta;

    /// <summary>Perturbed-parameter scratch for finite-difference Jacobian, length <c>n</c>.</summary>
    internal T[] PerturbedParameters => _perturbedParameters;

    /// <summary>Plus-perturbation residual scratch, length <c>m</c>.</summary>
    internal T[] PlusBuffer => _plusBuffer;

    /// <summary>Minus-perturbation residual scratch, length <c>m</c>.</summary>
    internal T[] MinusBuffer => _minusBuffer;

    /// <summary>Residual vector at current parameters, length <c>m</c>.</summary>
    internal T[] Residual => _residual;

    /// <summary>Residual vector at trial parameters, length <c>m</c>.</summary>
    internal T[] TrialResidual => _trialResidual;

    /// <summary>
    /// Contiguous row-major scratch for an analytic Jacobian writer, sized to the
    /// high-water <c>m·n</c> capacity. Used only on a shrunk grow-only pool, where the
    /// Jacobian slab's physical column stride exceeds the logical parameter count and a
    /// flat span over the slab would not be contiguous. Reusing this scratch across solves
    /// keeps the shrunk-pool analytic path allocation-free after warmup.
    /// </summary>
    internal T[] JacobianFlatScratch => _jacobianFlatScratch;

    /// <summary>Exact logical-length parameter array passed to <c>Func</c>-based callbacks.</summary>
    internal T[] CallbackParameters => _callbackParameters;

    /// <summary>
    /// Reconfigures the pool for a new <c>(parameterCount, residualCount)</c>. When either
    /// dimension exceeds the current allocated capacity, the underlying arrays are resized;
    /// when both are within capacity, the work buffers are reused. Changing the
    /// parameter count allocates an exact-length callback parameter array even when
    /// both dimensions remain within capacity.
    /// </summary>
    /// <param name="parameterCount">New parameter-vector dimension <c>n</c>. Must be positive.</param>
    /// <param name="residualCount">New residual-vector dimension <c>m</c>. Must be positive.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="parameterCount"/> or <paramref name="residualCount"/> is not positive.
    /// </exception>
    /// <remarks>
    /// Grow-only policy: capacity is monotonic across the pool's lifetime. Callers fitting
    /// many models of varying sizes against the same pool retain work-buffer capacity;
    /// each parameter-count change still allocates a callback parameter array.
    /// </remarks>
    public void Reset(int parameterCount, int residualCount)
    {
        ValidateDimensions(parameterCount, residualCount);

        var callbackParameters = _callbackParameters.Length == parameterCount
            ? _callbackParameters : new T[parameterCount];
        if (parameterCount > _capacityParameterCount || residualCount > _capacityResidualCount)
        {
            Allocate(Math.Max(parameterCount, _capacityParameterCount),
                Math.Max(residualCount, _capacityResidualCount));
        }

        _callbackParameters = callbackParameters;
        ParameterCount = parameterCount;
        ResidualCount = residualCount;
    }

    private void Allocate(int parameterCount, int residualCount)
    {
        var flatLength = checked(residualCount * parameterCount);
        // Tuple assignment evaluates every allocation before publishing any buffer.
        // A failed resize therefore leaves the previous pool and capacities intact.
        (_jacobian, _normalEquations, _gradient, _delta, _rhs, _y, _theta,
            _trialTheta, _effectiveDelta, _perturbedParameters, _plusBuffer,
            _minusBuffer, _residual, _trialResidual, _jacobianFlatScratch) =
            (new T[residualCount, parameterCount], new T[parameterCount, parameterCount],
                new T[parameterCount], new T[parameterCount], new T[parameterCount],
                new T[parameterCount], new T[parameterCount], new T[parameterCount],
                new T[parameterCount], new T[parameterCount], new T[residualCount],
                new T[residualCount], new T[residualCount], new T[residualCount],
                new T[flatLength]);
        _capacityParameterCount = parameterCount;
        _capacityResidualCount = residualCount;
    }

    private static void ValidateDimensions(int parameterCount, int residualCount)
    {
        if (parameterCount <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(parameterCount),
                parameterCount,
                "Parameter count must be positive.");
        }

        if (residualCount <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(residualCount),
                residualCount,
                "Residual count must be positive.");
        }
    }
}

/// <summary>
/// Pre-allocated scratch buffers for <see cref="LevenbergMarquardtSolver"/> iterations.
/// Reusing one <see cref="LevenbergMarquardtBuffers"/> instance across many
/// <see cref="LevenbergMarquardtSolver.Solve(System.Func{double[], double[]}, double[], LevenbergMarquardtBuffers, System.Func{double[], double[,]}, double[], double[])"/>
/// calls — provided the problem dimensions match — eliminates per-iteration managed-heap
/// allocation for hot-path consumers (bootstrap loops, real-time refits). One-shot callers
/// should use the
/// <see cref="LevenbergMarquardtSolver.Solve(System.Func{double[], double[]}, double[], System.Func{double[], double[,]}, double[], double[])"/>
/// overload instead; the solver constructs and discards a private buffer set internally with
/// no external impact.
/// </summary>
/// <remarks>
/// <para>
/// Legacy facade wrapping <see cref="LevenbergMarquardtBuffers{T}"/> with
/// <c>T = <see cref="double"/></c>. All operations delegate to the underlying generic
/// instance, which is accessible via the <see cref="Inner"/> property for internal helpers.
/// </para>
/// <para>
/// Thread safety: not thread-safe. One buffer instance per thread; each thread constructs
/// its own pool and reuses it across sequential solves on that thread. Concurrent solves
/// sharing a pool produce undefined results.
/// </para>
/// </remarks>
public sealed class LevenbergMarquardtBuffers
{
    private readonly LevenbergMarquardtBuffers<double> _inner;

    /// <summary>
    /// Initializes a buffer pool sized for a specific <c>(parameterCount, residualCount)</c> problem.
    /// </summary>
    /// <param name="parameterCount">Dimension <c>n</c> of the parameter vector <c>theta</c>. Must be positive.</param>
    /// <param name="residualCount">Dimension <c>m</c> of the residual vector <c>r(theta)</c>. Must be positive.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="parameterCount"/> or <paramref name="residualCount"/> is not positive.
    /// </exception>
    public LevenbergMarquardtBuffers(int parameterCount, int residualCount)
    {
        _inner = new LevenbergMarquardtBuffers<double>(parameterCount, residualCount);
    }

    /// <summary>
    /// Logical parameter count <c>n</c> the pool is currently configured for.
    /// May be less than the underlying allocated capacity after a shrinking <see cref="Reset(int, int)"/>.
    /// </summary>
    public int ParameterCount => _inner.ParameterCount;

    /// <summary>
    /// Logical residual count <c>m</c> the pool is currently configured for.
    /// May be less than the underlying allocated capacity after a shrinking <see cref="Reset(int, int)"/>.
    /// </summary>
    public int ResidualCount => _inner.ResidualCount;

    /// <summary>
    /// Exposes the underlying generic buffer instance so internal helpers
    /// (e.g., <c>FiniteDifferenceJacobian</c>, <c>DampedLinearSolve</c>) can
    /// access the typed storage without casting.
    /// </summary>
    internal LevenbergMarquardtBuffers<double> Inner => _inner;

    // Internal accessor properties — delegate to the generic instance.

    /// <summary>Jacobian matrix <c>J</c>, dimensioned <c>m x n</c>.</summary>
    internal double[,] Jacobian => _inner.Jacobian;

    /// <summary>Normal equations matrix <c>JtJ</c>, dimensioned <c>n x n</c>.</summary>
    internal double[,] NormalEquations => _inner.NormalEquations;

    /// <summary>Gradient vector <c>Jt * r</c>, length <c>n</c>.</summary>
    internal double[] Gradient => _inner.Gradient;

    /// <summary>Step vector <c>delta</c>, length <c>n</c>.</summary>
    internal double[] Delta => _inner.Delta;

    /// <summary>Right-hand side vector for Cholesky solve, length <c>n</c>.</summary>
    internal double[] Rhs => _inner.Rhs;

    /// <summary>Forward-substitution workspace, length <c>n</c>.</summary>
    internal double[] Y => _inner.Y;

    /// <summary>Current parameter vector, length <c>n</c>.</summary>
    internal double[] Theta => _inner.Theta;

    /// <summary>Trial parameter vector, length <c>n</c>.</summary>
    internal double[] TrialTheta => _inner.TrialTheta;

    /// <summary>Effective (post-bounds-clipping) step vector, length <c>n</c>.</summary>
    internal double[] EffectiveDelta => _inner.EffectiveDelta;

    /// <summary>Perturbed-parameter scratch for finite-difference Jacobian, length <c>n</c>.</summary>
    internal double[] PerturbedParameters => _inner.PerturbedParameters;

    /// <summary>Plus-perturbation residual scratch, length <c>m</c>.</summary>
    internal double[] PlusBuffer => _inner.PlusBuffer;

    /// <summary>Minus-perturbation residual scratch, length <c>m</c>.</summary>
    internal double[] MinusBuffer => _inner.MinusBuffer;

    /// <summary>Residual vector at current parameters, length <c>m</c>.</summary>
    internal double[] Residual => _inner.Residual;

    /// <summary>Residual vector at trial parameters, length <c>m</c>.</summary>
    internal double[] TrialResidual => _inner.TrialResidual;

    /// <summary>
    /// Reconfigures the pool for a new <c>(parameterCount, residualCount)</c>. When either
    /// dimension exceeds the current allocated capacity, the underlying arrays are resized;
    /// when both are within capacity, the work buffers are reused. Changing the
    /// parameter count allocates an exact-length callback parameter array even when
    /// both dimensions remain within capacity.
    /// </summary>
    /// <param name="parameterCount">New parameter-vector dimension <c>n</c>. Must be positive.</param>
    /// <param name="residualCount">New residual-vector dimension <c>m</c>. Must be positive.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="parameterCount"/> or <paramref name="residualCount"/> is not positive.
    /// </exception>
    /// <remarks>
    /// Grow-only policy: capacity is monotonic across the pool's lifetime. Callers fitting
    /// many models of varying sizes against the same pool retain work-buffer capacity;
    /// each parameter-count change still allocates a callback parameter array.
    /// </remarks>
    public void Reset(int parameterCount, int residualCount) =>
        _inner.Reset(parameterCount, residualCount);
}
