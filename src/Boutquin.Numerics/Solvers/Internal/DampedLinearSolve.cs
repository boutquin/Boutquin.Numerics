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

namespace Boutquin.Numerics.Solvers.Internal;

/// <summary>
/// Solves the Levenberg-Marquardt normal-equation step
/// <c>(JtJ + lambda * diag(JtJ)) * delta = -Jt*r</c>
/// using an in-place Cholesky factorization of the damped matrix,
/// generic over any <see cref="IFloatingPointIeee754{TSelf}"/> type.
/// </summary>
/// <typeparam name="T">
/// Floating-point element type. Must satisfy <see cref="IFloatingPointIeee754{TSelf}"/>
/// (Tier B generic-math constraint).
/// </typeparam>
/// <remarks>
/// <para>
/// The scaled-diagonal damping form (Marquardt 1963) is preferred over the
/// identity-damped form <c>(JtJ + lambda*I)</c> because it preserves the invariance
/// of the solver under rescaling of individual parameters: multiplying column
/// <c>j</c> of <c>J</c> by a constant <c>s</c> leaves the step <c>deltaj / s</c>
/// unchanged. This matters when parameters have very different magnitudes
/// (e.g., a rate fit where one parameter is <c>O(1e-3)</c> and another is <c>O(1)</c>).
/// </para>
/// <para>
/// All operations use generic <typeparamref name="T"/> arithmetic.
/// <see cref="TrySolve"/> borrows all working storage (<c>JtJ</c> accumulator,
/// gradient, step, substitution workspaces) from a <see cref="LevenbergMarquardtBuffers{T}"/>
/// pool so the inner LM iteration allocates nothing.
/// <see cref="TryInvertNormalEquations"/> is a one-shot helper used only at
/// convergence to build the parameter-covariance matrix and keeps its fresh allocations.
/// </para>
/// <para>
/// <see cref="TrySolve"/> forms the normal equations after power-of-two range scaling of
/// each Jacobian column and of the residual (<c>J = U·D</c>, <c>r = s·ρ</c>) and solves for
/// <c>D·δ/s</c>, unscaling each step component once at the end. The scales are powers of
/// two, so results are identical to the unscaled arithmetic whenever that arithmetic stays in
/// normal range; otherwise a representable <c>JtJ</c>, <c>Jt*r</c> or step is not lost to an
/// overflowing or flushed square. A step that is not representable is reported as a failure.
/// </para>
/// <para>
/// Scaled damping can restore positive definiteness when the normal-matrix diagonal
/// is positive. An exactly zero Jacobian column stays zero at every damping value;
/// a compatible structural null coordinate is supported with a zero step.
/// Other unusable pivots or non-finite arithmetic cause <see cref="TrySolve"/> to
/// return <see langword="false"/>. Increasing damping may help, but is not guaranteed to do so.
/// </para>
/// </remarks>
internal static class DampedLinearSolve<T>
    where T : IFloatingPointIeee754<T>
{
    // Largest finite binary exponent of T (1023 for double); 2^-e is finite for e >= -this.
    private static readonly int s_maxFiniteExponent = T.ILogB(T.BitDecrement(T.PositiveInfinity));

    /// <summary>
    /// Binary exponent <c>e</c> used to range-scale a block whose largest magnitude is
    /// <paramref name="maxAbs"/>: zero for an all-zero block, otherwise <c>ILogB(maxAbs)</c>
    /// raised where necessary so that the factor <c>2^-e</c> is itself finite. Scaled
    /// entries then have magnitude below 2, and applying the factor is a single exact (or,
    /// for a subnormal result, correctly rounded) multiplication.
    /// </summary>
    internal static int ScaleExponent(T maxAbs) =>
        T.IsZero(maxAbs) ? 0 : int.Max(T.ILogB(maxAbs), -s_maxFiniteExponent);

    /// <summary>Range-scaling exponent of the first <paramref name="length"/> entries of a vector.</summary>
    internal static int VectorScaleExponent(T[] vector, int length)
    {
        var maxAbs = T.Zero;
        for (var i = 0; i < length; i++)
        {
            maxAbs = T.Max(maxAbs, T.Abs(vector[i]));
        }

        return ScaleExponent(maxAbs);
    }

    /// <summary>Range-scaling exponent of the first <paramref name="rowCount"/> entries of a column.</summary>
    internal static int ColumnScaleExponent(T[,] matrix, int rowCount, int column)
    {
        var maxAbs = T.Zero;
        for (var i = 0; i < rowCount; i++)
        {
            maxAbs = T.Max(maxAbs, T.Abs(matrix[i, column]));
        }

        return ScaleExponent(maxAbs);
    }

    /// <summary>
    /// Attempts to solve <c>(JtJ + lambda*diag(JtJ)) * delta = -Jt*r</c> for the parameter step <c>delta</c>,
    /// writing into <c>buffers.Delta</c>.
    /// </summary>
    /// <param name="residual">Residual vector <c>r</c>, length <c>buffers.ResidualCount</c>.</param>
    /// <param name="lambda">Damping factor <c>lambda >= 0</c>.</param>
    /// <param name="buffers">
    /// Pool providing <c>Jacobian</c> (read), <c>NormalEquations</c>, <c>Gradient</c>,
    /// <c>Rhs</c>, <c>Y</c>, <c>Delta</c> (all written). All entries of each working
    /// array are overwritten on every call, so no prior-iteration state leaks into the
    /// current solve.
    /// </param>
    /// <param name="gradientAlreadyComputed">
    /// When <see langword="true"/>, the caller has already formed the current-point gradient
    /// <c>Jt*r</c> into <c>buffers.Gradient</c> (via the accepted-step / initial gradient
    /// ∞-norm check, which runs at the same J and r this solve reads), so this call leaves it
    /// in place instead of recomputing the identical vector. When <see langword="false"/> (the
    /// default, for callers that did not pre-fill the gradient) the gradient is recomputed
    /// here. The step itself is solved from the range-scaled right-hand side formed in this
    /// call, so the flag never changes the computed step.
    /// </param>
    /// <param name="lowerBounds">Optional lower bounds at <c>buffers.Theta</c>.</param>
    /// <param name="upperBounds">Optional upper bounds at <c>buffers.Theta</c>.</param>
    /// <returns><see langword="true"/> when a finite step was computed, including compatible
    /// structural null coordinates; otherwise <see langword="false"/>.</returns>
    public static bool TrySolve(
        T[] residual,
        T lambda,
        LevenbergMarquardtBuffers<T> buffers,
        bool gradientAlreadyComputed = false,
        ReadOnlySpan<T> lowerBounds = default,
        ReadOnlySpan<T> upperBounds = default)
    {
        var parameterCount = buffers.ParameterCount;
        var residualCount = buffers.ResidualCount;
        var jacobian = buffers.Jacobian;
        var a = buffers.NormalEquations;
        var g = buffers.Gradient;
        var rhs = buffers.Rhs;
        var y = buffers.Y;
        var delta = buffers.Delta;

        // Range scaling. Write J = U·D and r = s·ρ with power-of-two D = diag(2^eⱼ) and
        // s = 2^eᵣ taken from the column and residual maxima, so |U|, |ρ| < 2. With w = D·δ/s
        // the damped system becomes (UᵀU + lambda*diag(UᵀU))*w = -Uᵀρ, and δⱼ = 2^(eᵣ-eⱼ)·wⱼ.
        // Every scale is a power of two, so this reproduces the unscaled arithmetic exactly
        // whenever that arithmetic stays in normal range; when it does not, a representable
        // JtJ, Jt*r or step is no longer lost to an overflowing or flushed square.
        // `delta` carries the column factors 2^-eⱼ until the step is unscaled at the end.
        var residualExponent = VectorScaleExponent(residual, residualCount);
        var residualFactor = T.ScaleB(T.One, -residualExponent);
        for (var j = 0; j < parameterCount; j++)
        {
            delta[j] = T.ScaleB(T.One, -ColumnScaleExponent(jacobian, residualCount, j));
        }

        // Form UᵀU, the scaled right-hand side -Uᵀρ and (unless the caller pre-filled it)
        // g = Jt*r. Every [i, j] entry of `a` is written below, so the pool's prior-iteration
        // contents are discarded; g is recomputed here or kept from the caller's
        // gradient-∞-norm check, and the solve itself reads only the scaled right-hand side.
        for (var i = 0; i < parameterCount; i++)
        {
            if (!gradientAlreadyComputed)
            {
                var gi = T.Zero;
                for (var k = 0; k < residualCount; k++)
                {
                    gi += jacobian[k, i] * residual[k];
                }

                g[i] = gi;
            }

            var ci = T.Zero;
            for (var k = 0; k < residualCount; k++)
            {
                ci += (jacobian[k, i] * delta[i]) * (residual[k] * residualFactor);
            }

            rhs[i] = -ci;

            for (var j = i; j < parameterCount; j++)
            {
                var aij = T.Zero;
                for (var k = 0; k < residualCount; k++)
                {
                    aij += (jacobian[k, i] * delta[i]) * (jacobian[k, j] * delta[j]);
                }

                a[i, j] = aij;
                a[j, i] = aij;
            }
        }

        // Solve on the free coordinates. The scaled right-hand side has the sign of
        // -Jᵀr even when the unscaled gradient underflows. A bound with an outward
        // descent direction, or a fixed coordinate, contributes a zero step; deleting
        // its row and column prevents its coupling from distorting the free step.
        for (var i = 0; i < parameterCount; i++)
        {
            var fixedCoordinate = !lowerBounds.IsEmpty && !upperBounds.IsEmpty && lowerBounds[i] == upperBounds[i];
            var blockedAtLower = !lowerBounds.IsEmpty && buffers.Theta[i] <= lowerBounds[i] && rhs[i] < T.Zero;
            var blockedAtUpper = !upperBounds.IsEmpty && buffers.Theta[i] >= upperBounds[i] && rhs[i] > T.Zero;
            if (fixedCoordinate || blockedAtLower || blockedAtUpper)
            {
                rhs[i] = T.Zero;
                for (var j = 0; j < parameterCount; j++)
                {
                    a[i, j] = T.Zero;
                    a[j, i] = T.Zero;
                }
            }
        }

        // Apply Marquardt damping exactly: A <- A + lambda * diag(A). A structural
        // zero column remains zero; the factorization below treats that coordinate as
        // an unconstrained null-space direction and returns the minimum-norm delta 0.
        for (var i = 0; i < parameterCount; i++)
        {
            var diag = a[i, i];
            a[i, i] = diag + (lambda * diag);
        }

        // Cholesky factorization of the damped matrix (in place on the pool's
        // NormalEquations slab). The lower-triangular factor overwrites `a` below
        // the diagonal; entries above the diagonal are not read after this point.
        // If factorization fails, `delta` may hold a partial write — the caller
        // treats the call as a rejection and ignores `delta` on false returns.
        for (var j = 0; j < parameterCount; j++)
        {
            var sum = T.Zero;
            for (var k = 0; k < j; k++)
            {
                sum += a[j, k] * a[j, k];
            }

            var diag = a[j, j] - sum;
            if (!T.IsFinite(diag) || diag < T.Zero)
            {
                return false;
            }

            if (T.IsZero(diag))
            {
                if (!T.IsZero(rhs[j]))
                {
                    return false;
                }

                a[j, j] = T.Zero;
                for (var i = j + 1; i < parameterCount; i++)
                {
                    if (!T.IsZero(a[i, j]))
                    {
                        return false;
                    }

                    a[i, j] = T.Zero;
                }

                continue;
            }

            a[j, j] = T.Sqrt(diag);

            for (var i = j + 1; i < parameterCount; i++)
            {
                sum = T.Zero;
                for (var k = 0; k < j; k++)
                {
                    sum += a[i, k] * a[j, k];
                }

                a[i, j] = (a[i, j] - sum) / a[j, j];
            }
        }

        // Solve L*Lt*w = -Uᵀρ by forward/back substitution.
        // Forward substitution: L*y = rhs. `y[i]` depends only on y[0..i-1] and rhs[i],
        // all of which are already written in this call.
        for (var i = 0; i < parameterCount; i++)
        {
            var sum = rhs[i];
            for (var k = 0; k < i; k++)
            {
                sum -= a[i, k] * y[k];
            }

            if (T.IsZero(a[i, i]))
            {
                if (!T.IsZero(sum))
                {
                    return false;
                }

                y[i] = T.Zero;
                continue;
            }

            y[i] = sum / a[i, i];
        }

        // Back substitution in place: Lt*w = y. `w[i]` overwrites y[i] and depends only on
        // w[i+1..n-1] and y[i], all of which are written before use as `i` runs downward.
        for (var i = parameterCount - 1; i >= 0; i--)
        {
            var sum = y[i];
            for (var k = i + 1; k < parameterCount; k++)
            {
                sum -= a[k, i] * y[k];
            }

            if (T.IsZero(a[i, i]))
            {
                if (!T.IsZero(sum))
                {
                    return false;
                }

                y[i] = T.Zero;
                continue;
            }

            y[i] = sum / a[i, i];
        }

        // Undo the change of variables, δⱼ = 2^(eᵣ-eⱼ)·wⱼ, in one scaling per component. A step
        // that is not representable is not a solution: report failure so the caller takes
        // its rejected-step damping path instead of evaluating a non-finite trial.
        for (var i = 0; i < parameterCount; i++)
        {
            delta[i] = T.ScaleB(y[i], residualExponent + T.ILogB(delta[i]));
            if (!T.IsFinite(delta[i]))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Attempts to compute <c>scale² · (JtJ)^-1</c> by Cholesky at <c>lambda = 0</c>. Returns
    /// <see langword="null"/> when <c>JtJ</c> is not strictly positive definite
    /// or a scaled inverse entry is not finite.
    /// </summary>
    /// <remarks>
    /// Used to build the parameter-covariance matrix reported on converged results.
    /// Allocates fresh <typeparamref name="T"/><c>[,]</c> storage on every call by
    /// design — this is a one-shot terminal step per successful solve and does not
    /// affect the steady-state inner-loop allocation contract. The logical problem
    /// dimensions are taken from the caller (via <paramref name="residualCount"/> and
    /// <paramref name="parameterCount"/>) so the helper works correctly when passed
    /// a pool-backed Jacobian slab whose <see cref="Array.GetLength(int)"/> may exceed
    /// the logical size after a shrinking
    /// <see cref="LevenbergMarquardtBuffers{T}.Reset(int, int)"/>.
    /// The square-root variance scale is applied during triangular inversion,
    /// before inverse-factor products can overflow. A zero scale preserves the rank check
    /// and then yields a zero covariance for an exact fit.
    /// </remarks>
    public static T[,]? TryInvertNormalEquations(
        T[,] jacobian,
        int residualCount,
        int parameterCount,
        T scale)
    {
        var m = residualCount;
        var n = parameterCount;

        // J = U D. Form UᵀU so the column magnitudes never get squared
        // before their scale can cancel against the covariance scale.
        var columnScales = new T[n];
        for (var j = 0; j < n; j++)
        {
            for (var k = 0; k < m; k++)
            {
                columnScales[j] = T.Max(columnScales[j], T.Abs(jacobian[k, j]));
            }
            if (columnScales[j] == T.Zero || !T.IsFinite(columnScales[j]))
            {
                return null;
            }
        }

        var a = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var aij = T.Zero;
                for (var k = 0; k < m; k++)
                {
                    aij += (jacobian[k, i] / columnScales[i]) * (jacobian[k, j] / columnScales[j]);
                }

                a[i, j] = aij;
                a[j, i] = aij;
            }
        }

        // Cholesky into L stored below diagonal.
        for (var j = 0; j < n; j++)
        {
            var sum = T.Zero;
            for (var k = 0; k < j; k++)
            {
                sum += a[j, k] * a[j, k];
            }

            var diag = a[j, j] - sum;
            if (diag <= T.Zero || !T.IsFinite(diag))
            {
                return null;
            }

            a[j, j] = T.Sqrt(diag);

            for (var i = j + 1; i < n; i++)
            {
                sum = T.Zero;
                for (var k = 0; k < j; k++)
                {
                    sum += a[i, k] * a[j, k];
                }

                a[i, j] = (a[i, j] - sum) / a[j, j];
            }
        }

        // Form scale * L^-1 D^-1 without first materializing either inverse.
        var lInv = new T[n, n];
        for (var j = 0; j < n; j++)
        {
            lInv[j, j] = (scale / columnScales[j]) / a[j, j];
            for (var i = j + 1; i < n; i++)
            {
                var sum = T.Zero;
                for (var k = j; k < i; k++)
                {
                    sum -= a[i, k] * lInv[k, j];
                }

                lInv[i, j] = sum / a[i, i];
            }
        }

        // A^-1 = Lt^-1 * L^-1.
        var inverse = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                var sum = T.Zero;
                for (var k = i; k < n; k++)
                {
                    sum += lInv[k, i] * lInv[k, j];
                }

                if (!T.IsFinite(sum))
                {
                    return null;
                }

                inverse[i, j] = sum;
                inverse[j, i] = sum;
            }
        }

        return inverse;
    }
}

/// <summary>
/// Solves the Levenberg-Marquardt normal-equation step
/// <c>(JtJ + lambda * diag(JtJ)) * delta = -Jt*r</c>
/// using an in-place Cholesky factorization of the damped matrix.
/// </summary>
/// <remarks>
/// Legacy facade delegating to <see cref="DampedLinearSolve{T}"/> with
/// <c>T = <see cref="double"/></c>. Accepts the non-generic
/// <see cref="LevenbergMarquardtBuffers"/> and accesses its <see cref="LevenbergMarquardtBuffers.Inner"/>
/// property to obtain the underlying <see cref="LevenbergMarquardtBuffers{T}"/> instance.
/// </remarks>
internal static class DampedLinearSolve
{
    /// <summary>
    /// Attempts to solve <c>(JtJ + lambda*diag(JtJ)) * delta = -Jt*r</c> for the parameter step <c>delta</c>,
    /// writing into <c>buffers.Delta</c>.
    /// </summary>
    /// <param name="residual">Residual vector <c>r</c>, length <c>buffers.ResidualCount</c>.</param>
    /// <param name="lambda">Damping factor <c>lambda >= 0</c>.</param>
    /// <param name="buffers">
    /// Pool providing <c>Jacobian</c> (read), <c>NormalEquations</c>, <c>Gradient</c>,
    /// <c>Rhs</c>, <c>Y</c>, <c>Delta</c> (all written).
    /// </param>
    /// <returns><see langword="true"/> iff the damped matrix factored successfully and <c>delta</c> was computed.</returns>
    public static bool TrySolve(
        double[] residual,
        double lambda,
        LevenbergMarquardtBuffers buffers) =>
        DampedLinearSolve<double>.TrySolve(residual, lambda, buffers.Inner);

    /// <summary>
    /// Attempts to compute <c>(JtJ)^-1</c> by Cholesky at <c>lambda = 0</c>. Returns
    /// <see langword="null"/> when <c>JtJ</c> is not strictly positive definite.
    /// </summary>
    /// <remarks>
    /// Used to build the parameter-covariance matrix reported on converged results.
    /// Allocates fresh <c>double[,]</c> storage on every call by design — this is a
    /// one-shot terminal step per successful solve.
    /// </remarks>
    public static double[,]? TryInvertNormalEquations(
        double[,] jacobian,
        int residualCount,
        int parameterCount) =>
        DampedLinearSolve<double>.TryInvertNormalEquations(jacobian, residualCount, parameterCount, 1.0);
}
