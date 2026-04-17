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
/// When <c>JtJ + lambda*diag(JtJ)</c> is not positive definite (which can only happen
/// if <c>J</c> is rank deficient AND <c>lambda</c> is still small), <see cref="TrySolve"/>
/// returns <see langword="false"/>; the caller is expected to increase <c>lambda</c> and retry.
/// </para>
/// </remarks>
internal static class DampedLinearSolve<T>
    where T : IFloatingPointIeee754<T>
{
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
    /// <returns><see langword="true"/> iff the damped matrix factored successfully and <c>delta</c> was computed.</returns>
    public static bool TrySolve(
        T[] residual,
        T lambda,
        LevenbergMarquardtBuffers<T> buffers)
    {
        var parameterCount = buffers.ParameterCount;
        var residualCount = buffers.ResidualCount;
        var jacobian = buffers.Jacobian;
        var a = buffers.NormalEquations;
        var g = buffers.Gradient;
        var rhs = buffers.Rhs;
        var y = buffers.Y;
        var delta = buffers.Delta;

        // Form A = JtJ and g = Jt*r. Every [i, j] entry of `a` and every g[i] is
        // written below, so the pool's prior-iteration contents are discarded.
        for (var i = 0; i < parameterCount; i++)
        {
            var gi = T.Zero;
            for (var k = 0; k < residualCount; k++)
            {
                gi += jacobian[k, i] * residual[k];
            }

            g[i] = gi;

            for (var j = i; j < parameterCount; j++)
            {
                var aij = T.Zero;
                for (var k = 0; k < residualCount; k++)
                {
                    aij += jacobian[k, i] * jacobian[k, j];
                }

                a[i, j] = aij;
                a[j, i] = aij;
            }
        }

        // Apply Marquardt damping: A <- A + lambda * diag(A). Diagonal must stay
        // strictly positive; fall back to lambda*I where diag(A) ~ 0 so the matrix
        // remains PD for small-magnitude columns.
        for (var i = 0; i < parameterCount; i++)
        {
            var diag = a[i, i];
            var addend = diag > T.Zero ? lambda * diag : lambda;
            a[i, i] = diag + addend;
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
            if (diag <= T.Zero || !T.IsFinite(diag))
            {
                return false;
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

        // Right-hand side: we want delta = -(A+...)^-1 g. Solve A*delta = -g via
        // forward/back sub on L*Lt*delta = -g.
        for (var i = 0; i < parameterCount; i++)
        {
            rhs[i] = -g[i];
        }

        // Forward substitution: L*y = rhs. `y[i]` depends only on y[0..i-1] and rhs[i],
        // all of which are already written in this call.
        for (var i = 0; i < parameterCount; i++)
        {
            var sum = rhs[i];
            for (var k = 0; k < i; k++)
            {
                sum -= a[i, k] * y[k];
            }

            y[i] = sum / a[i, i];
        }

        // Back substitution: Lt*delta = y. `delta[i]` depends only on delta[i+1..n-1]
        // and y[i], all of which are written before use as `i` runs from n-1 downto 0.
        for (var i = parameterCount - 1; i >= 0; i--)
        {
            var sum = y[i];
            for (var k = i + 1; k < parameterCount; k++)
            {
                sum -= a[k, i] * delta[k];
            }

            delta[i] = sum / a[i, i];
        }

        return true;
    }

    /// <summary>
    /// Attempts to compute <c>(JtJ)^-1</c> by Cholesky at <c>lambda = 0</c>. Returns
    /// <see langword="null"/> when <c>JtJ</c> is not strictly positive definite.
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
    /// </remarks>
    public static T[,]? TryInvertNormalEquations(
        T[,] jacobian,
        int residualCount,
        int parameterCount)
    {
        var m = residualCount;
        var n = parameterCount;

        var a = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var aij = T.Zero;
                for (var k = 0; k < m; k++)
                {
                    aij += jacobian[k, i] * jacobian[k, j];
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

        // Invert L (lower-triangular inverse).
        var lInv = new T[n, n];
        for (var j = 0; j < n; j++)
        {
            lInv[j, j] = T.One / a[j, j];
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
        DampedLinearSolve<double>.TryInvertNormalEquations(jacobian, residualCount, parameterCount);
}
