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
/// Generic contract for multivariate nonlinear least-squares solvers.
/// </summary>
/// <remarks>
/// <para><b>Generic-math tier:</b> B (IEEE 754 transcendental). The user's residual
/// function typically involves transcendentals; the constraint ensures the caller
/// can construct such functions.</para>
/// </remarks>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
public interface IMultivariateLeastSquaresSolver<T>
    where T : IFloatingPointIeee754<T>
{
    /// <summary>
    /// Minimizes <c>½ Σᵢ rᵢ(θ)²</c> from an initial parameter estimate.
    /// </summary>
    MultivariateSolverResult<T> Solve(
        Func<T[], T[]> residuals,
        T[] initialGuess,
        Func<T[], T[,]>? jacobian = null,
        T[]? lowerBounds = null,
        T[]? upperBounds = null);
}

/// <summary>
/// Contract for multivariate nonlinear least-squares solvers. Given a vector-valued
/// residual function <c>r(θ) ∈ ℝ^m</c> and an initial parameter estimate
/// <c>θ₀ ∈ ℝ^n</c>, find <c>θ*</c> minimizing <c>½ Σᵢ rᵢ(θ)²</c>.
/// </summary>
/// <remarks>
/// <para>
/// The half-SSE convention is used throughout so that the gradient of the objective
/// is <c>∇f = Jᵀr</c> (not <c>2Jᵀr</c>) and the Gauss–Newton Hessian approximation
/// is <c>H ≈ JᵀJ</c>. This matches the Levenberg–Marquardt and Gauss–Newton literature.
/// </para>
/// <para>
/// Implementations must be deterministic — identical inputs (residuals, Jacobian,
/// initial guess, bounds, tolerances) yield bit-identical outputs. No RNG dependence.
/// </para>
/// </remarks>
public interface IMultivariateLeastSquaresSolver
{
    /// <summary>
    /// Minimizes <c>½ Σᵢ rᵢ(θ)²</c> from an initial parameter estimate.
    /// </summary>
    /// <param name="residuals">
    /// Residual function. Given a parameter vector <c>θ</c> of length <c>n</c>, returns
    /// the residual vector <c>r(θ)</c> of length <c>m</c>. The same <c>m</c> must be
    /// returned on every call. Throwing from inside this function is a user error and
    /// will propagate out of <see cref="Solve"/>.
    /// </param>
    /// <param name="initialGuess">
    /// Starting parameter vector <c>θ₀</c>. Length determines the problem dimension <c>n</c>.
    /// Must not be <see langword="null"/> and must contain only finite values.
    /// </param>
    /// <param name="jacobian">
    /// Optional analytic Jacobian. Given <c>θ</c>, returns <c>J[i, j] = ∂rᵢ/∂θⱼ</c>
    /// with shape <c>m × n</c>. When <see langword="null"/>, the solver computes <c>J</c>
    /// via central finite differences.
    /// </param>
    /// <param name="lowerBounds">
    /// Optional element-wise lower bounds on <c>θ</c>. When supplied, length must equal
    /// <paramref name="initialGuess"/>. Use <see cref="double.NegativeInfinity"/> for
    /// components that are effectively unbounded below.
    /// </param>
    /// <param name="upperBounds">
    /// Optional element-wise upper bounds on <c>θ</c>. When supplied, length must equal
    /// <paramref name="initialGuess"/>. Use <see cref="double.PositiveInfinity"/> for
    /// components that are effectively unbounded above.
    /// </param>
    /// <returns>Final parameter estimate, residuals, cost, and convergence diagnostics.</returns>
    MultivariateSolverResult Solve(
        Func<double[], double[]> residuals,
        double[] initialGuess,
        Func<double[], double[,]>? jacobian = null,
        double[]? lowerBounds = null,
        double[]? upperBounds = null);
}
