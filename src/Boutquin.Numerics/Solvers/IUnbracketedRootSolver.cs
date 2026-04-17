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
/// Generic contract for unbracketed root solvers operating on <typeparamref name="T"/>.
/// </summary>
/// <remarks>
/// <para><b>Generic-math tier:</b> B (IEEE 754 transcendental). See
/// <see cref="IBracketedRootSolver{T}"/> for the tier rationale.</para>
/// </remarks>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
public interface IUnbracketedRootSolver<T>
    where T : IFloatingPointIeee754<T>
{
    /// <summary>
    /// Solves <c>f(x) = 0</c> from an initial guess.
    /// </summary>
    RootSolverResult<T> Solve(Func<T, T> objective, T initialGuess);
}

/// <summary>
/// Contract for root solvers that accept a single initial estimate and do not
/// maintain a bracket across iterations. Convergence depends on the starting
/// value and local function behavior; divergence or oscillation is possible.
/// </summary>
/// <remarks>
/// <para>
/// Preferred over <see cref="IBracketedRootSolver"/> when a bracket is unknown or
/// expensive to establish (for example, implied-volatility inversion where the
/// root is known to be positive but bounds are not). Implementations do not
/// validate the quality of the initial guess; a poor guess may cause divergence
/// with no safety net. When robustness is required, pair with an outer bracket
/// search or use <see cref="NewtonRaphsonSolver"/> via <see cref="IBracketedRootSolver"/>.
/// </para>
/// <para>
/// The convergence <em>order</em> is per-implementation:
/// </para>
/// <list type="bullet">
///   <item><see cref="SecantSolver"/> — superlinear, order φ ≈ 1.618 (no derivative required).</item>
///   <item><see cref="MullerSolver"/> — superlinear, order ≈ 1.84; handles near-complex roots via <c>|disc|</c> under the square root.</item>
///   <item><see cref="NewtonRaphsonSolver"/> — quadratic when <c>f'</c> is well-behaved; diverges near stationary points.</item>
/// </list>
/// </remarks>
public interface IUnbracketedRootSolver
{
    /// <summary>
    /// Solves <c>f(x) = 0</c> using <paramref name="initialGuess"/> as the starting
    /// point for the iteration.
    /// </summary>
    /// <param name="objective">Scalar objective function whose root is sought.</param>
    /// <param name="initialGuess">
    /// Starting value for the iteration. Proximity to the root and local smoothness
    /// of the objective both influence whether the solver converges; implementations
    /// do not validate this parameter.
    /// </param>
    /// <returns>
    /// A <see cref="RootSolverResult"/>. When <see cref="RootSolverResult.Converged"/>
    /// is <see langword="false"/> the returned root is a best-effort last iterate and
    /// may reflect divergence, a vanishing-derivative stall, or iteration-cap exhaustion;
    /// consumers should inspect <see cref="RootSolverResult.FinalResidual"/> before use.
    /// </returns>
    RootSolverResult Solve(Func<double, double> objective, double initialGuess);
}
