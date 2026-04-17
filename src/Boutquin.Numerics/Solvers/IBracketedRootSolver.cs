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
/// Generic contract for bracketed root solvers operating on <typeparamref name="T"/>.
/// </summary>
/// <remarks>
/// <para><b>Generic-math tier:</b> B (IEEE 754 transcendental). The constraint
/// <see cref="IFloatingPointIeee754{TSelf}"/> ensures the caller can construct
/// objective functions using transcendentals (<c>T.Exp</c>, <c>T.Log</c>, etc.),
/// which is the typical use case for root finding. <c>decimal</c> callers use the
/// legacy <see cref="IBracketedRootSolver"/> facade.</para>
/// </remarks>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
public interface IBracketedRootSolver<T>
    where T : IFloatingPointIeee754<T>
{
    /// <summary>
    /// Solves <c>f(x) = 0</c> for <c>x ∈ [lowerBound, upperBound]</c>.
    /// </summary>
    RootSolverResult<T> Solve(Func<T, T> objective, T lowerBound, T upperBound);
}

/// <summary>
/// Contract for root solvers that require a bracketing interval
/// <c>[a, b]</c> satisfying <c>f(a) · f(b) &lt; 0</c> (strict sign change).
/// </summary>
/// <remarks>
/// <para>
/// By Bolzano's intermediate value theorem a continuous function with a sign change
/// on <c>[a, b]</c> has at least one root in the interval. All implementations
/// preserve this bracket invariant across iterations and therefore cannot diverge.
/// </para>
/// <para>
/// The universal guarantee at the interface level is <em>eventual convergence for
/// continuous sign-changing functions</em>. The convergence <em>order</em> and the
/// worst-case iteration bound are per-implementation contracts:
/// </para>
/// <list type="bullet">
///   <item><see cref="BisectionSolver"/> — linear convergence, one bit per iteration, bounded <c>⌈log₂((b−a)/tol)⌉</c>.</item>
///   <item><see cref="BrentSolver"/> — superlinear in smooth regions, <c>⌈log₂((b−a)/tol)⌉</c> worst case via halving-interval guarantee.</item>
///   <item><see cref="NewtonRaphsonSolver"/> — quadratic when the derivative is well-behaved, falls back to bisection otherwise (also implements <see cref="IUnbracketedRootSolver"/>).</item>
/// </list>
/// </remarks>
public interface IBracketedRootSolver
{
    /// <summary>
    /// Solves <c>f(x) = 0</c> for <c>x ∈ [<paramref name="lowerBound"/>, <paramref name="upperBound"/>]</c>
    /// given a bracket with opposite sign function values at the endpoints.
    /// </summary>
    /// <param name="objective">
    /// Scalar objective function whose root is sought. Must be continuous on the closed
    /// interval and satisfy <c>f(lowerBound) · f(upperBound) &lt; 0</c>.
    /// </param>
    /// <param name="lowerBound">Left endpoint of the bracketing interval.</param>
    /// <param name="upperBound">Right endpoint of the bracketing interval.</param>
    /// <returns>
    /// A <see cref="RootSolverResult"/> carrying the root estimate, convergence flag,
    /// iteration count, final residual <c>|f(root)|</c>, and an estimated error bound
    /// (final bracket half-width for bracket-based solvers).
    /// </returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown by implementations when <c>f(lowerBound)</c> and <c>f(upperBound)</c>
    /// have the same sign, indicating no sign-changing bracket.
    /// </exception>
    RootSolverResult Solve(Func<double, double> objective, double lowerBound, double upperBound);
}
