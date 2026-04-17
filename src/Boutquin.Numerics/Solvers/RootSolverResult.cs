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
/// Generic outcome of a root-finding operation for any floating-point type <typeparamref name="T"/>.
/// </summary>
/// <remarks>
/// <para><b>Generic-math tier:</b> Data holder — works for any <c>T</c> implementing
/// <see cref="IFloatingPoint{TSelf}"/>. The tier constraint on the producing solver
/// (Tier B, <see cref="IFloatingPointIeee754{TSelf}"/>) is stricter than the result's
/// own constraint.</para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
/// <param name="Root">Computed root estimate.</param>
/// <param name="Converged">Whether a convergence criterion was met within the iteration cap.</param>
/// <param name="Iterations">Number of main-loop iterations performed.</param>
/// <param name="FinalResidual">Value of <c>f(Root)</c> in function-value units.</param>
/// <param name="EstimatedError">Upper bound on the absolute root error.</param>
public sealed record RootSolverResult<T>(
    T Root,
    bool Converged,
    int Iterations,
    T FinalResidual,
    T EstimatedError)
    where T : IFloatingPoint<T>;

/// <summary>
/// Outcome of a root-finding operation, reporting the root estimate, convergence
/// status, iteration cost, final residual in function-value units, and an
/// estimated upper bound on the root error.
/// </summary>
/// <param name="Root">
/// Computed root estimate. When <paramref name="Converged"/> is <see langword="false"/>
/// this is the solver's last iterate and may be far from the true root (divergence,
/// stalled residual, or iteration-cap exhaustion). Always a finite <see cref="double"/>
/// — solvers that fail to produce a finite iterate terminate early rather than
/// returning <c>NaN</c>.
/// </param>
/// <param name="Converged">
/// <see langword="true"/> when at least one convergence criterion (residual, bracket
/// width, or step size, depending on solver) was met within the iteration cap.
/// <see langword="false"/> does not imply <paramref name="Root"/> is unusable — for
/// example, a diverged Newton iteration may still produce a reasonable approximation
/// — but callers should treat the result as best-effort and verify via
/// <paramref name="FinalResidual"/> or an independent check.
/// </param>
/// <param name="Iterations">Number of main-loop iterations performed (excludes initial function evaluations at the bounds).</param>
/// <param name="FinalResidual">
/// Value of <c>f(<paramref name="Root"/>)</c> in function-value units (i.e., absolute,
/// not relative). A small <c>|FinalResidual|</c> near the solver's <c>functionTolerance</c>
/// is the clearest success signal regardless of <paramref name="Converged"/>.
/// </param>
/// <param name="EstimatedError">
/// Upper bound on the absolute root error <c>|Root - root*|</c>, in input units.
/// Semantics vary by solver family:
/// <list type="bullet">
///   <item>Bracketed solvers (<see cref="BisectionSolver"/>, <see cref="BrentSolver"/>,
///         bracketed <see cref="NewtonRaphsonSolver"/>) — final bracket half-width <c>(b−a)/2</c>.</item>
///   <item>Unbracketed Secant / Muller — magnitude of the last step <c>|x_n − x_{n−1}|</c>,
///         which bounds the error only under local convergence assumptions.</item>
///   <item>Unbracketed Newton — last step size, or <see cref="double.NaN"/> when the
///         derivative vanished or produced a non-finite iterate (no bound available).</item>
/// </list>
/// </param>
public sealed record RootSolverResult(
    double Root,
    bool Converged,
    int Iterations,
    double FinalResidual,
    double EstimatedError);
