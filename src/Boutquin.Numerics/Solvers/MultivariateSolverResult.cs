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
/// Generic result of a multivariate nonlinear least-squares minimization.
/// </summary>
/// <remarks>
/// <para><b>Generic-math tier:</b> Data holder — works for any <c>T</c> implementing
/// <see cref="IFloatingPoint{TSelf}"/>.</para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed record MultivariateSolverResult<T>(
    T[] Parameters,
    T FinalCost,
    T[] FinalResiduals,
    T[,]? ParameterCovariance,
    int Iterations,
    bool Converged,
    LmTerminationReason TerminationReason,
    bool BoundsActive)
    where T : IFloatingPoint<T>
{
    private T[] _parameters = (T[])Parameters.Clone();
    private T[] _finalResiduals = (T[])FinalResiduals.Clone();
    private T[,]? _parameterCovariance = (T[,]?)ParameterCovariance?.Clone();

    /// <summary>Parameter estimate, copied on input and access.</summary>
    public T[] Parameters
    {
        get => (T[])_parameters.Clone();
        init => _parameters = (T[])value.Clone();
    }

    /// <summary>Final residual vector, copied on input and access.</summary>
    public T[] FinalResiduals
    {
        get => (T[])_finalResiduals.Clone();
        init => _finalResiduals = (T[])value.Clone();
    }

    /// <summary>Parameter covariance, copied on input and access.</summary>
    public T[,]? ParameterCovariance
    {
        get => (T[,]?)_parameterCovariance?.Clone();
        init => _parameterCovariance = (T[,]?)value?.Clone();
    }
}

/// <summary>
/// Reason the Levenberg–Marquardt iteration stopped.
/// </summary>
/// <remarks>
/// <para>
/// Exactly one termination reason is reported per <see cref="MultivariateSolverResult"/>.
/// <see cref="MultivariateSolverResult.BoundsActive"/> is an orthogonal flag and may be
/// <see langword="true"/> regardless of which termination reason applies.
/// </para>
/// </remarks>
public enum LmTerminationReason
{
    /// <summary>The absolute actual and nonnegative predicted reductions, relative to the
    /// current cost, are both at most <c>functionTolerance</c>. May follow an accepted or
    /// rejected trial; a clipped step cannot certify this stop.</summary>
    FunctionToleranceReached,

    /// <summary>The parameter step <c>‖δ‖</c> fell below <c>parameterTolerance</c> relative to <c>‖θ‖</c>.</summary>
    ParameterToleranceReached,

    /// <summary>The infinity norm of the gradient <c>‖Jᵀr‖∞</c> fell below <c>gradientTolerance</c>.</summary>
    GradientToleranceReached,

    /// <summary>The iteration cap was reached before any tolerance was satisfied.</summary>
    MaxIterationsReached,

    /// <summary>The damping parameter <c>λ</c> saturated at its ceiling; no acceptable step could be found.</summary>
    DampingOverflow,
}

/// <summary>
/// Result of a multivariate nonlinear least-squares minimization.
/// </summary>
/// <param name="Parameters">
/// Final parameter estimate <c>θ</c>. When <paramref name="Converged"/> is <see langword="false"/>
/// this is the last iterate (best cost seen) and may not be near an optimum.
/// </param>
/// <param name="FinalCost">
/// Half sum of squared residuals <c>½ Σᵢ rᵢ(θ)²</c> at <paramref name="Parameters"/>.
/// Relates to the residual sum of squares (RSS) via <c>RSS = 2·FinalCost</c>.
/// </param>
/// <param name="FinalResiduals">Residual vector <c>r(θ)</c> at <paramref name="Parameters"/>. Length equals the residual count.</param>
/// <param name="ParameterCovariance">
/// Approximate covariance of the parameter estimate,
/// <c>σ̂² · (JᵀJ)⁻¹</c> with <c>σ̂² = 2·FinalCost / (m − n)</c> (m = residual count,
/// n = parameter count). <see langword="null"/> in four cases: (1) the problem is
/// exactly- or under-determined (<c>m ≤ n</c>), where the <c>m − n</c> divisor for
/// <c>σ̂²</c> is non-positive and no residual degrees of freedom remain to estimate
/// it; (2) the problem is overdetermined (<c>m &gt; n</c>) but <c>JᵀJ</c> at the
/// returned point is numerically singular; (3) the solver did not meet a
/// convergence tolerance (<paramref name="Converged"/> is
/// <see langword="false"/>); or (4) a covariance entry is not finite in the
/// working floating-point type. A tolerance stop and an available covariance do not
/// certify stationarity or justify classical unconstrained inference at an active bound.
/// </param>
/// <param name="Iterations">
/// Number of accepted outer iterations — steps at which the cost strictly decreased
/// and the parameter vector advanced. Does not count step-rejected sub-iterations
/// (where damping was increased and the linear solve re-attempted at the same
/// parameter point); matches the MINPACK lmder convention for the reported
/// iteration count.
/// </param>
/// <param name="Converged">
/// <see langword="true"/> when termination was caused by one of the three tolerance
/// tests (<see cref="LmTerminationReason.FunctionToleranceReached"/>,
/// <see cref="LmTerminationReason.ParameterToleranceReached"/>,
/// <see cref="LmTerminationReason.GradientToleranceReached"/>);
/// <see langword="false"/> for <see cref="LmTerminationReason.MaxIterationsReached"/>
/// or <see cref="LmTerminationReason.DampingOverflow"/>.
/// </param>
/// <param name="TerminationReason">Which termination rule fired first. Exactly one value.</param>
/// <param name="BoundsActive">
/// <see langword="true"/> when any component of <paramref name="Parameters"/> is on a
/// user-supplied bound at termination. Orthogonal to <paramref name="TerminationReason"/> —
/// <paramref name="Converged"/> may still be <see langword="true"/> when the free
/// (non-bound-clipped) components satisfy the relevant tolerance.
/// </param>
public sealed record MultivariateSolverResult(
    double[] Parameters,
    double FinalCost,
    double[] FinalResiduals,
    double[,]? ParameterCovariance,
    int Iterations,
    bool Converged,
    LmTerminationReason TerminationReason,
    bool BoundsActive)
{
    private double[] _parameters = (double[])Parameters.Clone();
    private double[] _finalResiduals = (double[])FinalResiduals.Clone();
    private double[,]? _parameterCovariance = (double[,]?)ParameterCovariance?.Clone();

    /// <summary>Parameter estimate, copied on input and access.</summary>
    public double[] Parameters
    {
        get => (double[])_parameters.Clone();
        init => _parameters = (double[])value.Clone();
    }

    /// <summary>Final residual vector, copied on input and access.</summary>
    public double[] FinalResiduals
    {
        get => (double[])_finalResiduals.Clone();
        init => _finalResiduals = (double[])value.Clone();
    }

    /// <summary>Parameter covariance, copied on input and access.</summary>
    public double[,]? ParameterCovariance
    {
        get => (double[,]?)_parameterCovariance?.Clone();
        init => _parameterCovariance = (double[,]?)value?.Clone();
    }
}
