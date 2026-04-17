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
/// Central finite-difference Jacobian evaluator for vector-valued residuals,
/// generic over any <see cref="IFloatingPointIeee754{TSelf}"/> type.
/// Writes results into caller-owned storage so the helper itself allocates nothing.
/// </summary>
/// <typeparam name="T">
/// Floating-point element type. Must satisfy <see cref="IFloatingPointIeee754{TSelf}"/>
/// (Tier B generic-math constraint).
/// </typeparam>
/// <remarks>
/// <para>
/// Computes <c>J[i, j] = dri/dthetaj ~ (ri(theta + h*ej) - ri(theta - h*ej)) / (2*h)</c> using
/// an adaptive per-component step <c>h = max(1e-8, |thetaj|*sqrt(epsilon))</c> matching the
/// convention in <see cref="NewtonRaphsonSolver"/>. The <c>sqrt(epsilon)</c> scale balances
/// truncation error (<c>O(h^2)</c> for central differences) against roundoff,
/// yielding approximately eight correct digits for well-scaled smooth residuals.
/// </para>
/// <para>
/// The helper borrows scratch storage from the supplied
/// <see cref="LevenbergMarquardtBuffers{T}"/> pool — parameter-perturbation workspace
/// and plus/minus residual copies — so repeated calls within the LM iteration loop add no
/// managed-heap allocation. The caller's <c>residuals</c> callback still produces
/// allocations on each evaluation unless it is written to return a shared buffer;
/// that is outside this helper's control by design.
/// </para>
/// </remarks>
internal static class FiniteDifferenceJacobian<T>
    where T : IFloatingPointIeee754<T>
{
    // √(machine_epsilon) for the working type. Note: .NET's T.Epsilon is the smallest
    // positive value (subnormal for IEEE types), NOT machine epsilon in the numerical-
    // analysis sense. Machine epsilon = 2^-52 for double, 2^-23 for float.
    // We use per-type dispatch to match the original hardcoded SqrtEpsilon = 1.49e-8
    // for double and the correct value for other types.
    private static readonly T s_sqrtEpsilon = ComputeSqrtMachineEpsilon();

    private static T ComputeSqrtMachineEpsilon()
    {
        if (typeof(T) == typeof(double))
        {
            return (T)(object)1.4901161193847656e-8; // sqrt(2^-52)
        }

        if (typeof(T) == typeof(float))
        {
            return (T)(object)3.4526698e-4f; // sqrt(2^-23)
        }

        if (typeof(T) == typeof(Half))
        {
            return (T)(object)(Half)0.03125f; // sqrt(2^-10) ≈ 0.03125
        }

        // Fallback: compute numerically for unknown types.
        var one = T.One;
        var eps = one;
        var half = T.CreateChecked(0.5);
        while (one + eps * half > one)
        {
            eps *= half;
        }

        return T.Sqrt(eps);
    }

    // Absolute floor to prevent h = 0 at the origin.
    private static readonly T s_floor = T.CreateChecked(1e-8);

    // Constant 2 for the central-difference denominator.
    private static readonly T s_two = T.CreateChecked(2);

    /// <summary>
    /// Evaluates the Jacobian of <paramref name="residuals"/> at <paramref name="parameters"/>
    /// and writes the result into <c>buffers.Jacobian</c>.
    /// </summary>
    /// <param name="residuals">Residual function — must return a vector of length <c>buffers.ResidualCount</c>.</param>
    /// <param name="parameters">Parameter vector <c>theta</c> of length <c>buffers.ParameterCount</c>.</param>
    /// <param name="buffers">
    /// Pool providing <c>Jacobian</c> (m x n output), <c>PerturbedParameters</c> (n scratch),
    /// <c>PlusBuffer</c> / <c>MinusBuffer</c> (m scratch each). All entries of the Jacobian
    /// slab are overwritten on return; no prior contents are read.
    /// </param>
    public static void EvaluateInto(
        Func<T[], T[]> residuals,
        T[] parameters,
        LevenbergMarquardtBuffers<T> buffers)
    {
        var parameterCount = buffers.ParameterCount;
        var residualCount = buffers.ResidualCount;
        var perturbedParameters = buffers.PerturbedParameters;
        var plusBuffer = buffers.PlusBuffer;
        var minusBuffer = buffers.MinusBuffer;
        var jacobian = buffers.Jacobian;

        for (var j = 0; j < parameterCount; j++)
        {
            // Copy parameters into scratch, perturb +h.
            Array.Copy(parameters, perturbedParameters, parameterCount);
            var h = StepSize(parameters[j]);

            perturbedParameters[j] = parameters[j] + h;
            var plus = residuals(perturbedParameters);
            Array.Copy(plus, plusBuffer, residualCount);

            perturbedParameters[j] = parameters[j] - h;
            var minus = residuals(perturbedParameters);
            Array.Copy(minus, minusBuffer, residualCount);

            var scale = T.One / (s_two * h);
            for (var i = 0; i < residualCount; i++)
            {
                jacobian[i, j] = (plusBuffer[i] - minusBuffer[i]) * scale;
            }
        }
    }

    /// <summary>
    /// Computes the adaptive per-component step size for central finite differences.
    /// </summary>
    /// <param name="component">Current value of the parameter component.</param>
    /// <returns>Step size <c>h = max(floor, |component| * sqrt(epsilon))</c>.</returns>
    private static T StepSize(T component) =>
        T.Max(s_floor, T.Abs(component) * s_sqrtEpsilon);
}

/// <summary>
/// Central finite-difference Jacobian evaluator for vector-valued residuals.
/// Writes results into caller-owned storage so the helper itself allocates nothing.
/// </summary>
/// <remarks>
/// Legacy facade delegating to <see cref="FiniteDifferenceJacobian{T}"/> with
/// <c>T = <see cref="double"/></c>. Accepts the non-generic
/// <see cref="LevenbergMarquardtBuffers"/> and accesses its <see cref="LevenbergMarquardtBuffers.Inner"/>
/// property to obtain the underlying <see cref="LevenbergMarquardtBuffers{T}"/> instance.
/// </remarks>
internal static class FiniteDifferenceJacobian
{
    /// <summary>
    /// Evaluates the Jacobian of <paramref name="residuals"/> at <paramref name="parameters"/>
    /// and writes the result into <c>buffers.Jacobian</c>.
    /// </summary>
    /// <param name="residuals">Residual function — must return a vector of length <c>buffers.ResidualCount</c>.</param>
    /// <param name="parameters">Parameter vector <c>theta</c> of length <c>buffers.ParameterCount</c>.</param>
    /// <param name="buffers">
    /// Pool providing <c>Jacobian</c> (m x n output), <c>PerturbedParameters</c> (n scratch),
    /// <c>PlusBuffer</c> / <c>MinusBuffer</c> (m scratch each). All entries of the Jacobian
    /// slab are overwritten on return; no prior contents are read.
    /// </param>
    public static void EvaluateInto(
        Func<double[], double[]> residuals,
        double[] parameters,
        LevenbergMarquardtBuffers buffers) =>
        FiniteDifferenceJacobian<double>.EvaluateInto(residuals, parameters, buffers.Inner);
}
