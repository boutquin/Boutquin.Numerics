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
/// an adaptive per-component step <c>h = max(1e-8, |thetaj|*cbrt(epsilon))</c>, where
/// <c>epsilon</c> is the working type's machine epsilon, not its smallest subnormal.
/// Numerical Recipes (3rd ed.), §5.7 equations 5.7.8–9 balance central truncation
/// <c>O(h²)</c> against roundoff <c>O(epsilon/h)</c>. For smooth, well-scaled residuals,
/// the resulting theoretical relative-error scale is <c>epsilon^(2/3)</c>.
/// At exactly zero, a unit characteristic scale gives <c>h = cbrt(epsilon)</c>.
/// The floor and zero-point scale are heuristics; callback noise and
/// the function's own curvature can require an analytic Jacobian.
/// </para>
/// <para>
/// Narrower types have lower derivative accuracy. The theoretical scale is about
/// <c>4e-11</c> for <c>double</c>, <c>2e-5</c> for <c>float</c>, and <c>1e-2</c> for
/// <c>Half</c>; these are error scales under the smoothness assumptions, not bounds
/// for arbitrary callbacks. A bound can force a smaller or one-sided stencil,
/// changing its error balance. At a bound the one-sided step uses
/// <c>max(1, |thetaj|)*sqrt(epsilon)</c>, reduced to the available interval.
/// Supply an analytic Jacobian when accuracy matters.
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
    private static T DifferenceQuotient(T upper, T lower, T interval, bool multiplyReciprocal)
    {
        var difference = upper - lower;
        if (!T.IsFinite(difference))
        {
            // Finite opposite-sign values can overflow before a representable division.
            return upper / interval - lower / interval;
        }
        var reciprocal = T.One / interval;
        return multiplyReciprocal && T.IsFinite(reciprocal)
            ? difference * reciprocal
            : difference / interval;
    }

    // NR §5.7: the central-difference balance uses the cube root of machine epsilon.
    private static readonly T s_cbrtEpsilon = T.Cbrt(T.BitIncrement(T.One) - T.One);
    private static readonly T s_sqrtEpsilon = T.Sqrt(T.BitIncrement(T.One) - T.One);
    private static readonly T s_floor = T.CreateChecked(1e-8);

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
    /// <param name="lowerBounds">
    /// Optional element-wise lower bounds, same length as <paramref name="parameters"/> or
    /// empty for none. When the minus-side perturbation of a component would fall below its
    /// lower bound, that side is skipped and a one-sided (forward) difference toward the
    /// interior is used instead, matching the caller's own bound-clipping contract.
    /// </param>
    /// <param name="upperBounds">
    /// Optional element-wise upper bounds, same length as <paramref name="parameters"/> or
    /// empty for none. When the plus-side perturbation of a component would exceed its upper
    /// bound, that side is skipped and a one-sided (backward) difference toward the interior
    /// is used instead.
    /// </param>
    public static void EvaluateInto(
        Func<T[], T[]> residuals,
        T[] parameters,
        LevenbergMarquardtBuffers<T> buffers,
        ReadOnlySpan<T> lowerBounds = default,
        ReadOnlySpan<T> upperBounds = default)
    {
        var parameterCount = buffers.ParameterCount;
        var residualCount = buffers.ResidualCount;
        var perturbedParameters = buffers.CallbackParameters;
        var plusBuffer = buffers.PlusBuffer;
        var minusBuffer = buffers.MinusBuffer;
        var jacobian = buffers.Jacobian;

        for (var j = 0; j < parameterCount; j++)
        {
            // Copy parameters into scratch, perturb +h.
            Array.Copy(parameters, perturbedParameters, parameterCount);
            var upperBound = j < upperBounds.Length ? upperBounds[j] : T.PositiveInfinity;
            var lowerBound = j < lowerBounds.Length ? lowerBounds[j] : T.NegativeInfinity;
            if (lowerBound == upperBound)
            {
                // A fixed parameter has no feasible direction and contributes no step.
                for (var i = 0; i < residualCount; i++)
                {
                    jacobian[i, j] = T.Zero;
                }
                continue;
            }

            var h = StepSize(parameters[j]);
            var forwardRoom = upperBound - parameters[j];
            var backwardRoom = parameters[j] - lowerBound;
            if (forwardRoom == T.Zero || backwardRoom == T.Zero)
            {
                // A boundary stencil is first order: balance h against epsilon/h.
                h = T.Max(T.One, T.Abs(parameters[j])) * s_sqrtEpsilon;
            }
            if (forwardRoom > T.Zero)
            {
                h = T.Min(h, forwardRoom);
            }
            if (backwardRoom > T.Zero)
            {
                h = T.Min(h, backwardRoom);
            }
            var plusParameter = parameters[j] + h;
            var minusParameter = parameters[j] - h;
            var hasPlus = T.IsFinite(plusParameter) && plusParameter != parameters[j] && plusParameter <= upperBound;
            var hasMinus = T.IsFinite(minusParameter) && minusParameter != parameters[j] && minusParameter >= lowerBound;
            if (!hasPlus && !hasMinus)
            {
                throw new InvalidOperationException("No finite, distinct finite-difference perturbation is available.");
            }

            if (hasPlus)
            {
                perturbedParameters[j] = plusParameter;
                var plus = residuals(perturbedParameters);
                ValidateResidual(plus, residualCount);
                Array.Copy(plus, plusBuffer, residualCount);
            }

            if (hasMinus)
            {
                // Array callbacks can mutate every coordinate of their scratch input.
                Array.Copy(parameters, perturbedParameters, parameterCount);
                perturbedParameters[j] = minusParameter;
                var minus = residuals(perturbedParameters);
                ValidateResidual(minus, residualCount);
                Array.Copy(minus, minusBuffer, residualCount);
            }

            for (var i = 0; i < residualCount; i++)
            {
                jacobian[i, j] = hasPlus && hasMinus
                    ? DifferenceQuotient(plusBuffer[i], minusBuffer[i], plusParameter - minusParameter, true)
                    : hasPlus
                        ? DifferenceQuotient(plusBuffer[i], buffers.Residual[i], plusParameter - parameters[j], false)
                        : DifferenceQuotient(buffers.Residual[i], minusBuffer[i], parameters[j] - minusParameter, false);
            }
        }
    }

    /// <summary>
    /// Evaluates the Jacobian using an <see cref="LmResidualWriter{T}"/> callback and writes
    /// the result into <c>buffers.Jacobian</c>. The writer fills the pool's <c>PlusBuffer</c>
    /// and <c>MinusBuffer</c> directly — zero allocation on the callback path.
    /// </summary>
    /// <param name="writer">Buffer-fill residual callback.</param>
    /// <param name="parameters">Parameter vector of length <c>buffers.ParameterCount</c>.</param>
    /// <param name="buffers">Pool providing all scratch storage.</param>
    /// <param name="lowerBounds">
    /// Optional element-wise lower bounds, same length as <paramref name="parameters"/> or
    /// empty for none. See the array-callback overload for the one-sided-difference contract.
    /// </param>
    /// <param name="upperBounds">
    /// Optional element-wise upper bounds, same length as <paramref name="parameters"/> or
    /// empty for none. See the array-callback overload for the one-sided-difference contract.
    /// </param>
    public static void EvaluateInto(
        LmResidualWriter<T> writer,
        T[] parameters,
        LevenbergMarquardtBuffers<T> buffers,
        ReadOnlySpan<T> lowerBounds = default,
        ReadOnlySpan<T> upperBounds = default)
    {
        var parameterCount = buffers.ParameterCount;
        var residualCount = buffers.ResidualCount;
        var perturbedParameters = buffers.PerturbedParameters;
        var plusBuffer = buffers.PlusBuffer;
        var minusBuffer = buffers.MinusBuffer;
        var jacobian = buffers.Jacobian;

        var plusSpan = plusBuffer.AsSpan(0, residualCount);
        var minusSpan = minusBuffer.AsSpan(0, residualCount);

        // Slice the perturbed-parameter argument to the logical parameter count. On a shrunk
        // grow-only pool perturbedParameters is a high-water physical array longer than
        // parameterCount whose tail carries stale values from an earlier larger solve; a
        // contract-conforming writer that iterates parameters.Length would otherwise read that
        // stale tail, corrupting the finite-difference Jacobian. On a fresh/same-size pool this
        // slice is a no-op view over the whole array. The span is a view over the in-place-mutated
        // array, so it reflects each iteration's perturbation without re-slicing.
        var perturbedSpan = perturbedParameters.AsSpan(0, parameterCount);

        for (var j = 0; j < parameterCount; j++)
        {
            Array.Copy(parameters, perturbedParameters, parameterCount);
            var upperBound = j < upperBounds.Length ? upperBounds[j] : T.PositiveInfinity;
            var lowerBound = j < lowerBounds.Length ? lowerBounds[j] : T.NegativeInfinity;
            if (lowerBound == upperBound)
            {
                // A fixed parameter has no feasible direction and contributes no step.
                for (var i = 0; i < residualCount; i++)
                {
                    jacobian[i, j] = T.Zero;
                }
                continue;
            }

            var h = StepSize(parameters[j]);
            var forwardRoom = upperBound - parameters[j];
            var backwardRoom = parameters[j] - lowerBound;
            if (forwardRoom == T.Zero || backwardRoom == T.Zero)
            {
                // A boundary stencil is first order: balance h against epsilon/h.
                h = T.Max(T.One, T.Abs(parameters[j])) * s_sqrtEpsilon;
            }
            if (forwardRoom > T.Zero)
            {
                h = T.Min(h, forwardRoom);
            }
            if (backwardRoom > T.Zero)
            {
                h = T.Min(h, backwardRoom);
            }
            var plusParameter = parameters[j] + h;
            var minusParameter = parameters[j] - h;
            var hasPlus = T.IsFinite(plusParameter) && plusParameter != parameters[j] && plusParameter <= upperBound;
            var hasMinus = T.IsFinite(minusParameter) && minusParameter != parameters[j] && minusParameter >= lowerBound;
            if (!hasPlus && !hasMinus)
            {
                throw new InvalidOperationException("No finite, distinct finite-difference perturbation is available.");
            }

            if (hasPlus)
            {
                perturbedParameters[j] = plusParameter;
                plusSpan.Fill(T.NaN);
                writer(perturbedSpan, plusSpan);
                ValidateFinite(plusSpan);
            }

            if (hasMinus)
            {
                perturbedParameters[j] = minusParameter;
                minusSpan.Fill(T.NaN);
                writer(perturbedSpan, minusSpan);
                ValidateFinite(minusSpan);
            }

            for (var i = 0; i < residualCount; i++)
            {
                jacobian[i, j] = hasPlus && hasMinus
                    ? DifferenceQuotient(plusBuffer[i], minusBuffer[i], plusParameter - minusParameter, true)
                    : hasPlus
                        ? DifferenceQuotient(plusBuffer[i], buffers.Residual[i], plusParameter - parameters[j], false)
                        : DifferenceQuotient(buffers.Residual[i], minusBuffer[i], parameters[j] - minusParameter, false);
            }
        }
    }

    /// <summary>
    /// Computes the adaptive per-component step size for central finite differences.
    /// </summary>
    /// <param name="component">Current value of the parameter component.</param>
    /// <returns>Step size <c>h = max(1e-8, |component| * cbrt(epsilon)), or cbrt(epsilon) at zero</c>.</returns>
    private static T StepSize(T component) =>
        component == T.Zero ? s_cbrtEpsilon : T.Max(s_floor, T.Abs(component) * s_cbrtEpsilon);

    private static void ValidateResidual(T[] residual, int expectedCount)
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

        ValidateFinite(residual);
    }

    private static void ValidateFinite(ReadOnlySpan<T> values)
    {
        for (var i = 0; i < values.Length; i++)
        {
            if (!T.IsFinite(values[i]))
            {
                throw new InvalidOperationException($"Residual function returned a non-finite value at index {i}.");
            }
        }
    }
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
