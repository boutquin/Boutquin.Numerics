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

namespace Boutquin.Numerics.Interpolation;

/// <summary>
/// Resolves an <see cref="InterpolatorKind"/> discriminator to its stateless
/// singleton implementation of <see cref="IInterpolator"/>.
/// </summary>
/// <remarks>
/// <para>
/// Stateless interpolators are shared singletons because the <see cref="IInterpolator"/>
/// contract takes node arrays per call and precomputes nothing — allocation-free
/// resolution is the appropriate design. For repeated evaluations on the <em>same</em>
/// node set, consider the stateful variants (<see cref="CubicSplineInterpolator"/>,
/// <see cref="MonotoneCubicSpline"/>) which amortize coefficient setup across calls.
/// </para>
/// <para>
/// <see cref="InterpolatorKind.CubicSpline"/> is deliberately unsupported here:
/// <see cref="CubicSplineInterpolator"/> requires pre-construction with data points
/// and boundary conditions, so it cannot be exposed as a node-agnostic singleton
/// on <see cref="IInterpolator"/>. Construct it directly instead.
/// </para>
/// </remarks>
public static class InterpolatorFactory
{
    /// <summary>
    /// Returns the singleton <see cref="IInterpolator"/> for the specified kind.
    /// </summary>
    /// <param name="kind">The interpolation method to use.</param>
    /// <returns>The corresponding interpolator instance.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="kind"/> is not a recognized value.</exception>
    /// <exception cref="NotSupportedException">
    /// Thrown when <paramref name="kind"/> is <see cref="InterpolatorKind.CubicSpline"/>,
    /// which requires pre-construction with data points.
    /// </exception>
    public static IInterpolator Create(InterpolatorKind kind) => kind switch
    {
        InterpolatorKind.Linear => LinearInterpolator.Instance,
        InterpolatorKind.LogLinear => LogLinearInterpolator.Instance,
        InterpolatorKind.FlatForward => FlatForwardInterpolator.Instance,
        InterpolatorKind.MonotoneCubic => MonotoneCubicInterpolator.Instance,
        InterpolatorKind.MonotoneConvex => MonotoneConvexInterpolator.Instance,
        InterpolatorKind.CubicSpline => throw new NotSupportedException(
            "CubicSplineInterpolator requires pre-construction with data points. " +
            "Create it directly via new CubicSplineInterpolator(xs, ys)."),
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, $"Unknown interpolator kind: {kind}."),
    };
}

/// <summary>
/// Resolves an <see cref="InterpolatorKind"/> discriminator to its stateless
/// generic singleton implementation of <see cref="IInterpolator{T}"/>.
/// </summary>
/// <typeparam name="T">The floating-point type for interpolation.</typeparam>
/// <remarks>
/// <para>
/// <see cref="InterpolatorKind.LogLinear"/> and <see cref="InterpolatorKind.FlatForward"/>
/// require <see cref="IFloatingPointIeee754{TSelf}"/>. When <typeparamref name="T"/>
/// does not implement that interface (e.g. <see cref="decimal"/>), those kinds throw
/// <see cref="NotSupportedException"/> at runtime. Use <see cref="InterpolatorKind.Linear"/>
/// or <see cref="InterpolatorKind.MonotoneCubic"/> for non-IEEE754 types.
/// </para>
/// </remarks>
public static class InterpolatorFactory<T> where T : IFloatingPoint<T>
{
    /// <summary>
    /// Returns the singleton <see cref="IInterpolator{T}"/> for the specified kind.
    /// </summary>
    public static IInterpolator<T> Create(InterpolatorKind kind) => kind switch
    {
        InterpolatorKind.Linear => LinearInterpolator<T>.Instance,
        InterpolatorKind.LogLinear => ResolveIeee754Interpolator(kind),
        InterpolatorKind.FlatForward => ResolveIeee754Interpolator(kind),
        InterpolatorKind.MonotoneCubic => MonotoneCubicInterpolator<T>.Instance,
        InterpolatorKind.MonotoneConvex => MonotoneConvexInterpolator<T>.Instance,
        InterpolatorKind.CubicSpline => throw new NotSupportedException(
            "CubicSplineInterpolator<T> requires pre-construction with data points. " +
            "Create it directly via new CubicSplineInterpolator<T>(xs, ys)."),
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, $"Unknown interpolator kind: {kind}."),
    };

    private static IInterpolator<T> ResolveIeee754Interpolator(InterpolatorKind kind)
    {
        if (typeof(T) == typeof(double))
        {
            return kind == InterpolatorKind.LogLinear
                ? (IInterpolator<T>)(object)LogLinearInterpolator<double>.Instance
                : (IInterpolator<T>)(object)FlatForwardInterpolator<double>.Instance;
        }

        if (typeof(T) == typeof(float))
        {
            return kind == InterpolatorKind.LogLinear
                ? (IInterpolator<T>)(object)LogLinearInterpolator<float>.Instance
                : (IInterpolator<T>)(object)FlatForwardInterpolator<float>.Instance;
        }

        throw new NotSupportedException(
            $"{kind} requires IFloatingPointIeee754<T>, but {typeof(T).Name} does not implement it. " +
            "Use InterpolatorKind.Linear or InterpolatorKind.MonotoneCubic instead.");
    }
}
