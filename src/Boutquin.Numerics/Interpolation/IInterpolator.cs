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
/// Contract for single-evaluation interpolation of a scalar y-value at a target
/// x-coordinate given node arrays <c>(xs, ys)</c>. Implementations are expected
/// to be thread-safe and side-effect-free.
/// </summary>
/// <typeparam name="T">The floating-point type for interpolation.</typeparam>
/// <remarks>
/// Tier A: Requires only basic floating-point arithmetic.
/// </remarks>
public interface IInterpolator<T> where T : IFloatingPoint<T>
{
    /// <inheritdoc cref="IInterpolator.Name"/>
    string Name { get; }

    /// <inheritdoc cref="IInterpolator.Interpolate"/>
    T Interpolate(T x, ReadOnlySpan<T> xs, ReadOnlySpan<T> ys);
}

/// <summary>
/// Contract for single-evaluation interpolation of a scalar y-value at a target
/// x-coordinate given node arrays <c>(xs, ys)</c>. Implementations are expected
/// to be thread-safe and side-effect-free.
/// </summary>
/// <remarks>
/// <para>
/// Domain-agnostic by design: x and y are plain doubles. Domain layers convert
/// their types (dates → year fractions, discount factors ↔ log-DF, etc.) before
/// and after interpolation.
/// </para>
/// <para>
/// <b>Universal preconditions (all implementations):</b>
/// <list type="bullet">
///   <item><c>xs</c> is strictly increasing.</item>
///   <item><c>xs.Length == ys.Length</c> and both are non-empty.</item>
///   <item>All elements are finite (<c>NaN</c> / <c>Infinity</c> produce undefined output).</item>
/// </list>
/// </para>
/// <para>
/// <b>Per-implementation guarantees (not universal):</b>
/// <list type="bullet">
///   <item>Continuity class — <see cref="LinearInterpolator"/>, <see cref="LogLinearInterpolator"/>,
///         <see cref="FlatForwardInterpolator"/> are C⁰; <see cref="MonotoneCubicInterpolator"/>
///         is C¹ with monotonicity preservation; <see cref="MonotoneConvexInterpolator"/> is C¹
///         with non-negative-forward-rate guarantee; <see cref="CubicSplineInterpolator"/> is C².</item>
///   <item>Domain constraints on y — <see cref="LogLinearInterpolator"/> and
///         <see cref="FlatForwardInterpolator"/> require strictly positive y-values;
///         other implementations accept any finite values.</item>
///   <item>Extrapolation — all current stateless implementations clamp to the
///         boundary (flat extrapolation); <see cref="CubicSplineInterpolator"/>
///         extrapolates along the nearest cubic segment.</item>
///   <item>State — stateless implementations (<see cref="LinearInterpolator"/>,
///         <see cref="LogLinearInterpolator"/>, <see cref="FlatForwardInterpolator"/>,
///         <see cref="MonotoneCubicInterpolator"/>, <see cref="MonotoneConvexInterpolator"/>) are
///         exposed as singletons via <see cref="InterpolatorFactory"/>; stateful ones
///         (<see cref="CubicSplineInterpolator"/>, <see cref="MonotoneCubicSpline"/>)
///         precompute coefficients at construction and require direct instantiation.</item>
/// </list>
/// </para>
/// </remarks>
public interface IInterpolator
{
    /// <summary>
    /// Human-readable identifier for the interpolation scheme (e.g., "Linear",
    /// "LogLinear", "MonotoneCubic"). Intended for diagnostics, logging, and
    /// configuration round-tripping; not used for dispatch.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Evaluates the interpolant at <paramref name="x"/> given node arrays
    /// <paramref name="xs"/> (strictly increasing x-coordinates) and
    /// <paramref name="ys"/> (corresponding y-values).
    /// </summary>
    /// <param name="x">Target x-coordinate. Behavior outside <c>[xs[0], xs[^1]]</c> is per-implementation (typically flat extrapolation).</param>
    /// <param name="xs">Strictly-increasing node x-coordinates.</param>
    /// <param name="ys">Node y-values. Must have the same length as <paramref name="xs"/>; positivity constraints may apply per-implementation.</param>
    /// <returns>The interpolated y-value at <paramref name="x"/>.</returns>
    double Interpolate(double x, ReadOnlySpan<double> xs, ReadOnlySpan<double> ys);
}
