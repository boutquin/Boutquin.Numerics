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
/// Piecewise linear interpolation between adjacent nodes — C⁰ continuity, O(h²)
/// pointwise error on smooth data, no derivative continuity at nodes. Flat
/// extrapolation beyond the node range.
/// </summary>
/// <typeparam name="T">The floating-point type for interpolation.</typeparam>
/// <remarks>
/// Tier A: Requires only basic floating-point arithmetic.
/// </remarks>
public sealed class LinearInterpolator<T> : IInterpolator<T> where T : IFloatingPoint<T>
{
    /// <inheritdoc cref="LinearInterpolator.Instance"/>
    public static readonly LinearInterpolator<T> Instance = new();

    private LinearInterpolator() { }

    /// <inheritdoc/>
    public string Name => "Linear";

    /// <inheritdoc/>
    public T Interpolate(T x, ReadOnlySpan<T> xs, ReadOnlySpan<T> ys)
    {
        if (xs.Length != ys.Length)
        {
            throw new ArgumentException("xs and ys must have the same length.");
        }

        if (xs.Length == 0)
        {
            throw new ArgumentException("At least one node is required.");
        }

        if (x <= xs[0])
        {
            return ys[0];
        }

        if (x >= xs[^1])
        {
            return ys[^1];
        }

        var i = InterpolationHelper.FindInterval(x, xs);

        var t = (x - xs[i]) / (xs[i + 1] - xs[i]);
        return ys[i] + t * (ys[i + 1] - ys[i]);
    }
}

/// <summary>
/// Piecewise linear interpolation between adjacent nodes — C⁰ continuity, O(h²)
/// pointwise error on smooth data, no derivative continuity at nodes. Flat
/// extrapolation beyond the node range.
/// </summary>
/// <remarks>
/// <para>
/// Formula on <c>[xs[i], xs[i+1]]</c>: <c>y = ys[i] + t · (ys[i+1] − ys[i])</c>,
/// where <c>t = (x − xs[i]) / (xs[i+1] − xs[i])</c> is the normalized position in
/// the interval. The implementation is stateless and exposed as the singleton
/// <see cref="Instance"/>; all calls share no precomputed state.
/// </para>
/// <para>
/// Appropriate when the underlying data has discontinuous derivatives or when
/// sharp kinks at nodes are intentional (e.g., piecewise-linear payoff functions,
/// step changes in a policy variable). For smoother outputs on naturally smooth
/// data, prefer <see cref="MonotoneCubicInterpolator"/> (C¹) or
/// <see cref="CubicSplineInterpolator"/> (C²).
/// </para>
/// </remarks>
public sealed class LinearInterpolator : IInterpolator
{
    /// <summary>
    /// Shared singleton instance.
    /// </summary>
    public static readonly LinearInterpolator Instance = new();

    private LinearInterpolator() { }

    /// <inheritdoc />
    public string Name => "Linear";

    /// <inheritdoc />
    public double Interpolate(double x, ReadOnlySpan<double> xs, ReadOnlySpan<double> ys)
    {
        return LinearInterpolator<double>.Instance.Interpolate(x, xs, ys);
    }
}
