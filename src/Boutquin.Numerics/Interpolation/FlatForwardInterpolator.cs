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
/// Constant instantaneous forward rate between adjacent nodes. Given discount
/// factors <c>DF(t)</c>, the piecewise forward rate on <c>[tᵢ, tᵢ₊₁]</c> is
/// <c>f = (ln(DFᵢ) − ln(DFᵢ₊₁)) / (tᵢ₊₁ − tᵢ)</c>, and the interpolant evaluates
/// <c>DF(x) = DFᵢ · exp(−f · (x − tᵢ))</c>. Flat extrapolation beyond the node
/// range.
/// </summary>
/// <typeparam name="T">The floating-point type for interpolation.</typeparam>
/// <remarks>
/// Tier B: Requires IEEE 754 transcendental functions (Log, Exp).
/// </remarks>
public sealed class FlatForwardInterpolator<T> : IInterpolator<T> where T : IFloatingPointIeee754<T>
{
    /// <inheritdoc cref="FlatForwardInterpolator.Instance"/>
    public static readonly FlatForwardInterpolator<T> Instance = new();

    private FlatForwardInterpolator() { }

    /// <inheritdoc/>
    public string Name => "FlatForward";

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

        var dx = xs[i + 1] - xs[i];
        var f = (T.Log(ys[i]) - T.Log(ys[i + 1])) / dx;

        return ys[i] * T.Exp(-f * (x - xs[i]));
    }
}

/// <summary>
/// Constant instantaneous forward rate between adjacent nodes. Given discount
/// factors <c>DF(t)</c>, the piecewise forward rate on <c>[tᵢ, tᵢ₊₁]</c> is
/// <c>f = (ln(DFᵢ) − ln(DFᵢ₊₁)) / (tᵢ₊₁ − tᵢ)</c>, and the interpolant evaluates
/// <c>DF(x) = DFᵢ · exp(−f · (x − tᵢ))</c>. Flat extrapolation beyond the node
/// range.
/// </summary>
/// <remarks>
/// <para>
/// Mathematically equivalent to <see cref="LogLinearInterpolator"/> under
/// continuous compounding — both reproduce the same DF values at interior points.
/// The two differ only in which quantity is the primary output of the formula:
/// <see cref="FlatForwardInterpolator"/> exposes the forward rate <c>f</c> as
/// the load-bearing quantity, which is preferable when the same curve is
/// consumed downstream in rate form (e.g., pricing FRAs, computing carry).
/// </para>
/// <para>
/// Precondition: all y-values must be strictly positive (they are interpreted
/// as discount factors). Not validated; non-positive input produces <c>NaN</c>
/// via <c>Math.Log</c>.
/// </para>
/// <para>
/// Stateless; exposed as the singleton <see cref="Instance"/>. When adjacent
/// discount factors are equal (zero forward rate), the exponential in
/// <c>exp(−f · Δx)</c> evaluates to 1 and the interpolant returns <c>DFᵢ</c>
/// — the formula handles this case without special logic.
/// </para>
/// </remarks>
public sealed class FlatForwardInterpolator : IInterpolator
{
    /// <summary>
    /// Shared singleton instance.
    /// </summary>
    public static readonly FlatForwardInterpolator Instance = new();

    private FlatForwardInterpolator() { }

    /// <inheritdoc />
    public string Name => "FlatForward";

    /// <inheritdoc />
    public double Interpolate(double x, ReadOnlySpan<double> xs, ReadOnlySpan<double> ys)
    {
        return FlatForwardInterpolator<double>.Instance.Interpolate(x, xs, ys);
    }
}
