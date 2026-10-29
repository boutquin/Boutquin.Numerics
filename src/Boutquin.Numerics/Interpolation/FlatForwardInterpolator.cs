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
/// <para>
/// Tier B: Requires IEEE 754 transcendental functions (Log, Exp).
/// </para>
/// <para>
/// Precondition: all y-values must be strictly positive (they are interpreted as discount
/// factors). Positivity is not validated for performance; non-positive input propagates rather
/// than throwing: when <c>ys[i] &lt; 0</c>, <c>T.Log</c> returns <c>NaN</c>, which propagates to
/// the result. At interior queries a positive-left/zero-right interval returns zero;
/// a zero left node produces <c>NaN</c>, including a zero-zero interval, because the
/// log-linear exponent contains undefined infinity arithmetic. Domain layers must
/// enforce strict positivity upstream.
/// </para>
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

        InterpolationHelper.ValidateSortedAscending(xs, nameof(xs));
        InterpolationHelper.ValidateFiniteTarget(x);

        if (x <= xs[0])
        {
            return ys[0];
        }

        if (x >= xs[^1])
        {
            return ys[^1];
        }

        var i = InterpolationHelper.FindInterval(x, xs);

        var t = InterpolationHelper.LinearWeight(x, xs[i], xs[i + 1]);
        var lnYi = T.Log(ys[i]);
        return T.Exp(lnYi + t * (T.Log(ys[i + 1]) - lnYi));
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
/// The implementation returns the interpolated discount factor using the same
/// log-linear exponent; the constant forward rate describes the interval's model.
/// </para>
/// <para>
/// Precondition: all y-values must be strictly positive (they are interpreted
/// as discount factors). Not validated. At interior queries a positive-left/zero-right
/// interval returns zero; a zero left node or a negative node produces <c>NaN</c>.
/// </para>
/// <para>
/// Stateless; exposed as the singleton <see cref="Instance"/>. When adjacent
/// discount factors are equal (zero forward rate), the exponential in
/// <c>exp(−f · Δx)</c> evaluates to 1 and the interpolant returns <c>DFᵢ</c>
/// — the formula handles this case without special logic.
/// </para>
/// </remarks>
[Obsolete(
    "Use FlatForwardInterpolator<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
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
