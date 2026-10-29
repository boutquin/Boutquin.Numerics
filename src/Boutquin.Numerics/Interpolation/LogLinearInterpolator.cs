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
/// Linear interpolation in log space: <c>y(x) = exp(ln(ys[i]) + t · (ln(ys[i+1]) − ln(ys[i])))</c>
/// on <c>[xs[i], xs[i+1]]</c>, where <c>t = (x − xs[i]) / (xs[i+1] − xs[i])</c>.
/// Preserves strict positivity and produces constant instantaneous forward rates
/// between nodes when applied to discount factors, making it mathematically
/// equivalent to <see cref="FlatForwardInterpolator{T}"/> under continuous compounding.
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
/// a zero left node produces <c>NaN</c>, including a zero-zero interval, because
/// the exponent contains undefined infinity arithmetic.
/// Domain layers are expected to enforce strict positivity upstream.
/// </para>
/// </remarks>
public sealed class LogLinearInterpolator<T> : IInterpolator<T> where T : IFloatingPointIeee754<T>
{
    /// <inheritdoc cref="LogLinearInterpolator.Instance"/>
    public static readonly LogLinearInterpolator<T> Instance = new();

    private LogLinearInterpolator() { }

    /// <inheritdoc/>
    public string Name => "LogLinear";

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

        var lnYi = T.Log(ys[i]);
        var lnYi1 = T.Log(ys[i + 1]);
        var t = InterpolationHelper.LinearWeight(x, xs[i], xs[i + 1]);
        var lnY = lnYi + t * (lnYi1 - lnYi);

        return T.Exp(lnY);
    }
}

/// <summary>
/// Linear interpolation in log space: <c>y(x) = exp(ln(ys[i]) + t · (ln(ys[i+1]) − ln(ys[i])))</c>
/// on <c>[xs[i], xs[i+1]]</c>, where <c>t = (x − xs[i]) / (xs[i+1] − xs[i])</c>.
/// Preserves strict positivity and produces constant instantaneous forward rates
/// between nodes when applied to discount factors, making it mathematically
/// equivalent to <see cref="FlatForwardInterpolator"/> under continuous compounding.
/// </summary>
/// <remarks>
/// <para>
/// Precondition: all y-values must be strictly positive. The implementation does
/// not validate positivity for performance. Non-positive y-values propagate as
/// follows: when <c>ys[i] &lt; 0</c>, <c>Math.Log</c> returns <c>NaN</c>, which
/// propagates to the result. At interior queries a positive-left/zero-right interval
/// returns zero; a zero left node produces <c>NaN</c>, including a zero-zero interval.
/// Domain layers are expected to enforce strict positivity upstream.
/// </para>
/// <para>
/// The log-space formula evaluates the geometric blend without explicitly
/// forming the ratio <c>ys[i+1] / ys[i]</c>, which can overflow for widely
/// separated finite positive ordinates. Its accuracy still depends on the
/// working type's logarithm, exponential, and arithmetic rounding.
/// </para>
/// <para>Flat extrapolation beyond the node range. Stateless; exposed as the singleton <see cref="Instance"/>.</para>
/// </remarks>
[Obsolete(
    "Use LogLinearInterpolator<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class LogLinearInterpolator : IInterpolator
{
    /// <summary>
    /// Shared singleton instance.
    /// </summary>
    public static readonly LogLinearInterpolator Instance = new();

    private LogLinearInterpolator() { }

    /// <inheritdoc />
    public string Name => "LogLinear";

    /// <inheritdoc />
    public double Interpolate(double x, ReadOnlySpan<double> xs, ReadOnlySpan<double> ys)
    {
        return LogLinearInterpolator<double>.Instance.Interpolate(x, xs, ys);
    }
}
