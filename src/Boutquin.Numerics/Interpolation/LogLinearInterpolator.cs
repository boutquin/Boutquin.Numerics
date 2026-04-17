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
/// Tier B: Requires IEEE 754 transcendental functions (Log, Exp).
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
        var t = (x - xs[i]) / (xs[i + 1] - xs[i]);
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
/// not validate positivity for performance; <c>Math.Log</c> on a non-positive
/// value produces <c>NaN</c> or <c>-Infinity</c> and propagates to the result.
/// Domain layers are expected to enforce positivity upstream.
/// </para>
/// <para>
/// Interpolating in log space is numerically preferable to dividing discount
/// factors directly: the subtraction <c>ln(ys[i+1]) − ln(ys[i])</c> has bounded
/// relative error even when successive discount factors are very close to one
/// another, where the ratio form <c>ys[i+1] / ys[i]</c> can cancel significant
/// digits.
/// </para>
/// <para>Flat extrapolation beyond the node range. Stateless; exposed as the singleton <see cref="Instance"/>.</para>
/// </remarks>
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
