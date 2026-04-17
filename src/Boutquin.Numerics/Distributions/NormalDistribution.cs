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

namespace Boutquin.Numerics.Distributions;

/// <summary>
/// Standard normal (Gaussian) distribution <c>Z ~ N(0, 1)</c>: unified entry
/// point exposing the density <c>φ(x)</c>, cumulative <c>N(x) = P(Z ≤ x)</c>,
/// and inverse-cumulative (quantile) <c>N⁻¹(p)</c> functions. The CDF and
/// inverse-CDF delegate to <see cref="CumulativeNormal{T}"/> and
/// <see cref="InverseNormal{T}"/>, which hold the numerically-optimized rational
/// approximations; this façade centralizes the common use cases and keeps
/// call-sites independent of the backing algorithm choice.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> C (polynomial-approximation-bound). Accepts any
/// <typeparamref name="T"/> implementing <see cref="IFloatingPoint{TSelf}"/> at the
/// public surface; internally casts to <c>double</c> for the polynomial
/// approximation step and casts back.
/// </para>
/// <para>
/// Accuracy: PDF is exact to double precision; CDF and inverse-CDF are
/// accurate to ~14 significant digits (≈ double-precision limit). See
/// <see cref="CumulativeNormal{T}"/> (Laikov 2025, arXiv:2504.05068) and
/// <see cref="InverseNormal{T}"/> (Acklam 2000 with two Newton-Raphson polishing
/// iterations) for the underlying algorithms and their precision claims.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class NormalDistribution<T>
    where T : IFloatingPoint<T>
{
    private static readonly double s_reciprocalSqrt2Pi = 1.0 / Math.Sqrt(2.0 * Math.PI);

    /// <summary>
    /// Evaluates the standard normal probability density function
    /// phi(x) = exp(-x^2/2) / sqrt(2*pi) at the given point.
    /// </summary>
    /// <param name="x">The point at which to evaluate the density.</param>
    /// <returns>The probability density, always non-negative.</returns>
    public static T Pdf(T x)
    {
        var xd = double.CreateChecked(x);
        return T.CreateChecked(s_reciprocalSqrt2Pi * Math.Exp(-0.5d * xd * xd));
    }

    /// <summary>
    /// Evaluates the standard normal cumulative distribution function N(x) = P(Z &lt;= x).
    /// Delegates to <see cref="CumulativeNormal{T}.Evaluate"/>.
    /// </summary>
    /// <param name="x">The upper integration limit.</param>
    /// <returns>A value in [0, 1] representing the cumulative probability.</returns>
    public static T Cdf(T x)
    {
        return CumulativeNormal<T>.Evaluate(x);
    }

    /// <summary>
    /// Evaluates the inverse standard normal CDF (quantile function) N^{-1}(p).
    /// Delegates to <see cref="InverseNormal{T}.Evaluate"/>.
    /// </summary>
    /// <param name="p">Probability in (0, 1).</param>
    /// <returns>The z-score such that P(Z &lt;= z) = p.</returns>
    public static T InverseCdf(T p)
    {
        return InverseNormal<T>.Evaluate(p);
    }
}

/// <summary>
/// Legacy concrete-typed facade forwarding to <see cref="NormalDistribution{T}"/>
/// instantiated at <c>double</c>. Preserves the pre-migration public API.
/// </summary>
public static class NormalDistribution
{
    /// <inheritdoc cref="NormalDistribution{T}.Pdf"/>
    public static double Pdf(double x)
        => NormalDistribution<double>.Pdf(x);

    /// <inheritdoc cref="NormalDistribution{T}.Cdf"/>
    public static double Cdf(double x)
        => NormalDistribution<double>.Cdf(x);

    /// <inheritdoc cref="NormalDistribution{T}.InverseCdf"/>
    public static double InverseCdf(double p)
        => NormalDistribution<double>.InverseCdf(p);
}
