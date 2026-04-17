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
/// Evaluates the inverse standard normal CDF (quantile function) N^{-1}(p),
/// returning the z-score such that P(Z &lt;= z) = p.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> C (polynomial-approximation-bound). Accepts any
/// <typeparamref name="T"/> implementing <see cref="IFloatingPoint{TSelf}"/> at the
/// public surface; internally casts to <c>double</c> for Acklam's rational
/// approximation and casts back.
/// </para>
/// <para>
/// Uses Acklam's (2000) rational approximation as an initial estimate, followed
/// by two Newton-Raphson polishing iterations using <see cref="CumulativeNormal{T}"/>
/// and <see cref="NormalDistribution{T}.Pdf"/> for full double precision.
/// </para>
/// <para>
/// Reference: Peter J. Acklam, "An algorithm for computing the inverse normal
/// cumulative distribution function" (2000).
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class InverseNormal<T>
    where T : IFloatingPoint<T>
{
    // Acklam split point
    private const double PLow = 0.02425;

    // Central region numerator coefficients (a1..a6)
    private const double A1 = -3.969683028665376e+01;
    private const double A2 = 2.209460984245205e+02;
    private const double A3 = -2.759285104469687e+02;
    private const double A4 = 1.383577518672690e+02;
    private const double A5 = -3.066479806614716e+01;
    private const double A6 = 2.506628277459239e+00;

    // Central region denominator coefficients (b1..b5, b6 = 1)
    private const double B1 = -5.447609879822406e+01;
    private const double B2 = 1.615858368580409e+02;
    private const double B3 = -1.556989798598866e+02;
    private const double B4 = 6.680131188771972e+01;
    private const double B5 = -1.328068155288572e+01;

    // Lower tail numerator coefficients (c1..c6)
    private const double C1 = -7.784894002430293e-03;
    private const double C2 = -3.223964580411365e-01;
    private const double C3 = -2.400758277161838e+00;
    private const double C4 = -2.549732539343734e+00;
    private const double C5 = 4.374664141464968e+00;
    private const double C6 = 2.938163982698783e+00;

    // Lower tail denominator coefficients (d1..d4, d5 = 1)
    private const double D1 = 7.784695709041462e-03;
    private const double D2 = 3.224671290700398e-01;
    private const double D3 = 2.445134137142996e+00;
    private const double D4 = 3.754408661907416e+00;

    /// <summary>
    /// Returns the z-score such that P(Z &lt;= z) = <paramref name="p"/>
    /// for the standard normal distribution.
    /// </summary>
    /// <param name="p">Probability in the open interval (0, 1).</param>
    /// <returns>The quantile value z satisfying N(z) = p.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="p"/> is not in (0, 1).
    /// </exception>
    public static T Evaluate(T p)
    {
        var pd = double.CreateChecked(p);
        if (pd <= 0.0 || pd >= 1.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(p),
                p,
                "Probability must be in the open interval (0, 1).");
        }

        return T.CreateChecked(EvaluateCore(pd));
    }

    private static double EvaluateCore(double p)
    {
        double x;

        if (p < PLow)
        {
            // Lower tail region
            var q = Math.Sqrt(-2.0 * Math.Log(p));
            x = (((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6)
                / ((((D1 * q + D2) * q + D3) * q + D4) * q + 1.0);
        }
        else if (p <= 1.0 - PLow)
        {
            // Central region
            var q = p - 0.5;
            var r = q * q;
            x = q * (((((A1 * r + A2) * r + A3) * r + A4) * r + A5) * r + A6)
                / (((((B1 * r + B2) * r + B3) * r + B4) * r + B5) * r + 1.0);
        }
        else
        {
            // Upper tail — use symmetry
            return -EvaluateCore(1.0 - p);
        }

        for (var i = 0; i < 2; i++)
        {
            x -= (CumulativeNormal<double>.Evaluate(x) - p) / NormalDistribution<double>.Pdf(x);
        }

        return x;
    }
}

/// <summary>
/// Legacy concrete-typed facade forwarding to <see cref="InverseNormal{T}"/>
/// instantiated at <c>double</c>. Preserves the pre-migration public API.
/// </summary>
public static class InverseNormal
{
    /// <inheritdoc cref="InverseNormal{T}.Evaluate"/>
    public static double Evaluate(double p)
        => InverseNormal<double>.Evaluate(p);
}
