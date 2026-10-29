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
/// Evaluates the standard normal cumulative distribution function N(x) = P(Z &lt;= x).
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> C (polynomial-approximation-bound). Accepts any
/// <typeparamref name="T"/> implementing <see cref="IFloatingPoint{TSelf}"/> at the
/// public surface; internally casts to <c>double</c> for rational-function
/// evaluation and casts back. For <c>T = decimal</c>, conversion to <c>double</c>
/// can lose many decimal ULPs; the result has a double-precision ceiling as well
/// as the approximation's own rounding error.
/// </para>
/// <para>
/// Uses Cephes rational approximations for erf and erfc in the central region
/// and the normal-tail continued fraction for |x| &gt;= 4. The small tail is
/// evaluated directly, avoiding subtraction from a rounded erf near one.
/// Relative accuracy is checked against independent high-precision Gaussian
/// integrals; results are not guaranteed to be correctly rounded.
/// </para>
/// <para>
/// Finite tails whose exact probabilities round to 0 or 1 in double precision are
/// saturated before the approximation. A fused multiply-add retains the
/// rounding remainder in the Gaussian exponent, including in the tail.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class CumulativeNormal<T>
    where T : IFloatingPoint<T>
{
    // Cephes erf/erfc rational coefficients, adapted from scipy/xsf ndtr.h.
    // Copyright 1984, 1987, 1988, 1992 Stephen L. Moshier; Copyright 2024 SciPy.
    // BSD-3-Clause terms and source provenance are in THIRD-PARTY-NOTICES.txt.
    private const double ReciprocalSqrt2 = 0.707106781186547524400844362104849039;

    /// <summary>
    /// Returns the probability that a standard normal random variable is less than
    /// or equal to <paramref name="x"/> — that is, N(x) = P(Z &lt;= x).
    /// </summary>
    /// <param name="x">The upper integration limit of the standard normal density.</param>
    /// <returns>
    /// A value in [0, 1] representing the cumulative probability. IEEE limit
    /// behavior: <c>N(+∞) = 1</c>, <c>N(−∞) = 0</c>, and <c>N(NaN) = NaN</c>.
    /// </returns>
    public static T Evaluate(T x)
    {
        var xd = double.CreateChecked(x);

        // Short-circuit nonfinite inputs to the analytic CDF limits.
        if (!double.IsFinite(xd))
        {
            if (double.IsNaN(xd))
            {
                return T.CreateChecked(double.NaN);
            }

            return T.CreateChecked(xd > 0.0 ? 1.0 : 0.0);
        }

        // Beyond this point the exact normal tail rounds to its analytic limit in double
        // precision. Saturating also keeps the polynomial's powers of x² finite.
        if (xd >= 38.0)
        {
            return T.One;
        }

        if (xd <= -39.0)
        {
            return T.Zero;
        }

        var magnitude = Math.Abs(xd);
        var tail = magnitude >= 4.0 ? NormalTail(magnitude) : CentralTail(magnitude);
        return T.CreateChecked(xd < 0.0 ? tail : 1.0 - tail);
    }

    // Phi(x) - 1/2 without adding then subtracting 1/2. InverseNormal uses this
    // residual near its median, where the rounded CDF cannot resolve a correction.
    internal static double CenteredProbability(double x)
    {
        var z = 0.5 * x * x;
        var numerator = (((9.60497373987051638749 * z + 90.0260197203842689217) * z
            + 2232.00534594684319226) * z + 7003.32514112805075473) * z + 55592.3013010394962768;
        var denominator = ((((z + 33.5617141647503099647) * z + 521.357949780152679795) * z
            + 4594.32382970980127987) * z + 22629.0000613890934246) * z + 49267.3942608635921086;
        return 0.5 * x * ReciprocalSqrt2 * (numerator / denominator);
    }

    private static double CentralTail(double x)
    {
        var z = x * ReciprocalSqrt2;
        if (z < 1.0)
        {
            return 0.5 - CenteredProbability(x);
        }

        var numerator = (((((((2.46196981473530512524e-10 * z + 0.564189564831068821977) * z
            + 7.46321056442269912687) * z + 48.6371970985681366614) * z + 196.520832956077098242) * z
            + 526.445194995477358631) * z + 934.528527171957607540) * z + 1027.55188689515710272) * z
            + 557.535335369399327526;
        var denominator = (((((((z + 13.2281951154744992508) * z + 86.7072140885989742329) * z
            + 354.937778887819891062) * z + 975.708501743205489753) * z + 1823.90916687909736289) * z
            + 2246.33760818710981792) * z + 1656.66309194161350182) * z + 557.535340817727675546;
        return 0.5 * GaussianExponential(x) * (numerator / denominator);
    }

    private static double GaussianExponential(double x)
    {
        var exponent = -0.5 * x * x;
        var remainder = Math.FusedMultiplyAdd(-0.5 * x, x, -exponent);
        return Math.Exp(exponent) * Math.Exp(remainder);
    }

    // Mills' ratio: Q(x) = phi(x) / (x + 1/(x + 2/(x + 3/...))).
    // Forty levels are sufficient to converge at the x = 4 switch point.
    private static double NormalTail(double x)
    {
        var denominator = x;
        for (var i = 40; i >= 1; i--)
        {
            denominator = x + i / denominator;
        }

        return GaussianExponential(x) / Math.Sqrt(2.0 * Math.PI) / denominator;
    }
}

/// <summary>
/// Legacy concrete-typed facade forwarding to <see cref="CumulativeNormal{T}"/>
/// instantiated at <c>double</c>. Preserves the pre-migration public API.
/// </summary>
[Obsolete(
    "Use CumulativeNormal<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class CumulativeNormal
{
    /// <inheritdoc cref="CumulativeNormal{T}.Evaluate"/>
    public static double Evaluate(double x)
        => CumulativeNormal<double>.Evaluate(x);
}
