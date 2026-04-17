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
/// public surface; internally casts to <c>double</c> for the Laikov polynomial
/// evaluation and casts back. The cast loses at most one ULP of <typeparamref name="T"/>
/// and is below the approximation's published precision ceiling (~14 significant digits).
/// </para>
/// <para>
/// Uses the relationship N(x) = (1 + erf(x / sqrt(2))) / 2 with the error function
/// computed via Laikov's exponential-free global approximation (arXiv:2504.05068, 2025):
/// erf(x) = x / sqrt(x^2 + (p(x^2) / q(x^2))^32) where p and q are polynomials.
/// This provides ~48 bits of precision (~14 significant digits) across the entire range
/// with no branch points, replacing the earlier A&amp;S 7.1.26 approximation (~7 digits).
/// </para>
/// <para>
/// The algorithm is branchless and suitable for vectorized computation. It uses 17
/// coefficients, two polynomial evaluations, 5 squarings, one division, and one sqrt.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class CumulativeNormal<T>
    where T : IFloatingPoint<T>
{
    // Coefficients from Laikov (2025), arXiv:2504.05068, supplementary file v1.c, ferf().
    // Stored as exact IEEE 754 bit patterns to avoid decimal→binary round-trip loss.
    // c[0..3]  : pure q-polynomial coefficients
    // c[4]     : first p-polynomial coefficient
    // c[5..16] : interleaved (q, p) coefficient pairs
    private static readonly double s_c00 = BitConverter.Int64BitsToDouble(0x403CB2FF7B83F273);
    private static readonly double s_c01 = BitConverter.Int64BitsToDouble(0x40BE50A62F3A7A4D);
    private static readonly double s_c02 = BitConverter.Int64BitsToDouble(0x411A4315ECBB06C2);
    private static readonly double s_c03 = BitConverter.Int64BitsToDouble(0x4173D88BBAE026B6);
    private static readonly double s_c04 = BitConverter.Int64BitsToDouble(0x41657CC4BDE32AAF);
    private static readonly double s_c05 = BitConverter.Int64BitsToDouble(0x41B69EE60F7A66C0);
    private static readonly double s_c06 = BitConverter.Int64BitsToDouble(0x41AD43852D82E0E4);
    private static readonly double s_c07 = BitConverter.Int64BitsToDouble(0x41F5248E8C642C98);
    private static readonly double s_c08 = BitConverter.Int64BitsToDouble(0x41F0CFEDDB1CB1FB);
    private static readonly double s_c09 = BitConverter.Int64BitsToDouble(0x4224CB0372DE3573);
    private static readonly double s_c10 = BitConverter.Int64BitsToDouble(0x422145A5D6A78078);
    private static readonly double s_c11 = BitConverter.Int64BitsToDouble(0x4252D242E3B6388A);
    private static readonly double s_c12 = BitConverter.Int64BitsToDouble(0x425126882C83534F);
    private static readonly double s_c13 = BitConverter.Int64BitsToDouble(0x4270098836DDA156);
    private static readonly double s_c14 = BitConverter.Int64BitsToDouble(0x426DA78609B5DD31);
    private static readonly double s_c15 = BitConverter.Int64BitsToDouble(0x428CF4591BF6EAB5);
    private static readonly double s_c16 = BitConverter.Int64BitsToDouble(0x428CBC9A8F83AC35);

    private static readonly double s_reciprocalSqrt2 = 1.0 / Math.Sqrt(2.0);

    /// <summary>
    /// Returns the probability that a standard normal random variable is less than
    /// or equal to <paramref name="x"/> — that is, N(x) = P(Z &lt;= x).
    /// </summary>
    /// <param name="x">The upper integration limit of the standard normal density.</param>
    /// <returns>A value in [0, 1] representing the cumulative probability.</returns>
    public static T Evaluate(T x)
    {
        var xd = double.CreateChecked(x);
        return T.CreateChecked(0.5 * (1.0 + Erf(xd * s_reciprocalSqrt2)));
    }

    /// <summary>
    /// Computes erf(x) using Laikov's branchless exponential-free approximation.
    /// Structure: erf(x) = x / sqrt(x^2 + (p(x^2)/q(x^2))^32).
    /// </summary>
    private static double Erf(double x)
    {
        var z = x * x;

        // q(z): degree-10 polynomial, coefficients at indices 0,1,2,3 then odd indices 5,7,9,11,13,15
        var q = z + s_c00;
        q = q * z + s_c01;
        q = q * z + s_c02;
        q = q * z + s_c03;

        // p(z): degree-6 polynomial, coefficients at index 4 then even indices 6,8,10,12,14,16
        var p = s_c04;

        q = q * z + s_c05;
        p = p * z + s_c06;

        q = q * z + s_c07;
        p = p * z + s_c08;

        q = q * z + s_c09;
        p = p * z + s_c10;

        q = q * z + s_c11;
        p = p * z + s_c12;

        q = q * z + s_c13;
        p = p * z + s_c14;

        q = q * z + s_c15;
        p = p * z + s_c16;

        // phi = (p/q)^32
        p /= q;
        p *= p; // ^2
        p *= p; // ^4
        p *= p; // ^8
        p *= p; // ^16
        p *= p; // ^32

        // erf(x) = x / sqrt(x^2 + phi)
        return x / Math.Sqrt(z + p);
    }
}

/// <summary>
/// Legacy concrete-typed facade forwarding to <see cref="CumulativeNormal{T}"/>
/// instantiated at <c>double</c>. Preserves the pre-migration public API.
/// </summary>
public static class CumulativeNormal
{
    /// <inheritdoc cref="CumulativeNormal{T}.Evaluate"/>
    public static double Evaluate(double x)
        => CumulativeNormal<double>.Evaluate(x);
}
