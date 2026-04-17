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

namespace Boutquin.Numerics.MonteCarlo;

/// <summary>
/// Halton low-discrepancy sequence in arbitrary dimension. Each dimension
/// uses a different prime as the radix for the radical-inverse function.
/// Star-discrepancy O((log N)^d / N) — substantially faster integration
/// convergence than IID Monte Carlo for low-to-moderate dimension (d ≲ 8).
/// </summary>
/// <typeparam name="T">Floating-point type for the output coordinates.</typeparam>
/// <remarks>
/// <para>
/// Reference: Halton, J. H. (1960). "On the Efficiency of Certain
/// Quasi-Random Sequences of Points in Evaluating Multi-Dimensional
/// Integrals." Numerische Mathematik, 2(1), 84–90.
/// </para>
/// <para>
/// Above ~8 dimensions Halton suffers from correlation between higher-
/// indexed coordinates (large prime radices); use <see cref="SobolSequence{T}"/>
/// instead.
/// </para>
/// <para>
/// Tier C: Primes designed for double lattice; cast to/from double internally.
/// </para>
/// </remarks>
public sealed class HaltonSequence<T>
    where T : IFloatingPoint<T>
{
    private static readonly int[] s_primes = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
        157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    ];

    private readonly int _dimension;
    private long _index;

    /// <summary>Creates a Halton sequence over the given dimension.</summary>
    /// <param name="dimension">Number of dimensions (1 ≤ d ≤ 50).</param>
    /// <param name="skip">Number of leading points to skip (default 0). Skipping the first ~64 points is a common QMC practice to reduce low-index correlation artifacts.</param>
    public HaltonSequence(int dimension, long skip = 0)
    {
        if (dimension < 1 || dimension > s_primes.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(dimension), dimension, $"Dimension must lie in [1, {s_primes.Length}].");
        }

        if (skip < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(skip), skip, "Skip must be non-negative.");
        }

        _dimension = dimension;
        _index = skip;
    }

    /// <summary>Dimension of each sample.</summary>
    public int Dimension => _dimension;

    /// <summary>
    /// Returns the next d-dimensional Halton point, with each coordinate in [0, 1).
    /// </summary>
    public T[] Next()
    {
        _index++;
        var result = new T[_dimension];
        for (var d = 0; d < _dimension; d++)
        {
            result[d] = T.CreateChecked(RadicalInverse(_index, s_primes[d]));
        }

        return result;
    }

    private static double RadicalInverse(long n, int b)
    {
        var f = 1.0 / b;
        var r = 0.0;
        while (n > 0)
        {
            r += f * (n % b);
            n /= b;
            f /= b;
        }

        return r;
    }
}

/// <summary>
/// Halton low-discrepancy sequence in arbitrary dimension. Each dimension
/// uses a different prime as the radix for the radical-inverse function.
/// Star-discrepancy O((log N)^d / N) — substantially faster integration
/// convergence than IID Monte Carlo for low-to-moderate dimension (d ≲ 8).
/// </summary>
/// <remarks>
/// <para>
/// Reference: Halton, J. H. (1960). "On the Efficiency of Certain
/// Quasi-Random Sequences of Points in Evaluating Multi-Dimensional
/// Integrals." Numerische Mathematik, 2(1), 84–90.
/// </para>
/// <para>
/// Above ~8 dimensions Halton suffers from correlation between higher-
/// indexed coordinates (large prime radices); use <see cref="SobolSequence"/>
/// instead.
/// </para>
/// <para>
/// Tier C: Delegates to <see cref="HaltonSequence{T}"/> with T = <see cref="double"/>.
/// </para>
/// </remarks>
public sealed class HaltonSequence
{
    private readonly HaltonSequence<double> _impl;

    /// <summary>Creates a Halton sequence over the given dimension.</summary>
    /// <param name="dimension">Number of dimensions (1 ≤ d ≤ 50).</param>
    /// <param name="skip">Number of leading points to skip (default 0). Skipping the first ~64 points is a common QMC practice to reduce low-index correlation artifacts.</param>
    public HaltonSequence(int dimension, long skip = 0)
    {
        _impl = new HaltonSequence<double>(dimension, skip);
    }

    /// <summary>Dimension of each sample.</summary>
    public int Dimension => _impl.Dimension;

    /// <summary>
    /// Returns the next d-dimensional Halton point, with each coordinate in [0, 1).
    /// </summary>
    public double[] Next() => _impl.Next();
}
