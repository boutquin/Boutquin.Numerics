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
/// Sobol low-discrepancy sequence with the Joe-Kuo (2008) direction
/// numbers. Star-discrepancy O((log N)^d / N) — the same asymptotic rate
/// as Halton but with much better behavior in higher dimensions.
/// Recommended for QMC integration when d &gt; 8.
/// </summary>
/// <typeparam name="T">Floating-point type for the output coordinates.</typeparam>
/// <remarks>
/// <para>
/// Reference: Joe, S. &amp; Kuo, F. Y. (2008). "Constructing Sobol Sequences
/// with Better Two-Dimensional Projections." SIAM Journal on Scientific
/// Computing, 30(5), 2635–2654.
/// </para>
/// <para>
/// This implementation supports up to 32 dimensions out-of-the-box using
/// the embedded Joe-Kuo direction numbers. The first dimension uses the
/// trivial van der Corput sequence in base 2; higher dimensions use the
/// recurrence
/// <c>m_k = a_1·m_{k−1} ⊕ … ⊕ a_{s−1}·m_{k−s+1} ⊕ m_{k−s}·2^s ⊕ m_{k−s}</c>
/// for the appropriate primitive polynomial.
/// </para>
/// <para>
/// Tier C: Direction numbers designed for double lattice; cast to/from double internally.
/// </para>
/// </remarks>
public sealed class SobolSequence<T>
    where T : IFloatingPoint<T>
{
    // Joe-Kuo direction numbers for dimensions 1..32. Each row encodes:
    //  [degree s, polynomial coefficients (s−1 bits as integer), m_1, m_2, …, m_s]
    // We embed a compact subset; full Joe-Kuo tables go up to 21201 dimensions.
    private static readonly (int S, uint A, uint[] M)[] s_directionTable =
    [
        (0, 0u, []),                                 // dim 1 — handled directly via van der Corput.
        (1, 0u, [1u]),                                // dim 2
        (2, 1u, [1u, 3u]),                            // dim 3
        (3, 1u, [1u, 3u, 1u]),                        // dim 4
        (3, 2u, [1u, 1u, 1u]),                        // dim 5
        (4, 1u, [1u, 1u, 3u, 3u]),                    // dim 6
        (4, 4u, [1u, 3u, 5u, 13u]),                   // dim 7
        (5, 2u, [1u, 1u, 5u, 5u, 17u]),               // dim 8
        (5, 4u, [1u, 1u, 5u, 5u, 5u]),                // dim 9
        (5, 7u, [1u, 1u, 7u, 11u, 19u]),              // dim 10
        (5, 11u, [1u, 1u, 5u, 1u, 1u]),               // dim 11
        (5, 13u, [1u, 1u, 1u, 3u, 11u]),              // dim 12
        (5, 14u, [1u, 3u, 5u, 5u, 31u]),              // dim 13
        (6, 1u, [1u, 3u, 3u, 9u, 7u, 49u]),           // dim 14
        (6, 13u, [1u, 1u, 1u, 15u, 21u, 21u]),        // dim 15
        (6, 16u, [1u, 3u, 1u, 13u, 27u, 49u]),        // dim 16
        (6, 19u, [1u, 1u, 1u, 15u, 7u, 5u]),          // dim 17
        (6, 22u, [1u, 3u, 1u, 15u, 13u, 25u]),        // dim 18
        (6, 25u, [1u, 1u, 5u, 5u, 19u, 61u]),         // dim 19
        (7, 1u, [1u, 3u, 7u, 11u, 23u, 15u, 103u]),   // dim 20
        (7, 4u, [1u, 3u, 7u, 13u, 13u, 15u, 69u]),    // dim 21
        (7, 7u, [1u, 1u, 3u, 13u, 7u, 35u, 63u]),     // dim 22
        (7, 8u, [1u, 3u, 5u, 9u, 1u, 25u, 53u]),      // dim 23
        (7, 14u, [1u, 3u, 1u, 13u, 9u, 35u, 107u]),   // dim 24
        (7, 19u, [1u, 3u, 1u, 5u, 27u, 61u, 31u]),    // dim 25
        (7, 21u, [1u, 1u, 5u, 11u, 19u, 41u, 61u]),   // dim 26
        (7, 28u, [1u, 3u, 5u, 3u, 3u, 13u, 69u]),     // dim 27
        (7, 31u, [1u, 1u, 7u, 13u, 1u, 19u, 1u]),     // dim 28
        (7, 32u, [1u, 3u, 7u, 5u, 13u, 19u, 59u]),    // dim 29
        (7, 37u, [1u, 1u, 3u, 9u, 25u, 29u, 41u]),    // dim 30
        (7, 41u, [1u, 3u, 5u, 13u, 23u, 1u, 55u]),    // dim 31
        (7, 42u, [1u, 1u, 3u, 1u, 13u, 41u, 17u]),    // dim 32
    ];

    private const int Bits = 30; // resolution: 2^30 ≈ 1.07e9 distinct points.

    private readonly int _dimension;
    private readonly uint[][] _directions;
    private readonly uint[] _x;
    private uint _index;

    /// <summary>Creates a Sobol sequence over the given dimension.</summary>
    /// <param name="dimension">Number of dimensions (1 ≤ d ≤ 32).</param>
    /// <param name="skip">Number of leading points to skip. Recommended: power of two ≥ 64.</param>
    public SobolSequence(int dimension, long skip = 0)
    {
        if (dimension < 1 || dimension > s_directionTable.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(dimension), dimension,
                $"Dimension must lie in [1, {s_directionTable.Length}].");
        }

        if (skip < 0 || skip > uint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(skip), skip, "Skip must be in [0, 2^32).");
        }

        _dimension = dimension;
        _directions = new uint[dimension][];
        for (var d = 0; d < dimension; d++)
        {
            _directions[d] = BuildDirectionNumbers(d);
        }

        _x = new uint[dimension];
        _index = 0;
        for (var i = 0L; i < skip; i++)
        {
            Advance();
        }
    }

    /// <summary>Number of dimensions.</summary>
    public int Dimension => _dimension;

    /// <summary>Returns the next d-dimensional Sobol point in [0, 1).</summary>
    public T[] Next()
    {
        Advance();
        var result = new T[_dimension];
        for (var d = 0; d < _dimension; d++)
        {
            result[d] = T.CreateChecked(_x[d] / (double)(1u << Bits));
        }

        return result;
    }

    private void Advance()
    {
        _index++;
        // Find the rightmost zero bit of (_index − 1), 1-based.
        var c = 1;
        var value = _index - 1;
        while ((value & 1u) == 1u)
        {
            value >>= 1;
            c++;
        }

        if (c > Bits)
        {
            throw new InvalidOperationException(
                $"Sobol sequence index exceeded {1u << Bits} points.");
        }

        for (var d = 0; d < _dimension; d++)
        {
            _x[d] ^= _directions[d][c - 1];
        }
    }

    private static uint[] BuildDirectionNumbers(int dimensionIndex)
    {
        var dir = new uint[Bits];
        if (dimensionIndex == 0)
        {
            // Van der Corput in base 2: V_i = 1 << (Bits − i).
            for (var i = 0; i < Bits; i++)
            {
                dir[i] = 1u << (Bits - 1 - i);
            }

            return dir;
        }

        var (s, a, m) = s_directionTable[dimensionIndex];
        for (var i = 0; i < s; i++)
        {
            dir[i] = m[i] << (Bits - 1 - i);
        }

        for (var i = s; i < Bits; i++)
        {
            var v = dir[i - s] ^ (dir[i - s] >> s);
            for (var k = 1; k <= s - 1; k++)
            {
                if (((a >> (s - 1 - k)) & 1u) != 0u)
                {
                    v ^= dir[i - k];
                }
            }

            dir[i] = v;
        }

        return dir;
    }
}

/// <summary>
/// Sobol low-discrepancy sequence with the Joe-Kuo (2008) direction
/// numbers. Star-discrepancy O((log N)^d / N) — the same asymptotic rate
/// as Halton but with much better behavior in higher dimensions.
/// Recommended for QMC integration when d &gt; 8.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Joe, S. &amp; Kuo, F. Y. (2008). "Constructing Sobol Sequences
/// with Better Two-Dimensional Projections." SIAM Journal on Scientific
/// Computing, 30(5), 2635–2654.
/// </para>
/// <para>
/// This implementation supports up to 32 dimensions out-of-the-box using
/// the embedded Joe-Kuo direction numbers. The first dimension uses the
/// trivial van der Corput sequence in base 2; higher dimensions use the
/// recurrence
/// <c>m_k = a_1·m_{k−1} ⊕ … ⊕ a_{s−1}·m_{k−s+1} ⊕ m_{k−s}·2^s ⊕ m_{k−s}</c>
/// for the appropriate primitive polynomial.
/// </para>
/// <para>
/// Tier C: Delegates to <see cref="SobolSequence{T}"/> with T = <see cref="double"/>.
/// </para>
/// </remarks>
public sealed class SobolSequence
{
    private readonly SobolSequence<double> _impl;

    /// <summary>Creates a Sobol sequence over the given dimension.</summary>
    /// <param name="dimension">Number of dimensions (1 ≤ d ≤ 32).</param>
    /// <param name="skip">Number of leading points to skip. Recommended: power of two ≥ 64.</param>
    public SobolSequence(int dimension, long skip = 0)
    {
        _impl = new SobolSequence<double>(dimension, skip);
    }

    /// <summary>Number of dimensions.</summary>
    public int Dimension => _impl.Dimension;

    /// <summary>Returns the next d-dimensional Sobol point in [0, 1).</summary>
    public double[] Next() => _impl.Next();
}
