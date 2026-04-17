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

namespace Boutquin.Numerics.Random;

/// <summary>
/// PCG-XSL-RR 128/64 — permuted congruential generator with 128-bit state
/// and 64-bit output. Period 2^128, k-dimensionally equidistributed to k = 2,
/// passes BigCrush. The 128-bit state is advanced by a linear congruence;
/// each output applies an xorshift-low + right-rotate permutation to
/// the high-order bits.
/// </summary>
/// <remarks>
/// <para>
/// Reference: O'Neill, M. E. (2014). "PCG: A Family of Simple Fast
/// Space-Efficient Statistically Good Algorithms for Random Number
/// Generation." Technical Report HMC-CS-2014-0905, Harvey Mudd College.
/// </para>
/// <para>
/// Uses the standard default multiplier and increment constants from the
/// reference C++ implementation. Two generators constructed with different
/// <c>streamId</c> values produce statistically independent streams from
/// the same seed — useful for parallel Monte Carlo with reproducible output.
/// </para>
/// </remarks>
public sealed class Pcg64RandomSource : IRandomSource
{
    // Default multiplier from the reference PCG C++ library.
    // M = 47026247687942121848144207491837523525
    private const ulong MultiplierHigh = 2549297995355413924UL;
    private const ulong MultiplierLow = 4865540595714422341UL;

    // Default increment (odd). Only the low 127 bits matter; the low bit is fixed at 1.
    // INC = 117397592171526113268558934119004209487
    private const ulong DefaultIncrementHigh = 6364136223846793005UL;
    private const ulong DefaultIncrementLow = 1442695040888963407UL;

    private UInt128 _state;
    private readonly UInt128 _increment;

    /// <summary>
    /// Initializes a new generator seeded with <paramref name="seed"/> on the
    /// default stream.
    /// </summary>
    public Pcg64RandomSource(ulong seed)
        : this(seed, streamId: 0UL)
    {
    }

    /// <summary>
    /// Initializes a new generator with an explicit stream identifier.
    /// Different <paramref name="streamId"/> values yield independent
    /// streams from the same seed.
    /// </summary>
    /// <param name="seed">Seed material. Any 64-bit value, including zero, is valid.</param>
    /// <param name="streamId">Stream identifier. The low bit is shifted left and OR'd with 1 to form an odd increment.</param>
    public Pcg64RandomSource(ulong seed, ulong streamId)
    {
        var inc = streamId == 0UL
            ? new UInt128(DefaultIncrementHigh, DefaultIncrementLow)
            : new UInt128(0UL, streamId) << 1 | UInt128.One;
        _increment = inc;
        _state = UInt128.Zero;
        _ = NextULong();
        _state += new UInt128(0UL, seed);
        _ = NextULong();
    }

    /// <inheritdoc />
    public ulong NextULong()
    {
        var oldState = _state;
        // state = state * M + inc  (mod 2^128)
        _state = oldState * new UInt128(MultiplierHigh, MultiplierLow) + _increment;

        // XSL-RR permutation: fold the top 64 bits into the low 64 with xor,
        // then rotate right by the top 6 bits of the state.
        var xorshifted = (ulong)(oldState >> 64) ^ (ulong)oldState;
        var rot = (int)((oldState >> 122) & (UInt128)0x3FUL);
        return BitOperations.RotateRight(xorshifted, rot);
    }
}

/// <summary>
/// Generic PCG-XSL-RR 128/64 random source with type-parameterized floating-point output.
/// </summary>
/// <typeparam name="T">The floating-point type for random values.</typeparam>
/// <remarks>
/// Tier C: Wraps <see cref="Pcg64RandomSource"/> and casts from double.
/// </remarks>
public sealed class Pcg64RandomSource<T> : IRandomSource<T>
    where T : IFloatingPoint<T>
{
    private readonly Pcg64RandomSource _inner;

    /// <summary>
    /// Initializes a new generator seeded with <paramref name="seed"/> on the
    /// default stream.
    /// </summary>
    public Pcg64RandomSource(ulong seed)
    {
        _inner = new Pcg64RandomSource(seed);
    }

    /// <summary>
    /// Initializes a new generator with an explicit stream identifier.
    /// Different <paramref name="streamId"/> values yield independent
    /// streams from the same seed.
    /// </summary>
    /// <param name="seed">Seed material. Any 64-bit value, including zero, is valid.</param>
    /// <param name="streamId">Stream identifier. The low bit is shifted left and OR'd with 1 to form an odd increment.</param>
    public Pcg64RandomSource(ulong seed, ulong streamId)
    {
        _inner = new Pcg64RandomSource(seed, streamId);
    }

    /// <inheritdoc />
    public ulong NextULong() => _inner.NextULong();
}
