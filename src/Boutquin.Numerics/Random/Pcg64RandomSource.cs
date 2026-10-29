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
/// with reference-family empirical test results reported by O'Neill. This
/// implementation has not independently reproduced BigCrush. The state follows a linear congruence;
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
/// reference C++ implementation. Each draw advances the state and then
/// permutes the advanced state, as the reference C and C++ libraries do for
/// pcg64, so a <c>(seed, streamId)</c> pair with a non-zero <c>streamId</c>
/// reproduces <c>pcg64_srandom_r(seed, streamId)</c> output. Two generators constructed with different
/// non-zero <c>streamId</c> values select different odd increments from
/// the same seed. Different increments do not constitute a proof of statistical independence.
/// </para>
/// </remarks>
public sealed class Pcg64RandomSource : IRandomSource
{
    // Default 128-bit multiplier from the reference PCG C++ library (imneme/pcg-cpp,
    // PCG_128BIT_CONSTANT(2549297995355413924ULL, 4865540595714422341ULL)).
    // M = (high << 64) + low = 47026247687942121848144207491837523525.
    private const ulong MultiplierHigh = 2549297995355413924UL;
    private const ulong MultiplierLow = 4865540595714422341UL;

    // Default 128-bit increment from the reference PCG C++ library
    // (PCG_128BIT_CONSTANT(6364136223846793005ULL, 1442695040888963407ULL)).
    // INC = (high << 64) + low = 117397592171526113268558934119004209487.
    // NB: the high word 6364136223846793005 is the canonical PCG default-increment
    // HIGH word here — it is NOT the LCG multiplier (which appears in the separate
    // 64-bit pcg32 variant). The low word 1442695040888963407 is the classic odd
    // increment; together they are the verified pcg-cpp default increment, NOT
    // swapped. The constant is odd (low bit of the low word is 1), as required.
    private const ulong DefaultIncrementHigh = 6364136223846793005UL;
    private const ulong DefaultIncrementLow = 1442695040888963407UL;

    private UInt128 _state;
    private readonly UInt128 _increment;

    /// <summary>
    /// Initializes a new generator seeded with <paramref name="seed"/> on the
    /// default stream. This is exactly equivalent to
    /// <see cref="Pcg64RandomSource(ulong, ulong)"/> with <c>streamId = 0</c> —
    /// both use the full 128-bit default increment. Two generators created with
    /// the same seed via this convenience constructor and via the explicit
    /// constructor with <c>streamId = 0</c> therefore produce the <b>same</b>
    /// stream, not independent ones.
    /// </summary>
    public Pcg64RandomSource(ulong seed)
        : this(seed, streamId: 0UL)
    {
    }

    /// <summary>
    /// Initializes a new generator with an explicit stream identifier.
    /// Distinct non-zero <paramref name="streamId"/> values select different
    /// odd increments from the same seed; statistical independence is not guaranteed.
    /// </summary>
    /// <param name="seed">Seed material. Any 64-bit value, including zero, is valid.</param>
    /// <param name="streamId">Stream identifier. <b><c>streamId == 0</c> is the default stream</b> —
    /// the identical increment used by the parameterless-stream constructor <see cref="Pcg64RandomSource(ulong)"/>,
    /// <b>not</b> a distinct stream. It uses the full 128-bit default increment, but a generator with
    /// <c>streamId = 0</c> collides exactly with one built via the convenience constructor for the same seed.
    /// To assign different increments to parallel streams, give each worker
    /// a <b>distinct non-zero</b> id (reserve 0 for the single default stream), or use
    /// <see cref="Xoshiro256StarStarRandomSource"/> with <c>Jump()</c>. Non-zero values derive an odd
    /// increment of up to 65 bits, stored in <see cref="UInt128"/>: the full identifier is widened,
    /// shifted left by one bit, and OR'd with 1.</param>
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
        // state = state * M + inc  (mod 2^128)
        var state = _state * new UInt128(MultiplierHigh, MultiplierLow) + _increment;
        _state = state;

        // XSL-RR permutation of the NEW state (step, then output — pcg-c
        // pcg_setseq_128_xsl_rr_64_random_r): fold the top 64 bits into the low
        // 64 with xor, then rotate right by the top 6 bits of the state.
        var xorshifted = (ulong)(state >> 64) ^ (ulong)state;
        var rot = (int)((state >> 122) & (UInt128)0x3FUL);
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
    /// default stream. Equivalent to the explicit-stream constructor with
    /// <c>streamId = 0</c> — both use the full 128-bit default increment, so they
    /// produce the <b>same</b> stream for a given seed, not independent ones.
    /// </summary>
    public Pcg64RandomSource(ulong seed)
    {
        _inner = new Pcg64RandomSource(seed);
    }

    /// <summary>
    /// Initializes a new generator with an explicit stream identifier.
    /// Distinct non-zero <paramref name="streamId"/> values select different
    /// odd increments from the same seed; statistical independence is not guaranteed.
    /// </summary>
    /// <param name="seed">Seed material. Any 64-bit value, including zero, is valid.</param>
    /// <param name="streamId">Stream identifier. <b><c>streamId == 0</c> is the default stream</b> —
    /// the identical increment used by <see cref="Pcg64RandomSource{T}(ulong)"/>, <b>not</b> an independent
    /// stream: a generator with <c>streamId = 0</c> collides exactly with the convenience constructor for the
    /// same seed. To assign different increments to parallel streams, give each worker a <b>distinct non-zero</b> id
    /// (reserve 0 for the single default stream), or use <see cref="Xoshiro256StarStarRandomSource"/> with
    /// <c>Jump()</c>. Non-zero values derive an odd increment of up to 65 bits, stored in
    /// <see cref="UInt128"/>: the full identifier is widened, shifted left by one bit,
    /// and OR'd with 1.</param>
    public Pcg64RandomSource(ulong seed, ulong streamId)
    {
        _inner = new Pcg64RandomSource(seed, streamId);
    }

    /// <inheritdoc />
    public ulong NextULong() => _inner.NextULong();
}
