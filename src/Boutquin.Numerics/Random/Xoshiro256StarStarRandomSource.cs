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
/// xoshiro256** — 256-bit state, 64-bit output. The nonzero-state engine has
/// period 2^256 − 1 and is 4-dimensionally equidistributed (Blackman-Vigna 2018).
/// Use <see cref="Jump"/> to place stream starts 2^128 state transitions apart.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Blackman, D. &amp; Vigna, S. (2018). "Scrambled Linear
/// Pseudorandom Number Generators." arXiv:1805.01407.
/// </para>
/// <para>
/// Seeds are expanded from a single 64-bit value via SplitMix64 (Vigna's
/// recommended seed-expansion routine) so that zero and low-weight seeds
/// produce a nonzero state. Only a subset of the 256-bit states is reachable
/// from this 64-bit seed. The authors report empirical test results for their
/// generators; these are not a reproduction of BigCrush by this implementation.
/// The <see cref="Jump"/> method advances the generator by 2^128 steps.
/// Separated, nonoverlapping stream segments do not prove statistical independence.
/// </para>
/// </remarks>
public sealed class Xoshiro256StarStarRandomSource : IRandomSource
{
    private ulong _s0;
    private ulong _s1;
    private ulong _s2;
    private ulong _s3;

    /// <summary>Initializes the generator with the given 64-bit seed.</summary>
    public Xoshiro256StarStarRandomSource(ulong seed)
    {
        // SplitMix64 seed expansion (Vigna).
        var z = seed;
        _s0 = SplitMix64(ref z);
        _s1 = SplitMix64(ref z);
        _s2 = SplitMix64(ref z);
        _s3 = SplitMix64(ref z);
    }

    /// <inheritdoc />
    public ulong NextULong()
    {
        var result = BitOperations.RotateLeft(_s1 * 5UL, 7) * 9UL;

        var t = _s1 << 17;
        _s2 ^= _s0;
        _s3 ^= _s1;
        _s1 ^= _s2;
        _s0 ^= _s3;
        _s2 ^= t;
        _s3 = BitOperations.RotateLeft(_s3, 45);

        return result;
    }

    /// <summary>
    /// Advances the generator state by 2^128 calls to <see cref="NextULong"/>
    /// in O(64²) operations, without producing intermediate output. Used to
    /// place parallel stream starts 2^128 transitions apart. Keep each segment
    /// shorter than that distance and account for wraparound at the full
    /// <c>2^256 − 1</c> period when assigning segments. Nonoverlap does not
    /// establish statistical independence.
    /// </summary>
    /// <remarks>
    /// The four-word jump polynomial encodes <c>x^(2^128)</c> modulo the
    /// characteristic polynomial of the xoshiro256 state transition over GF(2).
    /// Evaluating it at the transition matrix advances the state by 2^128 steps.
    /// The constants come from the authors' <c>xoshiro256starstar.c</c> jump routine;
    /// changing them invalidates the 2^128 advance and must not be done.
    /// </remarks>
    public void Jump()
    {
        // Jump polynomial from the authors' xoshiro256starstar.c.
        // Encodes T^(2^128) in GF(2) where T is the xoshiro256 state-transition
        // matrix; XORing contributions from set bits effects the advance.
        var jump = new ulong[]
        {
            0x180ec6d33cfd0abaUL,
            0xd5a61266f0c9392cUL,
            0xa9582618e03fc9aaUL,
            0x39abdc4529b1661cUL,
        };

        ulong s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (var i = 0; i < jump.Length; i++)
        {
            for (var b = 0; b < 64; b++)
            {
                if ((jump[i] & (1UL << b)) != 0UL)
                {
                    s0 ^= _s0;
                    s1 ^= _s1;
                    s2 ^= _s2;
                    s3 ^= _s3;
                }

                _ = NextULong();
            }
        }

        _s0 = s0;
        _s1 = s1;
        _s2 = s2;
        _s3 = s3;
    }

    private static ulong SplitMix64(ref ulong z)
    {
        z += 0x9e3779b97f4a7c15UL;
        var x = z;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9UL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebUL;
        return x ^ (x >> 31);
    }
}

/// <summary>
/// Generic xoshiro256** random source with type-parameterized floating-point output.
/// </summary>
/// <typeparam name="T">The floating-point type for random values.</typeparam>
/// <remarks>
/// Tier C: Wraps <see cref="Xoshiro256StarStarRandomSource"/> and casts from double.
/// </remarks>
public sealed class Xoshiro256StarStarRandomSource<T> : IRandomSource<T>
    where T : IFloatingPoint<T>
{
    private readonly Xoshiro256StarStarRandomSource _inner;

    /// <summary>Initializes the generator with the given 64-bit seed.</summary>
    public Xoshiro256StarStarRandomSource(ulong seed)
    {
        _inner = new Xoshiro256StarStarRandomSource(seed);
    }

    /// <inheritdoc />
    public ulong NextULong() => _inner.NextULong();

    /// <summary>
    /// Advances the generator state by 2^128 calls to <see cref="IRandomSource.NextULong"/>
    /// in O(64²) operations, without producing intermediate output. Used to
    /// place parallel stream starts 2^128 transitions apart. Keep each segment
    /// shorter than that distance and account for wraparound at the full
    /// <c>2^256 − 1</c> period when assigning segments. Nonoverlap does not
    /// establish statistical independence.
    /// </summary>
    public void Jump() => _inner.Jump();
}
