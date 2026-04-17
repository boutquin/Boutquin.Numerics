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
/// xoshiro256** — 256-bit state, 64-bit output. Period 2^256 − 1. Passes
/// BigCrush. Marginally faster than PCG-64 on modern CPUs; slightly lower
/// equidistribution (2-dimensional vs PCG's stream-independence guarantee).
/// </summary>
/// <remarks>
/// <para>
/// Reference: Blackman, D. &amp; Vigna, S. (2018). "Scrambled Linear
/// Pseudorandom Number Generators." arXiv:1805.01407.
/// </para>
/// <para>
/// Seeds are expanded from a single 64-bit value via SplitMix64 (Vigna's
/// recommended seed-expansion routine) so that zero and low-weight seeds
/// still produce full-quality state. The <see cref="Jump"/> method advances
/// the generator by 2^128 steps, partitioning the output stream for parallel
/// simulation without correlation.
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
    /// spawn independent streams for parallel Monte Carlo: the parent thread
    /// seeds, each child thread <see cref="Jump"/>s before consuming, and the
    /// non-overlap of the 2^128-period segments guarantees statistical
    /// independence up to the full <c>2^256 − 1</c> period.
    /// </summary>
    /// <remarks>
    /// The four-word jump polynomial below is the characteristic polynomial of
    /// the xoshiro256 LFSR evaluated at <c>T^(2^128)</c>, reducing 2^128
    /// individual state transitions to a single matrix-vector multiply in
    /// GF(2). The constants are from Blackman &amp; Vigna 2018 (arXiv:1805.01407);
    /// changing them invalidates the 2^128 advance and must not be done.
    /// </remarks>
    public void Jump()
    {
        // Jump polynomial from Blackman-Vigna 2018, arXiv:1805.01407.
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
    /// spawn independent streams for parallel Monte Carlo: the parent thread
    /// seeds, each child thread <see cref="Jump"/>s before consuming, and the
    /// non-overlap of the 2^128-period segments guarantees statistical
    /// independence up to the full <c>2^256 − 1</c> period.
    /// </summary>
    public void Jump() => _inner.Jump();
}
