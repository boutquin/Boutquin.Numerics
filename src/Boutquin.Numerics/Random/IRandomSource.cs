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
/// Deterministic pseudo-random number source. Implementations produce
/// reproducible output given a fixed seed, independent of .NET runtime
/// version. Intended for Monte Carlo, bootstrap, and any other code that
/// needs cross-process / cross-runtime reproducibility.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="System.Random"/>'s internal state is documented as reproducible
/// within a given .NET runtime but is not guaranteed stable across major
/// versions. Use <see cref="IRandomSource"/> when simulation output must be
/// byte-identical across environments (golden test vectors, reproducible
/// research, cross-language verification).
/// </para>
/// <para>
/// Implementations are NOT thread-safe unless they explicitly document
/// otherwise. Instantiate one source per thread, or use
/// <see cref="Xoshiro256StarStarRandomSource.Jump"/> to partition the
/// stream across parallel workers.
/// </para>
/// </remarks>
public interface IRandomSource
{
    /// <summary>Generates the next 64-bit unsigned value.</summary>
    ulong NextULong();

    /// <summary>Generates the next 32-bit unsigned value.</summary>
    uint NextUInt() => (uint)(NextULong() >> 32);

    /// <summary>
    /// Generates a uniform <see cref="double"/> in [0, 1). Uses the top 53
    /// bits of the next 64-bit output as the mantissa.
    /// </summary>
    double NextDouble() => (NextULong() >> 11) * (1.0 / (1UL << 53));

    /// <summary>
    /// Generates a uniform integer in [0, <paramref name="upperExclusive"/>).
    /// Uses Lemire's nearly-divisionless rejection sampling over a 32-bit
    /// uniform (arXiv:1805.10941).
    /// </summary>
    int NextInt(int upperExclusive)
    {
        if (upperExclusive <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(upperExclusive), upperExclusive, "Upper bound must be positive.");
        }

        var range = (uint)upperExclusive;
        var x = NextUInt();
        var m = (ulong)x * range;
        var l = (uint)m;
        if (l < range)
        {
            var t = (uint)-(int)range % range;
            while (l < t)
            {
                x = NextUInt();
                m = (ulong)x * range;
                l = (uint)m;
            }
        }

        return (int)(m >> 32);
    }
}

/// <summary>
/// Generic pseudo-random number source that extends <see cref="IRandomSource"/>
/// with type-parameterized floating-point output.
/// </summary>
/// <typeparam name="T">The floating-point type for random values.</typeparam>
/// <remarks>
/// Tier C: Delegates to <see cref="IRandomSource.NextDouble"/> and casts to T.
/// </remarks>
public interface IRandomSource<T> : IRandomSource
    where T : IFloatingPoint<T>
{
    /// <summary>
    /// Generates a uniform value of type <typeparamref name="T"/> in [0, 1).
    /// </summary>
    T NextT() => T.CreateChecked(NextDouble());
}
