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

using Boutquin.Numerics.Random;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Random;

/// <summary>
/// Pins <see cref="Pcg64RandomSource"/> to the canonical pcg64 output published
/// with O'Neill's reference C library (pcg-c <c>test-high/expected/check-pcg64.out</c>,
/// "Round 1", generator seeded with <c>pcg64_srandom_r(&amp;rng, 42u, 54u)</c>).
/// pcg-c's <c>pcg_setseq_128_xsl_rr_64_random_r</c> advances the state and then
/// applies the XSL-RR permutation to the <b>new</b> state.
/// </summary>
public sealed class Pcg64RandomSourceOracleTests
{
    [Fact]
    public void NextULong_Seed42Stream54_MatchesPcgCReferenceOutput()
    {
        ulong[] expected =
        [
            0x86b1da1d72062b68UL, 0x1304aa46c9853d39UL, 0xa3670e9e0dd50358UL,
            0xf9090e529a7dae00UL, 0xc85b9fd837996f2cUL, 0x606121f8e3919196UL,
        ];

        var rng = new Pcg64RandomSource(seed: 42UL, streamId: 54UL);

        for (var i = 0; i < expected.Length; i++)
        {
            rng.NextULong().Should().Be(expected[i], "pcg64 draw {0} must match pcg-c check-pcg64.out", i);
        }
    }
}
