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

public sealed class C3PcgAuthorControls
{
    [Fact]
    public void DefaultZeroSeedMatchesCanonicalOneseq()
    {
        // Oracle: independently compiled pcg-c author header, including oneseq
        // initialization and the full 128-bit state. No library output captured.
        ulong[] expected = [0x01070196e695f8f1UL, 0x703ec840c59f4493UL, 0xe54954914b3a44faUL, 0x96130ff204b9285eUL, 0x7d9fdef535ceb21aUL, 0x666feed42e1219a0UL];
        var rng = new Pcg64RandomSource(0UL);
        foreach (var word in expected)
        {
            rng.NextULong().Should().Be(word);
        }
    }

    [Fact]
    public void DefaultSeed42MatchesCanonicalOneseq()
    {
        // Oracle: independently compiled pcg-c author header, including oneseq
        // initialization and the full 128-bit state. No library output captured.
        ulong[] expected = [0x287472e87ff5705aUL, 0xbbd190b04ed0b545UL, 0xb6cee3580db14880UL, 0xbf5f7d7e4c3d1864UL, 0x734eedbe7e50bbc5UL, 0xa5b6b5f867691c77UL];
        var rng = new Pcg64RandomSource(42UL);
        foreach (var word in expected)
        {
            rng.NextULong().Should().Be(word);
        }
    }

    [Fact]
    public void WideStreamIdentifierRetainsTheCarryBit()
    {
        // Oracle: independently compiled pcg-c author header, including oneseq
        // initialization and the full 128-bit state. No library output captured.
        ulong[] expected = [0x1f40946abc4fc90eUL, 0xc6497ef2b8d2a700UL, 0xfd6946ace7a4bf55UL, 0xcf7d386f0578f59bUL, 0x3fe3c2c6f4b4d1bfUL, 0x02dd869b63f6534aUL];
        var rng = new Pcg64RandomSource(42UL, ulong.MaxValue);
        foreach (var word in expected)
        {
            rng.NextULong().Should().Be(word);
        }
    }

    [Fact]
    public void ZeroOutputRotationMatchesCanonicalFold()
    {
        // Oracle: independently compiled pcg-c author header, including oneseq
        // initialization and the full 128-bit state. No library output captured.
        ulong[] expected = [0xd46d27beb3c424eaUL, 0x14fea5fc7a70798fUL, 0x2328d0f44b68d778UL, 0xc11e1742b68abe5fUL, 0x3dd14e24ec596b65UL, 0xde11ef08b448ce61UL];
        var rng = new Pcg64RandomSource(9UL, 54UL);
        foreach (var word in expected)
        {
            rng.NextULong().Should().Be(word);
        }
    }

}
