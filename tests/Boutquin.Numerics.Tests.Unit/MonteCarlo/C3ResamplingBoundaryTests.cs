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

using System.Reflection;
using Boutquin.Numerics.MonteCarlo;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class C3ResamplingBoundaryTests
{
    [Fact]
    public void Sobol_EmbeddedSeedsAndPolynomials_SatisfyFiniteFieldConditions()
    {
        var field = typeof(SobolSequence<double>).GetField("s_directionTable", BindingFlags.Static | BindingFlags.NonPublic)!;
        var table = ((int S, uint A, uint[] M)[])field.GetValue(null)!;
        foreach (var (degree, a, seeds) in table.Skip(1))
        {
            seeds.Length.Should().Be(degree);
            for (var i = 0; i < degree; i++)
            {
                (seeds[i] & 1u).Should().Be(1u);
                seeds[i].Should().BeLessThan(1u << (i + 1));
            }
            // A primitive degree-s GF(2) polynomial gives x exact order2^s-1.
            // Exhaust finite field multiplication; independent of Sobol recurrence.
            var polynomial = (1u << degree) | (a << 1) | 1u;
            var residue = 1u;
            for (var power = 1; power <= (1 << degree) - 1; power++)
            {
                residue <<= 1;
                if ((residue & (1u << degree)) != 0)
                {
                    residue ^= polynomial;
                }
                if (power < (1 << degree) - 1)
                {
                    residue.Should().NotBe(1u);
                }
            }
            residue.Should().Be(1u);
        }
    }

    [Fact]
    public void PolitisWhite_ExactCap_OverridesLargeCubicOptimum()
    {
        // n64, fallbackm12, cappedM13: G=-47595/6656,g=-881/6656.
        // b³=144978177600/776161>22³. ceil(min(24,64/3))=22.
        double[] pattern = [3, 4, -3, 2, 4, 3, -3, 0];
        PolitisWhiteBlockLength.EstimateDouble(Enumerable.Range(0, 64).Select(i => pattern[i % 8]).ToArray())
            .Should().Be(22);
    }

    [Fact]
    public void Percentile_Ordering_PreconditionRejectsUnsorted()
    {
        Action invalid = () => Percentile.Compute<decimal>([1m, 4m, 2m], .5);
        invalid.Should().Throw<ArgumentException>();
    }
}
