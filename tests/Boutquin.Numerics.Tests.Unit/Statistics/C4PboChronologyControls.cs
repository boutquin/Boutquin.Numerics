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
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

[Trait("Tier", "Unit")]
public sealed class C4PboChronologyControls
{
    [Theory]
    [InlineData(0, 0, 1)]
    [InlineData(1, 0, 2)]
    [InlineData(2, 0, 3)]
    [InlineData(3, 1, 2)]
    [InlineData(4, 1, 3)]
    [InlineData(5, 2, 3)]
    public void FourBlockTrainingSubsets_RetainOriginalBlockOrder(int index, int first, int second)
    {
        // Independent exhaustive 4 choose 2 list, not a second enumerator.
        var subsets = ProductionCombinations(4, 2);
        subsets.Should().HaveCount(6);
        subsets[index].Should().Equal(first, second);
    }

    [Theory]
    [InlineData(0, 1, 2, 3)]
    [InlineData(0, 2, 1, 3)]
    [InlineData(0, 3, 1, 2)]
    [InlineData(1, 2, 0, 3)]
    [InlineData(1, 3, 0, 2)]
    [InlineData(2, 3, 0, 1)]
    public void FourBlockTestSubsets_RetainOriginalComplementOrder(int a, int b, int first, int second)
    {
        ProductionComplement([a, b], 4).Should().Equal(first, second);
    }

    [Fact]
    public void SixBlockHeldOutSplit_PreservesChronologyWithGaps()
    {
        // A held-out nonadjacent split from the 20 six-block combinations.
        // Its three-row blocks represent training row ordinals
        // [3,4,5,9,10,11,15,16,17], test [0,1,2,6,7,8,12,13,14].
        // JoinedSharpe's inspected ascending block/within-block loops consume these
        // orders. This tests actual helper order, not an arbitrary-metric API.
        var subsets = ProductionCombinations(6, 3);
        subsets.Should().HaveCount(20);
        subsets[14].Should().Equal(1, 3, 5);
        ProductionComplement(subsets[14], 6).Should().Equal(0, 2, 4);
        // Complement must restore original order independently of subset order.
        ProductionComplement([5, 1, 3], 6).Should().Equal(0, 2, 4);
    }

    private static List<int[]> ProductionCombinations(int n, int k)
    {
        var method = typeof(ProbabilityOfBacktestOverfitting<double>).GetMethod(
            "Combinations", BindingFlags.Static | BindingFlags.NonPublic)!;
        return (List<int[]>)method.Invoke(null, new object[] { n, k })!;
    }

    private static int[] ProductionComplement(int[] subset, int n)
    {
        var method = typeof(ProbabilityOfBacktestOverfitting<double>).GetMethod(
            "Complement", BindingFlags.Static | BindingFlags.NonPublic)!;
        return (int[])method.Invoke(null, new object[] { subset, n })!;
    }
}
