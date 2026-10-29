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

using Boutquin.Numerics.LinearAlgebra;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class C2IndependentGaussianPivotTests
{
    [Fact]
    public void CandidateIndex_MapsUniformDrawToInclusiveHalfMaximumRowsOnly()
    {
        decimal[,] augmented = { { 100, 100, 0, 0, 0, 0 }, { 0, 5, 900, 0, 0, 0 },
            { 0, 10, 0, 0, 0, 0 }, { 0, -5, 0, 0, 0, 0 }, { 0, 4, 0, 0, 0, 0 } };
        var method = typeof(GaussianElimination<decimal>).GetMethod("SelectPivotRow", BindingFlags.Static | BindingFlags.NonPublic);
        method.Should().NotBeNull();
        for (var draw = 0; draw < 3; draw++)
        {
            var random = new FixedIndexRandom(draw);
            object?[] arguments = [augmented, 5, 1, random, 1m, 0m];
            object? result = null;
            FluentActions.Invoking(() => result = method!.Invoke(null, arguments)).Should().NotThrow();
            result.Should().Be(draw + 1);
            arguments[5].Should().Be(10m);
            random.Bounds.Should().Equal(3);
        }
    }

    [Fact]
    public void ExactHalfMaximumPivot_AttainsMultiplierTwoAndStageGrowthThree()
    {
        // Pivot1 is exactly half the column maximum2. The next row multiplier
        // is−2 and its trailing entry becomes2−(−2)*2=6, three times max|A|.
        decimal[,] augmented = { { 1, 2, 5 }, { -2, 2, 2 } };
        var random = new FixedIndexRandom(0);
        var actual = Eliminate(augmented, random, 0m);
        actual.Should().Equal(1m, 2m);
        augmented[0, 0].Should().Be(1);
        augmented[0, 1].Should().Be(2);
        augmented[1, 0].Should().Be(0);
        augmented[1, 1].Should().Be(6);
        augmented[1, 2].Should().Be(12);
        random.Bounds.Should().Equal(2);
    }

    [Fact]
    public void SeededPriorPivot_CanChangeLaterSingularityDecision()
    {
        // Input max2 fixes the selected floor at2e−14. First-row pivot gives
        // second diagonal−3e−14 (eligible); second-row pivot gives1.5e−14
        // (rejected). This is exact decimal Schur algebra, not a random oracle.
        decimal[,] a = { { 1, 0.000000000000015m }, { 2, 0 } };
        decimal[] b = [1.000000000000015m, 2m];
        var successes = 0;
        var rejections = 0;
        for (var seed = 0; seed < 32; seed++)
        {
            var expectedIndex = new System.Random(seed).Next(2);
            if (expectedIndex == 0)
            {
                decimal[]? result = null;
                FluentActions.Invoking(() => result = GaussianElimination<decimal>.Solve(a, b, seed)).Should().NotThrow();
                result.Should().Equal(1m, 1m);
                successes++;
            }
            else
            {
                FluentActions.Invoking(() => GaussianElimination<decimal>.Solve(a, b, seed)).Should().Throw<InvalidOperationException>();
                rejections++;
            }
        }
        successes.Should().BePositive();
        rejections.Should().BePositive();
    }

    private static decimal[] Eliminate(decimal[,] augmented, System.Random random, decimal floor)
    {
        var method = typeof(GaussianElimination<decimal>).GetMethod("EliminateAndBackSubstitute", BindingFlags.Static | BindingFlags.NonPublic);
        method.Should().NotBeNull();
        object? result = null;
        FluentActions.Invoking(() => result = method!.Invoke(null, [augmented, 2, random, floor])).Should().NotThrow();
        return (decimal[])result!;
    }

    private sealed class FixedIndexRandom(int index) : System.Random
    {
        internal List<int> Bounds { get; } = [];

        public override int Next(int maxValue)
        {
            Bounds.Add(maxValue);
            index.Should().BeLessThan(maxValue);
            return index;
        }
    }
}
