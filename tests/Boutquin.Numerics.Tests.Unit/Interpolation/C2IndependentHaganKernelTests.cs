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

using Boutquin.Numerics.Interpolation;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Interpolation;

public sealed class C2IndependentHaganKernelTests
{
    [Theory]
    [InlineData(2, -2, 1, 4, 3, 8)]
    [InlineData(1, -4, 1, 5, 1, 5)]
    [InlineData(1, -4, 7, 10, 23, 40)]
    [InlineData(3, -1, 1, 2, 25, 54)]
    [InlineData(3, -1, 7, 8, 1, 8)]
    [InlineData(2, 4, 1, 3, 11, 54)]
    [InlineData(2, 4, 2, 3, -4, 27)]
    [InlineData(2, 4, 5, 6, -8, 27)]
    [InlineData(0, 3, 1, 2, 0, 1)]
    [InlineData(3, 0, 1, 2, 0, 1)]
    [InlineData(0, 0, 1, 2, 0, 1)]
    public void DeviationPrimitive_MatchesIndependentRationalIntegral(
        int left, int right, int numerator, int denominator, int expectedNumerator, int expectedDenominator)
    {
        // Integrate the paper's piecewise quadratic on these fixed rational inputs.
        // II: eta=2/5; III: eta=3/4; IV: eta=2/3, A=-4/3.
        var expected = (decimal)expectedNumerator / expectedDenominator;
        Evaluate(left, right, (decimal)numerator / denominator).Should().BeApproximately(expected, 1e-25m);
        Evaluate(-left, -right, (decimal)numerator / denominator).Should().BeApproximately(-expected, 1e-25m);
    }

    [Theory]
    [InlineData(2, -2)]
    [InlineData(1, -4)]
    [InlineData(3, -1)]
    [InlineData(2, 4)]
    [InlineData(0, 3)]
    [InlineData(3, 0)]
    [InlineData(0, 0)]
    public void DeviationPrimitive_ConservesIntervalAverage(int left, int right)
    {
        // The deviation has integral zero over the complete interval, independently
        // of the public interpolator's node fast path.
        Evaluate(left, right, 0).Should().Be(0);
        Evaluate(left, right, 1).Should().BeApproximately(0, 1e-25m);
    }

    private static decimal Evaluate(decimal left, decimal right, decimal u)
    {
        var method = typeof(MonotoneConvexInterpolator<decimal>).GetMethod("IntegrateDeviation", BindingFlags.NonPublic | BindingFlags.Static);
        method.Should().NotBeNull();
        object? result = null;
        FluentActions.Invoking(() => result = method!.Invoke(null, [left, right, u])).Should().NotThrow();
        return (decimal)result!;
    }
}
