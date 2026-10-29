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

using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

[Trait("Tier", "Unit")]
public sealed class C4HullPrintedUpdateTests
{
    [Theory]
    [InlineData(.00012246, .004453, .000099967717589737)]
    [InlineData(.00009997, -.000735, .000078837971263425)]
    [InlineData(.00007884, .008015, .000077297283353425)]
    [InlineData(.00016729, .007827, .000142753947206697)]
    public void PrintedPreviousStates_ProduceIndependentlyRecomputedUpdates(
        double previousVariance, double previousReturn, double expected)
    {
        // Hull Table23.1 supplied rounded previous states for days4,5,6,1259.
        // Exact decimal arithmetic evaluates .0000039818 + .223793*u²
        // + .747577*h into the expected literals. Rounding of printed inputs
        // means these are not byte-exact historical unrounded path values.
        // This exercises conditional forecasting, not historical fitting,
        // the unknown intervening data, price conversion or a different seed.
        var fit = new GarchFitResult<double>(0, 0, .0000039818, .223793, .747577,
            [previousVariance], [previousReturn / Math.Sqrt(previousVariance)], 0, true);
        fit.ForecastVariance(1).Should().BeApproximately(expected, 1e-15);
    }
}
