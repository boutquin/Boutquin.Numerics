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
public sealed class C4RemainingLocalControls
{
    [Theory]
    [InlineData(.005, .03)]
    [InlineData(.009, .054)]
    [InlineData(.0128, .0768)]
    [InlineData(.0135, .081)]
    [InlineData(.045, .27)]
    [InlineData(.06, .36)]
    public void Bonferroni_PrimarySixValues_AreElementwiseSixfold(double p, double expected)
    {
        // A t distribution with one degree of freedom is exactly Cauchy:
        // two-sided tail p has t=cot(pi*p/2). Two observations give SR=t/sqrt(2).
        // Thus each published p is supplied independently through the scalar API.
        var sharpe = 1 / Math.Tan(Math.PI * p / 2) / Math.Sqrt(2);
        var result = HaircutSharpe<double>.Compute(sharpe, 6, 2, 1);
        result.BonferroniPValue.Should().BeApproximately(expected, 2e-12);
    }

    [Fact]
    public void Bic_CountsBothDirectedOffDiagonalEntries()
    {
        // Exact two-variable optima at lambda 0 and .6 (zero diagonal penalty).
        // The log-likelihood improvement is -10*log(1-.6^2)=4.4628710263.
        // Two ordered edges cost 2*log(10)=4.605170186; one costs 2.302585093.
        // Therefore the declared ordered-edge BIC selects the diagonal model.
        var result = GraphicalLassoSolver<double>.SolveWithBic(
            new double[,] { { 1, .6 }, { .6, 1 } }, 10, [0, .6],
            new GraphicalLassoOptions<double>(Weights: new double[,] { { 0, 1 }, { 1, 0 } }));
        result.Lambda.Should().Be(.6);
        result.Precision[0, 1].Should().BeApproximately(0, 1e-12);
    }

    [Fact]
    public void Bic_EqualScores_RetainFirstGridEntry()
    {
        // Both penalties exceed the same .6 soft-threshold, hence W=Theta=I,
        // zero edges and identical likelihood. The local strict comparison keeps first.
        var result = GraphicalLassoSolver<double>.SolveWithBic(
            new double[,] { { 1, .6 }, { .6, 1 } }, 10, [.7, .8],
            new GraphicalLassoOptions<double>(Weights: new double[,] { { 0, 1 }, { 1, 0 } }));
        result.Lambda.Should().Be(.7);
    }
    [Fact]
    public void Tail_DefaultQuantiles_PinEveryThresholdAndJointCount()
    {
        // Nineteen ordered observations have pseudo-ranks k/20.
        // Default cutoffs 1/20,2/20,3/20,4/20 include exactly 1,2,3,4 pairs.
        var values = Enumerable.Range(1, 19).Select(i => (decimal)i).ToArray();
        var result = LowerTailDependence<decimal>.Estimate(values, values);
        result.Threshold.Should().Equal(.05m, .10m, .15m, .20m);
        result.JointCount.Should().Equal(1, 2, 3, 4);
        result.Lambda.Should().Be(1m);
    }
}
