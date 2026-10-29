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

public sealed class OwnerNercomeScaleTests
{
    [Fact]
    public void Decimal_SelectorScoreDoesNotOverflowWhenCovarianceFits()
    {
        const decimal scale = 10000000000000m;
        decimal[,] rows = { { scale, 0 }, { 0, 2 * scale }, { 2 * scale, scale }, { -scale, 3 * scale } };
        decimal[,]? actual = null;
        var action = () => actual = new NercomeCovarianceEstimator<decimal>().Estimate(rows);
        action.Should().NotThrow("the covariance fits decimal even though an unscaled squared residual does not");
        actual![0, 0].Should().BeApproximately(263m / 150 * scale * scale, 10000000000m);
        actual[0, 1].Should().BeApproximately(-8m / 25 * scale * scale, 10000000000m);
        actual[1, 1].Should().BeApproximately(487m / 150 * scale * scale, 10000000000m);
    }

    [Theory]
    [InlineData(1e100)]
    [InlineData(1e-100)]
    public void Double_SelectorIsScaleEquivariantAcrossSquaredResidualRange(double scale)
    {
        double[,] rows = { { scale, 0 }, { 0, 2 * scale }, { 2 * scale, scale }, { -scale, 3 * scale } };
        var actual = new NercomeCovarianceEstimator<double>().Estimate(rows);
        (actual[0, 0] / scale / scale).Should().BeApproximately(263.0 / 150, 1e-11);
        (actual[0, 1] / scale / scale).Should().BeApproximately(-8.0 / 25, 1e-11);
        (actual[1, 1] / scale / scale).Should().BeApproximately(487.0 / 150, 1e-11);
    }

    [Fact]
    public void UnrepresentableCovariance_ThrowsForIeeeTypes()
    {
        var doubles = () => new NercomeCovarianceEstimator<double>().Estimate(new double[,] { { 1e200 }, { 1e200 } });
        doubles.Should().Throw<OverflowException>();
        var halves = () => new NercomeCovarianceEstimator<Half>().Estimate(new Half[,] { { (Half)300 }, { (Half)300 } });
        halves.Should().Throw<OverflowException>();
    }
}
