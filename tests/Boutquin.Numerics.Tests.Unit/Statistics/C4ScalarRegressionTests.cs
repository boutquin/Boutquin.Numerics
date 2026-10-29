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
public sealed class C4ScalarRegressionTests
{
    [Fact]
    public void Dsr_ZeroDecimalReturns_RefusesUndefinedSharpe()
    {
        Action act = () => DeflatedSharpeRatio<decimal>.ComputeFromReturns([0m, 0m, 0m, 0m], 1, 1);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Dsr_WideSharpe_PreservesFiniteNormalLimit()
    {
        // Normal moments, one trial, five observations: z tends to sqrt(8).
        var result = DeflatedSharpeRatio<double>.Compute(1e200, 1, 5, 0, 3, 1);
        result.PValue.Should().BeApproximately(0.0023388674905236283, 1e-14);
    }

    [Fact]
    public void Dsr_WideSharpe_UnitKurtosisRetainsNonzeroStandardError()
    {
        DeflatedSharpeRatio<double>.Compute(1e200, 1, 5, 0, 1, 1).PValue.Should().Be(0);
    }

    [Fact]
    public void Dsr_WideDoubleReturns_PreservesScaleInvariantMoments()
    {
        DsrResult<double>? result = null;
        var error = Record.Exception(() => result = DeflatedSharpeRatio<double>.ComputeFromReturns([1e308, 1e308, 1.1e308, 1.1e308], 1, 1));
        error.Should().BeNull("the mean and population standard deviation are representable");
        result!.DeflatedSharpe.Should().BeApproximately(21, 1e-12);
    }

    [Fact]
    public void Dsr_WideDecimalReturns_PreservesScaleInvariantMoments()
    {
        DsrResult<decimal>? result = null;
        var error = Record.Exception(() => result = DeflatedSharpeRatio<decimal>.ComputeFromReturns([20e27m, 21e27m, 22e27m, 23e27m], 1, 1));
        error.Should().BeNull("the mean and population standard deviation are representable");
        result!.DeflatedSharpe.Should().BeApproximately(19.2301846064982m, 1e-12m);
    }

    [Theory]
    [InlineData(1e200, 2.3527717270477075)]
    [InlineData(300, 2.352801788641642)]
    public void MinTrl_WideSharpe_PreservesFiniteRatio(double sharpe, double expected)
    {
        MinimumTrackRecordLength<double>.Compute(sharpe, 0, 0, 3, 0.05, 1).Should().BeApproximately(expected, 1e-12);
    }

    [Fact]
    public void MinTrl_HalfWideSharpe_PreservesFiniteRatio()
    {
        var result = MinimumTrackRecordLength<Half>.Compute((Half)300, (Half)0, (Half)0, (Half)3, 0.05, 1);
        ((double)result).Should().BeApproximately(2.352801788641642, 0.005);
    }

    [Theory]
    [InlineData(1e12)]
    [InlineData(1e15)]
    [InlineData(1e28)]
    public void Haircut_LargeDegreesOfFreedom_PreservesNormalLimit(double years)
    {
        var result = HaircutSharpe<double>.Compute(1 / Math.Sqrt(years), 1, years, 252);
        result.BonferroniPValue.Should().BeApproximately(0.31731050786291415, 1e-10);
    }
}
