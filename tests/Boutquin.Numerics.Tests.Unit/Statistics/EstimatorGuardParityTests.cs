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

public sealed class EstimatorGuardParityTests
{
    public static TheoryData<double> NonFiniteValues =>
        new() { double.NaN, double.PositiveInfinity, double.NegativeInfinity };

    [Theory]
    [MemberData(nameof(NonFiniteValues))]
    public void GenericEstimatorParameters_RejectNonFiniteValues(double value)
    {
        Action[] constructors =
        [
            () => _ = new ExponentiallyWeightedCovarianceEstimator<double>(value),
            () => _ = new NercomeCovarianceEstimator<double>(value),
            () => _ = new DetonedCovarianceEstimator<double>(value),
            () => _ = new DoublySparseEstimator<double>(value),
            () => _ = new PoetCovarianceEstimator<double>(thresholdMultiplier: value),
        ];

        foreach (var constructor in constructors)
        {
            constructor.Should().Throw<ArgumentOutOfRangeException>();
        }
    }

    [Theory]
    [MemberData(nameof(NonFiniteValues))]
    public void PoetGeneric_RejectsNonFiniteThresholds(double value)
    {
        FluentActions.Invoking(() => new PoetCovarianceEstimator<double>(thresholdMultiplier: value))
            .Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("thresholdMultiplier");
    }
}
