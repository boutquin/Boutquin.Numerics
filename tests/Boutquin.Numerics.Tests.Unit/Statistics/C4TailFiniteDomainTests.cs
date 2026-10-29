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

public sealed class C4TailFiniteDomainTests
{
    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void FirstMargin_NonFiniteValueIsRejected(double invalid)
    {
        Action estimate = () => LowerTailDependence<double>.Estimate(
            [1, invalid, 3, 4], [1, 2, 3, 4]);
        estimate.Should().Throw<ArgumentException>().WithParameterName("x").WithMessage("*finite*");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void SecondMargin_NonFiniteValueIsRejected(double invalid)
    {
        Action estimate = () => LowerTailDependence<double>.Estimate(
            [1, 2, 3, 4], [1, invalid, 3, 4]);
        estimate.Should().Throw<ArgumentException>().WithParameterName("y").WithMessage("*finite*");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void SingleColumn_NonFiniteValueIsRejectedBeforeDiagonalOverride(double invalid)
    {
        // A one-column result bypasses pair estimation, so its own finite-value
        // validation is required before returning the local diagonal override.
        Action estimate = () => LowerTailDependence<double>.EstimateMatrix([[1, invalid, 3, 4]]);
        estimate.Should().Throw<ArgumentException>().WithParameterName("columns").WithMessage("*finite*");
    }
}
