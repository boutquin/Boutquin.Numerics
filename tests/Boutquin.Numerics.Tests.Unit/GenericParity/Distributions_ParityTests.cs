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

using Boutquin.Numerics.Distributions;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.GenericParity;

public sealed class Distributions_ParityTests
{
    [Theory]
    [InlineData(-3.0, 0.0013499)]
    [InlineData(-1.5, 0.0668072)]
    [InlineData(0.0, 0.5)]
    [InlineData(1.5, 0.9331928)]
    [InlineData(3.0, 0.9986501)]
    public void CumulativeNormal_Decimal_MatchesDoubleWithinPrecisionCeiling(double xd, double _)
    {
        // Tier C computes in double; decimal output has a double-precision ceiling.
        var x = (decimal)xd;
        var expected = (decimal)CumulativeNormal<double>.Evaluate(xd);
        var actual = CumulativeNormal<decimal>.Evaluate(x);
        ((double)actual).Should().BeApproximately((double)expected, 1e-14);
    }

    [Theory]
    [InlineData(0.01)]
    [InlineData(0.25)]
    [InlineData(0.5)]
    [InlineData(0.75)]
    [InlineData(0.99)]
    public void InverseNormal_Decimal_MatchesDoubleWithinPrecisionCeiling(double pd)
    {
        var p = (decimal)pd;
        var expected = InverseNormal<double>.Evaluate(pd);
        var actual = InverseNormal<decimal>.Evaluate(p);
        ((double)actual).Should().BeApproximately(expected, 1e-10);
    }

    [Theory]
    [InlineData(-3.0)]
    [InlineData(-1.0)]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(3.0)]
    public void InverseNormal_Decimal_RoundTrip_WithCdf(double xd)
    {
        var x = (decimal)xd;
        var p = CumulativeNormal<decimal>.Evaluate(x);
        var z = InverseNormal<decimal>.Evaluate(p);
        ((double)z).Should().BeApproximately(xd, 1e-10);
    }
}
