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

namespace Boutquin.Numerics.Tests.Unit.Distributions;

public sealed class NormalAccuracyTests
{
    [Fact]
    public void Quantile_DecimalMedianNeighboursPreserveTheirSignAndMagnitude()
    {
        Assert.Equal(0.0000000000000000000000000003m,
            InverseNormal<decimal>.Evaluate(0.5000000000000000000000000001m));
        Assert.Equal(-0.0000000000000000000000000003m,
            InverseNormal<decimal>.Evaluate(0.4999999999999999999999999999m));
    }

    // Gaussian-integral and inverse-integral values from mpmath 1.3.0 at 200
    // decimal digits, using the exact binary64 input (not its rounded decimal).
    [Theory]
    [InlineData(-3.9, 4.8096344017602736e-5)]
    [InlineData(-3.85, 5.905891241892252e-5)]
    [InlineData(-1.2, 0.11506967022170828)]
    [InlineData(-1.0, 0.15865525393145705)]
    [InlineData(-7.5, 3.1908916729108963e-14)]
    public void Cdf_ResolvedTailRetainsRelativePrecision(double x, double expected)
    {
        var actual = CumulativeNormal<double>.Evaluate(x);
        Assert.InRange(Math.Abs((actual - expected) / expected), 0, 4e-15);
    }

    [Theory]
    [InlineData(0.49999999999999994, -1.3914582123358836e-16)]
    [InlineData(0.5000000000000001, 2.782916424671767e-16)]
    [InlineData(5.12e-5, -3.8848335967813585)]
    [InlineData(0.02425, -1.972961051311885)]
    [InlineData(1e-150, -26.122961190593983)]
    public void Quantile_CentralResidualAndTailRetainRelativePrecision(double p, double expected)
    {
        var actual = InverseNormal<double>.Evaluate(p);
        Assert.InRange(Math.Abs((actual - expected) / expected), 0, 2e-15);
    }
}
