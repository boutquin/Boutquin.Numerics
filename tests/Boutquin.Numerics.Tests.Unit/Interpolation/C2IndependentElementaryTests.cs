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

using Boutquin.Numerics.Interpolation;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Interpolation;

public sealed class C2IndependentElementaryTests
{
    [Fact]
    public void Linear_QuadraticError_AttainsSharpBound()
    {
        // f=x², endpoints2,6: h=4, max|f''|=2, sharp h²M/8=4.
        var value = LinearInterpolator<decimal>.Instance.Interpolate(4m, [2m, 6m], [4m, 36m]);
        value.Should().Be(20m);
        (value - 16m).Should().Be(4m);
    }

    [Fact]
    public void Linear_PiecewiseSlopes_AreContinuousWithKink()
    {
        var sut = LinearInterpolator<decimal>.Instance;
        // Dyadic widths keep interpolation weights exact in decimal: slopes +2 and -1.
        decimal[] xs = [0m, 2m, 4m];
        decimal[] ys = [0m, 4m, 2m];
        sut.Interpolate(1.75m, xs, ys).Should().Be(3.5m);
        sut.Interpolate(2m, xs, ys).Should().Be(4m);
        sut.Interpolate(2.25m, xs, ys).Should().Be(3.75m);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RawInterpolation_PowersOfTwo_DefineConstantForward(bool flatForward)
    {
        IInterpolator<double> sut = flatForward ? FlatForwardInterpolator<double>.Instance : LogLinearInterpolator<double>.Instance;
        double[] xs = [1, 5];
        double[] ys = [1d / 16, 1d / 256];
        // Independent power law d(t)=2^(-t-3), so f=ln2 and
        // r(t)=ln2+3ln2/t. These are exact dyadic discount values.
        double[] expected = [1d / 16, 1d / 32, 1d / 64, 1d / 128, 1d / 256];
        for (var i = 0; i < expected.Length; i++)
        {
            sut.Interpolate(i + 1, xs, ys).Should().BeApproximately(expected[i], 1e-16);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RawInterpolation_EqualNodesAndExterior_HaveZeroForward(bool flatForward)
    {
        IInterpolator<double> sut = flatForward ? FlatForwardInterpolator<double>.Instance : LogLinearInterpolator<double>.Instance;
        foreach (var x in new[] { -3d, 1d, 2d, 3d, 4d, 5d, 9d })
        {
            sut.Interpolate(x, [1d, 5d], [.25d, .25d]).Should().BeApproximately(.25, 1e-16);
        }
        sut.Interpolate(-3, [1d, 5d], [.25d, .125d]).Should().Be(.25);
        sut.Interpolate(9, [1d, 5d], [.25d, .125d]).Should().Be(.125);
    }
}
