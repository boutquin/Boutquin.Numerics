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

public sealed class ShrinkageScaledIntensityTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ConstantCorrelation_DoubleSmallPairsDecideIntensity(bool reverse)
    {
        var a = Math.ScaleB(1.0, 500);
        var c = Math.ScaleB(1.0, -500);
        var returns = new double[,] { { a, 0, c }, { -a, 0, -c }, { 0, a, -c }, { 0, -a, c } };
        var covariance = new LedoitWolfConstantCorrelationEstimator<double>().Estimate(Permute(returns, reverse));

        // rBar=0. The dominant pair's pi and gamma are identically zero;
        // the small pairs give pi/gamma=1, delta=1/4 and S13=2Ac/3.
        covariance[reverse ? 2 : 0, reverse ? 0 : 2].Should().BeApproximately(0.5, 2e-14);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ConstantCorrelation_DecimalSmallPairsDecideIntensity(bool reverse)
    {
        const decimal a = 100000000000000m;
        const decimal c = 0.00000000000001m;
        var returns = new decimal[,] { { a, 0, c }, { -a, 0, -c }, { 0, a, -c }, { 0, -a, c } };
        var covariance = new LedoitWolfConstantCorrelationEstimator<decimal>().Estimate(Permute(returns, reverse));

        covariance[reverse ? 2 : 0, reverse ? 0 : 2].Should().BeApproximately(0.5m, 0.000000000000000000000000001m);
    }

    [Fact]
    public void ConstantCorrelation_DoubleUnrepresentableIntensity_PreservesTargetProduct()
    {
        var a = Math.ScaleB(1.0, 500);
        var c = Math.ScaleB(1.0, -500);
        var returns = new double[,] { { a, 0, c }, { -a, 0, -c }, { 0, a, c }, { 0, -a, -c } };
        var covariance = new LedoitWolfConstantCorrelationEstimator<double>().Estimate(returns);

        // delta = 3c^2/[2(A^2+c^2)], F12 = 2sqrt(2)A^2/9.
        // delta underflows as a scalar, but delta*F12 is approximately sqrt(2)c^2/3.
        (covariance[0, 1] / (Math.Sqrt(2.0) / 3 * Math.ScaleB(1.0, -1000)))
            .Should().BeApproximately(1.0, 2e-13);
    }

    [Fact]
    public void ConstantCorrelation_DecimalUnrepresentableIntensity_PreservesTargetProduct()
    {
        const decimal a = 100000000000000m;
        const decimal c = 0.0000000000001m;
        var returns = new decimal[,] { { a, 0, c }, { -a, 0, -c }, { 0, a, c }, { 0, -a, -c } };
        var covariance = new LedoitWolfConstantCorrelationEstimator<decimal>().Estimate(returns);

        covariance[0, 1].Should().BeApproximately(0.0000000000000000000000000047m, 0.0000000000000000000000000001m);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Identity_DoubleFourthMomentAndTinyIntensity_PreserveSmallVariance(bool reverse)
    {
        var a = Math.ScaleB(1.0, 500);
        var c = Math.ScaleB(1.0, -500);
        var returns = new double[,] { { a, c }, { -a, c }, { a, -c }, { -a, -c } };
        var covariance = new LedoitWolfShrinkageEstimator<double>().Estimate(Permute(returns, reverse));

        // b^2=(Ac)^2/2, d^2=(A^2-c^2)^2/2, so delta=(Ac)^2/(A^2-c^2)^2.
        // Bessel-scaled small diagonal tends to 2c^2, rather than the sample's 4c^2/3.
        (covariance[reverse ? 0 : 1, reverse ? 0 : 1] / Math.ScaleB(1.0, -999))
            .Should().BeApproximately(1.0, 2e-13);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Identity_DecimalFourthMomentAndTinyIntensity_PreserveSmallVariance(bool reverse)
    {
        const decimal a = 100000000000000m;
        const decimal c = 0.0000000000001m;
        var returns = new decimal[,] { { a, c }, { -a, c }, { a, -c }, { -a, -c } };
        var covariance = new LedoitWolfShrinkageEstimator<decimal>().Estimate(Permute(returns, reverse));

        covariance[reverse ? 0 : 1, reverse ? 0 : 1]
            .Should().BeApproximately(0.0000000000000000000000000200m, 0.0000000000000000000000000002m);
    }

    private static T[,] Permute<T>(T[,] source, bool reverse)
    {
        if (!reverse)
        {
            return source;
        }
        var result = new T[source.GetLength(0), source.GetLength(1)];
        for (var i = 0; i < result.GetLength(0); i++)
        {
            for (var j = 0; j < result.GetLength(1); j++)
            {
                result[i, j] = source[i, result.GetLength(1) - j - 1];
            }
        }
        return result;
    }
}
