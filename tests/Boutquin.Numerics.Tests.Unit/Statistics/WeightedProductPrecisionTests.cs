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

public sealed class WeightedProductPrecisionTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Ewma_DecimalUnrepresentableWeight_PreservesPhysicalCovariance(bool reverseColumns)
    {
        // w = q^3 / (1 + q + q^2 + q^3), q=1e-14. For the only nonzero
        // observation A=1e14 the variance is w*(1-w)*A^2, about 1e-14.
        var returns = new decimal[4, 2];
        returns[0, reverseColumns ? 1 : 0] = 100000000000000m;
        returns[0, reverseColumns ? 0 : 1] = -50000000000000m;
        var covariance = new ExponentiallyWeightedCovarianceEstimator<decimal>(0.00000000000001m).Estimate(returns);

        covariance[reverseColumns ? 1 : 0, reverseColumns ? 1 : 0]
            .Should().BeApproximately(0.0000000000000099999999999999m, 0.0000000000000000000000000001m);
        covariance[0, 1].Should().BeApproximately(-0.00000000000000499999999999995m, 0.0000000000000000000000000001m);
        covariance[1, 0].Should().Be(covariance[0, 1]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Ewma_DoubleUnrepresentableWeight_PreservesPhysicalCovariance(bool reverseColumns)
    {
        // All nonzero inputs are at the guaranteed +/-500 binary exponent limits.
        // The oldest weight is 2^-1500 but the variance is approximately 2^-500.
        var amplitude = Math.ScaleB(1.0, 500);
        var returns = new double[4, 2];
        returns[0, reverseColumns ? 1 : 0] = amplitude;
        returns[0, reverseColumns ? 0 : 1] = -amplitude / 2;
        var covariance = new ExponentiallyWeightedCovarianceEstimator<double>(Math.ScaleB(1.0, -500)).Estimate(returns);

        covariance[reverseColumns ? 1 : 0, reverseColumns ? 1 : 0].Should().Be(Math.ScaleB(1.0, -500));
        covariance[0, 1].Should().Be(-Math.ScaleB(1.0, -501));
        covariance[1, 0].Should().Be(covariance[0, 1]);
    }

    [Fact]
    public void Ewma_DecimalLongOrdinaryDecay_PreservesOldOutlier()
    {
        var returns = new decimal[40, 1];
        returns[0, 0] = 100000000000000m;
        var covariance = new ExponentiallyWeightedCovarianceEstimator<decimal>(0.1m).Estimate(returns);

        // Exact geometric-weight formula, independently evaluated at 150 digits.
        covariance[0, 0].Should().BeApproximately(0.000000000009m, 0.0000000000000000000000000001m);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Oas_DecimalSmallOverlappingRow_RetainsAllProductDigits(bool reverseColumns)
    {
        var returns = new decimal[40, 2];
        returns[0, reverseColumns ? 1 : 0] = 100000000000000m;
        returns[1, reverseColumns ? 0 : 1] = 1000000000000m;
        returns[2, 0] = 1m;
        returns[2, 1] = 1m;
        var covariance = new OracleApproximatingShrinkageEstimator<decimal>().Estimate(returns);

        // S12=1/40. The original p=2 OAS formula gives rho=tr(S)^2 /
        // [40*(tr(S^2)-tr(S)^2/2)], hence this independently evaluated result.
        covariance[0, 1].Should().BeApproximately(0.0237494998999849979997499700m, 0.0000000000000000000000000001m);
        covariance[1, 0].Should().Be(covariance[0, 1]);
    }

    [Fact]
    public void Oas_DoubleSmallOverlappingRow_DoesNotVanishDuringNormalization()
    {
        var returns = new double[40, 2];
        returns[0, 0] = 1e150;
        returns[1, 1] = 1e148;
        returns[2, 0] = 1e-150;
        returns[2, 1] = 1e-150;
        var covariance = new OracleApproximatingShrinkageEstimator<double>().Estimate(returns);

        covariance[0, 1].Should().BeApproximately(2.3749499899984998e-302, 1e-315);
        covariance[1, 0].Should().Be(covariance[0, 1]);
    }
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void Oas_DecimalIndividuallyUnrepresentableMeanTerms_PreservesTheirSum(int columns)
    {
        var returns = new decimal[40, columns];
        for (var row = 0; row < 40; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                returns[row, column] = 0.00000000000001m;
            }
        }

        var covariance = new OracleApproximatingShrinkageEstimator<decimal>().Estimate(returns);
        // S_ij=1e-28. With two columns rho=.05, and .95e-28 rounds to 1e-28.
        // The individual physical product/count is only 2.5e-30 and must not round first.
        for (var i = 0; i < columns; i++)
        {
            for (var j = 0; j < columns; j++)
            {
                covariance[i, j].Should().Be(0.0000000000000000000000000001m);
            }
        }
    }

    [Fact]
    public void Ewma_DecimalIndividuallyUnrepresentableVarianceTerms_PreservesTheirSum()
    {
        var returns = new decimal[,] { { -0.00000000000001m }, { 0.00000000000001m },
            { -0.00000000000001m }, { 0.00000000000001m } };
        var covariance = new ExponentiallyWeightedCovarianceEstimator<decimal>(0.5m).Estimate(returns);

        // Weights [1,2,4,8]/15, mean=1e-14/3, exact variance=(8/9)*1e-28.
        covariance[0, 0].Should().Be(0.0000000000000000000000000001m);
    }

    [Fact]
    public void Ewma_DecimalMeanBelowResolution_PreservesCrossCovariance()
    {
        var returns = new decimal[,] { { -0.0000000000000000000000000001m, -100000000000000m },
            { 0.0000000000000000000000000001m, 100000000000000m },
            { -0.0000000000000000000000000001m, -100000000000000m },
            { 0.0000000000000000000000000001m, 100000000000000m } };
        var covariance = new ExponentiallyWeightedCovarianceEstimator<decimal>(0.5m).Estimate(returns);

        // The first weighted mean is 1e-28/3, below decimal resolution. Covariance
        // is (8/9)*1e-14 and must retain the center before amplification.
        covariance[0, 1].Should().BeApproximately(0.0000000000000088888888888889m, 0.0000000000000000000000000001m);
        covariance[1, 0].Should().Be(covariance[0, 1]);
    }

    [Fact]
    public void Oas_DecimalSubResolutionBlendTerms_AreAddedBeforeRounding()
    {
        var returns = new decimal[,] { { 0.00000000000001m, 0.00000000000001m },
            { 0.00000000000001m, 0.00000000000001m },
            { 0.00000000000001m, 0.00000000000001m },
            { 0.00000000000001m, 0.00000000000001m } };
        var covariance = new OracleApproximatingShrinkageEstimator<decimal>().Estimate(returns);

        // rho=.5: diagonal .5e-28+.5e-28=1e-28. The off-diagonal .5e-28
        // rounds to zero under decimal's midpoint-to-even rule.
        covariance[0, 0].Should().Be(0.0000000000000000000000000001m);
        covariance[1, 1].Should().Be(covariance[0, 0]);
        covariance[0, 1].Should().Be(0m);
        covariance[1, 0].Should().Be(0m);
    }

}
