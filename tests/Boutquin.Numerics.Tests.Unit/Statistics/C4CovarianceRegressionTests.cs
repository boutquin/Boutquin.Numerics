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
public sealed class C4CovarianceRegressionTests
{
    [Fact]
    public void Denoised_AllNoise_AveragesEntireResidualSubspace()
    {
        var panel = new double[1000, 3];
        double[,] root = { { 1, 0, 0 }, { .05, Math.Sqrt(1 - .05 * .05), 0 }, { 0, 0, 1 } };
        for (var j = 0; j < 3; j++)
        {
            for (var i = 0; i < 3; i++)
            {
                panel[2 * j, i] = Math.Sqrt(999.0 / 2) * root[i, j];
                panel[2 * j + 1, i] = -panel[2 * j, i];
            }
        }

        var result = new DenoisedCovarianceEstimator<double>().Estimate(panel);
        result[0, 1].Should().BeApproximately(0, 1e-12);
        for (var i = 0; i < 3; i++)
        {
            result[i, i].Should().BeApproximately(1, 1e-12);
        }
    }

    [Fact]
    public void Detoned_LongHalfPanel_MatchesExactWidenedInputs()
    {
        var random = new System.Random(41);
        var half = new Half[70000, 3];
        var wide = new double[70000, 3];
        for (var i = 0; i < 70000; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                half[i, j] = (Half)(random.NextDouble() - .5);
                wide[i, j] = (double)half[i, j];
            }
        }

        var expected = new DetonedCovarianceEstimator<double>().Estimate(wide);
        var result = new DetonedCovarianceEstimator<Half>().Estimate(half);
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                result[i, j].Should().Be((Half)expected[i, j]);
            }
        }
    }

    [Fact]
    public void Detoned_CollinearDecimal_PreservesNonnegativeQuadraticForm()
    {
        var panel = new decimal[100, 3];
        for (var i = 0; i < 100; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                panel[i, j] = (i % 2 == 0 ? -1 : 1) * (j + 1);
            }
        }

        var result = new DetonedCovarianceEstimator<decimal>().Estimate(panel);
        decimal[] vector = [1, .5m, 1m / 3];
        var quadratic = 0m;
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                quadratic += vector[i] * result[i, j] * vector[j];
            }
        }

        quadratic.Should().BeGreaterThanOrEqualTo(-1e-24m);
    }

    [Fact]
    public void Poet_ThresholdUsesSameBesselScaleAsCovariance()
    {
        double[,] panel = { { 1, 2, 3 }, { -1, 0, 2 }, { 2, -1, 0 }, { 0, 3, -2 }, { -2, -1, -3 }, { .5, -.5, 1 } };
        var result = new PoetCovarianceEstimator<double>().Estimate(panel);
        // Independent SVD residuals; primary 1/T covariance and threshold both
        // converted to Bessel units by 6/5, after primary soft thresholding.
        result[0, 2].Should().BeApproximately(1.7260771687916172, 1e-10);
    }

    [Fact]
    public void Qis_WideFiniteCovariance_PreservesScaleEquivariance()
    {
        var panel = new double[20, 3];
        var wide = new double[20, 3];
        for (var i = 0; i < 20; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                panel[i, j] = Math.Sin((i + 1) * (j + 2) * .71);
                wide[i, j] = panel[i, j] * 1.15e154;
            }
        }

        var expected = new QuadraticInverseShrinkageEstimator<double>().Estimate(panel);
        var result = new QuadraticInverseShrinkageEstimator<double>().Estimate(wide);
        (result[0, 1] / 1.3225e308).Should().BeApproximately(expected[0, 1], 1e-12);
    }

    [Fact]
    public void TracyWidom_WidePanel_ReversesWishartRoles()
    {
        var edge = TracyWidomDenoisedCovarianceEstimator<double>.TracyWidomCentering(8, 12)
            + TracyWidomDenoisedCovarianceEstimator<double>.TracyWidomShift(8, 12);
        edge.Should().BeApproximately(6.070036778283685, 1e-12);
    }
}
