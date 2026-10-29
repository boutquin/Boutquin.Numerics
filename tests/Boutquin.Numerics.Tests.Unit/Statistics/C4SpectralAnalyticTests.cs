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

public sealed class C4SpectralAnalyticTests
{
    [Fact]
    public void Oas_ThreeDimensionalInterior_MatchesRationalIntensity()
    {
        // Chen equation23: trace ratios give rho=21/37 at T=12,p=3.
        // The known-zero-mean convention gives S=diag(36,16,4)/12.
        double[,] panel =
        {
            { 3, 0, 0 }, { -3, 0, 0 }, { 0, 2, 0 }, { 0, -2, 0 }, { 0, 0, 1 }, { 0, 0, -1 },
            { 3, 0, 0 }, { -3, 0, 0 }, { 0, 2, 0 }, { 0, -2, 0 }, { 0, 0, 1 }, { 0, 0, -1 },
        };
        var actual = new OracleApproximatingShrinkageEstimator<double>().Estimate(panel);
        actual[0, 0].Should().BeApproximately(242.0 / 111, 1e-12);
        actual[1, 1].Should().BeApproximately(54.0 / 37, 1e-12);
        actual[2, 2].Should().BeApproximately(38.0 / 37, 1e-12);
        actual[0, 1].Should().Be(0);
    }

    [Fact]
    public void Oas_ThreeDimensionalClipping_MatchesScaledIdentity()
    {
        // Unclipped rho=21/19 at T=6,p=3; F=(14/9)I.
        double[,] panel = { { 3, 0, 0 }, { -3, 0, 0 }, { 0, 2, 0 }, { 0, -2, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
        var actual = new OracleApproximatingShrinkageEstimator<double>().Estimate(panel);
        for (var i = 0; i < 3; i++)
        {
            actual[i, i].Should().BeApproximately(14.0 / 9, 1e-12);
        }
    }

    [Fact]
    public void Qis_OneActiveEigenvalue_MatchesClosedFormWideSpectrum()
    {
        // Appendix A with p=2,n=1: S=2vv', v=(1,2), eigenvalues(0,10).
        // The one-point kernel has theta=0,Htheta=1/h. After trace correction,
        // null eigenvalue=10/(1+h^2), active eigenvalue=10h^2/(1+h^2).
        double[,] panel = { { 1, 2 }, { -1, -2 } };
        var actual = new QuadraticInverseShrinkageEstimator<double>().Estimate(panel);
        var hSquared = Math.Pow(2, -2.1);
        actual[0, 0].Should().BeApproximately((8 + 2 * hSquared) / (1 + hSquared), 1e-11);
        actual[1, 1].Should().BeApproximately((2 + 8 * hSquared) / (1 + hSquared), 1e-11);
        actual[0, 1].Should().BeApproximately(4 * (hSquared - 1) / (1 + hSquared), 1e-11);
        actual[1, 0].Should().BeApproximately(4 * (hSquared - 1) / (1 + hSquared), 1e-11);
        (actual[0, 0] + actual[1, 1]).Should().BeApproximately(10, 1e-11);
    }

    [Fact]
    public void Poet_OrthogonalResiduals_MatchesAnalyticSoftThreshold()
    {
        // Top factor has eigenvalue16/3 in direction(1,1,0)/sqrt2.
        // Residual observations are (0,0,0),(0,0,0),(1,-1,0),(-1,1,0).
        // Thus sigma12=-1/2,theta12=1/4 and Bessel residual12=-2/3.
        double[,] panel = { { 2, 2, 0 }, { -2, -2, 0 }, { 1, -1, 0 }, { -1, 1, 0 } };
        var actual = new PoetCovarianceEstimator<double>(1, 0.5).Estimate(panel);
        var threshold = (1 / Math.Sqrt(3) + Math.Sqrt(Math.Log(3) / 4)) / 3;
        actual[0, 1].Should().BeApproximately(2 + threshold, 1e-11);
        actual[1, 0].Should().BeApproximately(2 + threshold, 1e-11);
        actual[0, 0].Should().BeApproximately(10.0 / 3, 1e-11);
        actual[1, 1].Should().BeApproximately(10.0 / 3, 1e-11);
        actual[2, 2].Should().BeApproximately(0, 1e-11);
    }

    [Fact]
    public void Poet_ThresholdAboveResidual_RemovesOnlyResidualOffDiagonal()
    {
        double[,] panel = { { 2, 2, 0 }, { -2, -2, 0 }, { 1, -1, 0 }, { -1, 1, 0 } };
        var actual = new PoetCovarianceEstimator<double>(1, 2.0).Estimate(panel);
        actual[0, 1].Should().BeApproximately(8.0 / 3, 1e-11);
        actual[0, 0].Should().BeApproximately(10.0 / 3, 1e-11);
        actual[1, 1].Should().BeApproximately(10.0 / 3, 1e-11);
    }

    [Fact]
    public void EmpiricalTailMatrix_TiedMargins_PreservesJointRatioAndOverridesDiagonal()
    {
        double[] x = [1, 1, 1, 2];
        double[] y = [1, 2, 3, 4];
        var options = new TailOptions<double>(Quantiles: [0.5], Estimator: LowerTailDependenceEstimator.EmpiricalCopula);
        var actual = LowerTailDependence<double>.EstimateMatrix([x, y], options);
        // Joint count2/(4*.5)=1 both directions; self x has3/(4*.5)=1.5.
        // The explicit matrix convention still overrides the self entry to1.
        actual[0, 1].Should().Be(1);
        actual[1, 0].Should().Be(1);
        actual[0, 0].Should().Be(1);
        actual[1, 1].Should().Be(1);
    }
}
