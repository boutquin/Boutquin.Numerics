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
public sealed class C4LocalCovarianceControls
{
    [Fact]
    public void Oas_SmallUnits_MatchesIndependentRationalIntensity()
    {
        // OAS scale replacement: same exact scatter as the rational interior
        // panel, scaled by1e-14 in covariance. The independent intensity21/37
        // must survive; an absolute degeneracy guard forces the wrong target.
        var panel = new double[12, 3];
        for (var repeat = 0; repeat < 2; repeat++)
        {
            for (var j = 0; j < 3; j++)
            {
                panel[repeat * 6 + 2 * j, j] = (3 - j) * 1e-7;
                panel[repeat * 6 + 2 * j + 1, j] = -(3 - j) * 1e-7;
            }
        }

        var result = new OracleApproximatingShrinkageEstimator<double>().Estimate(panel);
        result[0, 0].Should().BeApproximately(242.0 / 111 * 1e-14, 1e-27);
        result[1, 1].Should().BeApproximately(54.0 / 37 * 1e-14, 1e-27);
        result[2, 2].Should().BeApproximately(38.0 / 37 * 1e-14, 1e-27);
    }

    [Fact]
    public void Denoised_OptionalLocalBlend_HasExactNonisotropicDiagonal()
    {
        // Explicit local fixed.1 post-pass, not an MLAM shrinkage formula.
        // Six signed coordinate observations give sample diag(2,8,18)/5.
        // Correlation isidentity, so denoising leaves it; targetmean28/15.
        // .9*S+.1*mean*I gives diag(41,122,257)/75.
        double[,] panel = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 2, 0 }, { 0, -2, 0 }, { 0, 0, 3 }, { 0, 0, -3 } };
        var result = new DenoisedCovarianceEstimator<double>(true).Estimate(panel);
        result[0, 0].Should().BeApproximately(41.0 / 75, 1e-13);
        result[1, 1].Should().BeApproximately(122.0 / 75, 1e-13);
        result[2, 2].Should().BeApproximately(257.0 / 75, 1e-13);
        result[0, 1].Should().BeApproximately(0, 1e-13);
    }

    [Fact]
    public void Denoised_TwoAssets_UsesDeclaredSampleOrLwFallback()
    {
        // Local small-dimension extension. Exact sample is[[20,4],[4,4]]/3;
        // opted-in scaled-identity LW intensity3/5 gives[[76,8],[8,44]]/15.
        double[,] panel = { { -3, -1 }, { -1, 1 }, { 1, -1 }, { 3, 1 } };
        var sample = new DenoisedCovarianceEstimator<double>(false).Estimate(panel);
        sample[0, 0].Should().BeApproximately(20.0 / 3, 1e-13);
        sample[0, 1].Should().BeApproximately(4.0 / 3, 1e-13);
        var shrunk = new DenoisedCovarianceEstimator<double>(true).Estimate(panel);
        shrunk[0, 0].Should().BeApproximately(76.0 / 15, 1e-13);
        shrunk[0, 1].Should().BeApproximately(8.0 / 15, 1e-13);
        shrunk[1, 1].Should().BeApproximately(44.0 / 15, 1e-13);
    }

    [Fact]
    public void TailQuantiles_PreserveOrderAndDuplicates_WithFirstRatio()
    {
        // Local finite-query contract, no limit-estimation claim. Pseudo-ranks
        // x=(.2,.4,.6,.8),y=(.2,.8,.4,.6), so jointcounts at(.6,.4,.6)
        // are(2,1,2), first marginal3 and first conditional ratio2/3.
        double[] x = [1, 2, 3, 4];
        double[] y = [1, 4, 2, 3];
        var result = LowerTailDependence<double>.Estimate(x, y, new TailOptions<double>(Quantiles: [.6, .4, .6]));
        result.Threshold.Should().Equal(.6, .4, .6);
        result.JointCount.Should().Equal(2, 1, 2);
        result.Lambda.Should().BeApproximately(2.0 / 3, 1e-15);
    }

    [Fact]
    public void GraphicalWeights_AsymmetricPenalty_IsRejected()
    {
        double[,] sample = { { 1, 0 }, { 0, 1 } };
        double[,] weights = { { 0, 1 }, { 2, 0 } };
        var options = new GraphicalLassoOptions<double>(Weights: weights);
        Action solve = () => GraphicalLassoSolver<double>.Solve(sample, .1, options);
        Action select = () => GraphicalLassoSolver<double>.SolveWithBic(sample, 20, [.1], options);
        solve.Should().Throw<ArgumentException>().WithMessage("*symmetric*");
        select.Should().Throw<ArgumentException>().WithMessage("*symmetric*");
    }
}
