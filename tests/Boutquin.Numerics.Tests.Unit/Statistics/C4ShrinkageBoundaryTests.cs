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
public sealed class C4ShrinkageBoundaryTests
{
    [Fact]
    public void ScaledIdentity_IntensityAboveOne_ClipsToExactTarget()
    {
        // LWID-CLIPB: S_T=diag(9/2,2), m=13/4, d²=25/8,
        // bbar²=97/16, raw ratio97/50>1. Bessel target=(13/3)I.
        double[,] panel = { { 3, 0 }, { -3, 0 }, { 0, 2 }, { 0, -2 } };
        var result = new LedoitWolfShrinkageEstimator<double>().Estimate(panel);
        result[0, 0].Should().BeApproximately(13.0 / 3, 1e-14);
        result[1, 1].Should().BeApproximately(13.0 / 3, 1e-14);
        result[0, 1].Should().Be(0);
    }

    [Fact]
    public void ScaledIdentity_ZeroDispersion_ReturnsUniqueCovariance()
    {
        // LWID-ZEROD: target equals sample, so any finite blend gives2/3I.
        // The quotient0/0 is unnecessary; delta0 versusdelta1 is unobservable.
        double[,] panel = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
        var result = new LedoitWolfShrinkageEstimator<double>().Estimate(panel);
        result[0, 0].Should().BeApproximately(2.0 / 3, 1e-14);
        result[1, 1].Should().BeApproximately(2.0 / 3, 1e-14);
        result[0, 1].Should().Be(0);
    }

    [Fact]
    public void ConstantCorrelation_NegativeNumerator_ClipsToSample()
    {
        // LWCC-LOW: independent80-digit evaluation of Honey AppendixB gives
        // pi=84468.1824 and rho=84612.26904085529..., so raw intensity<0.
        // Expected matrix is exact centered scatter/4, not a call to the
        // library sample estimator. Lower clipping must return this matrix.
        decimal[,] panel =
        {
            { -3, 8, 5 }, { -4, 2, -25 }, { -2, 10, 25 },
            { -3, 10, 5 }, { -4, -4, -10 },
        };
        decimal[,] expected = { { .7m, 4.3m, 15 }, { 4.3m, 37.2m, 82.5m }, { 15, 82.5m, 350 } };
        var result = new LedoitWolfConstantCorrelationEstimator<decimal>().Estimate(panel);
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                result[i, j].Should().BeApproximately(expected[i, j], 0.00000000000000000001m);
            }
        }
    }

    [Fact]
    public void SingleFactor_ZeroMarket_LocalDiagonalTargetHasExactBlend()
    {
        // LWSF-DEGENERATE, local extension only: every equal-weight market
        // return is0, so the selected zero-beta target is diagonal.
        // S_T=(5/2)[[1,-1],[-1,1]], pi9,rho9/2,gamma25/2,
        // delta9/100. Bessel diagonal10/3, offdiagonal-(10/3)*(91/100).
        // This controls current local policy; the paper excludes zero s00.
        double[,] panel = { { 1, -1 }, { -1, 1 }, { 2, -2 }, { -2, 2 } };
        var result = new LedoitWolfSingleFactorEstimator<double>().Estimate(panel);
        result[0, 0].Should().BeApproximately(10.0 / 3, 1e-14);
        result[1, 1].Should().BeApproximately(10.0 / 3, 1e-14);
        result[0, 1].Should().BeApproximately(-91.0 / 30, 1e-14);
    }
}
