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
public sealed class C4GraphicalLassoCertificateTests
{
    [Fact]
    public void ExplicitWeightedObjective_MatchesRationalKktCertificate()
    {
        // FHT weighted objective, explicitly Pii=0. This does not choose the
        // unresolved default diagonal penalty. Theta=[[2,-.5,0],[-.5,1.5,.3],
        // [0,.3,1]] is SPD (leading minors2,11/4,257/100).
        // Its exact cofactor inverse is W below. Construct S=W-P*Gamma,
        // P=.05*weights, Gamma01=-1, Gamma12=+1, Gamma02=-1/4.
        // Nonzero signs agree with Theta and the zero-edge subgradient is
        // strictly inside[-1,1]. Thus weighted KKT plus strict concavity of
        // logdet proves this unique optimum, independently of any solver.
        double[,] expectedCovariance =
        {
            { 141.0 / 257, 50.0 / 257, -15.0 / 257 },
            { 50.0 / 257, 200.0 / 257, -60.0 / 257 },
            { -15.0 / 257, -60.0 / 257, 275.0 / 257 },
        };
        double[,] sample =
        {
            { 141.0 / 257, 50.0 / 257 + .05, -15.0 / 257 + .025 },
            { 50.0 / 257 + .05, 200.0 / 257, -60.0 / 257 - .15 },
            { -15.0 / 257 + .025, -60.0 / 257 - .15, 275.0 / 257 },
        };
        double[,] weights = { { 0, 1, 2 }, { 1, 0, 3 }, { 2, 3, 0 } };
        double[,] expectedPrecision = { { 2, -.5, 0 }, { -.5, 1.5, .3 }, { 0, .3, 1 } };
        var result = GraphicalLassoSolver<double>.Solve(sample, .05,
            new GraphicalLassoOptions<double>(MaxIterations: 2000, Tolerance: 1e-12, Weights: weights));
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                result.Precision[i, j].Should().BeApproximately(expectedPrecision[i, j], 2e-7);
                result.Covariance[i, j].Should().BeApproximately(expectedCovariance[i, j], 2e-7);
            }
        }
    }
}
