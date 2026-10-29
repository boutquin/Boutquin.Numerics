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
public sealed class C4DscePartialSupportTests
{
    [Fact]
    public void PartialSignedSupport_MatchesAnalyticProjectorSum()
    {
        // DoublySparseEstimator-013/-014: selected MP/entry-threshold variant.
        // Balanced orthogonal signs give correlation D*C*D, where
        // C=[[1,.8,.6],[.8,1,.6],[.6,.6,1]] and D=diag(1,-1,1).
        // Its signal is lambda=(14+sqrt(88))/10 with direction (1,-1,r),
        // r=(sqrt(88)-4)/6. Threshold .57 keeps the first two magnitudes
        // (~.598) and removes the third (~.536). The unit retained direction
        // is w=(1,-1,0)/sqrt(2). Noise averaging gives mu=(3-lambda)/2.
        // Therefore the clean matrix is lambda*w*w' + mu*(I-v*v'), with
        // v=(1,-1,r)/sqrt(2+r*r). This projector identity does not run or
        // reproduce a numerical eigensolver. Original standard deviations
        // (1,2,3) then rescale the entries. No unit-diagonal normalization.
        var observations = new double[16, 3];
        var scale = Math.Sqrt(15.0 / 16);
        for (var row = 0; row < 16; row++)
        {
            var a = (row & 1) == 0 ? -1 : 1;
            var b = (row & 2) == 0 ? -1 : 1;
            var c = (row & 4) == 0 ? -1 : 1;
            observations[row, 0] = scale * a;
            observations[row, 1] = -2 * scale * (.8 * a + .6 * b);
            observations[row, 2] = 3 * scale * (.6 * a + .2 * b + Math.Sqrt(.6) * c);
        }

        var lambda = (14 + Math.Sqrt(88)) / 10;
        var mu = (3 - lambda) / 2;
        var r = (Math.Sqrt(88) - 4) / 6;
        var aSquared = 1 / (2 + r * r);
        var ab = r * aSquared;
        double[,] expected =
        {
            { lambda / 2 + mu * (1 - aSquared), 2 * (-lambda / 2 + mu * aSquared), -3 * mu * ab },
            { 2 * (-lambda / 2 + mu * aSquared), 4 * (lambda / 2 + mu * (1 - aSquared)), 6 * mu * ab },
            { -3 * mu * ab, 6 * mu * ab, 9 * mu * (1 - r * r * aSquared) },
        };
        var actual = new DoublySparseEstimator<double>(.57).Estimate(observations);
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                actual[i, j].Should().BeApproximately(expected[i, j], 1e-10);
            }
        }
    }
}
