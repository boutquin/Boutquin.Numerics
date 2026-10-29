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

using Boutquin.Numerics.MonteCarlo;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class C3ResamplingPolitisWhiteTests
{
    [Fact]
    public void PolitisWhite_ExactRationalCovariances_FirstWindow()
    {
        // Corrected author common-row sample covariances: G=702422/4959, g=444272/1653.
        double[] x = [8, 3, 14, 12, 18, 20, 16, 11, 5, 0, 0, 2, -4, -4, 0, 5, 15, 12, 12, 12, 9, 5, -7, -7, -13, 3, 7, 4, 12, 1, -1, -6, -16, -16, -6, -4, -5, 4, -6, -4, 10, 8, 11, 14, 15, 4, 0, 2, 0, 11, 24, 22, 12, 15, 7, 11, 7, -7, -12, -16, -1, 10, 20, 13];
        var expected = 2.6098282616481647;
        PolitisWhiteBlockLength.EstimateDouble(x).Should().BeApproximately(expected, 1e-12);
    }

    [Fact]
    public void PolitisWhite_ExactRationalCovariances_FallbackHorizon()
    {
        // Corrected author fallback M13: G=17364/5525, g=3514/5525.
        double[] pattern = [0, 0, 0, 1, 2, 3, 2, 1];
        var x = Enumerable.Range(0, 64).Select(i => pattern[i % 8]).ToArray();
        var expected = 11.604470744866159;
        PolitisWhiteBlockLength.EstimateDouble(x).Should().BeApproximately(expected, 1e-11);
    }

    [Fact]
    public void PolitisWhite_CorrectedPopulationFormula_ExactCube()
    {
        // G=3,g(0)=6,N2048: D=72, b³=2*9/72*2048=512, b=8.
        PolitisWhiteBlockLength.StationaryOptimum(3, 6, 2048).Should().BeApproximately(8, 1e-13);
        PolitisWhiteBlockLength.StationaryOptimum(-3, 6, 2048).Should().BeApproximately(8, 1e-13);
    }
}
