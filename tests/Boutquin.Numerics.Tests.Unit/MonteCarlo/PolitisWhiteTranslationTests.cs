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

public sealed class PolitisWhiteTranslationTests
{
    [Fact]
    public void EstimateDouble_TranslatedRamp_PreservesBlockLength()
    {
        var series = Enumerable.Range(0, 128).Select(i => 1e16 + 2 * (i % 16)).ToArray();

        PolitisWhiteBlockLength.EstimateDouble(series).Should().BeApproximately(33.858324503742359, 1e-12);
    }

    [Theory]
    [InlineData(64, 7.2118668796243934)]
    [InlineData(128, 14.725319703185178)]
    [InlineData(256, 48)]
    public void EstimateDouble_TranslatedSteps_MatchesDecimalOracle(int count, double expected)
    {
        var series = Enumerable.Range(0, count).Select(i => 1e16 + 2 * ((i / 8) % 4)).ToArray();

        PolitisWhiteBlockLength.EstimateDouble(series).Should().BeApproximately(expected, 1e-11);
    }

    [Fact]
    public void EstimateDouble_TranslatedOscillation_MatchesPaperFormula()
    {
        var series = Enumerable.Range(0, 200).Select(i => 1e16 + 2 * Math.Round(5 * Math.Sin(0.3 * i))).ToArray();

        PolitisWhiteBlockLength.EstimateDouble(series).Should().BeApproximately(43, 1e-12);
    }
}
