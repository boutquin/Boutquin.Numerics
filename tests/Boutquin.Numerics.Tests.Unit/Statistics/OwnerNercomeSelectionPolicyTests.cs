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

public sealed class OwnerNercomeSelectionPolicyTests
{
    [Fact]
    public void EqualZeroScores_RetainFirstPublishedCandidate()
    {
        // With one variable every projected covariance equals the complement moment,
        // so every selection score is exactly zero. The first published n=4 candidate
        // is m=3, leaving one observation and hence one of the four squared levels.
        // The final candidate m=2 leaves a two-element average, never one of these squares.
        var actual = new NercomeCovarianceEstimator<double>(null, 1, 0)
            .Estimate(new double[,] { { 1 }, { 2 }, { 4 }, { 8 } });
        actual[0, 0].Should().BeOneOf(1, 4, 16, 64);
    }
}
