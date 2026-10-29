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

public sealed class DistanceCorrelationTranslationTests
{
    [Fact]
    public void Compute_ExactlyAffineAtLargeOffset_ReturnsOne()
    {
        double[] x = [1e16, 1e16 + 2, 1e16 + 4, 1e16 + 6];
        double[] y = [0, 1, 2, 3];

        DistanceCorrelation<double>.Compute(x, y).Should().BeApproximately(1, 1e-14);
    }

    [Fact]
    public void Compute_BothSeriesTranslated_MatchesIndependentGolden()
    {
        double[] x = [0, 2, 4, 8, 12, 18, 24, 30];
        double[] y = [8, 0, 12, 2, 24, 4, 30, 18];
        // SRB doubly-centered distance formula evaluated in 70-digit decimal arithmetic.
        const double expected = 0.6518494559566712980269140176320893593376513379720926947322965195912174;

        DistanceCorrelation<double>.Compute(x.Select(v => v + 1e16).ToArray(),
            y.Select(v => v + 1e16).ToArray()).Should().BeApproximately(expected, 1e-14);
    }

    [Fact]
    public void Compute_OneSeriesTranslated_MatchesReferenceFormula()
    {
        double[] x = [1e16, 1e16 + 2, 1e16 + 4, 1e16 + 8];
        double[] y = [0, 1, 4, 9];
        // Independent 70-digit evaluation of the SRB biased distance-correlation formula.
        const double expected = 0.9917026572959774;

        DistanceCorrelation<double>.Compute(x, y).Should().BeApproximately(expected, 1e-14);
    }

    [Fact]
    public void Compute_FloatSeriesTranslated_MatchesReferenceFormula()
    {
        float[] x = [1e7f, 1e7f + 1, 1e7f + 2, 1e7f + 4];
        float[] y = [0, 1, 4, 9];

        DistanceCorrelation<float>.Compute(x, y).Should().BeApproximately(0.9917026572959774f, 1e-6f);
    }
}
