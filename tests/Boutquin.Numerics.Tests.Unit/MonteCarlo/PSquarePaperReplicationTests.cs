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

using System.Reflection;

using Boutquin.Numerics.MonteCarlo;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class PSquarePaperReplicationTests
{
    [Fact]
    public void TableI_ErratumStream_MatchesUnroundedPaperEquations()
    {
        // Jain & Chlamtac (1985), Table I, pp. 1080–1081; the explicit
        // p. 1078 erratum changes the second initial observation to 0.15.
        // Goldens independently evaluate the printed parabolic/linear
        // equations using Python fractions.Fraction, with no intermediate
        // rounding. Table I's arithmetic/rounding is not fully consistent:
        // e.g. its final q4 is 17.22, whereas those equations give 17.203904… .
        double[] stream =
        [
            0.02, 0.15, 0.74, 3.39, 0.83,
            22.37, 10.15, 15.43, 38.62, 15.92, 34.60, 10.28, 1.47,
            0.40, 0.05, 11.39, 0.27, 0.42, 0.09, 11.37,
        ];
        double[] expectedMedians =
        [
            0.74, 0.74, 2.1783333333333332, 4.752685185185185,
            4.752685185185185, 9.274704861111111, 9.274704861111111,
            9.274704861111111, 9.274704861111111, 6.297302000661376,
            6.297302000661376, 6.297302000661376, 6.297302000661376,
            4.440634353260337, 4.440634353260337,
        ];
        var estimator = new PSquareEstimator<double>(0.5);
        for (var i = 0; i < stream.Length; i++)
        {
            estimator.Add(stream[i]);
            if (i >= 5)
            {
                estimator.Estimate.Should().BeApproximately(expectedMedians[i - 5], 1e-12, "after observation {0}", i + 1);
            }
        }

        estimator.Count.Should().Be(20);

        // Inspect the full final marker state as printed in Table I, not only
        // the median, without widening the production API for diagnostics.
        var positions = (long[])typeof(PSquareEstimator<double>)
            .GetField("_n", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(estimator)!;
        var heights = (double[])typeof(PSquareEstimator<double>)
            .GetField("_q", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(estimator)!;
        positions.Should().Equal(1L, 6L, 10L, 16L, 20L);
        double[] expectedHeights = [0.02, 0.4938954475308642, 4.440634353260337, 17.20390427414021, 38.62];
        for (var i = 0; i < heights.Length; i++)
        {
            heights[i].Should().BeApproximately(expectedHeights[i], 1e-12, "marker {0}", i + 1);
        }
    }
}
