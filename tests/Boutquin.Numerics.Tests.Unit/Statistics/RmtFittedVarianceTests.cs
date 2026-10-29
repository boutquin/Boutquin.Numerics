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

public sealed class RmtFittedVarianceTests
{
    [Fact]
    public void Denoising_FitsNoiseVarianceBeforeSelectingSignalCount()
    {
        var panel = BuildPanel(17);
        var sample = new SampleCovarianceEstimator<double>().Estimate(panel);
        var (_, _, signalCount) = DenoisedCovarianceEstimator<double>.DenoiseCorrelation(sample, 500);
        // Independent NumPy eigvalsh + SciPy differential_evolution on the book's
        // 1000-point MP / Gaussian KDE (bandwidth .01) SSE gives sigma²=.30076066.
        // Its edge .521... selects 2 signals; fixing sigma²=1 selects only 1.
        signalCount.Should().Be(2, "the fitted noise variance must determine the MP edge");
    }

    private static double[,] BuildPanel(uint seed)
    {
        var state = seed;
        double Normal()
        {
            var value = -6.0;
            for (var k = 0; k < 12; k++)
            {
                state = unchecked(1664525 * state + 1013904223);
                value += state / 4294967296.0;
            }

            return value;
        }

        var panel = new double[500, 50];
        for (var i = 0; i < 500; i++)
        {
            var factor = Normal();
            for (var j = 0; j < 50; j++)
            {
                panel[i, j] = 1.5 * factor * (0.8 + 0.4 * j / 49) + Normal();
            }
        }

        return panel;
    }
}
