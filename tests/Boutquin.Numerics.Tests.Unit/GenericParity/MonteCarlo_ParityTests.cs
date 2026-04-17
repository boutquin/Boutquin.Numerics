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

namespace Boutquin.Numerics.Tests.Unit.GenericParity;

public sealed class MonteCarlo_ParityTests
{
    private static readonly decimal[] s_dataDecimal = { 1.5m, 2.3m, 3.1m, 4.7m, 5.9m, 6.2m, 7.8m, 8.4m, 9.1m, 10.0m };
    private static readonly double[] s_dataDouble = { 1.5, 2.3, 3.1, 4.7, 5.9, 6.2, 7.8, 8.4, 9.1, 10.0 };

    [Fact]
    public void BootstrapResampler_GenericMatchesLegacy_AtDecimal()
    {
        var blockSize = 1;
        var seed = 42;

        var legacy = BootstrapResampler.FromSeed(blockSize, seed);
        var generic = BootstrapResampler<decimal>.FromSeed(blockSize, seed);

        var legacyResult = legacy.Resample(s_dataDecimal);
        var genericResult = generic.Resample(s_dataDecimal);

        genericResult.Length.Should().Be(legacyResult.Length);
        for (var i = 0; i < legacyResult.Length; i++)
        {
            genericResult[i].Should().Be(legacyResult[i]);
        }
    }

    [Fact]
    public void PSquareEstimator_GenericMatchesLegacy_AtDouble()
    {
        var percentile = 0.5;

        var legacy = new PSquareEstimator(percentile);
        var generic = new PSquareEstimator<double>(percentile);

        foreach (var value in s_dataDouble)
        {
            legacy.Add(value);
            generic.Add(value);
        }

        var legacyEstimate = legacy.Estimate;
        var genericEstimate = generic.Estimate;

        genericEstimate.Should().BeApproximately(legacyEstimate, 1e-12);
    }

    [Fact]
    public void HaltonSequence_GenericMatchesLegacy_AtDouble()
    {
        var dimension = 2;
        var skip = 0;

        var legacy = new HaltonSequence(dimension, skip);
        var generic = new HaltonSequence<double>(dimension, skip);

        var legacyNext = legacy.Next();
        var genericNext = generic.Next();

        genericNext.Length.Should().Be(legacyNext.Length);
        for (var i = 0; i < legacyNext.Length; i++)
        {
            genericNext[i].Should().BeApproximately(legacyNext[i], 1e-12);
        }
    }

    [Fact]
    public void SobolSequence_GenericMatchesLegacy_AtDouble()
    {
        var dimension = 2;
        var skip = 0;

        var legacy = new SobolSequence(dimension, skip);
        var generic = new SobolSequence<double>(dimension, skip);

        var legacyNext = legacy.Next();
        var genericNext = generic.Next();

        genericNext.Length.Should().Be(legacyNext.Length);
        for (var i = 0; i < legacyNext.Length; i++)
        {
            genericNext[i].Should().BeApproximately(legacyNext[i], 1e-12);
        }
    }
}
