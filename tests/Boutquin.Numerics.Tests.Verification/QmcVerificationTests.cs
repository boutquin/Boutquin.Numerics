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

namespace Boutquin.Numerics.Tests.Verification;

public sealed class QmcVerificationTests : CrossLanguageVerificationBase
{
    // SciPy fixtures include index zero. Sobol emits it; Halton begins at index one.
    private const int HaltonOriginOffset = 1;

    [Fact]
    public void SobolSequence_MatchesReferencePointsExactly()
    {
        using var doc = LoadVector("qmc");
        var dim = doc.RootElement.GetProperty("dimension").GetInt32();
        var count = doc.RootElement.GetProperty("num_points").GetInt32();
        var expected = doc.RootElement.GetProperty("sobol")
            .EnumerateArray().Select(GetDoubleArray).ToArray();

        var sobol = new SobolSequence(dim);
        // Author point indices0..count-1, including the origin.
        for (var i = 0; i < count; i++)
        {
            var p = sobol.Next();
            for (var d = 0; d < dim; d++)
            {
                AssertScalarWithin(p[d], expected[i][d], PrecisionExact, $"Sobol point[{i}] dim {d}");
            }
        }
    }

    [Fact]
    public void SobolSequence_Dimension32Point2MatchesJoeKuo()
    {
        // SciPy 1.17.1's unscrambled Joe-Kuo D(6) table, after skipping the origin.
        var sobol = new SobolSequence(32, skip: 1);
        sobol.Next();
        var second = sobol.Next();
        Assert.Equal(0.25, second[31]);
    }

    [Fact]
    public void SobolSequence_AllEmbeddedRowsAndLaterBitsMatchSciPy()
    {
        using var doc = LoadVector("qmc_sobol_extended");
        var dimension = doc.RootElement.GetProperty("dimension").GetInt32();
        var indices = doc.RootElement.GetProperty("indices")
            .EnumerateArray().Select(item => item.GetInt32()).ToArray();
        var expected = doc.RootElement.GetProperty("sobol")
            .EnumerateArray().Select(GetDoubleArray).ToArray();

        var sobol = new SobolSequence(dimension, skip: 1);
        var vectorIndex = 0;
        for (var pointIndex = 1; pointIndex <= indices[^1]; pointIndex++)
        {
            var point = sobol.Next();
            if (pointIndex != indices[vectorIndex])
            {
                continue;
            }

            for (var d = 0; d < dimension; d++)
            {
                AssertScalarWithin(point[d], expected[vectorIndex][d], PrecisionExact,
                    $"Sobol point[{pointIndex}] dim {d}");
            }

            vectorIndex++;
        }
    }

    [Fact]
    public void HaltonSequence_MatchesReferencePointsExactly()
    {
        using var doc = LoadVector("qmc");
        var dim = doc.RootElement.GetProperty("dimension").GetInt32();
        var count = doc.RootElement.GetProperty("num_points").GetInt32();
        var expected = doc.RootElement.GetProperty("halton")
            .EnumerateArray().Select(GetDoubleArray).ToArray();

        var halton = new HaltonSequence(dim);
        for (var i = 0; i + HaltonOriginOffset < count; i++)
        {
            var p = halton.Next();
            for (var d = 0; d < dim; d++)
            {
                AssertScalarWithin(p[d], expected[i + HaltonOriginOffset][d], PrecisionExact, $"Halton point[{i + HaltonOriginOffset}] dim {d}");
            }
        }
    }

    [Fact]
    public void HaltonSequence_AllPrimeRadices_MatchSciPyPointAtIndexOne()
    {
        // Independent oracle: scipy.stats.qmc.Halton(d=50, scramble=False).random(2)[1].
        // Index one is 1 / radix in every dimension, so this pins all 50 radices.
        double[] expected = [
            0.5, 0.3333333333333333, 0.2, 0.14285714285714285, 0.09090909090909091,
            0.07692307692307693, 0.058823529411764705, 0.05263157894736842, 0.043478260869565216,
            0.034482758620689655, 0.03225806451612903, 0.02702702702702703, 0.024390243902439025,
            0.023255813953488372, 0.02127659574468085, 0.018867924528301886, 0.01694915254237288,
            0.01639344262295082, 0.014925373134328358, 0.014084507042253521, 0.0136986301369863,
            0.012658227848101266, 0.012048192771084338, 0.011235955056179775, 0.010309278350515464,
            0.009900990099009901, 0.009708737864077669, 0.009345794392523364, 0.009174311926605505,
            0.008849557522123894, 0.007874015748031496, 0.007633587786259542, 0.0072992700729927005,
            0.007194244604316547, 0.006711409395973154, 0.006622516556291391, 0.006369426751592357,
            0.006134969325153374, 0.005988023952095809, 0.005780346820809248, 0.00558659217877095,
            0.0055248618784530384, 0.005235602094240838, 0.0051813471502590676, 0.005076142131979695,
            0.005025125628140704, 0.004739336492890996, 0.004484304932735426, 0.004405286343612335,
            0.004366812227074236,
        ];

        var actual = new HaltonSequence(50).Next();
        for (var d = 0; d < expected.Length; d++)
        {
            AssertScalarWithin(actual[d], expected[d], PrecisionExact, $"Halton point[1] dim {d}");
        }
    }
}
