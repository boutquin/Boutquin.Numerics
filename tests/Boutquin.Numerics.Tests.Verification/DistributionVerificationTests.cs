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

using Boutquin.Numerics.Distributions;

namespace Boutquin.Numerics.Tests.Verification;

public sealed class DistributionVerificationTests : CrossLanguageVerificationBase
{
    [Theory]
    [InlineData("cdf", "x", 807)]
    [InlineData("ppf", "p", 1151)]
    public void NormalFunctions_MatchHighPrecisionGaussianOracles(string kind, string inputName, int count)
    {
        using var doc = LoadVector("distributions");
        var data = doc.RootElement.GetProperty("high_precision").GetProperty(kind);
        var inputs = GetDoubleArray(data.GetProperty(inputName));
        var expected = GetDoubleArray(data.GetProperty("y"));
        Assert.Equal(count, inputs.Length);
        Assert.Equal(count, expected.Length);
        for (var i = 0; i < count; i++)
        {
            var actual = kind == "cdf"
                ? CumulativeNormal<double>.Evaluate(inputs[i])
                : InverseNormal<double>.Evaluate(inputs[i]);
            var tolerance = 4e-15 * Math.Abs(expected[i]);
            AssertScalarWithin(actual, expected[i], tolerance, $"{kind}({inputs[i]:R})");
        }
    }

    [Fact]
    public void NormalCdf_MatchesScipyReferenceValues()
    {
        using var doc = LoadVector("distributions");
        var cdf = doc.RootElement.GetProperty("cdf");
        var xs = GetDoubleArray(cdf.GetProperty("x"));
        var expected = GetDoubleArray(cdf.GetProperty("y"));
        Assert.NotEmpty(xs);
        Assert.Equal(xs.Length, expected.Length);

        for (var i = 0; i < xs.Length; i++)
        {
            var actual = NormalDistribution.Cdf(xs[i]);
            // Scale the absolute error by the smaller tail probability: a fixed
            // absolute tolerance can accept zero (or one) for a finite tail.
            var tailProbability = Math.Min(expected[i], 1.0 - expected[i]);
            AssertScalarWithin(actual, expected[i], PrecisionNumeric * tailProbability, $"Φ({xs[i]})");
        }
    }

    [Fact]
    public void NormalInverseCdf_MatchesScipyReferenceValues()
    {
        using var doc = LoadVector("distributions");
        var ppf = doc.RootElement.GetProperty("ppf");
        var ps = GetDoubleArray(ppf.GetProperty("p"));
        var expected = GetDoubleArray(ppf.GetProperty("y"));
        Assert.NotEmpty(ps);
        Assert.Equal(ps.Length, expected.Length);

        for (var i = 0; i < ps.Length; i++)
        {
            var actual = NormalDistribution.InverseCdf(ps[i]);
            // Deep tails diverge slightly; statistical tolerance is ample.
            AssertScalarWithin(actual, expected[i], PrecisionStatistical, $"Φ⁻¹({ps[i]})");
        }
    }
}
