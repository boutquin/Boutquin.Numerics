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

namespace Boutquin.Numerics.Tests.Verification;

/// <summary>
/// Python cross-checks for <see cref="SampleSkewness{T}"/> and
/// <see cref="SampleExcessKurtosis{T}"/>. The reference generator
/// <c>tests/Verification/generate_sample_moments_vectors.py</c> uses
/// <c>scipy.stats.skew(data, bias=False)</c> and
/// <c>scipy.stats.kurtosis(data, fisher=True, bias=False)</c> — a fully
/// independent C/Cython implementation — on six synthetic datasets
/// covering normal, right-skewed, left-skewed, platykurtic, leptokurtic,
/// and small-sample regimes.
/// </summary>
/// <remarks>
/// Both the scipy reference and the Boutquin.Numerics two-pass algorithm
/// operate in double precision; shared algorithmic bugs cannot exist
/// because scipy uses a completely different computational path.
/// Tolerance is <see cref="CrossLanguageVerificationBase.PrecisionExact"/>
/// (1e-10 absolute) — achievable because both sides accumulate
/// double-precision floating-point with the same bias-correction formula
/// and the inputs are modest in magnitude.
/// </remarks>
public sealed class SampleMomentsVerificationTests : CrossLanguageVerificationBase
{
    [Fact]
    public void SampleSkewness_AllCases_MatchScipy()
    {
        using var doc = LoadVector("sample_moments");
        var cases = doc.RootElement.GetProperty("cases");

        foreach (var c in cases.EnumerateArray())
        {
            var name = c.GetProperty("name").GetString()!;
            var data = GetDoubleArray(c.GetProperty("data"));
            var expected = c.GetProperty("skewness").GetDouble();

            var actual = SampleSkewness<double>.Compute(data);

            AssertScalarWithin(actual, expected, PrecisionExact, $"skewness/{name}");
        }
    }

    [Fact]
    public void SampleExcessKurtosis_AllCases_MatchScipy()
    {
        using var doc = LoadVector("sample_moments");
        var cases = doc.RootElement.GetProperty("cases");

        foreach (var c in cases.EnumerateArray())
        {
            var name = c.GetProperty("name").GetString()!;
            var data = GetDoubleArray(c.GetProperty("data"));
            var expected = c.GetProperty("excess_kurtosis").GetDouble();

            var actual = SampleExcessKurtosis<double>.Compute(data);

            AssertScalarWithin(actual, expected, PrecisionExact, $"excess_kurtosis/{name}");
        }
    }
}
