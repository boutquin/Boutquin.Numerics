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

public sealed class CorrelationVerificationTests : CrossLanguageVerificationBase
{
    [Fact]
    public void Pearson_MatchesNumpyCorrcoef()
    {
        using var doc = LoadVector("correlation");
        var x = GetDecimalArray(doc.RootElement.GetProperty("x"));
        var y = GetDecimalArray(doc.RootElement.GetProperty("y"));
        var expected = doc.RootElement.GetProperty("pearson").GetDouble();

        var actual = PearsonCorrelation.Compute(x, y);
        AssertScalarWithin((double)actual, expected, PrecisionNumeric, "pearson");
    }

    [Fact]
    public void Spearman_MatchesScipy()
    {
        using var doc = LoadVector("correlation");
        var x = GetDecimalArray(doc.RootElement.GetProperty("x"));
        var y = GetDecimalArray(doc.RootElement.GetProperty("y"));
        var expected = doc.RootElement.GetProperty("spearman").GetDouble();

        var actual = RankCorrelation.Spearman(x, y);
        AssertScalarWithin((double)actual, expected, PrecisionNumeric, "spearman");
    }

    [Fact]
    public void KendallTauB_MatchesScipy()
    {
        using var doc = LoadVector("correlation");
        var x = GetDecimalArray(doc.RootElement.GetProperty("x"));
        var y = GetDecimalArray(doc.RootElement.GetProperty("y"));
        var expected = doc.RootElement.GetProperty("kendall_tau_b").GetDouble();

        var actual = RankCorrelation.KendallTauB(x, y);
        AssertScalarWithin((double)actual, expected, PrecisionNumeric, "kendall τ-b");
    }

    [Fact]
    public void DistanceCorrelation_MatchesSzekely2007Reference()
    {
        using var doc = LoadVector("correlation");
        var x = GetDoubleArray(doc.RootElement.GetProperty("x"));
        var y = GetDoubleArray(doc.RootElement.GetProperty("y"));
        var expected = doc.RootElement.GetProperty("distance_correlation").GetDouble();

        var actual = DistanceCorrelation.Compute(x, y);
        AssertScalarWithin(actual, expected, PrecisionNumeric, "distance correlation");
    }
}
