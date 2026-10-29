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

public sealed class LowerTailDependenceVerificationTests : CrossLanguageVerificationBase
{
    [Fact]
    public void MonotoneLambda_MatchesNumpyReference()
    {
        using var doc = LoadVector("lower_tail");
        var expected = doc.RootElement.GetProperty("monotone_lambda").GetDouble();

        var n = doc.RootElement.GetProperty("n").GetInt32();
        var x = Enumerable.Range(1, n).Select(i => (double)i).ToArray();
        var result = LowerTailDependence<double>.Estimate(x, x);

        AssertScalarWithin(result.Lambda, expected, PrecisionExact, "monotone_lambda");
    }

    [Fact]
    public void IndependentLambda_MatchesNumpyReference()
    {
        using var doc = LoadVector("lower_tail");
        var expected = doc.RootElement.GetProperty("independent_lambda").GetDouble();
        var x = GetDoubleArray(doc.RootElement.GetProperty("independent_x"));
        var y = GetDoubleArray(doc.RootElement.GetProperty("independent_y"));

        var result = LowerTailDependence<double>.Estimate(x, y);

        AssertScalarWithin(result.Lambda, expected, PrecisionExact, "independent_lambda");
    }
}
