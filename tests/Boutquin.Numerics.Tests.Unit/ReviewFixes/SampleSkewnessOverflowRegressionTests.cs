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

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// F-L1-04: sample skewness is scale-invariant, so scaling a data set by a large
/// constant must not change the computed skewness (and must not silently collapse it
/// to 0 because the intermediate Welford variance overflowed).
/// </summary>
public sealed class SampleSkewnessOverflowRegressionTests
{
    private static readonly double[] s_unitScaleData = [1, -1, 3, 0, 2, 7, -2];

    [Fact]
    public void Compute_LargeScaleDouble_MatchesUnitScaleSkewness()
    {
        var expected = SampleSkewness<double>.Compute(s_unitScaleData);

        var scaled = Array.ConvertAll(s_unitScaleData, v => v * 1e160);
        var actual = SampleSkewness<double>.Compute(scaled);

        actual.Should().BeApproximately(expected, 1e-9);
    }

    [Fact]
    public void Compute_LargeScaleDecimal_DoesNotThrow()
    {
        var unitScale = Array.ConvertAll(s_unitScaleData, v => (decimal)v);
        var expected = SampleSkewness<decimal>.Compute(unitScale);

        var scaled = Array.ConvertAll(s_unitScaleData, v => (decimal)v * 1_000_000_000_000_000m);

        var act = () => SampleSkewness<decimal>.Compute(scaled);

        act.Should().NotThrow();
        Math.Abs(SampleSkewness<decimal>.Compute(scaled) - expected).Should().BeLessThan(0.0000001m);
    }
}
