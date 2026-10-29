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

public sealed class ArenaR3B07RegressionTests
{
    [Fact]
    public void Spearman_LongFiniteHalfSeries_PreservesSelfCorrelation()
    {
        var values = new Half[65536];
        values[^1] = (Half)1;
        var act = () => RankCorrelation<Half>.Spearman(values, values);
        act.Should().NotThrow();
        act().Should().Be((Half)1);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Pearson_TranslatedDecimalSpread_PreservesSelfCorrelation(bool rolling)
    {
        decimal[] values = [1, 2, 3, 4, 10, 2.5m, 7];
        var shifted = values.Select(x => x + 1e20m).ToArray();
        var actual = rolling ? PearsonCorrelation<decimal>.Rolling(shifted, shifted, shifted.Length)[0]
            : PearsonCorrelation<decimal>.Compute(shifted, shifted);
        actual.Should().BeApproximately(1m, 1e-24m);
    }

    [Fact]
    public void Skewness_TranslatedDecimalSpread_MatchesIndependentOracle()
    {
        decimal[] values = [1, 2, 3, 4, 10, 2.5m, 7];
        var shifted = values.Select(x => x + 1e20m).ToArray();
        var act = () => SampleSkewness<decimal>.Compute(shifted);
        act.Should().NotThrow();
        act().Should().BeApproximately(1.1853364305393075024162952328m, 1e-24m);
    }

    [Fact]
    public void Kurtosis_TranslatedDecimalSpread_MatchesIndependentOracle()
    {
        decimal[] values = [1, 2, 3, 4, 10, 2.5m, 7];
        var shifted = values.Select(x => x + 1e20m).ToArray();
        var act = () => SampleExcessKurtosis<decimal>.Compute(shifted);
        act.Should().NotThrow();
        act().Should().BeApproximately(0.5843388413282408546348382167m, 1e-24m);
    }
}
