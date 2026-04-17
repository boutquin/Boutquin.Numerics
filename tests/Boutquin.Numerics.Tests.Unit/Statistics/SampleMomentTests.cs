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

public sealed class SampleMomentTests
{
    // -------------------------------------------------------------------------
    // SampleSkewness<T> tests
    // -------------------------------------------------------------------------

    [Fact]
    public void Skewness_SymmetricData_ReturnsNearZero()
    {
        // {1,2,3,4,5} is perfectly symmetric around 3; skewness should be 0.
        double[] data = [1.0, 2.0, 3.0, 4.0, 5.0];
        var result = SampleSkewness<double>.Compute(data);
        result.Should().BeApproximately(0.0, 1e-10);
    }

    [Fact]
    public void Skewness_RightSkewed_ReturnsPositive()
    {
        // Long right tail: the outlier 10 pulls the mean above the mode.
        double[] data = [1.0, 1.0, 1.0, 2.0, 10.0];
        var result = SampleSkewness<double>.Compute(data);
        result.Should().BePositive();
    }

    [Fact]
    public void Skewness_LeftSkewed_ReturnsNegative()
    {
        // Long left tail: the outlier -10 pulls the mean below the mode.
        double[] data = [-10.0, -2.0, -1.0, -1.0, -1.0];
        var result = SampleSkewness<double>.Compute(data);
        result.Should().BeNegative();
    }

    [Fact]
    public void Skewness_ThreeElements_MinimumValid()
    {
        // Three elements is the minimum; result should be a finite number.
        double[] data = [1.0, 2.0, 4.0];
        var result = SampleSkewness<double>.Compute(data);
        double.IsFinite(result).Should().BeTrue();
    }

    [Fact]
    public void Skewness_TwoElements_ThrowsArgumentException()
    {
        double[] data = [1.0, 2.0];
        var act = () => SampleSkewness<double>.Compute(data);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Skewness_EmptySpan_ThrowsArgumentException()
    {
        var act = () => SampleSkewness<double>.Compute([]);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Skewness_AllSameValue_ThrowsInvalidOperationException()
    {
        double[] data = [5.0, 5.0, 5.0, 5.0];
        var act = () => SampleSkewness<double>.Compute(data);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Skewness_GenericDoubleMatchesDecimal()
    {
        // Both type instantiations should agree on a well-conditioned sample.
        double[] dataDouble = [1.0, 2.0, 3.0, 4.0, 5.0];
        decimal[] dataDecimal = [1m, 2m, 3m, 4m, 5m];

        var resultDouble = SampleSkewness<double>.Compute(dataDouble);
        var resultDecimal = (double)SampleSkewness<decimal>.Compute(dataDecimal);

        resultDouble.Should().BeApproximately(resultDecimal, 1e-10);
    }

    [Fact]
    public void Skewness_KnownValue()
    {
        // For {2,8,0,4,1,9,9,0}: n=8, mean=4.125, s≈3.9799
        // Verified analytically: skewness ≈ 0.3305821804079748
        double[] data = [2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0];
        const double expected = 0.3305821804079748;
        var result = SampleSkewness<double>.Compute(data);
        result.Should().BeApproximately(expected, 1e-10);
    }

    // -------------------------------------------------------------------------
    // SampleExcessKurtosis<T> tests
    // -------------------------------------------------------------------------

    [Fact]
    public void ExcessKurtosis_NormalLikeSample_NearZero()
    {
        // {-3,-2,-1,0,1,2,3} is a small symmetric sample. With n=7 the bias-corrected
        // G₂ ≈ -1.2 (a platykurtic uniform-like distribution has negative excess kurtosis),
        // so |G₂| should be below 2 — far from a heavy-tailed positive value.
        double[] data = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        var result = SampleExcessKurtosis<double>.Compute(data);
        Math.Abs(result).Should().BeLessThan(2.0);
    }

    [Fact]
    public void ExcessKurtosis_Leptokurtic_ReturnsPositive()
    {
        // Heavy tails: two large outliers among many zeros → positive excess kurtosis.
        double[] data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, -1000.0];
        var result = SampleExcessKurtosis<double>.Compute(data);
        result.Should().BePositive();
    }

    [Fact]
    public void ExcessKurtosis_FourElements_MinimumValid()
    {
        // Four elements is the minimum; result should be a finite number.
        double[] data = [1.0, 2.0, 3.0, 5.0];
        var result = SampleExcessKurtosis<double>.Compute(data);
        double.IsFinite(result).Should().BeTrue();
    }

    [Fact]
    public void ExcessKurtosis_ThreeElements_ThrowsArgumentException()
    {
        double[] data = [1.0, 2.0, 3.0];
        var act = () => SampleExcessKurtosis<double>.Compute(data);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ExcessKurtosis_EmptySpan_ThrowsArgumentException()
    {
        var act = () => SampleExcessKurtosis<double>.Compute([]);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ExcessKurtosis_AllSameValue_ThrowsInvalidOperationException()
    {
        double[] data = [3.0, 3.0, 3.0, 3.0];
        var act = () => SampleExcessKurtosis<double>.Compute(data);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void ExcessKurtosis_GenericDoubleMatchesDecimal()
    {
        // Both type instantiations should agree closely on a well-conditioned sample.
        double[] dataDouble = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        decimal[] dataDecimal = [1m, 2m, 3m, 4m, 5m, 6m, 7m];

        var resultDouble = SampleExcessKurtosis<double>.Compute(dataDouble);
        var resultDecimal = (double)SampleExcessKurtosis<decimal>.Compute(dataDecimal);

        resultDouble.Should().BeApproximately(resultDecimal, 1e-8);
    }

    [Fact]
    public void ExcessKurtosis_KnownValue()
    {
        // For {2,8,0,4,1,9,9,0}: n=8, mean=4.125, s≈3.9799
        // Verified analytically: G₂ ≈ -2.098602258096086
        double[] data = [2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0];
        const double expected = -2.098602258096086;
        var result = SampleExcessKurtosis<double>.Compute(data);
        result.Should().BeApproximately(expected, 1e-10);
    }
}
