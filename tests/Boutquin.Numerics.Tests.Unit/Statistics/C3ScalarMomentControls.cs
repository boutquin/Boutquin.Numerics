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

public sealed class C3ScalarMomentControls
{
    [Fact]
    public void AdjustedMoments_HaveExactRationalFourPointOracle()
    {
        // Mean1, unbiased variance4, z=(-1/2,-1/2,-1/2,3/2).
        // Sum z^3=3 and sum z^4=21/4; G1=2, G2=4.
        double[] x = [0, 0, 0, 4];
        SampleSkewness<double>.Compute(x).Should().BeApproximately(2, 1e-14);
        // Reflecting the exact panel negates its cubic sum, not its squared sum.
        SampleSkewness<double>.Compute([-4, 0, 0, 0]).Should().BeApproximately(-2, 1e-14);
        SampleExcessKurtosis<double>.Compute(x).Should().BeApproximately(4, 1e-14);
        decimal[] d = [0m, 0m, 0m, 4m];
        SampleSkewness<decimal>.Compute(d).Should().BeApproximately(2m, 1e-25m);
        SampleExcessKurtosis<decimal>.Compute(d).Should().BeApproximately(4m, 1e-25m);
    }

    [Fact]
    public void MomentMinimums_RejectEachTooShortInput()
    {
        Action skew = () => SampleSkewness<double>.Compute([0, 1]);
        Action kurt = () => SampleExcessKurtosis<double>.Compute([0, 1, 2]);
        skew.Should().Throw<ArgumentException>();
        kurt.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ConstantMoments_RejectUndefinedStandardization()
    {
        Action skew = () => SampleSkewness<double>.Compute([3, 3, 3]);
        Action kurt = () => SampleExcessKurtosis<double>.Compute([3, 3, 3, 3]);
        skew.Should().Throw<InvalidOperationException>();
        kurt.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Pearson_ExactCenteredProductsAndRollingWindows()
    {
        // First window: Sxx=14/3,Syy=26/3,Sxy=-1/3.
        double[] x = [1, 2, 4];
        double[] y = [3, -1, 2];
        var first = -1 / Math.Sqrt(364);
        PearsonCorrelation<double>.Compute(x, y).Should().BeApproximately(first, 1e-14);
        double[] shiftedX = [1000000000001, 1000000000002, 1000000000004];
        PearsonCorrelation<double>.Compute(shiftedX, y).Should().BeApproximately(first, 1e-14);
        // Second window: Sxx=56/3,Syy=38/3,Sxy=44/3.
        var rolling = PearsonCorrelation<double>.Rolling([1, 2, 4, 8], [3, -1, 2, 4], 3);
        rolling.Length.Should().Be(2);
        rolling[0].Should().BeApproximately(first, 1e-14);
        rolling[1].Should().BeApproximately(44 / Math.Sqrt(2128), 1e-14);
    }

    [Fact]
    public void Pearson_RejectsMismatchesAndPreservesLocalDegeneratePolicy()
    {
        Action mismatch = () => PearsonCorrelation<double>.Compute([1, 2, 3], [1, 2]);
        mismatch.Should().Throw<ArgumentException>();
        PearsonCorrelation<double>.Compute([1, 2], [3, 4]).Should().Be(0);
        PearsonCorrelation<double>.Compute([3, 3, 3], [1, 2, 3]).Should().Be(0);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(-1)]
    public void Pearson_ClampsPerfectLinearPatterns(int sign)
    {
        // Both normalized centered squared sums are3/4. Their separate square
        // roots round down, so the unclipped double ratio exceeds1 by one ULP.
        PearsonCorrelation<double>.Compute([0, 0, 0, 4], [0, 0, 0, sign * 8]).Should().Be(sign);
    }
}
