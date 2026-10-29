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

/// <summary>
/// Range controls for three statistics reductions whose intermediate quantity left its type's range while the
/// returned quantity did not: the tie-group rank sum of the lower-tail pseudo-observations (an int sum that wraps
/// at 2^31), the automatic Newey-West pilot spectral sums (raw moment units, squared), and the graphical lasso
/// mean-change normalisation (p x p formed in Half). Every expected value is derived from the estimator's
/// stated homogeneity or from exact integer arithmetic, never from the code under test.
/// </summary>
public sealed class DependenceAndSpectralRangeTests
{
    // F-L1-01. The pseudo-observation of the top-ranked singleton of a series of 2^30 observations has
    // start = end = 2^30 - 1, so the rank sum start + end + 2 is exactly 2^31, one past int.MaxValue. A full
    // public-entry run needs about 15 GB, so the sum is pinned at its seam: it must be exact in long.
    [Fact]
    public void RankSum_AtTopRankOfTwoPowerThirtySeries_IsExactlyTwoPowerThirtyOne()
    {
        const int top = (1 << 30) - 1;

        LowerTailDependence<double>.RankSum(top, top).Should().Be(2147483648L);
        LowerTailDependence<Half>.RankSum(top, top).Should().Be(2147483648L);
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(5, 9)]
    [InlineData(1_073_741_823, 1_073_741_823)]
    [InlineData(1_500_000_000, 2_000_000_000)]
    [InlineData(int.MaxValue - 2, int.MaxValue - 2)]
    public void RankSum_EqualsExactIntegerSumForAnyIndices(int start, int end)
    {
        LowerTailDependence<float>.RankSum(start, end).Should().Be((long)start + end + 2);
    }

    // F-L1-02. The automatic selector consumes only the ratio s1 / s0 of the pilot spectral sums, which is
    // invariant to the scale of the moment series, yet the sums were accumulated in raw units (scale squared).
    // The estimator is homogeneous of degree two, so scaling a series by an exact power of two c scales the
    // result by c^2; the expected value below is the unscaled estimate times c^2, with no shared arithmetic.
    // Two superposed sinusoids rounded to quarters: exact binary fractions with strong serial dependence, so the
    // pilot window is nonzero and the spectral sums are large in raw units. The seed shifts the phase.
    private static double[] QuarterSeries(int seed, int count)
    {
        var values = new double[count];
        for (var i = 0; i < count; i++)
        {
            var t = i + (seed * 7);
            values[i] = Math.Round((Math.Sin(1.7 * t) * 8) + (Math.Sin(0.31 * t) * 5)) / 4.0;
        }

        return values;
    }

    [Theory]
    [InlineData(11, 510)]
    [InlineData(12, 511)]
    public void MeanVarianceAutomatic_Double_NearTopOfRange_IsHomogeneousOfDegreeTwo(int seed, int exponent)
    {
        var series = QuarterSeries(seed, 100);
        var scaled = series.Select(v => Math.ScaleB(v, exponent)).ToArray();

        var baseline = NeweyWestVariance<double>.MeanVarianceAutomatic(series);
        var actual = NeweyWestVariance<double>.MeanVarianceAutomatic(scaled);

        var expected = Math.ScaleB(baseline, 2 * exponent);
        double.IsFinite(expected).Should().BeTrue();
        actual.Should().BeApproximately(expected, Math.Abs(expected) * 1e-12);
    }

    [Fact]
    public void MeanVarianceAutomatic_Float_NearTopOfRange_IsHomogeneousOfDegreeTwo()
    {
        var series = QuarterSeries(13, 100).Select(v => (float)v).ToArray();
        var scaled = series.Select(v => MathF.ScaleB(v, 62)).ToArray();

        var baseline = NeweyWestVariance<float>.MeanVarianceAutomatic(series);
        var actual = NeweyWestVariance<float>.MeanVarianceAutomatic(scaled);

        var expected = MathF.ScaleB(baseline, 124);
        float.IsFinite(expected).Should().BeTrue();
        actual.Should().BeApproximately(expected, Math.Abs(expected) * 1e-5f);
    }

    [Fact]
    public void MeanVarianceAutomatic_Decimal_NearTopOfRange_IsHomogeneousOfDegreeTwo()
    {
        var series = QuarterSeries(14, 100).Select(v => (decimal)v).ToArray();
        var factor = 70_368_744_177_664m; // 2^46, exact in decimal
        var scaled = series.Select(v => v * factor).ToArray();

        var baseline = NeweyWestVariance<decimal>.MeanVarianceAutomatic(series);
        var actual = NeweyWestVariance<decimal>.MeanVarianceAutomatic(scaled);

        var expected = baseline * factor * factor;
        actual.Should().BeApproximately(expected, Math.Abs(expected) * 1e-15m);
    }

    // F-L1-04. For Half, p x p = 65536 rounds to +Infinity once p >= 256, so the mean absolute change was
    // divided by Infinity, became 0, and the sweep certified convergence after one iteration. The same
    // 8 x 8 AR(1) block inside an identity panel converges in several sweeps for p = 255 and for double at any p.
    private static T[,] IdentityWithArBlock<T>(int p, double rho, int block)
        where T : System.Numerics.IFloatingPointIeee754<T>
    {
        var s = new T[p, p];
        for (var i = 0; i < p; i++)
        {
            s[i, i] = T.One;
        }

        for (var i = 0; i < block; i++)
        {
            for (var j = 0; j < block; j++)
            {
                s[i, j] = T.CreateChecked(Math.Pow(rho, Math.Abs(i - j)));
            }
        }

        return s;
    }

    [Fact]
    public void GraphicalLasso_Half_PanelOfTwoFiftySix_DoesNotCertifyConvergenceAfterOneSweep()
    {
        var reference = GraphicalLassoSolver<double>.Solve(IdentityWithArBlock<double>(256, 0.8, 8), 0.02);
        var half = GraphicalLassoSolver<Half>.Solve(IdentityWithArBlock<Half>(256, 0.8, 8), (Half)0.02);

        reference.Iterations.Should().BeGreaterThan(1);
        // Half is unsupported: an exhausted inner solve may honestly stop without
        // convergence. The regression forbids a false one-sweep success certificate.
        if (half.Converged)
        {
            half.Iterations.Should().BeGreaterThan(1);
            ((double)half.Covariance[0, 7]).Should().BeApproximately(reference.Covariance[0, 7], 1e-3);
        }
    }
}
