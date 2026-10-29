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
using Boutquin.Numerics.Random;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class ReleaseQualificationBootstrapTests
{
    [Theory]
    [InlineData(1e-12, 2.0444154168105063e-13)]
    [InlineData(1e-15, 2.0444154167983716e-16)]
    [InlineData(1e-17, 2.0444154167983596e-18)]
    [InlineData(1e-20, 2.0444154167983592e-21)]
    [InlineData(1e-30, 2.0444154167983595e-31)]
    public void HarrellDavis_TinyTailMassSurvivesLargeGap(double p, double expectedWeight)
    {
        // mpmath 80 digits: betainc(3*p,3*(1-p),.5,1,regularized=True),
        // with p converted from its exact binary-double value.
        var actual = HarrellDavisPercentile.Compute<double>([0, 1e150], p);
        (actual / 1e150).Should().BeApproximately(expectedWeight, expectedWeight * 1e-10);
        var actualDecimal = HarrellDavisPercentile.Compute<decimal>([0m, 1e14m], p);
        ((double)actualDecimal / 1e14).Should().BeApproximately(expectedWeight, Math.Max(1e-42, expectedWeight * 1e-10));
    }

    [Fact]
    public void HarrellDavis_MirroredTinyMassSurvivesNearUnitUpdate()
    {
        const double expectedWeight = 2.0427813608653692e-16;
        const double p = 0.999999999999999;
        (HarrellDavisPercentile.Compute<double>([-1e150, 0], p) / -1e150)
            .Should().BeApproximately(expectedWeight, 1e-26);
        ((double)HarrellDavisPercentile.Compute<decimal>([-1e14m, 0m], p) / -1e14)
            .Should().BeApproximately(expectedWeight, 1e-26);
    }

    [Theory]
    [InlineData(50)]
    [InlineData(1000000)]
    public void HarrellDavis_SymmetricStepHasExactMedian(int count)
    {
        // Equal beta shapes place exactly half their mass on either side of 1/2.
        var sample = new double[count];
        Array.Fill(sample, 1.0, count / 2, count / 2);
        HarrellDavisPercentile.Compute<double>(sample, .5).Should().BeApproximately(.5, 1e-12);
    }

    [Fact]
    public void HarrellDavis_LargeNonsymmetricRampMatchesScipyBetaWeights()
    {
        const int count = 3000000;
        var sample = Enumerable.Range(0, count).Select(i => (double)i / count).ToArray();
        // scipy.special.betainc interval masses, independently accumulated by numpy.dot.
        HarrellDavisPercentile.Compute<double>(sample, .3)
            .Should().BeApproximately(.2999998333333333, 1e-10);
    }

    [Fact]
    public void FastDoubleBootstrap_PrimeTrialCountRetainsEveryExactQuantileTie()
    {
        const int count = 101;
        foreach (var tail in new[] { PValueTail.RightTail, PValueTail.LeftTail })
        {
            for (var k = 1; k < count; k++)
            {
                var sign = tail == PValueTail.RightTail ? 1m : -1m;
                var exactQuantile = sign * (count - k) * (count - 1);
                var observed = exactQuantile + sign;
                var call = 0;
                decimal Statistic(ReadOnlySpan<decimal> sample)
                {
                    var b = call / 2;
                    var inner = call++ % 2 == 1;
                    return inner ? sign * b * count : b < k ? sign * 1e6m : exactQuantile;
                }
                var actual = new FastDoubleBootstrap(count, new Pcg64RandomSource(9UL))
                    .PValue([1m], observed, Statistic, tail);
                actual.Should().Be(k / (decimal)count, "B=101 and k={0}: the exact Type-7 quantile is an outer atom", k);
            }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FastDoubleBootstrap_NonDyadicInnerQuantileRetainsStrictTies(bool imposeNull)
    {
        // B=50, outer=[10,20*.02,29*(-1)], observed=5 => p*=1/50.
        // Inner=[49*0,1], index=49*.98=48.02, exact Q*=.02.
        // Only outer 10 strictly exceeds Q*, so the final p is 1/50.
        var call = 0;
        decimal Statistic(ReadOnlySpan<decimal> values)
        {
            var index = call / 2;
            var inner = call++ % 2 == 1;
            return inner ? index == 49 ? 1m : 0m : index == 0 ? 10m : index <= 20 ? .02m : -1m;
        }
        var engine = new FastDoubleBootstrap(50, new Pcg64RandomSource(9UL));
        var actual = imposeNull
            ? engine.PValueImposingNull([1m], 5m, Statistic, static _ => { }, PValueTail.RightTail)
            : engine.PValue([1m], 5m, Statistic, PValueTail.RightTail);
        actual.Should().Be(.02m);
        call.Should().Be(100);
    }
}
