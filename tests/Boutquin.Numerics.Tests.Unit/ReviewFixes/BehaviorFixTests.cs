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

using Boutquin.Numerics.Collections;
using Boutquin.Numerics.Distributions;
using Boutquin.Numerics.Interpolation;
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Behavioral regression tests from the 2026-06-20 code review (findings #2, #10, #12, #34).
/// </summary>
public sealed class BehaviorFixTests
{
    // --- RollingWindow empty-window indexer message (review finding #34) ---

    [Fact]
    public void RollingWindow_EmptyIndexer_MessageIsNotNonsensical()
    {
        var window = new RollingWindow<double>(4);
        var act = () => _ = window[0];
        act.Should().Throw<ArgumentOutOfRangeException>()
            .Which.Message.Should().NotContain("0 and -1");
    }

    // --- DistanceCorrelation n<3 consistency with Pearson/Rank (review finding #12) ---

    [Fact]
    public void DistanceCorrelation_TwoPoints_ReturnsZeroLikePearsonAndRank()
    {
        var x = new double[] { 1.0, 2.0 };
        var y = new double[] { 1.0, 5.0 };
        // Distance correlation is undefined for n < 3 (always |1| on two points);
        // Pearson and Rank both return 0 for n < 3, so this must too.
        DistanceCorrelation<double>.Compute(x, y).Should().Be(0.0);
    }

    // --- InverseNormal extreme-tail underflow (review finding #2) ---

    [Fact]
    public void InverseNormal_ExtremeSmallProbability_ReturnsFiniteValue()
    {
        // p just above the smallest positive double: the Newton polish divides by
        // Pdf(x), which can underflow to 0 at |x| ~ 38 and yield NaN.
        var result = InverseNormal<double>.Evaluate(double.Epsilon);
        double.IsFinite(result).Should().BeTrue();
        result.Should().BeLessThan(0.0);
    }

    [Fact]
    public void InverseNormal_ExtremeLargeProbability_ReturnsFiniteValue()
    {
        var result = InverseNormal<double>.Evaluate(1.0 - 1e-16);
        double.IsFinite(result).Should().BeTrue();
        // Mirror the small-p sign check: p just below 1 must yield a large POSITIVE
        // z-score (true value ≈ 8.2), not just any finite number.
        result.Should().BeGreaterThan(0.0);
    }

    // --- MonotoneConvex numerically-zero origin acceptance (review finding #10) ---

    [Fact]
    public void MonotoneConvex_NumericallyZeroOrigin_IsAccepted()
    {
        // A caller computing the origin as (t - t) may get a tiny non-zero value
        // instead of literal 0; the virtual-origin guard should tolerate it.
        var xs = new double[] { 1e-15, 1.0, 2.0, 3.0 };
        var ys = new double[] { 1e-15, 0.10, 0.22, 0.33 };
        var act = () => MonotoneConvexInterpolator<double>.Instance.Interpolate(1.5, xs, ys);
        act.Should().NotThrow();
    }

    [Fact]
    public void MonotoneConvex_RealNonZeroOrigin_StillThrows()
    {
        var xs = new double[] { 0.5, 1.0, 2.0, 3.0 };
        var ys = new double[] { 0.0, 0.10, 0.22, 0.33 };
        var act = () => MonotoneConvexInterpolator<double>.Instance.Interpolate(1.5, xs, ys);
        act.Should().Throw<ArgumentException>();
    }
}
