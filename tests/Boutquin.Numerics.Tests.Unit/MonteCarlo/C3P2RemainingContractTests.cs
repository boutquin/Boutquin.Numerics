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
using FluentAssertions;
namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class C3P2RemainingContractTests
{
    [Fact]
    public void StartupUsesExactTypeSevenAndEmptyZero()
    {
        var e = new PSquareEstimator<double>(.25);
        Assert.Equal(0, e.Estimate);
        e.Add(8); Assert.Equal(8, e.Estimate);
        e.Add(0); Assert.Equal(2, e.Estimate);
        e.Add(4); Assert.Equal(2, e.Estimate);
        e.Add(12); Assert.Equal(3, e.Estimate);
        Assert.Equal(4, e.Count);
    }
    [Fact]
    public void SixthObservationWithoutEligibleMoveKeepsMarkerHeights()
    {
        var e = new PSquareEstimator<double>(.5);
        foreach (var x in new double[] { 0, 1, 2, 3, 4 })
        {
            e.Add(x);
        }
        e.Add(2.5);
        // At j=6, discrepancies(.25,.5,-.25) are all strictly inside(-1,1).
        var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        Assert.Equal(new double[] { 0, 1, 2, 3, 4 }, (double[])typeof(PSquareEstimator<double>).GetField("_q", flags)!.GetValue(e)!);
        Assert.Equal(new long[] { 1, 2, 3, 5, 6 }, (long[])typeof(PSquareEstimator<double>).GetField("_n", flags)!.GetValue(e)!);
        Assert.Equal(6, e.Count);
    }
    [Fact]
    public void DomainsRejectEndpointsAndNonfiniteWithoutAdvancing()
    {
        foreach (var p in new double[] { 0, 1, -1, 2, double.NaN, double.PositiveInfinity, double.NegativeInfinity })
        { Action create = () => _ = new PSquareEstimator<double>(p); create.Should().Throw<ArgumentOutOfRangeException>(); }
        var e = new PSquareEstimator<double>(.5);
        foreach (var x in new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity })
        { Action add = () => e.Add(x); add.Should().Throw<ArgumentOutOfRangeException>(); Assert.Equal(0, e.Count); }
    }
}
