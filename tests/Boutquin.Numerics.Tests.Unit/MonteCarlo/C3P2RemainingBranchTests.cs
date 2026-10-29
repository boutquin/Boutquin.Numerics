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

using System.Reflection;
using Boutquin.Numerics.MonteCarlo;
namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class C3P2RemainingBranchTests
{
    [Fact]
    public void PredictorEqualityAtEitherNeighbourRequiresLinearFallback()
    {
        const BindingFlags flags = BindingFlags.NonPublic | BindingFlags.Instance;
        foreach (var direction in new[] { -1, 1 })
        {
            var e = new PSquareEstimator<double>(.5);
            var q = (double[])typeof(PSquareEstimator<double>).GetField("_q", flags)!.GetValue(e)!;
            var n = (long[])typeof(PSquareEstimator<double>).GetField("_n", flags)!.GetValue(e)!;
            new double[] { -1, 0, direction == 1 ? 1 : 0, 1, 2 }.CopyTo(q, 0);
            new long[] { 1, 2, 3, 4, 5 }.CopyTo(n, 0);
            // Equal-spaced quadratic predictor in the requested direction equals
            // the neighbouring repeated height exactly, so strict acceptance fails.
            object[] args = { 2, direction, 0d };
            Assert.False((bool)typeof(PSquareEstimator<double>).GetMethod("TryParabolic", flags)!.Invoke(e, args)!);
            Assert.Equal(direction == 1 ? 1d : 0d, (double)typeof(PSquareEstimator<double>).GetMethod("Linear", flags)!.Invoke(e, new object[] { 2, direction })!);
        }
    }
    [Fact]
    public void LinearFallbackUsesSignedNeighbourGapAndPositionDistance()
    {
        const BindingFlags flags = BindingFlags.NonPublic | BindingFlags.Instance;
        var e = new PSquareEstimator<double>(.5);
        var q = (double[])typeof(PSquareEstimator<double>).GetField("_q", flags)!.GetValue(e)!;
        var n = (long[])typeof(PSquareEstimator<double>).GetField("_n", flags)!.GetValue(e)!;
        new double[] { 0, 1, 4, 10, 12 }.CopyTo(q, 0);
        new long[] { 1, 2, 5, 7, 8 }.CopyTo(n, 0);
        var method = typeof(PSquareEstimator<double>).GetMethod("Linear", flags)!;
        Assert.Equal(7d, (double)method.Invoke(e, new object[] { 2, 1 })!);
        Assert.Equal(3d, (double)method.Invoke(e, new object[] { 2, -1 })!);
    }
}
