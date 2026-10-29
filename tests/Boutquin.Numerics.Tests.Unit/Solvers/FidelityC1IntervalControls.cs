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

using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

public sealed class FidelityC1IntervalControls
{
    [Theory]
    [InlineData(1.0)]
    [InlineData(-1.0)]
    public void BisectionHalvesTheRootContainingIntervalForEitherOrientation(double sign)
    {
        // The root 3/10 lies successively in [0,1], [0,1/2], [1/4,1/2],
        // [1/4,3/8]. Multiplying f by -1 must leave every abscissa unchanged.
        var visits = new List<double>();
        var result = new BisectionSolver<double>(1e-30, 1e-30, 3).Solve(x =>
        {
            visits.Add(x);
            return sign * (x - 0.3);
        }, 0, 1);
        Assert.Equal(new double[] { 0, 1, 0.5, 0.25, 0.375, 0.3125 }, visits);
        Assert.False(result.Converged);
        Assert.Equal(3, result.Iterations);
        Assert.Equal(0.3125, result.Root);
        Assert.Equal(0.0625, result.EstimatedError);
        Assert.Equal(sign * (0.3125 - 0.3), result.FinalResidual);
    }

    [Fact]
    public void BisectionExactMidpointRootReturnsWithoutFurtherHalving()
    {
        var calls = 0;
        var result = new BisectionSolver<double>(1e-30, 1e-30, 5).Solve(x =>
        {
            calls++;
            return x - 0.5;
        }, 0, 1);
        Assert.Equal(3, calls);
        Assert.True(result.Converged);
        Assert.Equal(1, result.Iterations);
        Assert.Equal(0.5, result.Root);
        Assert.Equal(0, result.FinalResidual);
    }
}
