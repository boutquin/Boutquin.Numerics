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

[Trait("Tier", "Unit")]
public sealed class FidelityC1PoolOwnershipControls
{
    [Fact]
    public void IndependentPoolsKeepNestedSolvesAndReturnedResultsIsolated()
    {
        var outerPool = new LevenbergMarquardtBuffers<double>(1, 1);
        var innerPool = new LevenbergMarquardtBuffers<double>(1, 1);
        Assert.NotSame(outerPool.Residual, innerPool.Residual);
        Assert.NotSame(outerPool.Theta, innerPool.Theta);
        Assert.NotSame(outerPool.Jacobian, innerPool.Jacobian);
        var solver = new LevenbergMarquardtSolver<double>(1, 1e-30, 1e-30, 1e-30, 3);
        static void Inner(ReadOnlySpan<double> x, Span<double> r) => r[0] = x[0] - 4;
        static void Jacobian(ReadOnlySpan<double> _, Span<double> j) => j[0] = 1;
        var nestedCalls = 0;
        void Outer(ReadOnlySpan<double> x, Span<double> r)
        {
            var inner = solver.Solve(Inner, [0], innerPool, Jacobian);
            // r=y-4 and lambda3 gives y=1, r=-3, cost9/2 independently.
            Assert.Equal(1, inner.Parameters[0]);
            Assert.Equal(4.5, inner.FinalCost);
            nestedCalls++;
            r[0] = x[0] - 1;
        }
        var result = solver.Solve(Outer, [0], outerPool, Jacobian);
        Assert.True(nestedCalls >= 2);
        Assert.Equal(0.25, result.Parameters[0]);
        Assert.Equal(-0.75, result.FinalResiduals[0]);
        Assert.Equal(9.0 / 32, result.FinalCost);
        // Reusing one pool must not mutate a previously returned independent result.
        solver.Solve(Inner, [0], outerPool, Jacobian);
        Assert.Equal(0.25, result.Parameters[0]);
        Assert.Equal(-0.75, result.FinalResiduals[0]);
    }
}
