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
public sealed class FidelityC1BrentAccuracyControls
{
    [Theory]
    [InlineData(-80)]
    [InlineData(0)]
    [InlineData(80)]
    public void FlatContinuousDyadicRootMeetsComputedZeroAbscissaBound(int exponent)
    {
        // Exact dyadic root5/4*2^exponent; all arithmetic remains finite and normal.
        // The small slope makes both initial residuals much smaller than the requested
        // abscissa tolerance, without making either endpoint close to the root.
        // This checks this conditioned family, not a universal convergence theorem.
        var scale = Math.ScaleB(1, exponent);
        var root = 1.25 * scale;
        var tolerance = 1e-6 * scale;
        var result = new BrentSolver<double>(tolerance, 20).Solve(x => 1e-20 * (x - root), scale, 2 * scale);
        var machineEpsilon = Math.ScaleB(1, -52);
        var bound = tolerance + 6 * machineEpsilon * Math.Abs(root);
        Assert.True(result.Converged);
        Assert.InRange(Math.Abs(result.Root - root), 0, bound);
        Assert.Equal(0, result.FinalResidual);
    }
}
