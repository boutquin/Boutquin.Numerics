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
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class C4GlassoInnerCycleTests
{
    [Fact]
    public void CoupledLassoCoordinates_ReachIndependentUniqueOptimum()
    {
        // FHT eq.3–4: W11=[[2,1],[1,2]], s12=[3,3], lambda1.
        // At beta=[2/3,2/3], W11*beta-s12=[-1,-1]; both coordinates
        // are positive, so the L1 KKT gradient is exactlyzero. W11 is SPD
        // (eigenvalues1,3), proving this is the unique optimum. A single
        // cold-start coordinate sweep gives[1,1/2], not the optimum.
        double[,] block = { { 2, 1 }, { 1, 2 } };
        double[] cross = [3, 3];
        double[] coefficients = [0, 0];
        var solve = typeof(GraphicalLassoSolver<double>).GetMethod(
            "SolveLassoSubproblem", BindingFlags.Static | BindingFlags.NonPublic)!;
        var converged = (bool)solve.Invoke(null,
            new object?[] { block, cross, coefficients, 1.0, 2, 3, null })!;

        coefficients[0].Should().BeApproximately(2.0 / 3, 1e-8);
        coefficients[1].Should().BeApproximately(2.0 / 3, 1e-8);
        converged.Should().BeTrue();
    }
}
