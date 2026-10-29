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

using Boutquin.Numerics.LinearAlgebra.Internal;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

[Trait("Tier", "Unit")]
public sealed class FidelityC1PivotSelectionControls
{
    [Fact]
    public void PivotSelectsStrictlyLargestNormRatherThanFirstColumn()
    {
        // Orthogonal columns have norms1,2,4. Every maximum is unique, so
        // the required order2,1,0 is independent of any equal-norm tie policy.
        double[,] a = { { 1, 0, 0 }, { 0, 2, 0 }, { 0, 0, 4 }, { 0, 0, 0 } };
        var permutation = new int[3];
        Assert.True(HouseholderQr<double>.Factor(a, new double[4], new double[3], permutation));
        Assert.Equal(new[] { 2, 1, 0 }, permutation);
    }

    [Fact]
    public void LaterPivotUsesTrailingNormRatherThanOriginalColumnNorm()
    {
        // Initial norms10,sqrt82,2 select column0. Removing its row leaves
        // remaining norms1,2, so column2 must precede column1 despite sqrt82>2.
        double[,] a = { { 10, 9, 0 }, { 0, 1, 0 }, { 0, 0, 2 }, { 0, 0, 0 } };
        var permutation = new int[3];
        Assert.True(HouseholderQr<double>.Factor(a, new double[4], new double[3], permutation));
        Assert.Equal(new[] { 0, 2, 1 }, permutation);
    }
}
