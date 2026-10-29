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
using Boutquin.Numerics.LinearAlgebra.Internal;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

public sealed class FidelityC1QrEuclideanNormControl
{
    [Fact]
    public void PivotNormUsesSumOfSquaresOverSpecifiedRows()
    {
        var method = typeof(HouseholderQr<double>).GetMethod("ColumnNorm", BindingFlags.Static | BindingFlags.NonPublic)!;
        var norm = method.CreateDelegate<Func<double[,], int, int, int, double>>();
        double[,] panel = { { 12, 99 }, { 3, 99 }, { 4, 99 } };
        // Pythagorean triples distinguish Euclidean norm from L1, maximum,
        // squared norm, the wrong column, and the wrong trailing row range.
        Assert.Equal(13, norm(panel, 0, 0, 3));
        Assert.Equal(5, norm(panel, 0, 1, 3));
        double[,] pivotPanel = { { 5, 3 }, { 0, 3 }, { 0, 0 } };
        var permutation = new int[2];
        Assert.True(HouseholderQr<double>.Factor(pivotPanel, new double[3], new double[2], permutation));
        // ||column0||2=5>sqrt18, although ||column0||1=5<6.
        Assert.Equal(new[] { 0, 1 }, permutation);
    }
}
