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

using Boutquin.Numerics.LinearAlgebra;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class C2IndependentJacobiStoppingTests
{
    [Fact]
    public void DecimalEigenvectors_HaveWorkingPrecisionResiduals()
    {
        decimal[,] a = { { 2, .3m, .5m }, { .3m, 3, .2m }, { .5m, .2m, 4 } };
        var result = JacobiEigenDecomposition<decimal>.Decompose(a);
        for (var k = 0; k < 3; k++)
        {
            for (var i = 0; i < 3; i++)
            {
                var residual = -result.Values[k] * result.Vectors[i, k];
                for (var j = 0; j < 3; j++) { residual += a[i, j] * result.Vectors[j, k]; }
                Math.Abs(residual).Should().BeLessThan(5e-27m);
            }
        }
    }

    [Fact]
    public void DiagonalInput_HasExactlyPermutedIdentityEigenvectors()
    {
        double[,] a = { { 2, 0, 0 }, { 99, 4, 0 }, { -100, 88, 1 } };
        var result = JacobiEigenDecomposition<double>.Decompose(a);
        result.Values.Should().Equal(4, 2, 1);
        double[,] expected = { { 0, 1, 0 }, { 1, 0, 0 }, { 0, 0, 1 } };
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++) { result.Vectors[i, j].Should().Be(expected[i, j]); }
        }
    }

    [Fact]
    public void ExactOffNormEquality_StopsBeforeAnyRotation()
    {
        // Upper offdiagonals 3,4,5 times 1e-28 have full Frobenius magnitude
        // sqrt(2*(9+16+25))*1e-28=1e-27, equal to the decimal threshold.
        decimal[,] a = { { 1, 3e-28m, 4e-28m },
            { 3e-28m, 2, 5e-28m }, { 4e-28m, 5e-28m, 3 } };
        var result = JacobiEigenDecomposition<decimal>.Decompose(a);
        result.Values.Should().Equal(3m, 2m, 1m);
        decimal[,] expected = { { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 } };
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++) { result.Vectors[i, j].Should().Be(expected[i, j]); }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReciprocalPhiBranch_HasIndependentAnalyticTangent(bool negative)
    {
        // The eigenproblem for [[1,1],[1,21]] gives t=sqrt(101)−10.
        // Constant evaluated independently at80decimal digits, then rounded once.
        const double Tangent = .0498756211208902702192649127595761869;
        var t = negative ? -Tangent : Tangent;
        double[,] a = negative ? new double[,] { { 21, 1 }, { 1, 1 } } : new double[,] { { 1, 1 }, { 1, 21 } };
        double[,] v = { { 1, 0 }, { 0, 1 } };
        Rotate(a, v);
        var c = 1 / Math.Sqrt(1 + t * t);
        v[0, 0].Should().BeApproximately(c, 2e-15);
        v[0, 1].Should().BeApproximately(t * c, 2e-15);
        v[1, 0].Should().BeApproximately(-t * c, 2e-15);
        a[0, 1].Should().Be(0);
        a[0, 0].Should().BeApproximately((negative ? 21 : 1) - t, 2e-15);
        a[1, 1].Should().BeApproximately((negative ? 1 : 21) + t, 2e-15);
    }

    [Theory]
    [InlineData(1e-20)]
    [InlineData(1e-200)]
    public void TinyAngleBranch_RetainsRepresentableEigenvectorComponent(double offDiagonal)
    {
        // For diagonal difference1, the small root is b+O(b³). The correction
        // is below one double ULP on both inputs; inspect a normalized component.
        double[,] a = { { 1, offDiagonal }, { offDiagonal, 2 } };
        double[,] v = { { 1, 0 }, { 0, 1 } };
        Rotate(a, v);
        (v[0, 1] / offDiagonal).Should().BeApproximately(1, 2e-15);
        (v[1, 0] / offDiagonal).Should().BeApproximately(-1, 2e-15);
        a[0, 1].Should().Be(0);
    }

    private static void Rotate(double[,] a, double[,] v)
    {
        var method = typeof(JacobiEigenDecomposition<double>).GetMethod("ApplyRotation", BindingFlags.Static | BindingFlags.NonPublic);
        method.Should().NotBeNull();
        FluentActions.Invoking(() => method!.Invoke(null, [a, v, 2, 0, 1])).Should().NotThrow();
    }
}
