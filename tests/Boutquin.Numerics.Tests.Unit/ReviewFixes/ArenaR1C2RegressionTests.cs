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

using Boutquin.Numerics.LinearAlgebra;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

public sealed class ArenaR1C2RegressionTests
{
    [Fact]
    public void GenericPcaResult_DefensivelyCopiesArraysOnIngressAndEgress()
    {
        var eigenvalues = new[] { 1.0 };
        var eigenvectors = new[,] { { 1.0 } };
        var result = new PcaResult<double>
        {
            Eigenvalues = eigenvalues,
            Eigenvectors = eigenvectors,
            ExplainedVarianceRatio = [1.0],
            CumulativeExplainedVariance = [1.0],
            Mean = [0.0],
        };

        eigenvalues[0] = 99.0;
        eigenvectors[0, 0] = 100.0;
        result.Eigenvalues[0].Should().Be(1.0);
        result.Project(new[,] { { 3.0 } }, 1)[0, 0].Should().Be(3.0);
        var exposedValues = result.Eigenvalues;
        var exposedVectors = result.Eigenvectors;
        exposedValues[0] = 77.0;
        exposedVectors[0, 0] = 200.0;

        result.Eigenvalues[0].Should().Be(1.0);
        result.Project(new[,] { { 3.0 } }, 1)[0, 0].Should().Be(3.0);
    }

    [Fact]
    public void LegacyPcaResult_DefensivelyCopiesArraysOnIngressAndEgress()
    {
        var eigenvalues = new[] { 1m };
        var eigenvectors = new[,] { { 1m } };
        var result = new PcaResult
        {
            Eigenvalues = eigenvalues,
            Eigenvectors = eigenvectors,
            ExplainedVarianceRatio = [1m],
            CumulativeExplainedVariance = [1m],
            Mean = [0m],
        };

        eigenvalues[0] = 99m;
        eigenvectors[0, 0] = 100m;
        result.Eigenvalues[0].Should().Be(1m);
        result.Project(new[,] { { 3m } }, 1)[0, 0].Should().Be(3m);
        var exposedValues = result.Eigenvalues;
        var exposedVectors = result.Eigenvectors;
        exposedValues[0] = 77m;
        exposedVectors[0, 0] = 200m;

        result.Eigenvalues[0].Should().Be(1m);
        result.Project(new[,] { { 3m } }, 1)[0, 0].Should().Be(3m);
    }

    [Fact]
    public void GenericEigenResult_DefensivelyCopiesArraysOnIngressAndEgress()
    {
        var values = new[] { 2.0 };
        var vectors = new[,] { { 1.0 } };
        var result = new EigenResult<double>(values, vectors);

        values[0] = 99.0;
        vectors[0, 0] = 100.0;
        result.Values[0].Should().Be(2.0);
        result.Vectors[0, 0].Should().Be(1.0);
        var exposedValues = result.Values;
        var exposedVectors = result.Vectors;
        exposedValues[0] = 77.0;
        exposedVectors[0, 0] = 200.0;

        result.Values[0].Should().Be(2.0);
        result.Vectors[0, 0].Should().Be(1.0);
    }

    [Fact]
    public void LegacyEigenResult_DefensivelyCopiesArraysOnIngressAndEgress()
    {
        var values = new[] { 2m };
        var vectors = new[,] { { 1m } };
        var result = new EigenResult(values, vectors);

        values[0] = 99m;
        vectors[0, 0] = 100m;
        result.Values[0].Should().Be(2m);
        result.Vectors[0, 0].Should().Be(1m);
        var exposedValues = result.Values;
        var exposedVectors = result.Vectors;
        exposedValues[0] = 77m;
        exposedVectors[0, 0] = 200m;

        result.Values[0].Should().Be(2m);
        result.Vectors[0, 0].Should().Be(1m);
    }

    [Fact]
    public void GenericPivotedCholeskyResult_DefensivelyCopiesArraysOnIngressAndEgress()
    {
        var lower = new[,] { { 2.0, 0.0 }, { 1.0, 3.0 } };
        var permutation = new[] { 0, 1 };
        var result = new PivotedCholeskyResult<double>(lower, permutation, 2);

        lower[0, 0] = 99.0;
        permutation[0] = 1;
        result.Lower[0, 0].Should().Be(2.0);
        result.Permutation[0].Should().Be(0);
        var exposedLower = result.Lower;
        var exposedPermutation = result.Permutation;
        exposedLower[0, 0] = 77.0;
        exposedPermutation[0] = 1;

        result.Lower[0, 0].Should().Be(2.0);
        result.Permutation[0].Should().Be(0);
    }

    [Fact]
    public void LegacyPivotedCholeskyResult_DefensivelyCopiesArraysOnIngressAndEgress()
    {
        var lower = new[,] { { 2m, 0m }, { 1m, 3m } };
        var permutation = new[] { 0, 1 };
        var result = new PivotedCholeskyResult(lower, permutation, 2);

        lower[0, 0] = 99m;
        permutation[0] = 1;
        result.Lower[0, 0].Should().Be(2m);
        result.Permutation[0].Should().Be(0);
        var exposedLower = result.Lower;
        var exposedPermutation = result.Permutation;
        exposedLower[0, 0] = 77m;
        exposedPermutation[0] = 1;

        result.Lower[0, 0].Should().Be(2m);
        result.Permutation[0].Should().Be(0);
    }
}
