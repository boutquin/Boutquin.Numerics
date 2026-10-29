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

public sealed class C2IndependentJacobiRotationTests
{
    [Theory]
    [InlineData(1, 4, 2, 0.5)]
    [InlineData(4, 1, 2, -0.5)]
    [InlineData(2, 2, 1, 1)]
    public void SingleRotation_MatchesIndependentDenseSimilarity(double app, double aqq, double apq, double tangent)
    {
        // The first two blocks have exact roots0,5 and |t|=1/2; the equal block has roots1,3 and t=1.
        const int N = 5;
        const int P = 1;
        const int Q = 3;
        double[,] input = { { 7, .25, .5, .75, 1 }, { .25, app, 1.25, apq, 1.5 },
            { .5, 1.25, 8, 1.75, 2 }, { .75, apq, 1.75, aqq, 2.25 }, { 1, 1.5, 2, 2.25, 9 } };
        var actual = (double[,])input.Clone();
        var vectors = new double[N, N];
        var rotation = new double[N, N];
        for (var i = 0; i < N; i++) { vectors[i, i] = 1; rotation[i, i] = 1; }
        var cosine = 1 / Math.Sqrt(1 + tangent * tangent);
        var sine = tangent * cosine;
        rotation[P, P] = cosine;
        rotation[Q, Q] = cosine;
        rotation[P, Q] = sine;
        rotation[Q, P] = -sine;
        var method = typeof(JacobiEigenDecomposition<double>).GetMethod("ApplyRotation", BindingFlags.NonPublic | BindingFlags.Static);
        method.Should().NotBeNull();
        FluentActions.Invoking(() => method!.Invoke(null, [actual, vectors, N, P, Q])).Should().NotThrow();
        var beforeOff = 0d;
        var afterOff = 0d;
        for (var i = 0; i < N; i++)
        {
            for (var j = i; j < N; j++)
            {
                var expected = 0d;
                for (var k = 0; k < N; k++)
                {
                    for (var l = 0; l < N; l++) { expected += rotation[k, i] * input[k, l] * rotation[l, j]; }
                }
                actual[i, j].Should().BeApproximately(expected, 4e-15, $"entry ({i},{j}) of J transpose A J");
                if (i != j) { beforeOff += 2 * input[i, j] * input[i, j]; afterOff += 2 * actual[i, j] * actual[i, j]; }
            }
            for (var j = 0; j < N; j++) { vectors[i, j].Should().BeApproximately(rotation[i, j], 1e-15); }
        }
        actual[P, Q].Should().Be(0);
        (beforeOff - afterOff).Should().BeApproximately(2 * apq * apq, 3e-14);
    }
}
