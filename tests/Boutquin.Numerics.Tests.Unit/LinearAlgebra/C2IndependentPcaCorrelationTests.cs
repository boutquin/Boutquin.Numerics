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

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class C2IndependentPcaCorrelationTests
{
    [Fact]
    public void CorrelationMatrix_HasExactTraceAndExplainedVariance()
    {
        var result = PrincipalComponentAnalysis<double>.Decompose(new double[,] { { 1, .5 }, { .5, 1 } });
        result.Eigenvalues[0].Should().BeApproximately(1.5, 1e-14);
        result.Eigenvalues[1].Should().BeApproximately(.5, 1e-14);
        result.Eigenvalues.Sum().Should().BeApproximately(2, 1e-14);
        result.ExplainedVarianceRatio[0].Should().BeApproximately(.75, 1e-14);
        result.ExplainedVarianceRatio[1].Should().BeApproximately(.25, 1e-14);
    }

    [Fact]
    public void StandardizedOrthogonalPanel_HasUnitSampleVarianceInEveryComponent()
    {
        // Independent signed coordinate pairs: raw sample variances18/5,8/5,2/5.
        // Dividing by each sample SD gives identity correlation, trace3 and three
        // equal unit eigenvalues regardless of the basis chosen in that eigenspace.
        decimal[,] data = { { 13, 20, 30 }, { 7, 20, 30 }, { 10, 22, 30 },
            { 10, 18, 30 }, { 10, 20, 31 }, { 10, 20, 29 } };
        var result = PrincipalComponentAnalysis.FromReturns(data, standardize: true);
        foreach (var value in result.Eigenvalues)
        {
            value.Should().BeApproximately(1m, 1e-23m);
        }
        result.Eigenvalues.Sum().Should().BeApproximately(3m, 1e-23m);
        foreach (var ratio in result.ExplainedVarianceRatio)
        {
            ratio.Should().BeApproximately(1m / 3m, 1e-23m);
        }
    }
}
