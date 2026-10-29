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

using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class HalfCovarianceRegressionTests
{
    [Fact]
    public void Qis_LongHalfPanel_ReturnsFiniteCovariance()
    {
        var result = new QuadraticInverseShrinkageEstimator<Half>().Estimate(CreateLongPanel(70000));
        AssertFinite(result);
    }

    [Fact]
    public void Denoised_LongHalfPanel_ReturnsFiniteCovariance()
    {
        Half[,]? result = null;
        var error = Record.Exception(() => result = new DenoisedCovarianceEstimator<Half>().Estimate(CreateLongPanel(65000)));
        error.Should().BeNull();
        AssertFinite(result!);
    }

    [Fact]
    public void DoublySparse_LongHalfPanel_ReturnsFiniteCovariance()
    {
        Half[,]? result = null;
        var error = Record.Exception(() => result = new DoublySparseEstimator<Half>().Estimate(CreateLongPanel(65000)));
        error.Should().BeNull();
        AssertFinite(result!);
    }

    [Fact]
    public void Oas_OffsetTinyHalfVariances_MatchesClosedFormIntensity()
    {
        var panel = new Half[100, 2];
        for (var i = 0; i < 100; i++)
        {
            panel[i, 0] = (Half)(1 + (i % 2 == 0 ? 1 : -1) / 1024.0);
            panel[i, 1] = (Half)(1 + (i % 4 < 2 ? 3 : -3) / 1024.0);
        }

        // Known-zero-mean S=[[1+v,1],[1,1+9v]], v=2^-20.
        // p=2 gives rho=tr(S)^2/(100*(tr(S²)-tr(S)^2/2)).
        const double v = 1.0 / (1024 * 1024);
        var trace = 2 + 10 * v;
        var dispersion = 2 + 32 * v * v;
        var rho = trace * trace / (100 * dispersion);
        var result = new OracleApproximatingShrinkageEstimator<Half>().Estimate(panel);
        result[0, 0].Should().Be((Half)((1 - rho) * (1 + v) + rho * (1 + 5 * v)));
        result[1, 1].Should().Be((Half)((1 - rho) * (1 + 9 * v) + rho * (1 + 5 * v)));
        result[0, 1].Should().Be((Half)(1 - rho));
    }

    private static Half[,] CreateLongPanel(int count)
    {
        var panel = new Half[count, 3];
        for (var i = 0; i < count; i++)
        {
            panel[i, 0] = (Half)((i % 5) - 2);
            panel[i, 1] = (Half)(((i * 3) % 7) - 3);
            panel[i, 2] = (Half)(((i * 7) % 11) - 5);
        }

        return panel;
    }

    private static void AssertFinite(Half[,] matrix)
    {
        matrix.GetLength(0).Should().Be(3);
        matrix.GetLength(1).Should().Be(3);
        // Independent discrete-uniform variances of the mod-5, mod-7 and mod-11
        // panel columns are 2, 4 and 10; finite samples and shrinkage perturb them slightly.
        double[] variances = [2, 4, 10];
        for (var i = 0; i < 3; i++)
        {
            ((double)matrix[i, i]).Should().BeApproximately(variances[i], .03);
        }

        foreach (var value in matrix)
        {
            Half.IsFinite(value).Should().BeTrue();
        }
    }
}
