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

public sealed class Arena20260928SolverTypeContractTests
{
    [Fact]
    public void Lm_Half_RejectsUnsupportedThresholdRangeExplicitly()
    {
        var error = Assert.Throws<NotSupportedException>(() => new LevenbergMarquardtSolver<Half>(
            20, (Half)0.01, (Half)0.01, (Half)0.01, (Half)0.01));
        Assert.Contains("threshold", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Qp_Half_RejectsUnderflowedThresholdsExplicitly(bool meanVariance)
    {
        Half[,] covariance = { { (Half)1, (Half)0 }, { (Half)0, (Half)1 } };
        var error = Assert.Throws<NotSupportedException>(() => meanVariance
            ? ActiveSetQpSolver<Half>.SolveMeanVariance(covariance, new Half[] { (Half)0, (Half)0 }, (Half)1, (Half)0, (Half)1)
            : ActiveSetQpSolver<Half>.SolveMinVariance(covariance, (Half)0, (Half)1));
        Assert.Contains("threshold", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SupportedTypes_KeepTheirNativeArithmeticPaths()
    {
        var lm = new LevenbergMarquardtSolver<float>(100, 1e-6f, 1e-6f, 1e-6f, 1e-3f);
        var result = lm.Solve(x => new[] { x[0] - 3 }, new[] { 0f }, jacobian: _ => new float[,] { { 1 } });
        Assert.True(result.Converged);
        Assert.InRange(Math.Abs(result.Parameters[0] - 3), 0, 1e-5f);
        float[,] single = { { 1, 0 }, { 0, 1 } };
        decimal[,] precise = { { 1, 0 }, { 0, 1 } };
        Assert.Equal(new[] { 0.5f, 0.5f }, ActiveSetQpSolver<float>.SolveMinVariance(single, 0, 1));
        Assert.Equal(new[] { 0.5m, 0.5m }, ActiveSetQpSolver<decimal>.SolveMinVariance(precise, 0, 1));
    }
}
