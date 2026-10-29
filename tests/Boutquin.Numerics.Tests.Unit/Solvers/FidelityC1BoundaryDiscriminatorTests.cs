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
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

[Trait("Tier", "Unit")]
public sealed class FidelityC1BoundaryDiscriminatorTests
{
    [Theory]
    [InlineData(0, 2, 5, 8)]
    [InlineData(7, 5, 2, -1)]
    public void Muller_ZeroDenominatorRepeatsPreviousDisplacement(double first, double second, double third, double expected)
    {
        // Muller 1956 p214: lambda_next=1 multiplies h_i. It is not an absolute unit step.
        var visits = new List<double>();
        double Objective(double x) { visits.Add(x); return 1; }
        var result = new MullerSolver<double>(1e-12, 1).Solve(Objective, first, second, third);
        visits.Should().Equal(first, second, third, expected);
        result.Root.Should().Be(expected);
        result.Converged.Should().BeFalse();
    }

    [Fact]
    public void Muller_ResidualRatioExactlyTenDoesNotBacktrack()
    {
        // Seed values 3,2,1 imply the linear interpolation root 3. The cubic term
        // vanishes at the seeds and makes f(3)=10, exactly ten times f(2).
        var visits = new List<double>();
        double Objective(double x)
        {
            visits.Add(x);
            return 3 - x + (5.0 / 3) * x * (x - 1) * (x - 2);
        }
        var result = new MullerSolver<double>(1e-12, 1).Solve(Objective, 0, 1, 2);
        visits.Should().Equal(0, 1, 2, 3);
        result.Root.Should().Be(3);
        result.FinalResidual.Should().Be(10);
    }

    [Theory]
    [InlineData("equal-best")]
    [InlineData("equal-second")]
    [InlineData("equal-expansion")]
    public void NelderMead_ExactEqualityFollowsLagariasBranch(string scenario)
    {
        var visits = new List<double[]>();
        double Objective(double[] x)
        {
            visits.Add((double[])x.Clone());
            return (x[0], x[1]) switch
            {
                (0, 0) => 0,
                (1, 0) => 1,
                (0, 1) => 2,
                (1, -1) => scenario switch { "equal-best" => 0, "equal-second" => 1, _ => -1 },
                (1.5, -2) => -1,
                (0.75, -0.5) => 0.5,
                _ => 10,
            };
        }
        var result = NelderMeadMinimizer<double>.Minimize(Objective, [0, 0],
            new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1, XTolerance: 0, FTolerance: 0));
        visits[3].Should().Equal(1, -1);
        if (scenario == "equal-best")
        {
            visits.Should().HaveCount(8, "f_best <= f_reflection < f_second accepts reflection without expansion");
            result.Argmin.Should().Equal(new double[] { 0, 0 }, "the retained best precedes an equal new vertex");
        }
        else if (scenario == "equal-second")
        {
            visits.Should().HaveCount(9);
            visits[4].Should().Equal(new double[] { 0.75, -0.5 }, "equality to second-worst triggers outside contraction");
        }
        else
        {
            visits.Should().HaveCount(9);
            visits[4].Should().Equal(1.5, -2);
            result.Argmin.Should().Equal(new double[] { 1, -1 }, "expansion must improve strictly over reflection");
        }
    }
}
