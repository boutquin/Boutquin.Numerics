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

public sealed class LevenbergMarquardtRangeTests
{
    [Theory]
    [InlineData(false, 1e200, 2e109, -1e109)]
    [InlineData(true, 1e200, 2e109, -1e109)]
    [InlineData(false, 1e160, 1e150, -2e150)]
    [InlineData(true, 1e160, 1e150, -2e150)]
    public void Solve_OverflowedGradient_ConvergesToTheRepresentableOptimum(bool writer, double slope, double a, double b)
    {
        // Two residuals slope·θ+a and slope·θ+b with Jacobian slope. At θ₀ = 0 the exact gradient
        // slope·(a+b) is 1e309 and -1e310, beyond the double range, but the residuals, the
        // Jacobian, the cost and the optimum θ = -(a+b)/(2·slope) (-5e-92 and 5e-11) are all
        // representable. These four cases formerly asserted an InvalidOperationException from
        // the gradient; the gradient is only a stopping-test quantity, and the step is solved in
        // range-scaled variables, so the representable optimum must be returned. The first
        // damped step is the Gauss–Newton step divided by 1 + initialDamping (1e-3), so the
        // result is within initialDamping of the closed-form optimum in relative terms.
        var solver = new LevenbergMarquardtSolver<double>(100, 1e-12, 1e-12, 0, 1e-3);
        var optimum = -(a + b) / (2 * slope);

        var result = writer
            ? solver.Solve((p, r) => { r[0] = slope * p[0] + a; r[1] = slope * p[0] + b; }, [0.0], 2,
                (_, j) => { j[0] = slope; j[1] = slope; })
            : solver.Solve(p => [slope * p[0] + a, slope * p[0] + b], [0.0], _ => new[,] { { slope }, { slope } });

        result.Converged.Should().BeTrue();
        result.Iterations.Should().BeGreaterThan(0, "an unrepresentable gradient never satisfies a gradient tolerance");
        Math.Abs(result.Parameters[0] - optimum).Should().BeLessThanOrEqualTo(2e-3 * Math.Abs(optimum));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Solve_SolutionBeyondTheRange_StillRejectsArithmeticFailure(bool writer)
    {
        // Jacobian 1e-300, optimum θ = 1.5e308 from θ₀ = -1.5e308: the exact Gauss–Newton step
        // 3e308 exceeds the double range, so the trial point is not representable, its residual
        // overflows to infinity, and the problem is genuinely unrepresentable. This is the
        // arithmetic failure the documented InvalidOperationException still covers.
        var solver = new LevenbergMarquardtSolver<double>(100, 1e-12, 1e-12, 0, 1e-3);
        Action act = () =>
        {
            if (writer)
            {
                solver.Solve((p, r) => r[0] = 1e-300 * (p[0] - 1.5e308), [-1.5e308], 1, (_, j) => j[0] = 1e-300);
            }
            else
            {
                solver.Solve(p => [1e-300 * (p[0] - 1.5e308)], [-1.5e308], _ => new[,] { { 1e-300 } });
            }
        };
        act.Should().Throw<InvalidOperationException>().WithMessage("*non-finite*");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Solve_LargeInactiveParameter_DoesNotStopAfterFirstStep(bool writer)
    {
        var solver = new LevenbergMarquardtSolver<double>(100, 1e-14, 1e-210, 0, 1e-3);
        var result = writer
            ? solver.Solve((p, r) => r[0] = p[1] - 10, [1e200, 0], 1, (_, j) => { j[0] = 0; j[1] = 1; })
            : solver.Solve(p => [p[1] - 10], [1e200, 0], _ => new[,] { { 0.0, 1.0 } });
        result.Parameters[1].Should().BeApproximately(10, 1e-9);
        result.Iterations.Should().BeGreaterThan(1);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void Solve_NarrowOrFixedBounds_FitsFreeCoordinate(bool writer, bool fixedCoordinate)
    {
        var width = fixedCoordinate ? 0 : 1e-9;
        var target = width * 0.2;
        var solver = new LevenbergMarquardtSolver<double>(100, 1e-14, 1e-14, 1e-12, 1e-3);
        MultivariateSolverResult<double>? result = null;
        Action act = () => result = writer
            ? solver.Solve((p, r) => { r[0] = (p[0] - target) * 1e9; r[1] = p[1] - 2; }, [width * 0.5, 0], 2,
                lowerBounds: [0, -10], upperBounds: [width, 10])
            : solver.Solve(p => [(p[0] - target) * 1e9, p[1] - 2], [width * 0.5, 0],
                lowerBounds: [0, -10], upperBounds: [width, 10]);
        act.Should().NotThrow("valid narrow and fixed bounds must permit fitting free coordinates");
        result!.Parameters[0].Should().BeApproximately(target, 1e-18);
        result.Parameters[1].Should().BeApproximately(2, 1e-8);
    }

    [Fact]
    public void SolverGuide_GenericLmExamples_SupplyRequiredArguments()
    {
        var guide = ReadDocument("solvers.md");
        guide.Should().NotContain("new LevenbergMarquardtSolver<double>()");
        guide.Should().NotContain("new LevenbergMarquardtSolver<float>()");
    }

    [Fact]
    public void SolverGuides_LmSupport_ExcludesHalf()
    {
        var row = ReadDocument("solvers.md").Split('\n').Single(line => line.StartsWith("| Multivariate nonlinear LS", StringComparison.Ordinal));
        row.Should().NotContain("`Half`");
        ReadDocument("tier-constraints.md").Should().Contain("does not support `Half`");
    }

    private static string ReadDocument(string name)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Boutquin.Numerics.slnx")))
        {
            directory = directory.Parent;
        }
        return File.ReadAllText(Path.Combine(directory!.FullName, "docs", name));
    }
}
