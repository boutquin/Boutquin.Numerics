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
public sealed class Arena20260928NelderMeadTests
{
    [Theory]
    [InlineData("Reflection", 0)]
    [InlineData("Reflection", -1)]
    [InlineData("Expansion", 1)]
    [InlineData("Expansion", 0)]
    [InlineData("Contraction", 0)]
    [InlineData("Contraction", 1)]
    [InlineData("Shrink", 0)]
    [InlineData("Shrink", 1)]
    [InlineData("XTolerance", -1)]
    [InlineData("FTolerance", -1)]
    public void InvalidFiniteOption_IsRefusedBeforeObjective(string name, double value)
    {
        var options = new SimplexOptions<double>();
        options = name switch
        {
            "Reflection" => options with { Reflection = value },
            "Expansion" => options with { Expansion = value },
            "Contraction" => options with { Contraction = value },
            "Shrink" => options with { Shrink = value },
            "XTolerance" => options with { XTolerance = value },
            _ => options with { FTolerance = value },
        };
        var calls = 0;
        Action solve = () => NelderMeadMinimizer<double>.Minimize(x => { calls++; return (x[0] - 10) * (x[0] - 10); }, [0], options);

        solve.Should().Throw<ArgumentOutOfRangeException>().Which.ParamName.Should().Be(name);
        calls.Should().Be(0);
    }

    [Fact]
    public void ExpansionNotGreaterThanReflection_IsRefused()
    {
        Action solve = () => NelderMeadMinimizer<double>.Minimize(x => x[0] * x[0], [1],
            new SimplexOptions<double>(Reflection: 3, Expansion: 2));
        solve.Should().Throw<ArgumentOutOfRangeException>().Which.ParamName.Should().Be("Expansion");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EqualNewVertices_PreserveExistingBest(bool shrink)
    {
        var visited = new List<double[]>();
        double Objective(double[] x)
        {
            visited.Add((double[])x.Clone());
            return (x[0], x[1]) switch
            {
                (0, 0) => 2,
                (1, 0) => 0,
                (0, 1) => 1,
                (1, 1) when !shrink => 1,
                (0.75, 0.75) when !shrink => 0,
                (0.5, 0) when shrink => 0,
                (0.5, 0.5) when shrink => 0,
                _ => 10,
            };
        }
        var result = NelderMeadMinimizer<double>.Minimize(Objective, [0, 0],
            new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1));

        // Lagarias §2: new equals follow existing equals; shrink retains old best.
        // Initial ranks are (1,0), (0,1), (0,0), so centroid=(.5,.5).
        visited[3].Should().Equal(1, 1);
        visited[4].Should().Equal(shrink ? [0.25, 0.25] : [0.75, 0.75]);
        result.Argmin.Should().Equal(1, 0);
        result.MinValue.Should().Be(0);
    }
    [Theory]
    [InlineData("reflection")]
    [InlineData("expansion")]
    [InlineData("rejected-expansion")]
    [InlineData("outside-equality")]
    [InlineData("inside")]
    [InlineData("inside-equality-shrink")]
    public void StandardCoefficientsAndComparisons_FollowPrimaryBranch(string branch)
    {
        var visited = new List<double[]>();
        double Objective(double[] x)
        {
            visited.Add((double[])x.Clone());
            return (x[0], x[1]) switch
            {
                (0, 0) => 0,
                (1, 0) => 1,
                (0, 1) => 2,
                (1, -1) => branch switch
                {
                    "reflection" => 0.5,
                    "expansion" or "rejected-expansion" => -1,
                    "outside-equality" => 1.5,
                    _ => 2,
                },
                (1.5, -2) => branch == "expansion" ? -2 : -0.5,
                (0.75, -0.5) => 1.5,
                (0.25, 0.5) => branch == "inside" ? 1 : 2,
                _ => 0.5,
            };
        }
        var result = NelderMeadMinimizer<double>.Minimize(Objective, [0, 0],
            new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1));

        // Lagarias equations2.4–2.7, with centroid(.5,0) and worst(0,1).
        visited[3].Should().Equal(1, -1);
        if (branch == "reflection") { visited.Should().HaveCount(6); }
        else if (branch is "expansion" or "rejected-expansion")
        {
            visited.Should().HaveCount(7);
            visited[4].Should().Equal(1.5, -2);
            result.Argmin.Should().Equal(branch == "expansion" ? [1.5, -2] : [1, -1]);
        }
        else if (branch == "outside-equality")
        {
            visited.Should().HaveCount(7, "outside equality is accepted, without shrinking");
            visited[4].Should().Equal(0.75, -0.5);
        }
        else
        {
            visited[4].Should().Equal(0.25, 0.5);
            if (branch == "inside") { visited.Should().HaveCount(7); }
            else
            {
                visited.Should().HaveCount(11, "inside equality must shrink");
                visited[5].Should().Equal(0.5, 0);
                visited[6].Should().Equal(0, 0.5);
            }
        }
    }

    [Theory]
    [InlineData("Rosenbrock")]
    [InlineData("Powell")]
    [InlineData("Helical")]
    public void OriginalPaperStarts_ReachDefiningZeroMinimum(string problem)
    {
        // Nelder–Mead1965 pp309–310: these three nonnegative functions have minimum0.
        // This checks the stated start and optimum, not table evaluation counts:
        // the API uses axial initialization and diameter/spread stopping.
        double[] start = problem switch
        {
            "Rosenbrock" => [-1.2, 1],
            "Powell" => [3, -1, 0, 1],
            _ => [-1, 0, 0],
        };
        double Objective(double[] x)
        {
            if (problem == "Rosenbrock") { return 100 * Math.Pow(x[1] - x[0] * x[0], 2) + Math.Pow(1 - x[0], 2); }
            if (problem == "Powell")
            {
                return Math.Pow(x[0] + 10 * x[1], 2) + 5 * Math.Pow(x[2] - x[3], 2)
                    + Math.Pow(x[1] - 2 * x[2], 4) + 10 * Math.Pow(x[0] - x[3], 4);
            }
            var theta = Math.Atan2(x[1], x[0]) / (2 * Math.PI);
            if (x[0] < 0 && theta < 0) { theta += 1; }
            return 100 * (Math.Pow(x[2] - 10 * theta, 2) + Math.Pow(Math.Sqrt(x[0] * x[0] + x[1] * x[1]) - 1, 2)) + x[2] * x[2];
        }
        var result = NelderMeadMinimizer<double>.Minimize(Objective, start,
            new SimplexOptions<double>(MaxIterations: 10000, XTolerance: 1e-10, FTolerance: 0, InitialStep: 0.5));

        result.Converged.Should().BeTrue();
        result.MinValue.Should().BeLessThan(1e-16);
        double[] optimum = problem switch { "Rosenbrock" => [1, 1], "Powell" => [0, 0, 0, 0], _ => [1, 0, 0] };
        for (var i = 0; i < optimum.Length; i++)
        {
            result.Argmin[i].Should().BeApproximately(optimum[i], problem == "Powell" ? 1e-4 : 1e-7);
        }
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(10)]
    public void FourthPowerPaperFamily_HasZeroMinimum(int dimension)
    {
        // Nelder–Mead1965 p311: sum x_i^4 from all ones, dimensions2–10.
        var result = NelderMeadMinimizer<double>.Minimize(x => x.Sum(v => v * v * v * v),
            Enumerable.Repeat(1.0, dimension).ToArray(),
            new SimplexOptions<double>(MaxIterations: 20000, XTolerance: 1e-10, FTolerance: 0, InitialStep: 0.5));
        result.Converged.Should().BeTrue();
        result.MinValue.Should().BeLessThan(1e-16);
        result.Argmin.Should().OnlyContain(x => Math.Abs(x) < 1e-4);
    }
}
