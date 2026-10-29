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

public sealed class GaussianEliminationTests
{
    [Theory]
    [InlineData(14)]
    [InlineData(24)]
    public void SmallDecimalSystem_PreservesItsUnitScaleSolution(int decimalPlaces)
    {
        var scale = new decimal(1, 0, 0, false, (byte)decimalPlaces);
        decimal[,] a = { { 3 * scale, scale, 2 * scale }, { scale, 4 * scale, scale }, { 2 * scale, scale, 5 * scale } };
        decimal[] b = [7 * scale, -4 * scale, 15 * scale];
        for (var seed = 0; seed < 20; seed++)
        {
            var actual = GaussianElimination<decimal>.Solve(a, b, seed);
            actual[0].Should().BeApproximately(1m, 1e-25m);
            actual[1].Should().BeApproximately(-2m, 1e-25m);
            actual[2].Should().BeApproximately(3m, 1e-25m);
        }
    }

    [Fact]
    public void Solve_MinimumPositiveDiagonal_DoesNotSelectZeroPivot()
    {
        const decimal tiny = 0.0000000000000000000000000001m;
        decimal[]? result = null;
        var act = () => result = GaussianElimination<decimal>.Solve(
            new[,] { { tiny, 0m }, { 0m, tiny } }, [tiny, tiny], seed: 0);

        act.Should().NotThrow("a nonsingular diagonal system must not select an exact zero pivot");
        result.Should().Equal(1m, 1m);
    }

    [Fact]
    public void Solve_3x3System_ReturnsCorrectSolution()
    {
        // 2x + y - z = 8
        // -3x - y + 2z = -11
        // -2x + y + 2z = -3
        // Solution: x=2, y=3, z=-1
        var a = new decimal[,]
        {
            { 2, 1, -1 },
            { -3, -1, 2 },
            { -2, 1, 2 },
        };
        var b = new decimal[] { 8, -11, -3 };

        var x = GaussianElimination.Solve(a, b, seed: 42);

        ((double)x[0]).Should().BeApproximately(2.0, 1e-10);
        ((double)x[1]).Should().BeApproximately(3.0, 1e-10);
        ((double)x[2]).Should().BeApproximately(-1.0, 1e-10);
    }

    [Fact]
    public void Solve_SingularMatrix_Throws()
    {
        var a = new decimal[,] { { 1, 2 }, { 2, 4 } };
        var b = new decimal[] { 3, 6 };

        var act = () => GaussianElimination.Solve(a, b, seed: 42);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Solve_DeterministicWithSeed_ProducesSameResult()
    {
        var a = new decimal[,]
        {
            { 4, 3, 2 },
            { 3, 5, 1 },
            { 2, 1, 6 },
        };
        var b = new decimal[] { 1, 2, 3 };

        var x1 = GaussianElimination.Solve(a, b, seed: 123);
        var x2 = GaussianElimination.Solve(a, b, seed: 123);

        for (var i = 0; i < x1.Length; i++)
        {
            x1[i].Should().Be(x2[i]);
        }
    }

    [Fact]
    public void Solve_IdentityMatrix_ReturnsRhs()
    {
        var a = new decimal[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        var b = new decimal[] { 5, 10, 15 };

        var x = GaussianElimination.Solve(a, b, seed: 42);

        ((double)x[0]).Should().BeApproximately(5.0, 1e-10);
        ((double)x[1]).Should().BeApproximately(10.0, 1e-10);
        ((double)x[2]).Should().BeApproximately(15.0, 1e-10);
    }

    [Fact]
    public void Solve_DimensionMismatch_Throws()
    {
        var a = new decimal[,] { { 1, 2 }, { 3, 4 } };
        var b = new decimal[] { 1, 2, 3 };

        var act = () => GaussianElimination.Solve(a, b);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Solve_NonFiniteCoefficient_Throws()
    {
        var a = new double[,] { { double.NaN } };
        var b = new[] { 1.0 };

        var act = () => GaussianElimination<double>.Solve(a, b, seed: 42);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Solve_NonFiniteRightHandSide_Throws()
    {
        var a = new double[,] { { 1 } };
        var b = new[] { double.NaN };

        var act = () => GaussianElimination<double>.Solve(a, b, seed: 42);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Solve_OverflowedPivot_Throws()
    {
        var a = new double[,] { { 1e308, 1e308 }, { 1e308, -1e308 } };
        var b = new[] { 1.0, 1.0 };

        var act = () => GaussianElimination<double>.Solve(a, b, seed: 42);

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void SelectPivotRow_TwoEligibleCandidates_VariesAcrossSeeds()
    {
        // Column 0 has two eligible candidates: aug[0,0]=6, aug[1,0]=10. columnMax=10,
        // threshold=0.5*10=5; both 6 and 10 are >= 5, so SelectPivotRow's candidate set is
        // {0, 1} and the outcome is driven entirely by the RNG. This is the mechanism the
        // class's own XML remarks (and the pivot-choice-for-all-seeds theory) claim is
        // "randomized" — but every OTHER test in this suite asserts only the seed-invariant
        // final solution, so a build that silently always returns the first candidate (no
        // randomization at all) passes every one of them. This test observes the pivot
        // CHOICE directly, via reflection into the private selection method, across a range
        // of seeds, and requires that both eligible rows are actually chosen at least once.
        var aug = new double[,]
        {
            { 6, 0, 0 },
            { 10, 0, 0 },
        };

        var chosenRows = new HashSet<int>();
        for (var seed = 0; seed < 50; seed++)
        {
            var row = InvokeSelectPivotRow(aug, n: 2, col: 0, new System.Random(seed), pivotFloor: 1e-10, out _);
            chosenRows.Add(row);
        }

        chosenRows.Should().BeEquivalentTo(new[] { 0, 1 },
            "the randomized pivot selection must actually draw from both eligible candidates " +
            "across seeds, not silently collapse to a single deterministic choice");
    }

    private static int InvokeSelectPivotRow(
        double[,] aug, int n, int col, System.Random rng, double pivotFloor, out double columnMax)
    {
        var method = typeof(GaussianElimination<double>).GetMethod(
            "SelectPivotRow", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("SelectPivotRow method not found via reflection.");

        var args = new object?[] { aug, n, col, rng, pivotFloor, null };
        try
        {
            var result = (int)method.Invoke(null, args)!;
            columnMax = (double)args[5]!;
            return result;
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            throw ex.InnerException;
        }
    }

    [Fact]
    public void XmlDocs_DoNotClaimRandomizedPivotStaysWithinPartialPivotingStabilityFloor()
    {
        // reference.md §15 (Known faithful-omission): "The randomized top-50% choice is a repo
        // heuristic OUTSIDE partial pivoting's stability guarantee: an eligible pivot >= 0.5*max
        // allows multipliers |l| <= 2 ... growth factor can exceed partial pivoting's 2^(n-1)
        // worst case, up to 3^(n-1) ... Any source text claiming the heuristic stays 'within the
        // partial-pivoting stability floor' is wrong." Audited (reference §15): a 2x2 system with
        // both rows eligible and multiplier 2 overflows for some seeds and not others — the prior
        // XML claimed the opposite. This test pins the source text, not runtime behavior, because
        // the defect IS the text; a doc-only regression here is a real regression.
        var source = File.ReadAllText(RepositoryPath(
            "src", "Boutquin.Numerics", "LinearAlgebra", "GaussianElimination.cs"));

        source.Should().NotContain("without leaving the partial-pivoting stability floor",
            "the randomized top-50% pivot is OUTSIDE partial pivoting's stability guarantee, not within it");
        source.Should().NotContain("preserving the partial-pivoting stability floor",
            "an eligible pivot at 0.5*columnMax allows multipliers up to 2, twice partial pivoting's bound");
        source.Should().NotContain("it carries no proven growth-factor bound beyond partial pivoting's worst case",
            "the growth-factor bound IS known and stated (up to 3^(n-1)), and it exceeds partial pivoting's 2^(n-1)");
        source.Should().NotContain("the solution is numerically identical up to floating-point noise regardless of seed",
            "different seeds can pick different eligible pivots and diverge, including one seed overflowing " +
            "and another succeeding, on ill-scaled inputs");

        source.Should().Contain("multipliers up to 2",
            "the corrected docs must state the actual multiplier bound the randomized heuristic allows");
        source.Should().Contain("3ⁿ⁻¹",
            "the corrected docs must state the actual growth-factor ceiling (Higham/GVL-style exponent notation)");
    }

    private static string RepositoryPath(params string[] parts)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Boutquin.Numerics.slnx")))
        {
            directory = directory.Parent;
        }

        directory.Should().NotBeNull("tests must run beneath the repository checkout");
        return Path.Combine(new[] { directory!.FullName }.Concat(parts).ToArray());
    }
}
