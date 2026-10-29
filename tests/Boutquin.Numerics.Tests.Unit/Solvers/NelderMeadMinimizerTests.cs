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

/// <summary>
/// Unit tests for <see cref="NelderMeadMinimizer{T}"/>.
/// </summary>
public sealed class NelderMeadMinimizerTests
{
    [Fact]
    public void Minimize_LargeFiniteCentroid_ConvergesWithoutNonFiniteTrials()
    {
        const double scale = 1e308;
        var nonFiniteCalls = 0;
        double Objective(double[] x)
        {
            if (x.Any(value => !double.IsFinite(value)))
            {
                nonFiniteCalls++;
            }
            return Math.Pow(x[0] / scale - 0.8, 2) + Math.Pow(x[1] / scale - 0.9, 2);
        }

        var result = NelderMeadMinimizer<double>.Minimize(
            Objective, [scale, scale],
            new SimplexOptions<double>(XTolerance: scale * 1e-10, FTolerance: 0.0));

        nonFiniteCalls.Should().Be(0, "the finite simplex has a representable centroid");
        result.Converged.Should().BeTrue();
        (result.Argmin[0] / scale).Should().BeApproximately(0.8, 1e-8);
        (result.Argmin[1] / scale).Should().BeApproximately(0.9, 1e-8);
    }

    [Fact]
    public void Minimize_NegativeInfinityAfterShrink_DoesNotReportConvergence()
    {
        static double Objective(double[] x) => x[0] switch
        {
            0.0 => 0.0,
            0.05 => 1.0,
            0.025 => double.NegativeInfinity,
            _ => 2.0,
        };

        var result = NelderMeadMinimizer<double>.Minimize(
            Objective, [0.0], new SimplexOptions<double>(MaxIterations: 2, XTolerance: 0.0));

        result.Converged.Should().BeFalse("a non-finite worst vertex has no finite objective spread");
    }

    [Fact]
    public void SimplexResult_CopiesArgminOnBothEdges()
    {
        var source = new[] { 1.25, -2.5 };
        var result = new SimplexResult<double>(source, 0.0, 1, true);

        source[0] = 99.0;
        result.Argmin[0].Should().Be(1.25);

        var exposed = result.Argmin;
        exposed[1] = 88.0;
        result.Argmin[1].Should().Be(-2.5);
    }

    public static TheoryData<string, double> NonFiniteOptionValues => new()
    {
        { nameof(SimplexOptions<double>.XTolerance), double.NaN },
        { nameof(SimplexOptions<double>.FTolerance), double.PositiveInfinity },
        { nameof(SimplexOptions<double>.InitialStep), double.NegativeInfinity },
        { nameof(SimplexOptions<double>.Reflection), double.NaN },
        { nameof(SimplexOptions<double>.Expansion), double.PositiveInfinity },
        { nameof(SimplexOptions<double>.Contraction), double.NegativeInfinity },
        { nameof(SimplexOptions<double>.Shrink), double.NaN },
    };

    [Theory]
    [MemberData(nameof(NonFiniteOptionValues))]
    public void Minimize_RejectsNonFiniteOptions(string optionName, double value)
    {
        var options = optionName switch
        {
            nameof(SimplexOptions<double>.XTolerance) => new SimplexOptions<double>(XTolerance: value),
            nameof(SimplexOptions<double>.FTolerance) => new SimplexOptions<double>(FTolerance: value),
            nameof(SimplexOptions<double>.InitialStep) => new SimplexOptions<double>(InitialStep: value),
            nameof(SimplexOptions<double>.Reflection) => new SimplexOptions<double>(Reflection: value),
            nameof(SimplexOptions<double>.Expansion) => new SimplexOptions<double>(Expansion: value),
            nameof(SimplexOptions<double>.Contraction) => new SimplexOptions<double>(Contraction: value),
            nameof(SimplexOptions<double>.Shrink) => new SimplexOptions<double>(Shrink: value),
            _ => throw new InvalidOperationException($"Unknown option: {optionName}"),
        };

        var act = () => NelderMeadMinimizer<double>.Minimize(x => x[0] * x[0], [1.0], options);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(optionName);
    }

    /// <summary>
    /// Every sibling iteration-count parameter in this solver family
    /// (<see cref="BisectionSolver{T}"/>, <see cref="BrentSolver{T}"/>,
    /// <see cref="SecantSolver{T}"/>, <see cref="MullerSolver{T}"/>,
    /// <see cref="NewtonRaphsonSolver{T}"/>, <see cref="LevenbergMarquardtSolver{T}"/>) rejects
    /// a non-positive <c>maxIterations</c> via <c>ArgumentOutOfRangeException.ThrowIfNegativeOrZero</c>.
    /// <see cref="SimplexOptions{T}.MaxIterations"/> was the lone family member with no such
    /// guard: <c>while (iter &lt; maxIter)</c> is false immediately for
    /// <c>maxIter &lt;= 0</c>, so the loop body never runs and the initial simplex is returned
    /// as a normal-looking <c>Converged = false</c> result instead of a diagnostic exception.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    public void Minimize_RejectsNonPositiveMaxIterations(int maxIterations)
    {
        var options = new SimplexOptions<double>(MaxIterations: maxIterations);

        var act = () => NelderMeadMinimizer<double>.Minimize(x => x[0] * x[0], [1.0], options);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName(nameof(SimplexOptions<double>.MaxIterations));
    }

    // AC-1: Rosenbrock minimum [1,1] to within 1e-4 from start [0,0]
    [Fact]
    public void Minimize_Rosenbrock_RecoversMinimum()
    {
        // f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at [1,1]
        static double Rosenbrock(double[] x)
        {
            var a = 1.0 - x[0];
            var b = x[1] - x[0] * x[0];
            return a * a + 100.0 * b * b;
        }

        var result = NelderMeadMinimizer<double>.Minimize(Rosenbrock, [0.0, 0.0]);

        result.Converged.Should().BeTrue("Rosenbrock should converge with default options");
        result.Argmin[0].Should().BeApproximately(1.0, 1e-4, "x* should be ~1");
        result.Argmin[1].Should().BeApproximately(1.0, 1e-4, "y* should be ~1");
        result.MinValue.Should().BeLessThan(1e-7, "f(x*) should be ~0");
    }

    // AC-2: Sphere minimum [0,...,0] in 3D to within 1e-6
    [Fact]
    public void Minimize_Sphere3D_RecoversMinimum()
    {
        static double Sphere(double[] x) => x[0] * x[0] + x[1] * x[1] + x[2] * x[2];

        var result = NelderMeadMinimizer<double>.Minimize(Sphere, [1.0, 2.0, -1.0]);

        result.Converged.Should().BeTrue("Sphere should converge with default options");
        result.Argmin[0].Should().BeApproximately(0.0, 1e-6, "x* should be ~0");
        result.Argmin[1].Should().BeApproximately(0.0, 1e-6, "y* should be ~0");
        result.Argmin[2].Should().BeApproximately(0.0, 1e-6, "z* should be ~0");
        result.MinValue.Should().BeLessThan(1e-12);
    }

    // AC-3: Beale minimum [3, 0.5] to within 1e-3 from start [0,0]
    [Fact]
    public void Minimize_Beale_RecoversMinimum()
    {
        // f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
        // minimum at [3, 0.5], f=0
        static double Beale(double[] x)
        {
            var t1 = 1.5 - x[0] + x[0] * x[1];
            var t2 = 2.25 - x[0] + x[0] * x[1] * x[1];
            var t3 = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
            return t1 * t1 + t2 * t2 + t3 * t3;
        }

        var result = NelderMeadMinimizer<double>.Minimize(
            Beale,
            [0.0, 0.0],
            new SimplexOptions<double>(MaxIterations: 5000));

        result.Converged.Should().BeTrue("Beale should converge");
        result.Argmin[0].Should().BeApproximately(3.0, 1e-3, "x* should be ~3");
        result.Argmin[1].Should().BeApproximately(0.5, 1e-3, "y* should be ~0.5");
    }

    // AC-4: Returns Converged=false (not exception) on iteration cap with MaxIterations=1
    [Fact]
    public void Minimize_IterationCap_ReturnsFalseNotException()
    {
        static double Sphere(double[] x) => x[0] * x[0] + x[1] * x[1];

        var opts = new SimplexOptions<double>(MaxIterations: 1);
        var act = () => NelderMeadMinimizer<double>.Minimize(Sphere, [1.0, 1.0], opts);

        var result = act.Should().NotThrow().Subject;
        result.Converged.Should().BeFalse("should not converge in 1 iteration");
        result.Iterations.Should().Be(2);
    }

    // AC-5: Deterministic across two identical calls
    [Fact]
    public void Minimize_DeterministicAcrossTwoCalls()
    {
        static double Rosenbrock(double[] x)
        {
            var a = 1.0 - x[0];
            var b = x[1] - x[0] * x[0];
            return a * a + 100.0 * b * b;
        }

        var r1 = NelderMeadMinimizer<double>.Minimize(Rosenbrock, [0.5, 0.5]);
        var r2 = NelderMeadMinimizer<double>.Minimize(Rosenbrock, [0.5, 0.5]);

        r1.Argmin[0].Should().Be(r2.Argmin[0], "results must be identical");
        r1.Argmin[1].Should().Be(r2.Argmin[1]);
        r1.MinValue.Should().Be(r2.MinValue);
        r1.Iterations.Should().Be(r2.Iterations);
        r1.Converged.Should().Be(r2.Converged);
    }

    // AC-6: ArgumentException on empty initialGuess
    [Fact]
    public void Minimize_EmptyGuess_ThrowsArgumentException()
    {
        static double Sphere(double[] x) => x[0] * x[0];

        var act = () => NelderMeadMinimizer<double>.Minimize(Sphere, []);
        act.Should().Throw<ArgumentException>("empty guess is invalid");
    }

    // Covers F-VS3-04: the null-objective guard (ArgumentNullException.ThrowIfNull(objective)
    // at NelderMeadMinimizer.cs line ~154) was previously untested — only the empty-guess
    // guard above had coverage. Pins the documented ArgumentNullException contract.
    [Fact]
    public void Minimize_NullObjective_ThrowsArgumentNullException()
    {
        var act = () => NelderMeadMinimizer<double>.Minimize(null!, [1.0, 1.0]);
        act.Should().Throw<ArgumentNullException>("objective is required");
    }

    // Additional: options null fields use defaults (smoke test)
    [Fact]
    public void Minimize_NullOptions_UsesDefaults()
    {
        static double Sphere(double[] x) => x[0] * x[0] + x[1] * x[1];
        var result = NelderMeadMinimizer<double>.Minimize(Sphere, [1.0, 1.0], null);
        result.Converged.Should().BeTrue("sphere should converge with default options");
    }

    // M-3 guard: a non-finite objective value at a mid-iteration vertex must be
    // ranked worst (as if +∞), never sorted as best.
    [Fact]
    public void Minimize_NonFiniteObjectiveMidIteration_DoesNotCorruptOrdering()
    {
        // The initial simplex for a 2-D problem evaluates 3 vertices (calls 1-3),
        // which are already guarded (non-finite there throws ArgumentException).
        // Force every mid-iteration evaluation in calls 4-30 to return NaN
        // regardless of the point, then confirm the search still converges to the
        // true Sphere minimum at (0,0) once real evaluations resume. Pre-fix,
        // Array.Sort's Comparer<T>.Default.Compare treats NaN as LESS than every
        // other value (IComparable<double> convention: NaN.CompareTo(x) = -1 for
        // any finite x), so a NaN-tagged vertex sorts as order[0] = "best" — and
        // the shrink step's unconditional `fVals[order[j]] = objective(vj)` (no
        // finiteness check) is exactly how a NaN gets stored into fVals in the
        // first place. Once corrupted, fSpread (= fVals[worst] - fVals[best])
        // involves a NaN operand and "fSpread <= fTol" is permanently false, so
        // the search runs to the iteration cap without ever converging.
        var evalCount = 0;
        double Objective(double[] x)
        {
            evalCount++;
            if (evalCount is >= 4 and <= 30)
            {
                return double.NaN;
            }

            return x[0] * x[0] + x[1] * x[1]; // Sphere, minimum at (0,0)
        }

        var result = NelderMeadMinimizer<double>.Minimize(Objective, [5.0, 5.0]);

        result.Converged.Should().BeTrue("the NaN vertex must be ranked worst, never corrupting best-tracking");
        result.Argmin[0].Should().BeApproximately(0.0, 1e-3);
        result.Argmin[1].Should().BeApproximately(0.0, 1e-3);
        double.IsFinite(result.MinValue).Should().BeTrue();
    }

    // Verify decimal type works (Tier A — no transcendentals in minimizer itself)
    [Fact]
    public void Minimize_Decimal_SphereConverges()
    {
        static decimal Sphere(decimal[] x) => x[0] * x[0] + x[1] * x[1];
        var result = NelderMeadMinimizer<decimal>.Minimize(Sphere, [1m, 2m]);
        result.Converged.Should().BeTrue("sphere with decimal should converge");
        result.Argmin[0].Should().BeApproximately(0m, 1e-6m);
        result.Argmin[1].Should().BeApproximately(0m, 1e-6m);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Minimize_NonFiniteInitialGuess_ThrowsArgumentException(double coordinate)
    {
        var act = () => NelderMeadMinimizer<double>.Minimize(_ => 0.0, [coordinate]);

        act.Should().Throw<ArgumentException>()
            .WithMessage("*index 0*finite*")
            .WithParameterName("initialGuess");
    }

    [Fact]
    public void Minimize_PositiveInitialStepLostToRounding_RejectsStep()
    {
        var act = () => NelderMeadMinimizer<double>.Minimize(
            x => x[0] * x[0],
            [1e20],
            new SimplexOptions<double>(InitialStep: 1e-20));

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("options");
    }

    [Fact]
    public void Minimize_DefaultInitialStepOverflow_RejectsVertexBeforeObjective()
    {
        var sawNonFiniteCoordinate = false;
        double Objective(double[] x)
        {
            sawNonFiniteCoordinate |= !double.IsFinite(x[0]);
            return 0.0;
        }

        var act = () => NelderMeadMinimizer<double>.Minimize(Objective, [double.MaxValue]);

        act.Should().Throw<ArgumentException>().WithParameterName("initialGuess");
        sawNonFiniteCoordinate.Should().BeFalse();
    }
}
