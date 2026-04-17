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

namespace Boutquin.Numerics.Tests.Verification;

public sealed class SolverVerificationTests : CrossLanguageVerificationBase
{
    private static Func<double, double> ResolveFunction(string name) => name switch
    {
        "cubic" => x => x * x * x - x - 2.0,
        "quadratic" => x => x * x - 2.0,
        "log" => x => Math.Log(x) - 1.0,
        "cosine" => x => Math.Cos(x) - x,
        "exp_minus_linear" => x => Math.Exp(-x) - x,
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Unknown test function."),
    };

    [Fact]
    public void BrentSolver_MatchesReferenceRoots()
    {
        using var doc = LoadVector("solvers");
        var cases = doc.RootElement.GetProperty("cases");

        foreach (var c in cases.EnumerateArray())
        {
            var name = c.GetProperty("name").GetString()!;
            var bracket = GetDoubleArray(c.GetProperty("bracket"));
            var expected = c.GetProperty("root").GetDouble();
            var f = ResolveFunction(name);

            var result = new BrentSolver(PrecisionExact).Solve(f, bracket[0], bracket[1]);
            result.Converged.Should().BeTrue($"Brent must converge for {name}");
            AssertScalarWithin(result.Root, expected, PrecisionNumeric, $"Brent/{name}");
        }
    }

    [Fact]
    public void BisectionSolver_MatchesReferenceRoots()
    {
        using var doc = LoadVector("solvers");
        var cases = doc.RootElement.GetProperty("cases");

        foreach (var c in cases.EnumerateArray())
        {
            var name = c.GetProperty("name").GetString()!;
            var bracket = GetDoubleArray(c.GetProperty("bracket"));
            var expected = c.GetProperty("root").GetDouble();
            var f = ResolveFunction(name);

            var result = new BisectionSolver(PrecisionExact).Solve(f, bracket[0], bracket[1]);
            result.Converged.Should().BeTrue($"Bisection must converge for {name}");
            AssertScalarWithin(result.Root, expected, PrecisionNumeric, $"Bisection/{name}");
        }
    }

    // ---------------------------------------------------------------------
    //  Phase 4 — Newton (bracketed + unbracketed), Secant, Muller (§2.4).
    //  Reference roots are mathematical constants independent of the
    //  solver used to find them. Muller's stored roots are validated at
    //  generation time against ``mpmath.findroot(..., solver='muller')``
    //  to eliminate the shared-implementation-bug risk a numpy hand-port
    //  would carry.
    // ---------------------------------------------------------------------

    [Fact]
    public void NewtonRaphsonSolver_Bracketed_MatchesReferenceRoots()
    {
        using var doc = LoadVector("solvers");
        var cases = doc.RootElement.GetProperty("cases");

        foreach (var c in cases.EnumerateArray())
        {
            var name = c.GetProperty("name").GetString()!;
            var bracket = GetDoubleArray(c.GetProperty("bracket"));
            var expected = c.GetProperty("root").GetDouble();
            var f = ResolveFunction(name);

            var result = new NewtonRaphsonSolver().Solve(f, bracket[0], bracket[1]);
            result.Converged.Should().BeTrue($"Newton (bracketed) must converge for {name}");
            AssertScalarWithin(result.Root, expected, PrecisionNumeric, $"Newton-bracketed/{name}");
        }
    }

    [Fact]
    public void NewtonRaphsonSolver_Unbracketed_MatchesReferenceRoots()
    {
        using var doc = LoadVector("solvers");
        var cases = doc.RootElement.GetProperty("cases");

        foreach (var c in cases.EnumerateArray())
        {
            var name = c.GetProperty("name").GetString()!;
            var initialGuess = c.GetProperty("initial_guess").GetDouble();
            var expected = c.GetProperty("root").GetDouble();
            var f = ResolveFunction(name);

            var result = new NewtonRaphsonSolver().Solve(f, initialGuess);
            result.Converged.Should().BeTrue($"Newton (unbracketed) must converge for {name}");
            AssertScalarWithin(result.Root, expected, PrecisionNumeric, $"Newton-unbracketed/{name}");
        }
    }

    [Fact]
    public void SecantSolver_MatchesReferenceRoots()
    {
        using var doc = LoadVector("solvers");
        var cases = doc.RootElement.GetProperty("cases");

        foreach (var c in cases.EnumerateArray())
        {
            var name = c.GetProperty("name").GetString()!;
            var initialGuess = c.GetProperty("initial_guess").GetDouble();
            var expected = c.GetProperty("root").GetDouble();
            var f = ResolveFunction(name);

            var result = new SecantSolver().Solve(f, initialGuess);
            result.Converged.Should().BeTrue($"Secant must converge for {name}");
            AssertScalarWithin(result.Root, expected, PrecisionNumeric, $"Secant/{name}");
        }
    }

    [Fact]
    public void MullerSolver_MatchesReferenceRoots()
    {
        using var doc = LoadVector("solvers");
        var cases = doc.RootElement.GetProperty("cases");

        foreach (var c in cases.EnumerateArray())
        {
            var name = c.GetProperty("name").GetString()!;
            var initialGuess = c.GetProperty("initial_guess").GetDouble();
            var expected = c.GetProperty("root").GetDouble();
            var f = ResolveFunction(name);

            var result = new MullerSolver().Solve(f, initialGuess);
            result.Converged.Should().BeTrue($"Muller must converge for {name}");
            AssertScalarWithin(result.Root, expected, PrecisionNumeric, $"Muller/{name}");
        }
    }
}
