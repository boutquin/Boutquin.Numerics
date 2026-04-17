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

using BenchmarkDotNet.Attributes;
using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.BenchMark;

/// <summary>
/// Compares root-finder throughput on a fixed test function with a known
/// root, across tight and loose tolerances. Each solver is constructed
/// once per benchmark run; <c>Solve</c> is the hot path.
/// </summary>
[MemoryDiagnoser]
public class SolverBenchmarks
{
    [Params(1e-10, 1e-6)]
    public double Tolerance { get; set; }

    private BisectionSolver _bisection = null!;
    private BrentSolver _brent = null!;
    private NewtonRaphsonSolver _newton = null!;
    private SecantSolver _secant = null!;

    // f(x) = x^3 - x - 2; root ≈ 1.5213797...
    private static double F(double x) => x * x * x - x - 2.0;
    private static double FPrime(double x) => 3.0 * x * x - 1.0;

    [GlobalSetup]
    public void Setup()
    {
        _bisection = new BisectionSolver(Tolerance);
        _brent = new BrentSolver(Tolerance);
        _newton = new NewtonRaphsonSolver(Tolerance, Tolerance, derivative: FPrime);
        _secant = new SecantSolver(Tolerance);
    }

    [Benchmark(Baseline = true)]
    public RootSolverResult Bisection() => _bisection.Solve(F, 1.0, 2.0);

    [Benchmark]
    public RootSolverResult Brent() => _brent.Solve(F, 1.0, 2.0);

    [Benchmark]
    public RootSolverResult Newton() => _newton.Solve(F, 1.0, 2.0);

    [Benchmark]
    public RootSolverResult Secant() => _secant.Solve(F, 1.0, 2.0);
}
