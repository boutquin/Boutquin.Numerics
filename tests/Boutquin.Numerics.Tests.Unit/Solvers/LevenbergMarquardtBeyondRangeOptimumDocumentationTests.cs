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

using System.Text.RegularExpressions;
using Boutquin.Numerics.Solvers;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// Pins the documented limit of the Levenberg–Marquardt range guarantee: only range-rejected
/// steps withhold the function/parameter-tolerance stops, so an in-domain, numerically
/// rank-deficient problem whose exact optimum lies beyond the representable range can still
/// be reported converged at a non-stationary point. The guides and XML must say so.
/// </summary>
public sealed class LevenbergMarquardtBeyondRangeOptimumDocumentationTests
{
    private const string NarrowedClaim = "non-stationary point of a numerically rank-deficient problem";

    [Fact]
    public void Solve_InDomainRankDeficientProblemWithBeyondRangeOptimum_StillCertifiesFunctionTolerance()
    {
        // Every J entry, y entry and start is inside [1e-150, 1e150]. J has exact full column rank, so the
        // exact least-squares optimum is unique and has a coordinate beyond double.MaxValue (Fraction normal
        // equations), hence the gradient is nonzero at every representable θ. No trial step is rejected for
        // range, so the MINPACK ftol test certifies the solve: Converged = true at a non-stationary point.
        double a = 7.09463555193594e-149;
        double[,] j =
        {
            { a, a },
            { a, 7.094635551997094e-149 },
            { 1.2981160496994943e-148, 1.2981160497327508e-148 },
        };
        double[] y = [-5.028428610250318e149, -1.625787374324429e149, 3.7155321943112776e149];
        var solver = new LevenbergMarquardtSolver<double>(400, 1e-12, 1e-12, 0, 1e-3);

        var result = solver.Solve(
            theta => Residuals(j, y, theta),
            [1.644551756559363e-05, -36.60244821524093],
            _ => j);

        result.Converged.Should().BeTrue();
        result.TerminationReason.Should().Be(LmTerminationReason.FunctionToleranceReached);
        result.Parameters.Should().OnlyContain(p => double.IsFinite(p) && Math.Abs(p) > 1e290);
    }

    [Fact]
    public void SolverGuide_BeyondRangeOptimum_DoesNotPromiseUnconditionalNonConvergence()
    {
        var guide = Normalize(ReadRepositoryFile("docs", "solvers.md"));

        guide.Should().NotContain("A problem whose optimum lies beyond the range therefore ends with");
        guide.Should().Contain(NarrowedClaim);
        guide.Should().Contain("`FunctionToleranceReached` or `ParameterToleranceReached`");
    }

    [Fact]
    public void SolverXmlDocumentation_ToleranceStops_CarryTheRankDeficientBeyondRangeCaveat()
    {
        var source = Normalize(ReadRepositoryFile("src", "Boutquin.Numerics", "Solvers", "LevenbergMarquardtSolver.cs"));

        // Generic class remarks, its two tolerance <param> texts, the legacy facade remarks and its two
        // tolerance <param> texts: six surfaces, each carrying the caveat once.
        Regex.Matches(source, Regex.Escape(NarrowedClaim)).Count.Should().Be(6);
        source.Should().NotContain("optimum lies beyond the range therefore");
    }

    private static double[] Residuals(double[,] j, double[] y, double[] theta)
    {
        var r = new double[y.Length];
        for (var i = 0; i < r.Length; i++)
        {
            r[i] = (j[i, 0] * theta[0]) + (j[i, 1] * theta[1]) - y[i];
        }

        return r;
    }

    // Collapses line wrapping and the XML-comment prefix so a claim can be matched across lines.
    private static string Normalize(string text) =>
        Regex.Replace(text, @"\s*(?:///)?\s*\r?\n\s*(?:///)?\s*", " ").Replace("  ", " ");

    private static string ReadRepositoryFile(params string[] relative)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Boutquin.Numerics.slnx")))
        {
            directory = directory.Parent;
        }

        return File.ReadAllText(Path.Combine([directory!.FullName, .. relative]));
    }
}
