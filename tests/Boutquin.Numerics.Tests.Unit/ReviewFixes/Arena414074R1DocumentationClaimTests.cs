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

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

using Boutquin.Numerics.Interpolation;
using Boutquin.Numerics.MonteCarlo;
using Boutquin.Numerics.Solvers;
using Boutquin.Numerics.Statistics;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Executes the claims that the public README, the solver guide and the paper-replication
/// catalogue make about runnable behavior: snippet inputs are fed to the public
/// constructors, catalogue procedures are replayed against production, and every
/// "Implemented" row must name a test method that exists in its linked file. Each test
/// reads the shipped document text, so a stale claim fails at its assertion.
/// </summary>
public sealed class Arena414074R1DocumentationClaimTests
{
    private const string CatalogueDocument = "docs/paper-replication-tests.md";

    private static readonly string s_root = FindRepositoryRoot();

    // ------------------------------------------------------------------
    // Solver guide: the MINPACK ftol gain-ratio clause is implemented.
    // ------------------------------------------------------------------

    [Fact]
    public void SolversGuide_FunctionTolerance_DescribesTheImplementedGainRatioGate()
    {
        var guide = ReadRepositoryText("docs/solvers.md");
        var bullet = guide.Split('\n').Single(l => l.StartsWith("- **Relative function tolerance**", StringComparison.Ordinal));
        var termination = guide.Split('\n').Single(l => l.StartsWith("| `FunctionToleranceReached`", StringComparison.Ordinal));

        bullet.Should().NotContain("omits that third clause",
            because: "all four public Solve paths gate FunctionToleranceReached on half the gain ratio being at most one");
        bullet.Should().Contain("½·ratio ≤ 1").And.Contain("gain ratio");
        termination.Should().Contain("gain ratio",
            because: "the termination table must state the full stopping rule, including the third MINPACK clause");
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.1)]
    public void FunctionTolerance_GainRatioGate_DecidesTerminationOnEveryPublicPath(double curvature)
    {
        // r(x) = [1-c*x², s*(1+x)] at x=0 with damping1 gives step -1/2.
        // The first residual has zero initial derivative, so curvature changes only
        // actual reduction. This isolates the gain gate without dominant initial damping.
        // The expected termination comes from the documented three-clause rule evaluated in
        // closed form here, not from the solver.
        const double FunctionTolerance = 0.25;
        const double Scale = 0.01;
        const double Step = -0.5;
        var trialResidual = 1 - curvature * Step * Step;
        var cost = (1 + Scale * Scale) / 2;
        var actualReduction = cost - (trialResidual * trialResidual + Scale * Scale * (1 + Step) * (1 + Step)) / 2;
        var predictedReduction = Scale * Scale * (-Step - Step * Step / 2);
        var ratio = actualReduction / predictedReduction;
        var gateOpen = Math.Abs(actualReduction / cost) <= FunctionTolerance
            && predictedReduction / cost <= FunctionTolerance
            && ratio / 2 <= 1;
        var expected = gateOpen ? LmTerminationReason.FunctionToleranceReached : LmTerminationReason.MaxIterationsReached;

        double F(double x) => 1 - curvature * x * x;
        double J(double x) => -2 * curvature * x;
        var solver = new LevenbergMarquardtSolver<double>(1, FunctionTolerance, 1e-20, 1e-20, 1);
        var pool = new LevenbergMarquardtBuffers<double>(1, 2);
        Func<double[], double[]> residual = p => [F(p[0]), Scale * (1 + p[0])];
        Func<double[], double[,]> jacobian = p => new double[,] { { J(p[0]) }, { Scale } };
        LmResidualWriter<double> residualWriter = (p, r) => { r[0] = F(p[0]); r[1] = Scale * (1 + p[0]); };
        LmJacobianWriter<double> jacobianWriter = (p, j) => { j[0] = J(p[0]); j[1] = Scale; };

        MultivariateSolverResult<double>[] results =
        [
            solver.Solve(residual, new[] { 0.0 }, jacobian),
            solver.Solve(residual, new[] { 0.0 }, pool, jacobian),
            solver.Solve(residualWriter, new[] { 0.0 }, 2, jacobianWriter),
            solver.Solve(residualWriter, new[] { 0.0 }, pool, jacobianWriter),
        ];

        (curvature == 0.0).Should().Be(gateOpen, "the pair must straddle the gain-ratio gate");
        foreach (var result in results)
        {
            result.TerminationReason.Should().Be(expected, $"curvature {curvature}: gain ratio {ratio:R}");
            result.Iterations.Should().Be(1);
        }
    }

    // ------------------------------------------------------------------
    // README snippets: the Q-spline example and the bootstrap callbacks.
    // ------------------------------------------------------------------

    [Fact]
    public void ReadmeInterpolationSnippet_SplineExamples_SatisfyEveryConstructorInputDomain()
    {
        var block = ReadmeCodeBlock("Interpolation");
        var xs = ReadmeDoubleArray(block, "xs");
        var ys = ReadmeDoubleArray(block, "ys");
        var boundaries = Regex.Matches(block, @"new CubicSplineInterpolator\(xs, ys, SplineBoundary\.(\w+)\)")
            .Select(m => m.Groups[1].Value)
            .ToArray();

        boundaries.Should().Contain("QSpline");
        for (var i = 0; i < xs.Length; i++)
        {
            ys[i].Should().Be(xs[i] * xs[i], "the snippet documents quadratic ordinates");
        }

        foreach (var name in boundaries)
        {
            var boundary = (SplineBoundary)typeof(SplineBoundary).GetProperty(name)!.GetValue(null)!;
            var construct = () => new CubicSplineInterpolator(xs, ys, boundary);
            construct.Should().NotThrow($"the README {name} example must pass the constructor's knot-count rule");
        }

        // Jarre Theorem 2: a Q-spline reproduces polynomials of degree <= 3 between the knots.
        new CubicSplineInterpolator(xs, ys, SplineBoundary.QSpline).Interpolate(2.5)
            .Should().BeApproximately(2.5 * 2.5, 1e-10);
    }

    [Fact]
    public void ReadmeBootstrapSnippet_StatisticCallbacks_DoNotApplyLinqToSpanParameters()
    {
        var block = ReadmeCodeBlock("Monte Carlo Bootstrap");
        var parameters = Regex.Matches(block, @"\b([A-Za-z_]\w*)\s*=>")
            .Select(m => m.Groups[1].Value)
            .Concat(Regex.Matches(block, @"\(\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*\)\s*=>")
                .SelectMany(m => m.Groups[1].Value.Split(',', StringSplitOptions.TrimEntries)))
            .Distinct()
            .ToArray();

        parameters.Should().Contain("paths", "the snippet passes statistic callbacks to Run and PValueImposingNull");
        foreach (var parameter in parameters)
        {
            // Run, RunMulti, PValueImposingNull and Subsampler.Run hand their callbacks a
            // ReadOnlySpan<decimal> or Span<decimal>, which has no LINQ extension methods.
            Regex.IsMatch(block, $@"\b{Regex.Escape(parameter)}\.(Average|Sum|Min|Max|Count|Any|All|Select|Where|OrderBy|ToArray|ToList|First|Last|Aggregate)\(")
                .Should().BeFalse($"callback parameter '{parameter}' is a span and cannot call LINQ");
        }
    }

    // ------------------------------------------------------------------
    // Catalogue: every Implemented row names an existing test method.
    // ------------------------------------------------------------------

    [Fact]
    public void Catalogue_ImplementedRows_NameATestMethodThatExistsInTheLinkedFile()
    {
        var missing = new List<string>();
        var rows = 0;
        foreach (var line in ReadRepositoryText(CatalogueDocument).Split('\n')
            .Where(l => l.StartsWith("Status: **Implemented**", StringComparison.Ordinal)))
        {
            rows++;
            var match = Regex.Match(line, @"\[[^\]]+\]\(([^)]+)\)\s*::\s*`([^`]+)`");
            if (!match.Success)
            {
                missing.Add($"unparseable status line: {line}");
                continue;
            }

            var path = Path.GetFullPath(Path.Combine(s_root, "docs", match.Groups[1].Value));
            var name = match.Groups[2].Value;
            if (!File.Exists(path))
            {
                missing.Add($"{match.Groups[1].Value} does not exist (row names {name})");
                continue;
            }

            var declaration = $@"\b(?:void|Task)\s+{Regex.Escape(name)}\s*(?:<[^>]*>)?\s*\(";
            if (!Regex.IsMatch(File.ReadAllText(path), declaration))
            {
                missing.Add($"{name} is not declared in {match.Groups[1].Value}");
            }
        }

        rows.Should().BeGreaterThanOrEqualTo(204, "the catalogue header states 204 implemented entries");
        missing.Should().BeEmpty("a status line that names an absent method documents no test");
    }

    [Fact]
    public void CatalogueRow118_QSplineGrids_AreAcceptedAndReproduceLowDegreePolynomials()
    {
        // Jarre Theorem 2: degree <= 3 polynomials are reproduced between knots.
        var row = CatalogueRow(118);
        var grids = Regex.Matches(row, @"[Xx]s: \[([^\]]+)\]").Select(m => ParseNumbers(m.Groups[1].Value)).ToArray();
        grids.Should().HaveCount(2, "row #118 lists a uniform and an irregular grid");
        foreach (var xs in grids)
        {
            foreach (var cubic in new[] { false, true })
            {
                var ys = xs.Select(x => Polynomial(x, cubic)).ToArray();
                var build = () => new CubicSplineInterpolator(xs, ys, SplineBoundary.QSpline);
                build.Should().NotThrow($"row #118 grid [{string.Join(',', xs)}] must satisfy the QSpline knot-count rule");
                var spline = build();
                for (var i = 0; i < xs.Length - 1; i++)
                {
                    foreach (var fraction in new[] { 0.25, 0.5, 0.75 })
                    {
                        var x = xs[i] + fraction * (xs[i + 1] - xs[i]);
                        spline.Interpolate(x).Should().BeApproximately(Polynomial(x, cubic), 1e-10);
                    }
                }
            }
        }

        static double Polynomial(double x, bool useCubic) => useCubic
            ? x * x * x - 2 * x * x + 3 * x - 1
            : 2 * x * x - 3 * x + 1;
    }

    [Fact]
    public void CatalogueRow119_QSplineQuarticGoldens_MatchProductionOnTheListedGrids()
    {
        var row = CatalogueRow(119);
        row.Should().Contain("Absolute tolerance: 1e-10");
        var checkedValues = 0;
        foreach (var kind in new[] { "Uniform", "Irregular" })
        {
            var grid = Regex.Match(row, $@"{kind} xs: \[([^\]]+)\]; query x: ([0-9.]+) and ([0-9.]+)");
            var values = Regex.Match(row, $@"{kind} values respectively: ([^\n]+)");
            grid.Success.Should().BeTrue($"row #119 lists the {kind} grid and its two queries");
            values.Success.Should().BeTrue($"row #119 lists the {kind} goldens");
            var xs = ParseNumbers(grid.Groups[1].Value);
            double[] queries = [double.Parse(grid.Groups[2].Value, CultureInfo.InvariantCulture), double.Parse(grid.Groups[3].Value, CultureInfo.InvariantCulture)];
            var goldens = ParseNumbers(values.Groups[1].Value);
            goldens.Should().HaveCount(2);

            var ys = xs.Select(x => x * x * x * x).ToArray();
            var build = () => new CubicSplineInterpolator(xs, ys, SplineBoundary.QSpline);
            build.Should().NotThrow($"row #119 {kind} grid must satisfy the QSpline knot-count rule");
            var spline = build();
            for (var i = 0; i < queries.Length; i++)
            {
                spline.Interpolate(queries[i]).Should().BeApproximately(goldens[i], 1e-10, $"{kind} golden at x={queries[i]}");
                checkedValues++;
            }
        }

        checkedValues.Should().Be(4);
    }

    [Fact]
    public void CatalogueRow37_DetoningAllNoisePanel_MatchesTheProductionCorrelation()
    {
        var row = CatalogueRow(37);
        var returns = ParseMatrix(Regex.Match(row, @"returns \(observations × variables\): (\[\[.*?\]\])").Groups[1].Value);
        var alpha = double.Parse(Regex.Match(row, @"detoningAlpha: ([0-9.]+)").Groups[1].Value, CultureInfo.InvariantCulture);
        var expectedMatch = Regex.Match(row, @"Output correlation equals (\[\[.*?\]\]) within ([0-9e.\-]+)");
        expectedMatch.Success.Should().BeTrue("row #37 states the output correlation and its tolerance");
        var expected = ParseMatrix(expectedMatch.Groups[1].Value);
        var tolerance = double.Parse(expectedMatch.Groups[2].Value, CultureInfo.InvariantCulture);
        var signalCount = int.Parse(Regex.Match(row, @"SignalCount: (\d+)").Groups[1].Value, CultureInfo.InvariantCulture);

        var sample = new SampleCovarianceEstimator<double>().Estimate(returns);
        DenoisedCovarianceEstimator<double>.DenoiseCorrelation(sample, returns.GetLength(0)).SignalCount.Should().Be(signalCount);
        var actual = ToCorrelation(new DetonedCovarianceEstimator<double>(alpha).Estimate(returns));

        for (var i = 0; i < expected.GetLength(0); i++)
        {
            for (var j = 0; j < expected.GetLength(1); j++)
            {
                actual[i, j].Should().BeApproximately(expected[i, j], tolerance, $"entry [{i},{j}] of the documented output");
            }
        }
    }

    // ------------------------------------------------------------------
    // Catalogue: Sobol rows follow the origin-emitting stream.
    // ------------------------------------------------------------------

    [Fact]
    public void CatalogueRow95_SobolProcedure_ReplaysAgainstTheReferenceVector()
    {
        var row = CatalogueRow(95);
        var inputs = Regex.Match(row, @"dimension: (\d+); skip: (\d+)");
        var indices = Regex.Match(row, @"at array indices (\d+)\.\.(\d+)");
        inputs.Success.Should().BeTrue();
        indices.Success.Should().BeTrue("row #95 states which /sobol array indices are compared");
        var dimension = int.Parse(inputs.Groups[1].Value, CultureInfo.InvariantCulture);
        var skip = long.Parse(inputs.Groups[2].Value, CultureInfo.InvariantCulture);
        var first = int.Parse(indices.Groups[1].Value, CultureInfo.InvariantCulture);
        var last = int.Parse(indices.Groups[2].Value, CultureInfo.InvariantCulture);

        using var vector = JsonDocument.Parse(ReadRepositoryText("tests/Verification/vectors/qmc.json"));
        var sobol = vector.RootElement.GetProperty("sobol").EnumerateArray()
            .Select(point => point.EnumerateArray().Select(c => c.GetDouble()).ToArray())
            .ToArray();

        // Replay the row exactly as written: construct with the documented skip, emit
        // (last - first + 1) points, and compare the j-th emitted point with /sobol[first + j].
        var sequence = new SobolSequence(dimension, skip);
        for (var j = 0; j <= last - first; j++)
        {
            var point = sequence.Next();
            for (var d = 0; d < dimension; d++)
            {
                point[d].Should().BeApproximately(sobol[first + j][d], 1e-10, $"emitted point {j}, coordinate {d}, against /sobol[{first + j}]");
            }
        }

        row.Should().NotContain("omits the origin", "the default stream emits author index 0, the origin");
        row.Should().Contain("origin").And.Contain("skip: 1");
        new SobolSequence(dimension).Next().Should().OnlyContain(c => c == 0.0);
    }

    [Fact]
    public void CatalogueRow96_SobolDimension32Procedure_ProducesTheDocumentedCoordinate()
    {
        var row = CatalogueRow(96);
        var inputs = Regex.Match(row, @"dimension: (\d+); skip: (\d+); call Next twice");
        var output = Regex.Match(row, @"coordinate (\d+) \(zero-based index (\d+)\): exactly ([0-9.]+)");
        inputs.Success.Should().BeTrue("row #96 states dimension, skip and the call count");
        output.Success.Should().BeTrue("row #96 states the asserted coordinate");

        var sequence = new SobolSequence(
            int.Parse(inputs.Groups[1].Value, CultureInfo.InvariantCulture),
            long.Parse(inputs.Groups[2].Value, CultureInfo.InvariantCulture));
        sequence.Next();
        var second = sequence.Next();

        second[int.Parse(output.Groups[2].Value, CultureInfo.InvariantCulture)]
            .Should().Be(double.Parse(output.Groups[3].Value, CultureInfo.InvariantCulture),
                "the documented skip must reproduce the documented second emitted point");
    }

    // ------------------------------------------------------------------
    // Catalogue: Jarre Tables 1-5 assert only what the tests assert.
    // ------------------------------------------------------------------

    [Theory]
    [InlineData(121)]
    [InlineData(122)]
    [InlineData(123)]
    [InlineData(124)]
    [InlineData(125)]
    public void CatalogueJarreRows_AssertedOracleMaxima_MatchProduction(int row)
    {
        var text = CatalogueRow(row);
        var lines = text.Split('\n');
        var label = Array.FindIndex(lines, l => l.StartsWith("Independent oracle maxima actually asserted", StringComparison.Ordinal));
        label.Should().BeGreaterThanOrEqualTo(0);

        var asserted = new List<(string Name, double[] Maxima)>();
        var first = label + 1;
        while (lines[first].Trim().Length == 0)
        {
            first++;
        }

        for (var i = first; i < lines.Length && lines[i].Trim().Length > 0; i++)
        {
            var entry = Regex.Match(lines[i], @"^- (NAT|NAK|Q|RNAK): \[(.+)\]$");
            entry.Success.Should().BeTrue($"unexpected line under the asserted label: {lines[i]}");
            asserted.Add((entry.Groups[1].Value, ParseNumbers(entry.Groups[2].Value)));
        }

        asserted.Should().NotBeEmpty();
        var table = row - 121;
        Func<double, double> function = table switch
        {
            3 => x => 1 / (1 + x * x),
            4 => x => 1 / (1 + Math.Exp(-x)),
            _ => Math.Sin,
        };

        using var vector = JsonDocument.Parse(ReadRepositoryText("tests/Verification/vectors/jarre_splines.json"));
        var cases = vector.RootElement.GetProperty("tables")[table].GetProperty("cases");
        foreach (var (name, maxima) in asserted)
        {
            maxima.Should().HaveCount(5);
            for (var index = 0; index < 5; index++)
            {
                var knots = cases[index].GetProperty("knots").EnumerateArray().Select(e => e.GetDouble()).ToArray();
                var values = cases[index].GetProperty("values").EnumerateArray().Select(e => e.GetDouble()).ToArray();
                var boundary = name switch
                {
                    "NAT" => SplineBoundary<double>.Natural,
                    "NAK" => SplineBoundary<double>.NotAKnot,
                    "Q" => SplineBoundary<double>.QSpline,
                    _ => SplineBoundary<double>.RevisedNotAKnot,
                };
                var actual = MaximumError(knots, function, new CubicSplineInterpolator<double>(knots, values, boundary));
                actual.Should().BeApproximately(maxima[index], 5e-13,
                    $"row #{row} {name} at {knots.Length} knots is listed as asserted against production");
            }
        }
    }

    [Theory]
    [InlineData(121)]
    [InlineData(122)]
    [InlineData(123)]
    [InlineData(124)]
    [InlineData(125)]
    public void CatalogueJarreRows_AttributeQAndRnakToTheOctoberAuthorSource(int row)
    {
        var text = CatalogueRow(row);
        var lines = text.Split('\n');

        text.Should().NotContain("implementing Sections 4–4.1 of arXiv:2507.05083v1 (July 2025), including both safeguards",
            because: "production implements the JCAM Appendix and October 29 2025 author listing; the July preprint is historical");
        text.Should().Contain("historical");
        text.Should().Contain("AuthenticatedOctoberAuthorCoefficients_MatchBetweenKnots");

        for (var i = 0; i < lines.Length; i++)
        {
            if (!Regex.IsMatch(lines[i], @"^- (Q|RNAK): \["))
            {
                continue;
            }

            var labelLine = lines.Take(i).Last(l => l.Trim().Length > 0 && !Regex.IsMatch(l, @"^- (NAT|NAK|Q|RNAK): \["));
            (labelLine.Contains("retained for comparison", StringComparison.Ordinal) || labelLine.Contains("not asserted", StringComparison.Ordinal))
                .Should().BeTrue($"the Q/RNAK list '{lines[i][..Math.Min(40, lines[i].Length)]}' sits under '{labelLine}'");
        }

        ReadRepositoryText("tests/Boutquin.Numerics.Tests.Verification/JarreSplineVerificationTests.cs")
            .Should().Contain("void AuthenticatedOctoberAuthorCoefficients_MatchBetweenKnots(");
    }

    // ------------------------------------------------------------------
    // Helpers.
    // ------------------------------------------------------------------

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Boutquin.Numerics.slnx")))
        {
            directory = directory.Parent;
        }

        return directory?.FullName
            ?? throw new DirectoryNotFoundException("Boutquin.Numerics.slnx was not found above the test output directory.");
    }

    internal static string ReadRepositoryText(string relativePath) =>
        File.ReadAllText(Path.Combine(s_root, relativePath), Encoding.UTF8).Replace("\r\n", "\n");

    internal static string ReadmeCodeBlock(string heading)
    {
        // README repeats some headings (a feature list, then the usage examples), so take the
        // occurrence whose section opens with a csharp fence.
        var readme = ReadRepositoryText("README.md");
        var marker = $"\n### {heading}\n";
        for (var headingIndex = readme.IndexOf(marker, StringComparison.Ordinal);
            headingIndex >= 0;
            headingIndex = readme.IndexOf(marker, headingIndex + marker.Length, StringComparison.Ordinal))
        {
            var sectionEnd = readme.IndexOf("\n### ", headingIndex + marker.Length, StringComparison.Ordinal);
            var open = readme.IndexOf("```csharp\n", headingIndex, StringComparison.Ordinal);
            if (open < 0 || (sectionEnd >= 0 && open > sectionEnd))
            {
                continue;
            }

            var close = readme.IndexOf("\n```", open + "```csharp\n".Length, StringComparison.Ordinal);
            close.Should().BeGreaterThan(open);
            return readme[(open + "```csharp\n".Length)..close];
        }

        throw new InvalidOperationException($"README has no '### {heading}' section with a csharp example.");
    }

    private static double[] ReadmeDoubleArray(string block, string name)
    {
        var match = Regex.Match(block, $@"double\[\] {name} = \[([^\]]*)\];");
        match.Success.Should().BeTrue($"the README snippet declares double[] {name}");
        return ParseNumbers(match.Groups[1].Value);
    }

    private static string CatalogueRow(int number)
    {
        var lines = ReadRepositoryText(CatalogueDocument).Split('\n');
        var start = Array.FindIndex(lines, l => l.StartsWith($"#### Test #{number}:", StringComparison.Ordinal));
        start.Should().BeGreaterThanOrEqualTo(0, $"the catalogue has Test #{number}");
        var end = start + 1;
        while (end < lines.Length
            && !lines[end].StartsWith("#### ", StringComparison.Ordinal)
            && !lines[end].StartsWith("### ", StringComparison.Ordinal)
            && !lines[end].StartsWith("## ", StringComparison.Ordinal)
            && !lines[end].StartsWith("<a id=", StringComparison.Ordinal))
        {
            end++;
        }

        return string.Join('\n', lines[start..end]);
    }

    private static double[] ParseNumbers(string list) =>
        list.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(ParseRational)
            .ToArray();

    private static double ParseRational(string token)
    {
        var text = token.Replace('−', '-').Trim();
        var parts = text.Split('/');
        return parts.Length == 2
            ? double.Parse(parts[0], CultureInfo.InvariantCulture) / double.Parse(parts[1], CultureInfo.InvariantCulture)
            : double.Parse(text, CultureInfo.InvariantCulture);
    }

    private static double[,] ParseMatrix(string text)
    {
        var rows = Regex.Matches(text.Replace('−', '-'), @"\[([^\[\]]+)\]")
            .Select(m => ParseNumbers(m.Groups[1].Value))
            .ToArray();
        var matrix = new double[rows.Length, rows[0].Length];
        for (var i = 0; i < rows.Length; i++)
        {
            for (var j = 0; j < rows[i].Length; j++)
            {
                matrix[i, j] = rows[i][j];
            }
        }

        return matrix;
    }

    private static double[,] ToCorrelation(double[,] covariance)
    {
        var n = covariance.GetLength(0);
        var correlation = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                correlation[i, j] = covariance[i, j] / Math.Sqrt(covariance[i, i] * covariance[j, j]);
            }
        }

        return correlation;
    }

    private static double MaximumError(double[] knots, Func<double, double> function, CubicSplineInterpolator<double> spline)
    {
        // Same search as the verification suite: 16 panels per interval, golden-section refined,
        // so the figure is a supremum estimate rather than a midpoint sample.
        const int Panels = 16;
        const double GoldenFraction = 0.6180339887498948482;
        var maximum = 0.0;
        double Error(double x) => Math.Abs(spline.Interpolate(x) - function(x));
        for (var i = 0; i < knots.Length - 1; i++)
        {
            for (var panel = 0; panel < Panels; panel++)
            {
                var lo = knots[i] + (knots[i + 1] - knots[i]) * panel / Panels;
                var hi = knots[i] + (knots[i + 1] - knots[i]) * (panel + 1) / Panels;
                maximum = Math.Max(maximum, Math.Max(Error(lo), Error(hi)));
                var left = hi - GoldenFraction * (hi - lo);
                var right = lo + GoldenFraction * (hi - lo);
                var leftError = Error(left);
                var rightError = Error(right);
                for (var iteration = 0; iteration < 64; iteration++)
                {
                    if (leftError < rightError)
                    {
                        lo = left;
                        left = right;
                        leftError = rightError;
                        right = lo + GoldenFraction * (hi - lo);
                        rightError = Error(right);
                    }
                    else
                    {
                        hi = right;
                        right = left;
                        rightError = leftError;
                        left = hi - GoldenFraction * (hi - lo);
                        leftError = Error(left);
                    }
                }

                maximum = Math.Max(maximum, Math.Max(leftError, rightError));
            }
        }

        return maximum;
    }
}
