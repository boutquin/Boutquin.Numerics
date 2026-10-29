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

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Boutquin.Numerics.Solvers;
using Microsoft.Win32.SafeHandles;

namespace Boutquin.Numerics.Tests.Unit.Solvers.NistStRD;

/// <summary>
/// Opt-in reproduce-or-bound stress harness over the numerically-hard NIST solves.
/// It executes the nonlinear Levenberg–Marquardt problems (Lanczos1/2/3, Eckerle4,
/// Bennett5, MGH17) and the ill-conditioned ordinary-least-squares regressions
/// (Pontius, Filip, Wampler4, Wampler5 — at both <c>double</c> and <c>decimal</c>)
/// under controlled CPU oversubscription. Each solve is checked against its NIST
/// certified values; any assertion breach or non-assertion host anomaly is written to
/// a gitignored evidence directory. The run finishes with one of three recorded
/// dispositions — <see cref="StressDisposition.LogicReproduced"/>,
/// <see cref="StressDisposition.HostAnomalyObserved"/>, or
/// <see cref="StressDisposition.BoundedNonRepro"/>.
/// </summary>
/// <remarks>
/// <para>
/// This type carries no <c>[Fact]</c>/<c>[Theory]</c> of its own, and every invocation of
/// <see cref="Run(StressConfig)"/> is gated behind the environment variable
/// <c>BOUTQUIN_NUMERICS_STRESS=1</c>, so it never runs in the default <c>dotnet test</c>
/// invocation. Two callers drive it, both under that gate: the companion console runner
/// (<c>tools/Boutquin.Numerics.StressRunner</c>, additionally reachable via the local
/// <c>--stress</c> flag) executes the full stress sweep, and the targeted capture-path
/// regressions in <see cref="NistStRDStressCaptureTests"/> exercise <see cref="Run(StressConfig)"/>
/// directly (with a failure injected) — those facts carry
/// <see cref="StressGatedFactAttribute"/>, so they are skipped unless the gate variable is
/// set and CI runs them explicitly in its stress-gate step. Datasets are reused verbatim
/// from <see cref="NistStRDData"/> and <see cref="NistLinearRegressionData"/> — no dataset
/// is duplicated here.
/// </para>
/// </remarks>
public sealed class NistStRDStressHarness
{
    /// <summary>The environment variable that gates whether the stress pass executes.</summary>
    public const string GateEnvironmentVariable = "BOUTQUIN_NUMERICS_STRESS";

    /// <summary>Configuration for a single stress run.</summary>
    /// <param name="RepetitionsPerTarget">How many times each target solve is executed.</param>
    /// <param name="ThreadMultiplier">Concurrency is <c>ThreadMultiplier × ProcessorCount</c>; must be ≥ 2 for oversubscription.</param>
    /// <param name="OutputDirectory">Gitignored parent directory that receives one distinct directory per run.</param>
    /// <param name="InjectFailureFor">Verification affordance: when set to a prefixed target key, forces an impossibly tight tolerance for exactly that target so the capture path is exercised. The key MUST carry a family prefix — <c>LM:&lt;name&gt;</c> for a nonlinear Levenberg–Marquardt target, <c>OLSD:&lt;name&gt;</c> for an ordinary-least-squares double target, or <c>OLSM:&lt;name&gt;</c> for an OLS decimal target (for example <c>OLSD:Pontius</c>). A non-<see langword="null"/> value that matches no target is rejected with an <see cref="ArgumentException"/> rather than run as a vacuous clean pass. <see langword="null"/> for a normal run.</param>
    /// <param name="MinimumExecutions">The execution floor the run must meet (default 5000).</param>
    public sealed record StressConfig(
        int RepetitionsPerTarget,
        int ThreadMultiplier,
        string OutputDirectory,
        string? InjectFailureFor = null,
        int MinimumExecutions = 5000);

    /// <summary>The recorded exit disposition of a stress run.</summary>
    public enum StressDisposition
    {
        /// <summary>At least one deterministic numeric assertion tripped on reproducible inputs.</summary>
        LogicReproduced,

        /// <summary>Zero logic failures, but at least one non-assertion host anomaly was captured.</summary>
        HostAnomalyObserved,

        /// <summary>≥ MinimumExecutions with zero logic failures and zero host anomalies.</summary>
        BoundedNonRepro,
    }

    /// <summary>The outcome of a stress run.</summary>
    public sealed record StressReport(
        int ExecutionCount,
        int TargetCount,
        int RepetitionsPerTarget,
        int ThreadMultiplier,
        int MaxDegreeOfParallelism,
        int ProcessorCount,
        int LogicFailures,
        int HostAnomalies,
        StressDisposition Disposition,
        IReadOnlyList<string> BundlePaths,
        double ElapsedSeconds,
        bool MeetsExecutionFloor,
        string SummaryPath);

    private sealed record Target(string Name, string DataHashHex, Func<CapturedFailure?> RunOnce);

    // Internal (not private) so the test-only Run(config, writeBundle, writeSummary) seam below
    // — which needs CapturedFailure in its Func<> parameter types — can reference it.
    internal sealed record CapturedFailure(
        string ProblemName,
        string Kind,
        string FailingField,
        string StartVector,
        string DataHashHex,
        string ActualValue,
        string CertifiedValue,
        string Tolerance,
        string LmTermination,
        int LmIterations,
        string HostState);

    /// <summary>Executes the stress pass and writes the summary plus any evidence bundles.</summary>
    /// <param name="config">The run configuration.</param>
    /// <returns>A structured report of the run, including its recorded disposition.</returns>
    public static StressReport Run(StressConfig config) => Run(config, WriteBundle, WriteSummary);

    /// <summary>
    /// Test-only seam: identical to <see cref="Run(StressConfig)"/> but with the evidence
    /// writers injectable, so a captured-failure-then-persistence-throws window (a real I/O
    /// exception arriving after a logic failure is already in the in-memory bundle list — e.g.
    /// a disk-full or permission fault) can be reproduced deterministically without depending on
    /// filesystem-specific fault injection. Production always calls the public overload above,
    /// which passes the real <see cref="WriteBundle"/>/<see cref="WriteSummary"/> writers, so
    /// this seam changes no production behavior on its own.
    /// </summary>
    internal static StressReport Run(
        StressConfig config,
        Func<string, int, CapturedFailure, string> writeBundle,
        Func<StressConfig, string, int, int, int, int, int, int, StressDisposition, bool, double, IReadOnlyList<string>, string> writeSummary)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(writeBundle);
        ArgumentNullException.ThrowIfNull(writeSummary);

        var targets = BuildTargets(config.InjectFailureFor, out var injectMatched);
        if (config.InjectFailureFor is not null && !injectMatched)
        {
            throw new ArgumentException(
                $"--inject-failure target '{config.InjectFailureFor}' matched no stress target. " +
                "The key must carry a family prefix — LM:<name> (nonlinear), OLSD:<name> (OLS double), " +
                "or OLSM:<name> (OLS decimal), for example OLSD:Pontius.",
                nameof(config));
        }

        if (config.RepetitionsPerTarget <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(config), "Repetitions per target must be positive.");
        }

        if (config.ThreadMultiplier < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(config), "The thread multiplier must be at least two.");
        }

        var possibleExecutions = (long)targets.Count * config.RepetitionsPerTarget;
        if (config.MinimumExecutions <= 0 || config.MinimumExecutions > possibleExecutions || possibleExecutions > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(config),
                "The execution floor must be positive and reachable within the configured repetitions.");
        }

        var processorCount = Environment.ProcessorCount;
        var requestedDop = (long)config.ThreadMultiplier * processorCount;
        if (requestedDop > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(config),
                "The thread multiplier produces an unsupported degree of parallelism.");
        }

        var maxDop = (int)requestedDop;
        if (OutputPathContainsFile(config.OutputDirectory))
        {
            throw new ArgumentException("The output directory path contains an existing file.", nameof(config));
        }

        try
        {
            CreateOutputDirectoryDurably(config.OutputDirectory);
        }
        catch (IOException ex) when (OutputPathContainsFile(config.OutputDirectory))
        {
            // Another process may replace a path component after the preflight check.
            throw new ArgumentException("The output directory path contains an existing file.", nameof(config), ex);
        }
        var runDirectory = Path.Combine(config.OutputDirectory, $"run-{Guid.NewGuid():N}");
        Directory.CreateDirectory(runDirectory);
        SyncDirectory(config.OutputDirectory);

        // Build the full work list: every (target, repetition) pair.
        var work = new List<Target>(targets.Count * config.RepetitionsPerTarget);
        for (var r = 0; r < config.RepetitionsPerTarget; r++)
        {
            work.AddRange(targets);
        }

        var logicFailures = 0;
        var hostAnomalies = 0;
        var executions = 0;
        var bundleIndex = -1;
        var bundles = new ConcurrentBag<(int Index, CapturedFailure Failure)>();

        var stopwatch = Stopwatch.StartNew();
        Parallel.ForEach(
            work,
            new ParallelOptions { MaxDegreeOfParallelism = maxDop },
            target =>
            {
                Interlocked.Increment(ref executions);
                try
                {
                    var failure = target.RunOnce();
                    if (failure is not null)
                    {
                        Interlocked.Increment(ref logicFailures);
                        var index = Interlocked.Increment(ref bundleIndex);
                        bundles.Add((index, failure));

                        // Emit the capture line AT CAPTURE TIME, inside this parallel work
                        // item — not deferred to the post-loop foreach below. A mid-sweep
                        // process death (SIGKILL/SIGSEGV/OOM, the exact host classes
                        // pr-verify.yml's 134/137/139 arm names) must not be able to erase the
                        // record that this failure was captured; Console.Error.WriteLine is
                        // safe to call concurrently from multiple work items.
                        Console.Error.WriteLine(
                            $"NIST-STRESS-CAPTURED index={index:D4} problem={failure.ProblemName} kind={failure.Kind}");
                    }
                }
                catch (Exception ex)
                {
                    // A non-assertion throw from the solve is a host/process anomaly, not a
                    // deterministic logic failure — record it as such so the disposition
                    // distinguishes the two classes.
                    Interlocked.Increment(ref hostAnomalies);
                    var index = Interlocked.Increment(ref bundleIndex);
                    var failure = new CapturedFailure(
                        ProblemName: target.Name,
                        Kind: "HOST-ANOMALY",
                        FailingField: "exception",
                        StartVector: "(see dataset accessor)",
                        DataHashHex: target.DataHashHex,
                        ActualValue: "n/a",
                        CertifiedValue: "n/a",
                        Tolerance: "n/a",
                        LmTermination: "n/a",
                        LmIterations: -1,
                        HostState: $"{ex.GetType().FullName}: {ex.Message}");
                    bundles.Add((index, failure));

                    // Same at-capture-time guarantee as the logic-failure branch above.
                    Console.Error.WriteLine(
                        $"NIST-STRESS-CAPTURED index={index:D4} problem={failure.ProblemName} kind={failure.Kind}");
                }
            });
        stopwatch.Stop();

        var disposition = logicFailures > 0
            ? StressDisposition.LogicReproduced
            : hostAnomalies > 0
                ? StressDisposition.HostAnomalyObserved
                : StressDisposition.BoundedNonRepro;

        // Emit the disposition and every captured failure to stderr BEFORE attempting to
        // persist them. The disposition is already fully determined at this point (the
        // Parallel.ForEach above has completed) — if the evidence-write step below throws (a
        // disk-full or permission fault reaching the writer), the exception must never escape
        // this method and be classified by the caller as a retriable host-process anomaly,
        // which would silently discard the only record of a deterministic logic failure and let
        // CI retry a failure that is guaranteed to recur. This line guarantees the disposition
        // reaches the process's own log stream even if every subsequent write fails.
        Console.Error.WriteLine(
            $"NIST-STRESS-DISPOSITION prelim={ToDispositionLabel(disposition)} " +
            $"logic-failures={logicFailures} host-anomalies={hostAnomalies} captured={bundles.Count}");

        var bundlePaths = new List<string>();
        foreach (var (index, failure) in bundles.OrderBy(b => b.Index))
        {
            // The NIST-STRESS-CAPTURED line for this index was already emitted at capture time,
            // inside the Parallel.ForEach body above — not here, so a mid-sweep process death
            // cannot erase it (see the fix for the finding this comment accompanies).
            try
            {
                bundlePaths.Add(writeBundle(runDirectory, index, failure));
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // A persistence fault must never turn an already-captured logic failure into an
                // unhandled process abort. The at-capture-time line already recorded the
                // failure; this line records that its on-disk bundle could not be written, and
                // the run proceeds — the disposition computed above is unaffected because it was
                // derived from logicFailures/hostAnomalies, not from bundlePaths.
                Console.Error.WriteLine(
                    $"NIST-STRESS-BUNDLE-PERSIST-FAILED index={index:D4} problem={failure.ProblemName}: " +
                    $"{ex.GetType().FullName}: {ex.Message}");
            }
        }

        var meetsFloor = executions >= config.MinimumExecutions;
        string summaryPath;
        try
        {
            summaryPath = writeSummary(
                config, runDirectory, targets.Count, processorCount, maxDop, executions,
                logicFailures, hostAnomalies, disposition, meetsFloor,
                stopwatch.Elapsed.TotalSeconds, bundlePaths);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // Preserve numeric-failure precedence, but never acknowledge a clean pass
            // when its required summary could not be persisted.
            Console.Error.WriteLine(
                $"NIST-STRESS-SUMMARY-PERSIST-FAILED: {ex.GetType().FullName}: {ex.Message}");
            hostAnomalies++;
            if (logicFailures == 0)
            {
                disposition = StressDisposition.HostAnomalyObserved;
            }
            summaryPath = Path.Combine(runDirectory, "stress-summary.txt");
        }

        return new StressReport(
            ExecutionCount: executions,
            TargetCount: targets.Count,
            RepetitionsPerTarget: config.RepetitionsPerTarget,
            ThreadMultiplier: config.ThreadMultiplier,
            MaxDegreeOfParallelism: maxDop,
            ProcessorCount: processorCount,
            LogicFailures: logicFailures,
            HostAnomalies: hostAnomalies,
            Disposition: disposition,
            BundlePaths: bundlePaths,
            ElapsedSeconds: stopwatch.Elapsed.TotalSeconds,
            MeetsExecutionFloor: meetsFloor,
            SummaryPath: summaryPath);
    }

    private static bool OutputPathContainsFile(string path)
    {
        for (var component = Path.GetFullPath(path); component is not null; component = Path.GetDirectoryName(component))
        {
            if (File.Exists(component))
            {
                return true;
            }
        }

        return false;
    }

    // ── Target catalogue (datasets reused from the existing providers) ──────

    private static IReadOnlyList<Target> BuildTargets(string? injectFor, out bool injectMatched)
    {
        var targets = new List<Target>();
        var matched = false;

        // Levenberg–Marquardt nonlinear targets. Tolerances mirror NistStRDTests exactly.
        matched |= AddLm(targets, injectFor, "Lanczos1", NistStRDData.Lanczos1(), SixExponential, 500, 1e-3, 0.01);
        matched |= AddLm(targets, injectFor, "Lanczos2", NistStRDData.Lanczos2(), SixExponential, 500, 1e-3, 0.01);
        matched |= AddLm(targets, injectFor, "Lanczos3", NistStRDData.Lanczos3(), SixExponential, 500, 1e-2, 0.05);
        matched |= AddLm(targets, injectFor, "Eckerle4", NistStRDData.Eckerle4(), Eckerle4Model, 500, 1e-3, 0.05);
        matched |= AddLm(targets, injectFor, "Bennett5", NistStRDData.Bennett5(), Bennett5Model, 1000, 1e-2, 0.05);
        matched |= AddLm(targets, injectFor, "MGH17", NistStRDData.MGH17(), Mgh17Model, 2000, 1e-2, 0.05);

        // Ill-conditioned OLS targets at double precision. Tolerances mirror NistLinearRegressionTests.
        matched |= AddOlsDouble(targets, injectFor, "Pontius", NistLinearRegressionData.Pontius(), NistLinearRegressionData.DoubleCoefficientToleranceFor("Pontius"));
        matched |= AddOlsDouble(targets, injectFor, "Filip", NistLinearRegressionData.Filip(), NistLinearRegressionData.DoubleCoefficientToleranceFor("Filip"));
        matched |= AddOlsDouble(targets, injectFor, "Wampler4", NistLinearRegressionData.Wampler4(), NistLinearRegressionData.DoubleCoefficientToleranceFor("Wampler4"));
        matched |= AddOlsDouble(targets, injectFor, "Wampler5", NistLinearRegressionData.Wampler5(), NistLinearRegressionData.DoubleCoefficientToleranceFor("Wampler5"));

        // Same four at decimal precision. Tolerances mirror NistLinearRegressionDecimalTests.
        matched |= AddOlsDecimal(targets, injectFor, "Pontius", NistLinearRegressionData.Pontius(), NistLinearRegressionData.DecimalCoefficientToleranceFor("Pontius"));
        matched |= AddOlsDecimal(targets, injectFor, "Filip", NistLinearRegressionData.Filip(), NistLinearRegressionData.DecimalCoefficientToleranceFor("Filip"));
        matched |= AddOlsDecimal(targets, injectFor, "Wampler4", NistLinearRegressionData.Wampler4(), NistLinearRegressionData.DecimalCoefficientToleranceFor("Wampler4"));
        matched |= AddOlsDecimal(targets, injectFor, "Wampler5", NistLinearRegressionData.Wampler5(), NistLinearRegressionData.DecimalCoefficientToleranceFor("Wampler5"));

        injectMatched = matched;
        return targets;
    }

    private static bool AddLm(
        List<Target> targets,
        string? injectFor,
        string name,
        NistStRDData.Problem data,
        Func<NistStRDData.Problem, double[], double[]> model,
        int maxIterations,
        double parameterTolerance,
        double sseTolerance)
    {
        var hash = HashDoubles(data.X, data.Y);
        // Injection = an impossibly tight parameter tolerance for exactly this target,
        // guaranteeing a captured logic failure on an otherwise-correct solve.
        var isInjected = string.Equals(injectFor, "LM:" + name, StringComparison.Ordinal);
        var effectiveParamTol = isInjected ? 1e-30 : parameterTolerance;

        targets.Add(new Target(name + " (LM)", hash, () =>
            CheckLm(name, data, model, maxIterations, effectiveParamTol, sseTolerance, hash)));
        return isInjected;
    }

    private static bool AddOlsDouble(
        List<Target> targets,
        string? injectFor,
        string name,
        NistLinearRegressionData.Problem problem,
        double coefficientTolerance)
    {
        var hash = HashMatrixAndVector(problem.X, problem.Y);
        var isInjected = string.Equals(injectFor, "OLSD:" + name, StringComparison.Ordinal);
        var effectiveTol = isInjected ? 1e-30 : coefficientTolerance;

        targets.Add(new Target(name + " (OLS double)", hash, () =>
            CheckOlsDouble(name, problem, effectiveTol, hash)));
        return isInjected;
    }

    private static bool AddOlsDecimal(
        List<Target> targets,
        string? injectFor,
        string name,
        NistLinearRegressionData.Problem problem,
        decimal coefficientTolerance)
    {
        var hash = HashMatrixAndVector(problem.X, problem.Y);
        var isInjected = string.Equals(injectFor, "OLSM:" + name, StringComparison.Ordinal);
        var effectiveTol = isInjected ? 1e-30m : coefficientTolerance;

        targets.Add(new Target(name + " (OLS decimal)", hash, () =>
            CheckOlsDecimal(name, problem, effectiveTol, hash)));
        return isInjected;
    }

    // ── Numeric checks — mirror the certified-value assertions, capturing not throwing ──

    private static CapturedFailure? CheckLm(
        string name,
        NistStRDData.Problem data,
        Func<NistStRDData.Problem, double[], double[]> model,
        int maxIterations,
        double parameterTolerance,
        double sseTolerance,
        string dataHash)
    {
        var solver = new LevenbergMarquardtSolver(maxIterations: maxIterations);
        var result = solver.Solve(theta => model(data, theta), data.Start2);

        var start = string.Join(", ", data.Start2.Select(v => v.ToString("G17", CultureInfo.InvariantCulture)));
        CapturedFailure Fail(string field, string actual, string certified, string tol) => new(
            ProblemName: name,
            Kind: "LOGIC",
            FailingField: field,
            StartVector: start,
            DataHashHex: dataHash,
            ActualValue: actual,
            CertifiedValue: certified,
            Tolerance: tol,
            LmTermination: result.TerminationReason.ToString(),
            LmIterations: result.Iterations,
            HostState: "deterministic-assertion");

        if (!result.Converged)
        {
            return Fail("convergence", result.TerminationReason.ToString(), "Converged=true", "n/a");
        }

        if (result.TerminationReason is not (LmTerminationReason.FunctionToleranceReached
            or LmTerminationReason.ParameterToleranceReached
            or LmTerminationReason.GradientToleranceReached))
        {
            return Fail("termination-reason", result.TerminationReason.ToString(), "a tolerance reason", "n/a");
        }

        var sse = 2.0 * result.FinalCost;
        if (data.CertifiedSse < 1e-18)
        {
            if (!(sse < 1e-18))
            {
                return Fail("SSE (absolute floor)", Fmt(sse), Fmt(data.CertifiedSse), "1E-18");
            }
        }
        else
        {
            var rel = Math.Abs(sse - data.CertifiedSse) / data.CertifiedSse;
            if (!(rel < sseTolerance))
            {
                return Fail("SSE (relative)", $"rel={Fmt(rel)} (sse={Fmt(sse)})", Fmt(data.CertifiedSse), Fmt(sseTolerance));
            }
        }

        for (var i = 0; i < data.Certified.Length; i++)
        {
            var certified = data.Certified[i];
            var estimate = result.Parameters[i];
            var rel = certified == 0.0 ? Math.Abs(estimate) : Math.Abs((estimate - certified) / certified);
            if (!(rel < parameterTolerance))
            {
                return Fail($"coefficient b{i + 1}", $"{Fmt(estimate)} (rel={Fmt(rel)})", Fmt(certified), Fmt(parameterTolerance));
            }
        }

        return null;
    }

    private static CapturedFailure? CheckOlsDouble(
        string name,
        NistLinearRegressionData.Problem problem,
        double coefficientTolerance,
        string dataHash)
    {
        var result = OrdinaryLeastSquares.Fit(problem.X, problem.Y, problem.IncludeIntercept);
        for (var i = 0; i < problem.Coefficients.Length; i++)
        {
            var certified = problem.Coefficients[i];
            var actual = result.Coefficients[i];
            var rel = certified == 0.0 ? Math.Abs(actual) : Math.Abs((actual - certified) / certified);
            if (!(rel < coefficientTolerance))
            {
                return new CapturedFailure(
                    ProblemName: name,
                    Kind: "LOGIC",
                    FailingField: $"coefficient b{i}",
                    StartVector: $"includeIntercept={problem.IncludeIntercept}",
                    DataHashHex: dataHash,
                    ActualValue: $"{Fmt(actual)} (rel={Fmt(rel)})",
                    CertifiedValue: Fmt(certified),
                    Tolerance: Fmt(coefficientTolerance),
                    LmTermination: "n/a (OLS)",
                    LmIterations: -1,
                    HostState: "deterministic-assertion");
            }
        }

        return null;
    }

    private static CapturedFailure? CheckOlsDecimal(
        string name,
        NistLinearRegressionData.Problem problem,
        decimal coefficientTolerance,
        string dataHash)
    {
        var x = ToDecimalMatrix(problem.X);
        var y = ToDecimalArray(problem.Y);
        var result = OrdinaryLeastSquares<decimal>.Fit(x, y, problem.IncludeIntercept);
        for (var i = 0; i < problem.Coefficients.Length; i++)
        {
            var certified = (decimal)problem.Coefficients[i];
            var actual = result.Coefficients[i];
            var rel = certified == 0m ? Math.Abs(actual) : Math.Abs((actual - certified) / certified);
            if (!(rel < coefficientTolerance))
            {
                return new CapturedFailure(
                    ProblemName: name,
                    Kind: "LOGIC",
                    FailingField: $"coefficient b{i}",
                    StartVector: $"includeIntercept={problem.IncludeIntercept}",
                    DataHashHex: dataHash,
                    ActualValue: $"{actual} (rel={rel:E3})",
                    CertifiedValue: certified.ToString(CultureInfo.InvariantCulture),
                    Tolerance: coefficientTolerance.ToString(CultureInfo.InvariantCulture),
                    LmTermination: "n/a (OLS decimal)",
                    LmIterations: -1,
                    HostState: "deterministic-assertion");
            }
        }

        return null;
    }

    // ── Residual models (the math, not the data) ────────────────────────────

    private static double[] SixExponential(NistStRDData.Problem data, double[] t)
    {
        var r = new double[data.X.Length];
        for (var i = 0; i < data.X.Length; i++)
        {
            var x = data.X[i];
            var model = (t[0] * Math.Exp(-t[1] * x))
                      + (t[2] * Math.Exp(-t[3] * x))
                      + (t[4] * Math.Exp(-t[5] * x));
            r[i] = data.Y[i] - model;
        }

        return r;
    }

    private static double[] Eckerle4Model(NistStRDData.Problem data, double[] t)
    {
        var r = new double[data.X.Length];
        for (var i = 0; i < data.X.Length; i++)
        {
            var centered = (data.X[i] - t[2]) / t[1];
            r[i] = data.Y[i] - ((t[0] / t[1]) * Math.Exp(-0.5 * centered * centered));
        }

        return r;
    }

    private static double[] Bennett5Model(NistStRDData.Problem data, double[] t)
    {
        var r = new double[data.X.Length];
        for (var i = 0; i < data.X.Length; i++)
        {
            r[i] = data.Y[i] - (t[0] * Math.Pow(t[1] + data.X[i], -1.0 / t[2]));
        }

        return r;
    }

    private static double[] Mgh17Model(NistStRDData.Problem data, double[] t)
    {
        var r = new double[data.X.Length];
        for (var i = 0; i < data.X.Length; i++)
        {
            var x = data.X[i];
            r[i] = data.Y[i] - (t[0] + (t[1] * Math.Exp(-x * t[3])) + (t[2] * Math.Exp(-x * t[4])));
        }

        return r;
    }

    // ── Evidence bundle + summary writers ───────────────────────────────────

    private static string WriteBundle(string dir, int index, CapturedFailure f)
    {
        var path = Path.Combine(dir, $"bundle-{index:D4}-{Sanitize(f.ProblemName)}.txt");
        var sb = new StringBuilder();
        sb.AppendLine("NIST-STRESS EVIDENCE BUNDLE");
        sb.AppendLine("===========================");
        Line(sb, $"timestamp           : {DateTimeOffset.UtcNow:O}");
        Line(sb, $"problem             : {f.ProblemName}");
        Line(sb, $"kind                : {f.Kind}");
        Line(sb, $"failing-field       : {f.FailingField}");
        Line(sb, $"start-vector/inputs : {f.StartVector}");
        Line(sb, $"data-hash (SHA-256) : {f.DataHashHex}");
        Line(sb, $"actual              : {f.ActualValue}");
        Line(sb, $"certified           : {f.CertifiedValue}");
        Line(sb, $"failing-tolerance   : {f.Tolerance}");
        Line(sb, $"lm-termination      : {f.LmTermination}");
        Line(sb, $"lm-iterations       : {f.LmIterations}");
        Line(sb, $"managed-thread-id   : {Environment.CurrentManagedThreadId}");
        Line(sb, $"processor-count     : {Environment.ProcessorCount}");
        Line(sb, $"host/exit-state     : {f.HostState}");
        Line(sb, $"process-id          : {Environment.ProcessId}");
        WriteDurablyAndReplace(path, sb.ToString());
        return path;
    }

    private static string WriteSummary(
        StressConfig config,
        string runDirectory,
        int targetCount,
        int processorCount,
        int maxDop,
        int executions,
        int logicFailures,
        int hostAnomalies,
        StressDisposition disposition,
        bool meetsFloor,
        double elapsedSeconds,
        IReadOnlyList<string> bundlePaths)
    {
        var path = Path.Combine(runDirectory, "stress-summary.txt");
        var sb = new StringBuilder();
        sb.AppendLine("NIST/LM SOLVER STRESS RUN — SUMMARY");
        sb.AppendLine("===================================");
        Line(sb, $"timestamp            : {DateTimeOffset.UtcNow:O}");
        Line(sb, $"target-count         : {targetCount}");
        Line(sb, $"repetitions-per-tgt  : {config.RepetitionsPerTarget}");
        Line(sb, $"execution-count      : {executions}");
        Line(sb, $"minimum-executions   : {config.MinimumExecutions}");
        Line(sb, $"meets-execution-floor: {meetsFloor}");
        Line(sb, $"processor-count      : {processorCount}");
        Line(sb, $"thread-multiplier    : {config.ThreadMultiplier}");
        Line(sb, $"max-degree-parallel  : {maxDop}");
        Line(sb, $"logic-failures       : {logicFailures}");
        Line(sb, $"host-anomalies       : {hostAnomalies}");
        Line(sb, $"disposition          : {ToDispositionLabel(disposition)}");
        Line(sb, $"elapsed-seconds      : {elapsedSeconds:F2}");
        Line(sb, $"gate-env-variable    : {GateEnvironmentVariable}");
        if (config.InjectFailureFor is not null)
        {
            Line(sb, $"inject-failure-for   : {config.InjectFailureFor}");
        }

        Line(sb, $"bundle-count         : {bundlePaths.Count}");
        foreach (var b in bundlePaths)
        {
            Line(sb, $"  bundle             : {b}");
        }

        WriteDurablyAndReplace(path, sb.ToString());
        return path;
    }

    private static void WriteDurablyAndReplace(string path, string contents)
    {
        var tempPath = Path.Combine(
            Path.GetDirectoryName(path)!,
            $".{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");
        try
        {
            using (var stream = new FileStream(
                       tempPath,
                       FileMode.CreateNew,
                       FileAccess.Write,
                       FileShare.None,
                       bufferSize: 4096,
                       FileOptions.WriteThrough))
            {
                using (var writer = new StreamWriter(
                           stream,
                           new UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
                           bufferSize: 4096,
                           leaveOpen: true))
                {
                    writer.Write(contents);
                    writer.Flush();
                }

                stream.Flush(flushToDisk: true);
            }

            File.Move(tempPath, path, overwrite: true);
            SyncDirectory(Path.GetDirectoryName(path)!);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    private static void SyncDirectory(string directory)
    {
        if (OperatingSystem.IsWindows())
        {
            return;
        }

        var descriptor = UnixDirectoryApi.s_open(directory, 0);
        if (descriptor < 0)
        {
            throw new IOException($"Could not open directory '{directory}' for durability sync (errno {Marshal.GetLastPInvokeError()}).");
        }

        using var directoryHandle = new SafeFileHandle((IntPtr)descriptor, ownsHandle: true);
        RandomAccess.FlushToDisk(directoryHandle);
    }

    private static void CreateOutputDirectoryDurably(string outputDirectory)
    {
        var missingDirectories = new Stack<string>();
        var currentDirectory = Path.GetFullPath(outputDirectory);
        while (!Directory.Exists(currentDirectory))
        {
            missingDirectories.Push(currentDirectory);
            currentDirectory = Path.GetDirectoryName(currentDirectory)!;
        }

        Directory.CreateDirectory(outputDirectory);
        foreach (var missingDirectory in missingDirectories)
        {
            SyncDirectory(Path.GetDirectoryName(missingDirectory)!);
        }
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl, CharSet = CharSet.Ansi, SetLastError = true)]
    private delegate int OpenDirectoryNative(string path, int flags);

    private static class UnixDirectoryApi
    {
        private static readonly IntPtr s_libc = NativeLibrary.Load(
            OperatingSystem.IsMacOS() ? "libSystem.B.dylib" : "libc.so.6");

        internal static readonly OpenDirectoryNative s_open =
            Marshal.GetDelegateForFunctionPointer<OpenDirectoryNative>(NativeLibrary.GetExport(s_libc, "open"));
    }

    /// <summary>Maps a disposition to the spec's recorded exit label.</summary>
    /// <param name="disposition">The run disposition.</param>
    /// <returns>One of <c>LOGIC-REPRODUCED</c>, <c>HOST-ANOMALY-OBSERVED</c>, or <c>BOUNDED-NON-REPRO</c>.</returns>
    public static string ToDispositionLabel(StressDisposition disposition) => disposition switch
    {
        StressDisposition.LogicReproduced => "LOGIC-REPRODUCED",
        StressDisposition.HostAnomalyObserved => "HOST-ANOMALY-OBSERVED",
        StressDisposition.BoundedNonRepro => "BOUNDED-NON-REPRO",
        _ => disposition.ToString(),
    };

    // ── Helpers ─────────────────────────────────────────────────────────────

    private static void Line(StringBuilder sb, FormattableString line) =>
        sb.AppendLine(line.ToString(CultureInfo.InvariantCulture));

    private static string Fmt(double v) => v.ToString("G17", CultureInfo.InvariantCulture);

    private static string Sanitize(string name)
    {
        var sb = new StringBuilder(name.Length);
        foreach (var c in name)
        {
            sb.Append(char.IsLetterOrDigit(c) ? c : '-');
        }

        return sb.ToString();
    }

    private static string HashDoubles(params double[][] arrays)
    {
        var bytes = new List<byte>();
        foreach (var array in arrays)
        {
            foreach (var value in array)
            {
                bytes.AddRange(BitConverter.GetBytes(value));
            }
        }

        return Convert.ToHexString(SHA256.HashData(bytes.ToArray()));
    }

    private static string HashMatrixAndVector(double[,] matrix, double[] vector)
    {
        var bytes = new List<byte>();
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < cols; j++)
            {
                bytes.AddRange(BitConverter.GetBytes(matrix[i, j]));
            }
        }

        foreach (var value in vector)
        {
            bytes.AddRange(BitConverter.GetBytes(value));
        }

        return Convert.ToHexString(SHA256.HashData(bytes.ToArray()));
    }

    private static decimal[,] ToDecimalMatrix(double[,] source)
    {
        var rows = source.GetLength(0);
        var cols = source.GetLength(1);
        var result = new decimal[rows, cols];
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < cols; j++)
            {
                result[i, j] = (decimal)source[i, j];
            }
        }

        return result;
    }

    private static decimal[] ToDecimalArray(double[] source)
    {
        var result = new decimal[source.Length];
        for (var i = 0; i < source.Length; i++)
        {
            result[i] = (decimal)source[i];
        }

        return result;
    }
}
