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

using System.Diagnostics;
using System.Text;

using Boutquin.Numerics.Tests.Unit.Solvers.NistStRD;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>Regression coverage for the round-one integration fixes.</summary>
[Collection(NistNonlinearSolveCollection.Name)]
public sealed class ArenaR1C1IntegrationRegressionTests
{
    /// <summary>A stress run cannot claim or delete an unrelated file in the caller's output directory.</summary>
    [StressGatedFact]
    public void StressRun_PreservesPreexistingCallerBundle()
    {
        var dir = CreateScratchDirectory();
        try
        {
            var userBundle = Path.Combine(dir, "bundle-999999.txt");
            File.WriteAllText(userBundle, "sentinel user data");

            _ = NistStRDStressHarness.Run(new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                MinimumExecutions: 1));

            File.Exists(userBundle).Should().BeTrue(
                "a caller-supplied output directory can contain unrelated matching files");
            File.ReadAllText(userBundle).Should().Be("sentinel user data",
                "a caller-supplied output directory can contain unrelated matching files");
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    /// <summary>A rerun must leave the previous completed report and its bundle intact.</summary>
    [StressGatedFact]
    public void StressRun_RerunPreservesCompletedFailureEvidence()
    {
        var dir = CreateScratchDirectory();
        try
        {
            var previous = NistStRDStressHarness.Run(new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                InjectFailureFor: "OLSD:Pontius",
                MinimumExecutions: 1));
            var oldSummary = File.ReadAllBytes(previous.SummaryPath);
            var oldBundlePath = previous.BundlePaths.Single();
            var oldBundle = File.ReadAllBytes(oldBundlePath);

            var current = NistStRDStressHarness.Run(new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                MinimumExecutions: 1));

            current.SummaryPath.Should().NotBe(previous.SummaryPath);
            File.ReadAllBytes(previous.SummaryPath).Should().Equal(oldSummary);
            File.ReadAllBytes(oldBundlePath).Should().Equal(oldBundle,
                "an interrupted later run must not erase the only completed failure evidence");
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    /// <summary>Missing CLI values are configuration errors, not unhandled process exceptions.</summary>
    [Theory]
    [InlineData("--reps")]
    [InlineData("--threads")]
    [InlineData("--min")]
    [InlineData("--out")]
    [InlineData("--inject-failure")]
    public void StressRunner_MissingOptionValueReturnsConfigurationError(string option)
    {
        var result = RunStressRunner(option);

        result.ExitCode.Should().Be(2);
        result.StandardError.Should().Contain("CONFIG-ERROR:");
    }

    /// <summary>Malformed numeric CLI values are configuration errors, not host anomalies.</summary>
    [Theory]
    [InlineData("--reps")]
    [InlineData("--threads")]
    [InlineData("--min")]
    public void StressRunner_MalformedNumericValueReturnsConfigurationError(string option)
    {
        var result = RunStressRunner(option, "not-a-number");

        result.ExitCode.Should().Be(2);
        result.StandardError.Should().Contain("CONFIG-ERROR:");
    }

    /// <summary>A multiplier below the documented minimum must be refused before any run output is created.</summary>
    [Fact]
    public void StressRunner_RejectsThreadMultiplierBelowTwo()
    {
        var dir = CreateScratchDirectory();
        try
        {
            var output = Path.Combine(dir, "invalid-thread-output");
            var result = RunStressRunner("--stress", "--threads", "0", "--reps", "1", "--min", "1", "--out", output);

            result.ExitCode.Should().Be(2, "a multiplier below two is invalid configuration");
            result.StandardError.Should().Contain("CONFIG-ERROR:");
            Directory.Exists(output).Should().BeFalse("invalid settings must not create stress evidence");
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    /// <summary>A file cannot be used as the output directory and must be reported as invalid configuration.</summary>
    [Fact]
    public void StressRunner_ExistingFileOutputReturnsConfigurationError()
    {
        var dir = CreateScratchDirectory();
        try
        {
            var output = Path.Combine(dir, "occupied-output");
            File.WriteAllText(output, "caller data");

            var result = RunStressRunner("--stress", "--reps", "1", "--min", "1", "--out", output);

            result.ExitCode.Should().Be(2, "an existing file is an invalid output directory");
            result.StandardError.Should().Contain("CONFIG-ERROR:");
            File.ReadAllText(output).Should().Be("caller data");
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    private static (int ExitCode, string StandardError) RunStressRunner(params string[] args) =>
        RunStressRunner(LaunchDotnetProcess, args);

    /// <summary>
    /// Test-only seam: identical to <see cref="RunStressRunner(string[])"/> but with the process
    /// launcher injectable, so the bounded-retry behaviour below can be pinned deterministically
    /// against a fake launcher rather than depending on reproducing a real SIGABRT/SIGKILL/SIGSEGV
    /// from host CPU oversubscription (this class's own <c>dotnet</c> child-process spawns are
    /// demonstrably flaky under the exact concurrency shape <c>pr-verify.yml</c> uses — see
    /// F-L4-04). Production always calls the public overload above, which passes the real
    /// <see cref="LaunchDotnetProcess"/> launcher.
    /// </summary>
    private static (int ExitCode, string StandardError) RunStressRunner(
        Func<string[], (int ExitCode, string StandardError)> launch, string[] args)
    {
        var (exitCode, standardError) = launch(args);
        if (IsAbnormalHostExitCode(exitCode))
        {
            // A signal-terminated child process (128+SIGABRT=134, 128+SIGKILL=137,
            // 128+SIGSEGV=139, …) under host CPU oversubscription from a concurrently-running
            // xUnit test collection is a transient host anomaly, not the deterministic
            // configuration-error exit code (2) every caller in this class asserts. Retrying
            // once before reporting absorbs that transient contention instead of failing the
            // config-error assertion for a reason unrelated to the configuration under test.
            (exitCode, standardError) = launch(args);
        }

        return (exitCode, standardError);
    }

    /// <summary>
    /// True for a POSIX signal-terminated exit code (128 + signal number), which every exit
    /// code this class's callers assert (0, 1, 2) is always below.
    /// </summary>
    private static bool IsAbnormalHostExitCode(int exitCode) => exitCode >= 128;

    private static (int ExitCode, string StandardError) LaunchDotnetProcess(string[] args)
    {
        var runner = RepositoryPath("tools", "Boutquin.Numerics.StressRunner", "bin", "Release", "net10.0", "Boutquin.Numerics.StressRunner.dll");
        var dotnetRoot = Environment.GetEnvironmentVariable("DOTNET_ROOT");
        var dotnet = dotnetRoot is not null
            ? Path.Combine(dotnetRoot, "dotnet")
            : OperatingSystem.IsMacOS() ? "/usr/local/share/dotnet/dotnet" : "dotnet";
        var start = new ProcessStartInfo(dotnet)
        {
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
        };
        start.ArgumentList.Add(runner);
        foreach (var arg in args)
        {
            start.ArgumentList.Add(arg);
        }

        using var process = Process.Start(start)!;
        var standardError = process.StandardError.ReadToEnd();
        _ = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        return (process.ExitCode, standardError);
    }

    /// <summary>
    /// A transient host-anomaly exit code (a signal-terminated child process, e.g. 134 for
    /// SIGABRT) from the FIRST launch must be retried once rather than reported as-is; the
    /// caller-visible result must be the retry's outcome. This pins F-L4-04's fix without
    /// depending on reproducing a real SIGABRT under host CPU oversubscription.
    /// </summary>
    [Fact]
    public void RunStressRunner_RetriesOnceOnAbnormalHostExitCode_AndReturnsTheRetryResult()
    {
        var attempt = 0;
        (int ExitCode, string StandardError) FakeLaunch(string[] args)
        {
            attempt++;
            return attempt == 1
                ? (134, "simulated SIGABRT under host CPU oversubscription")
                : (2, "CONFIG-ERROR: simulated genuine configuration failure");
        }

        var result = RunStressRunner(FakeLaunch, ["--stress", "--threads", "0"]);

        attempt.Should().Be(2, "an abnormal (signal-terminated) first exit code must trigger exactly one retry");
        result.ExitCode.Should().Be(2, "the caller-visible result must be the retry's outcome, not the transient anomaly");
        result.StandardError.Should().Contain("CONFIG-ERROR:");
    }

    /// <summary>
    /// A second consecutive abnormal exit code must NOT trigger a second retry — the bound is
    /// exactly one retry, not an unbounded loop that could itself contend for host CPU.
    /// </summary>
    [Fact]
    public void RunStressRunner_DoesNotRetryMoreThanOnce()
    {
        var attempt = 0;
        (int ExitCode, string StandardError) FakeLaunch(string[] args)
        {
            attempt++;
            return (134, "simulated SIGABRT on every attempt");
        }

        var result = RunStressRunner(FakeLaunch, ["--stress", "--threads", "0"]);

        attempt.Should().Be(2, "exactly one retry is attempted, even when it also comes back abnormal");
        result.ExitCode.Should().Be(134, "the second (retry) attempt's result is reported as-is, with no further retries");
    }

    /// <summary>
    /// A mid-sweep hard kill (SIGKILL, matching the finding's own <c>timeout -s KILL</c>
    /// reproduction) must not erase the record that a logic failure was captured. The harness
    /// must emit the <c>NIST-STRESS-CAPTURED</c> stderr line at capture time, inside the
    /// parallel loop, rather than deferring every capture line to after the whole sweep
    /// completes — otherwise a process death partway through a long sweep leaves no trace that
    /// any failure was ever captured, and a subsequent clean retry passes the gate.
    /// </summary>
    [StressGatedFact]
    public void StressRunner_KilledMidSweep_HasAlreadyStreamedCapturedLineForObservedFailures()
    {
        var dir = CreateScratchDirectory();
        try
        {
            var output = Path.Combine(dir, "kill-mid-sweep-output");
            var runner = RepositoryPath("tools", "Boutquin.Numerics.StressRunner", "bin", "Release", "net10.0", "Boutquin.Numerics.StressRunner.dll");
            var dotnetRoot = Environment.GetEnvironmentVariable("DOTNET_ROOT");
            var dotnet = dotnetRoot is not null
                ? Path.Combine(dotnetRoot, "dotnet")
                : OperatingSystem.IsMacOS() ? "/usr/local/share/dotnet/dotnet" : "dotnet";

            var start = new ProcessStartInfo(dotnet)
            {
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                UseShellExecute = false,
            };
            start.ArgumentList.Add(runner);
            start.ArgumentList.Add("--stress");
            start.ArgumentList.Add("--reps");
            start.ArgumentList.Add("20000");
            start.ArgumentList.Add("--min");
            start.ArgumentList.Add("1");
            start.ArgumentList.Add("--inject-failure");
            start.ArgumentList.Add("OLSD:Pontius");
            start.ArgumentList.Add("--out");
            start.ArgumentList.Add(output);

            var sync = new object();
            var capturedLineSeenBeforeKill = false;

            using var process = new Process { StartInfo = start, EnableRaisingEvents = true };
            process.ErrorDataReceived += (_, e) =>
            {
                if (e.Data is not null && e.Data.Contains("NIST-STRESS-CAPTURED", StringComparison.Ordinal))
                {
                    lock (sync)
                    {
                        capturedLineSeenBeforeKill = true;
                    }
                }
            };

            process.Start();
            process.BeginErrorReadLine();
            process.BeginOutputReadLine();

            // 20,000 repetitions of an always-failing target under >= 2x-processor
            // oversubscription (the default thread multiplier) captures at a rate of hundreds
            // per second (per the finding's own probe evidence); 1.5s is ample time for several
            // captures to stream to stderr while leaving the sweep itself still in progress.
            var exitedOnItsOwn = process.WaitForExit(1500);
            if (!exitedOnItsOwn)
            {
                process.Kill(entireProcessTree: true);
                process.WaitForExit(10_000);
            }

            lock (sync)
            {
                capturedLineSeenBeforeKill.Should().BeTrue(
                    "at least one NIST-STRESS-CAPTURED line must reach stderr DURING the sweep " +
                    "(emitted at capture time, inside the parallel loop) so that a mid-sweep " +
                    "process kill cannot erase the record that a logic failure was captured; " +
                    "the sweep was killed before it could reach its own post-loop persistence " +
                    "phase, so any line observed here was necessarily emitted at capture time");
            }
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    /// <summary>The CI stress gate must execute every opt-in integration regression.</summary>
    [Fact]
    public void StressWorkflow_ExecutesGatedIntegrationRegressions()
    {
        var workflow = File.ReadAllText(RepositoryPath(".github", "workflows", "pr-verify.yml"));
        var filter = workflow.Split('\n').Single(line => line.Contains("--filter", StringComparison.Ordinal));
        filter.Should().Contain("FullyQualifiedName~ArenaR1C1IntegrationRegressionTests",
            "the CI stress gate must execute the opt-in summary and bundle regressions");
    }

    /// <summary>The harness must persist directory metadata after run creation and file replacement.</summary>
    [Fact]
    public void StressHarness_SyncsDirectoryAfterEntryChanges()
    {
        var source = File.ReadAllText(RepositoryPath("tests", "Boutquin.Numerics.Tests.Unit", "Solvers", "NistStRD", "NistStRDStressHarness.cs"));
        source.Should().Contain("RandomAccess.FlushToDisk(directoryHandle)",
            "file data flush alone does not persist the renamed directory entry");
        source.Should().Contain("File.Move(tempPath, path, overwrite: true);\n            SyncDirectory(Path.GetDirectoryName(path)!)",
            "the replacement entry must be synced before the writer returns");
        source.Should().Contain("Directory.CreateDirectory(runDirectory);\n        SyncDirectory(config.OutputDirectory)",
            "creating a distinct run directory also changes the parent directory entry set");
    }

    /// <summary>A newly created output path must persist every new directory entry, including nested parents.</summary>
    [Fact]
    public void StressHarness_SyncsParentsOfNewOutputDirectories()
    {
        var source = File.ReadAllText(RepositoryPath("tests", "Boutquin.Numerics.Tests.Unit", "Solvers", "NistStRD", "NistStRDStressHarness.cs"));

        source.Should().Contain("CreateOutputDirectoryDurably(config.OutputDirectory)",
            "the run must use the creation path that persists new output-directory entries");
        source.Should().Contain("SyncDirectory(Path.GetDirectoryName(missingDirectory)!)",
            "each new directory entry belongs to its containing directory, including nested output paths");
    }

    /// <summary>The host-anomaly wrapper also creates fresh output and attempt directory entries.</summary>
    [Fact]
    public void StressWorkflow_SyncsParentsOfNewHostAnomalyDirectories()
    {
        var workflow = File.ReadAllText(RepositoryPath(".github", "workflows", "pr-verify.yml"));

        workflow.Should().Contain("mkdir -p \"$attempt_dir\"\n            sync \"$(dirname \"$OUT_ROOT\")\"\n            sync \"$OUT_ROOT\"",
            "the wrapper must persist both new directory entries before acknowledging host-anomaly evidence");
    }

    /// <summary>The CI wrapper must recognize Unix process-abort exit classes and persist evidence before retrying.</summary>
    [Fact]
    public void StressWorkflow_RecognizesProcessAbortExitClassesBeforeRetry()
    {
        var workflow = File.ReadAllText(RepositoryPath(".github", "workflows", "pr-verify.yml"));

        workflow.Should().Contain("134|137|139)", "SIGABRT, SIGKILL/OOM, and SIGSEGV are process-level host anomalies");
        workflow.Should().Contain("write_host_anomaly", "the wrapper must persist host metadata when the child cannot write a bundle");
        var captureIndex = workflow.IndexOf("write_host_anomaly 1", StringComparison.Ordinal);
        captureIndex.Should().BeGreaterThan(-1);
        captureIndex.Should().BeLessThan(
            workflow.IndexOf("run_stress 2", captureIndex, StringComparison.Ordinal),
            "attempt-1 process-abort evidence must be captured before retrying");
        var renameIndex = workflow.IndexOf("mv \"$temp_path\" \"$final_path\"", StringComparison.Ordinal);
        workflow.IndexOf("sync \"$attempt_dir\"", renameIndex, StringComparison.Ordinal).Should().BeGreaterThan(
            renameIndex, "the wrapper must persist the renamed host-anomaly directory entry");
    }

    /// <summary>The advertised all-vector entry point must invoke every consumed standalone generator.</summary>
    [Theory]
    [InlineData("generate_nelder_mead_vectors", "gen_nelder_mead()")]
    [InlineData("generate_garch_vectors", "gen_garch()")]
    [InlineData("generate_graphical_lasso_vectors", "gen_graphical_lasso()")]
    [InlineData("generate_lower_tail_vectors", "gen_lower_tail()")]
    public void GenerateVectors_InvokesEveryConsumedStandaloneGenerator(string moduleName, string invocation)
    {
        var orchestrator = File.ReadAllText(RepositoryPath("tests", "Verification", "generate_vectors.py"));

        orchestrator.Should().Contain($"from {moduleName} import generate", "the orchestrator must import every consumed generator");
        orchestrator.Should().Contain(invocation, "the orchestrator must execute every imported generator before reporting success");
    }

    /// <summary>Writing a new run must leave open readers on pre-existing caller files.</summary>
    [StressGatedFact]
    public void StressRun_PreservesOpenReadersOnCallerFiles()
    {
        const string sentinel = "complete-previous-evidence";
        var dir = CreateScratchDirectory();
        var summaryPath = Path.Combine(dir, "stress-summary.txt");
        var bundlePath = Path.Combine(dir, "bundle-0000-Pontius.txt");
        try
        {
            File.WriteAllText(summaryPath, sentinel);
            File.WriteAllText(bundlePath, sentinel);
            using var oldSummary = OpenExistingForConcurrentReplacement(summaryPath);
            using var oldBundle = OpenExistingForConcurrentReplacement(bundlePath);

            var report = NistStRDStressHarness.Run(new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                InjectFailureFor: "OLSD:Pontius",
                MinimumExecutions: 1));

            ReadAll(oldSummary).Should().Be(sentinel, "the new run does not replace a caller's summary file");
            ReadAll(oldBundle).Should().Be(sentinel, "the new run does not replace a caller's bundle file");
            File.ReadAllText(summaryPath).Should().Be(sentinel);
            File.ReadAllText(bundlePath).Should().Be(sentinel);
            File.ReadAllText(report.SummaryPath).Should().Contain("LOGIC-REPRODUCED");
            File.ReadAllText(report.BundlePaths.Single()).Should().Contain("NIST-STRESS EVIDENCE BUNDLE");
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    /// <summary>Reusing an output directory must isolate bundles produced by each run.</summary>
    [StressGatedFact]
    public void StressRun_IsolatesBundlesWhenOutputDirectoryIsReused()
    {
        var dir = CreateScratchDirectory();
        try
        {
            var failed = NistStRDStressHarness.Run(new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                InjectFailureFor: "OLSD:Pontius",
                MinimumExecutions: 1));
            failed.BundlePaths.Should().ContainSingle();

            var clean = NistStRDStressHarness.Run(new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                MinimumExecutions: 1));

            clean.BundlePaths.Should().BeEmpty();
            clean.SummaryPath.Should().NotBe(failed.SummaryPath);
            File.Exists(failed.BundlePaths.Single()).Should().BeTrue(
                "the completed failure bundle belongs to the previous run");
            Directory.GetFiles(dir, "bundle-*.txt").Should().BeEmpty(
                "run files belong inside distinct child directories, not the shared parent");
            File.ReadAllText(clean.SummaryPath).Should().Contain("bundle-count         : 0");
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    /// <summary>An invalid rerun must preserve the last completed failure evidence.</summary>
    [StressGatedFact]
    public void StressRun_InvalidConfigurationPreservesPreviousEvidence()
    {
        var dir = CreateScratchDirectory();
        try
        {
            var previous = NistStRDStressHarness.Run(new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                InjectFailureFor: "OLSD:Pontius",
                MinimumExecutions: 1));
            var summary = File.ReadAllBytes(previous.SummaryPath);
            var bundle = File.ReadAllBytes(previous.BundlePaths.Single());

            var invalid = new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 0,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                MinimumExecutions: 1);
            var act = () => NistStRDStressHarness.Run(invalid);

            act.Should().Throw<ArgumentException>("zero repetitions cannot satisfy the execution floor");
            File.ReadAllBytes(previous.SummaryPath).Should().Equal(summary);
            File.ReadAllBytes(previous.BundlePaths.Single()).Should().Equal(bundle);
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    /// <summary>An unreachable execution floor must be rejected before prior evidence is touched.</summary>
    [StressGatedFact]
    public void StressRun_UnreachableFloorPreservesPreviousEvidence()
    {
        var dir = CreateScratchDirectory();
        try
        {
            var previous = NistStRDStressHarness.Run(new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                InjectFailureFor: "OLSD:Pontius",
                MinimumExecutions: 1));
            var summary = File.ReadAllBytes(previous.SummaryPath);
            var bundle = File.ReadAllBytes(previous.BundlePaths.Single());

            var invalid = new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                MinimumExecutions: int.MaxValue);
            var act = () => NistStRDStressHarness.Run(invalid);

            act.Should().Throw<ArgumentException>("the requested floor exceeds all possible executions");
            File.ReadAllBytes(previous.SummaryPath).Should().Equal(summary);
            File.ReadAllBytes(previous.BundlePaths.Single()).Should().Equal(bundle);
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    /// <summary>An overflowing parallelism setting must be rejected before prior evidence is touched.</summary>
    [StressGatedFact]
    public void StressRun_OverflowingThreadMultiplierPreservesPreviousEvidence()
    {
        var dir = CreateScratchDirectory();
        try
        {
            var previous = NistStRDStressHarness.Run(new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                InjectFailureFor: "OLSD:Pontius",
                MinimumExecutions: 1));
            var summary = File.ReadAllBytes(previous.SummaryPath);
            var bundlePath = previous.BundlePaths.Single();
            var bundle = File.ReadAllBytes(bundlePath);

            var invalid = new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: int.MaxValue,
                OutputDirectory: dir,
                MinimumExecutions: 1);
            var act = () => NistStRDStressHarness.Run(invalid);

            act.Should().Throw<ArgumentOutOfRangeException>("derived parallelism cannot fit in a positive Int32");
            File.ReadAllBytes(previous.SummaryPath).Should().Equal(summary);
            File.Exists(bundlePath).Should().BeTrue("an invalid replacement cannot delete completed failure evidence");
            File.ReadAllBytes(bundlePath).Should().Equal(bundle);
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    private static FileStream OpenExistingForConcurrentReplacement(string path) =>
        new(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);

    private static string ReadAll(FileStream stream)
    {
        stream.Position = 0;
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, leaveOpen: true);
        return reader.ReadToEnd();
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

    private static string CreateScratchDirectory()
    {
        var dir = Path.Combine(Path.GetTempPath(), "boutquin-numerics-integration-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void TryDeleteDirectory(string dir)
    {
        try
        {
            if (Directory.Exists(dir))
            {
                Directory.Delete(dir, recursive: true);
            }
        }
        catch (IOException)
        {
            // Best-effort cleanup must not hide the assertion result.
        }
    }
}
