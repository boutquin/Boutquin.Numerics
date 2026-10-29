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
using System.Security.Cryptography;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers.NistStRD;

/// <summary>
/// Regression coverage for the <see cref="NistStRDStressHarness"/> failure-capture path.
/// The stress harness is otherwise driven only by the opt-in console runner; these tests
/// invoke <see cref="NistStRDStressHarness.Run(NistStRDStressHarness.StressConfig)"/> directly (with a failure injected) so the
/// evidence-capture behaviour is exercised as an ordinary, deterministic-enough unit test —
/// pinning the two capture-path defects the arena confirmed:
/// <list type="bullet">
/// <item>the bundle-index race (duplicate indices silently overwrote evidence on disk); and</item>
/// <item>the silent no-op of an <c>--inject-failure</c> key that matches no target.</item>
/// </list>
/// Because each fact drives <see cref="NistStRDStressHarness.Run(NistStRDStressHarness.StressConfig)"/> under CPU
/// oversubscription — the same heavy nonlinear/ill-conditioned workload
/// <see cref="NistNonlinearSolveCollection"/> exists to isolate — the class joins that
/// bounded, non-parallel collection and every fact is gated by
/// <see cref="StressGatedFactAttribute"/> so the harness runs only under
/// <c>BOUTQUIN_NUMERICS_STRESS=1</c> (opt-in), never in the default <c>dotnet test</c>
/// suite. CI runs them explicitly in the stress-gate step where that variable is set.
/// </summary>
[Collection(NistNonlinearSolveCollection.Name)]
public sealed class NistStRDStressCaptureTests
{
    /// <summary>
    /// Under CPU oversubscription every injected logic failure must produce its own distinct,
    /// on-disk evidence bundle. When the per-bundle index is generated non-atomically
    /// (<c>bundles.Add((bundles.Count, …))</c>), two concurrent captures can read the same
    /// <c>Count</c>, collide on <c>bundle-{index:D4}-…</c>, and one bundle silently overwrites
    /// the other — so the number of files on disk drops below the reported logic-failure count
    /// while the summary still claims a full bundle set. High reps × high oversubscription trips
    /// the race reliably; the assertions check distinctness and disk-count, never exact indices.
    /// </summary>
    [StressGatedFact]
    public void Run_WithInjectedFailureUnderOversubscription_WritesOneDistinctBundlePerFailure()
    {
        const int reps = 300;
        var dir = CreateScratchDirectory();
        try
        {
            var config = new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: reps,
                ThreadMultiplier: 4,
                OutputDirectory: dir,
                InjectFailureFor: "OLSD:Pontius",
                MinimumExecutions: 1);

            var report = NistStRDStressHarness.Run(config);

            // The injected target fails on every repetition — one logic failure per rep.
            report.LogicFailures.Should().Be(
                reps,
                "the injected OLSD:Pontius target must trip its assertion on every repetition");

            // Every failure must yield its own bundle path, and they must all be distinct —
            // a duplicated path is the direct signature of the non-atomic index race.
            report.BundlePaths.Should().HaveCount(report.LogicFailures);
            report.BundlePaths.Should().OnlyHaveUniqueItems(
                "each captured failure must own a unique bundle filename; duplicate indices " +
                "collide the bundle-{index:D4}-{name}.txt path and silently overwrite evidence");

            // The count on disk must equal the reported failure count — no silent evidence loss.
            var filesOnDisk = Directory.GetFiles(Path.GetDirectoryName(report.SummaryPath)!, "bundle-*.txt");
            filesOnDisk.Should().HaveCount(
                report.LogicFailures,
                "the evidence bundles on disk must match the reported logic-failure count; " +
                "fewer files means a collided index overwrote a captured failure");
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    /// <summary>
    /// A non-<see langword="null"/> <c>InjectFailureFor</c> that matches no target must be
    /// rejected as a configuration error, not run as a vacuous clean pass. The documented keys
    /// carry a family prefix (<c>LM:</c> / <c>OLSD:</c> / <c>OLSM:</c>); a bare target name
    /// (<c>Pontius</c>) matches nothing, and silently returning BOUNDED-NON-REPRO would let an
    /// operator misread a no-op as "capture path verified — all clean".
    /// </summary>
    [StressGatedFact]
    public void Run_WithInjectKeyMatchingNoTarget_ThrowsConfigurationError()
    {
        var dir = CreateScratchDirectory();
        try
        {
            var config = new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                InjectFailureFor: "Pontius",
                MinimumExecutions: 1);

            var act = () => NistStRDStressHarness.Run(config);

            act.Should().Throw<ArgumentException>(
                    "an inject key that matches no target is a configuration error, not a clean run")
                .WithMessage("*matched no*");
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    /// <summary>
    /// The documented prefixed inject key must actually fire: <c>OLSD:Pontius</c> forces the
    /// injected OLS-double target to trip, yielding a LOGIC-REPRODUCED disposition with captured
    /// evidence — the positive counterpart to the zero-match rejection above.
    /// </summary>
    [StressGatedFact]
    public void Run_WithPrefixedInjectKey_ReproducesLogicFailureAndCaptures()
    {
        var dir = CreateScratchDirectory();
        try
        {
            var config = new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                InjectFailureFor: "OLSD:Pontius",
                MinimumExecutions: 1);

            var report = NistStRDStressHarness.Run(config);

            report.Disposition.Should().Be(NistStRDStressHarness.StressDisposition.LogicReproduced);
            report.LogicFailures.Should().Be(1);
            report.BundlePaths.Should().ContainSingle().Which.Should().Match(p => File.Exists(p));
            var problem = NistLinearRegressionData.Pontius();
            var bytes = new byte[(problem.X.Length + problem.Y.Length) * sizeof(double)];
            Buffer.BlockCopy(problem.X, 0, bytes, 0, problem.X.Length * sizeof(double));
            Buffer.BlockCopy(problem.Y, 0, bytes, problem.X.Length * sizeof(double), problem.Y.Length * sizeof(double));
            File.ReadAllText(report.BundlePaths.Single()).Should().Contain(
                $"data-hash (SHA-256) : {Convert.ToHexString(SHA256.HashData(bytes))}");
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    /// <summary>
    /// A captured logic failure must survive a persistence fault. <see cref="NistStRDStressHarness.Run(NistStRDStressHarness.StressConfig)"/>
    /// keeps every captured failure in memory until after the full sweep completes, then
    /// writes each bundle; an I/O exception from that write step (disk-full, permission fault)
    /// must not escape as an unhandled exception — the caller (the console runner) would then
    /// crash with an anomaly-shaped exit code, and CI's exit-134/137/139 arm classifies that as
    /// a retriable host anomaly and re-runs the sweep, silently discarding the only record that
    /// a deterministic logic failure was ever captured. Uses the internal
    /// <c>Run(config, writeBundle, writeSummary)</c> seam to inject a throwing bundle writer —
    /// this is deterministic and portable, unlike reproducing a real filesystem permission
    /// fault at the exact moment a failure has already been captured.
    /// </summary>
    [StressGatedFact]
    public void Run_WhenBundlePersistenceThrows_DoesNotCrashAndPreservesDisposition()
    {
        var dir = CreateScratchDirectory();
        try
        {
            var config = new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                InjectFailureFor: "OLSD:Pontius",
                MinimumExecutions: 1);

            static string ThrowingWriteBundle(string runDirectory, int index, NistStRDStressHarness.CapturedFailure failure) =>
                throw new IOException("simulated bundle-write failure (disk-full/permission fault)");

            static string FakeWriteSummary(
                NistStRDStressHarness.StressConfig cfg, string runDirectory, int targetCount, int processorCount,
                int maxDop, int executions, int logicFailures, int hostAnomalies,
                NistStRDStressHarness.StressDisposition disposition, bool meetsFloor, double elapsedSeconds,
                IReadOnlyList<string> bundlePaths) => Path.Combine(runDirectory, "stress-summary.txt");

            var act = () => NistStRDStressHarness.Run(config, ThrowingWriteBundle, FakeWriteSummary);

            var report = act.Should().NotThrow(
                    "a persistence fault on an already-captured logic failure must never escape " +
                    "as an unhandled exception; the disposition was already determined before " +
                    "any write was attempted")
                .Subject;

            report.Disposition.Should().Be(
                NistStRDStressHarness.StressDisposition.LogicReproduced,
                "the disposition is derived from the in-memory logic-failure count, which the " +
                "failed write must not retroactively change");
            report.LogicFailures.Should().Be(1);
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    /// <summary>
    /// Sibling of <see cref="Run_WhenBundlePersistenceThrows_DoesNotCrashAndPreservesDisposition"/>:
    /// a persistence fault from the SUMMARY writer (rather than the per-bundle writer) must also
    /// not escape as an unhandled exception. Both guards were added in the same commit with the
    /// identical rationale (a persistence fault after a logic failure is already captured must
    /// never be misread by the caller as a retriable host anomaly), but only the bundle-write
    /// guard had a regression test — this closes that gap using the same
    /// <c>Run(config, writeBundle, writeSummary)</c> seam, this time with a throwing
    /// <c>writeSummary</c> and a bundle writer that succeeds normally.
    /// </summary>
    [StressGatedFact]
    public void Run_WhenSummaryPersistenceThrows_DoesNotCrashAndPreservesDisposition()
    {
        var dir = CreateScratchDirectory();
        try
        {
            var config = new NistStRDStressHarness.StressConfig(
                RepetitionsPerTarget: 1,
                ThreadMultiplier: 2,
                OutputDirectory: dir,
                InjectFailureFor: "OLSD:Pontius",
                MinimumExecutions: 1);

            static string FakeWriteBundle(string runDirectory, int index, NistStRDStressHarness.CapturedFailure failure) =>
                Path.Combine(runDirectory, $"bundle-{index:D4}-{failure.ProblemName}.txt");

            static string ThrowingWriteSummary(
                NistStRDStressHarness.StressConfig cfg, string runDirectory, int targetCount, int processorCount,
                int maxDop, int executions, int logicFailures, int hostAnomalies,
                NistStRDStressHarness.StressDisposition disposition, bool meetsFloor, double elapsedSeconds,
                IReadOnlyList<string> bundlePaths) =>
                throw new IOException("simulated summary-write failure (disk-full/permission fault)");

            var act = () => NistStRDStressHarness.Run(config, FakeWriteBundle, ThrowingWriteSummary);

            var report = act.Should().NotThrow(
                    "a persistence fault while writing the summary must never escape as an " +
                    "unhandled exception; the disposition and logic-failure count were already " +
                    "determined before the summary write was attempted")
                .Subject;

            report.Disposition.Should().Be(
                NistStRDStressHarness.StressDisposition.LogicReproduced,
                "the disposition is derived from the in-memory logic-failure count, which a " +
                "failed summary write must not retroactively change");
            report.LogicFailures.Should().Be(1);
            report.SummaryPath.Should().NotBeNullOrEmpty(
                "the report must still carry a fallback summary path even though the real " +
                "write failed, so the caller receives a complete, non-crashing StressReport");
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    [StressGatedFact]
    public void Run_WhenCleanSummaryPersistenceFails_ReportsHostAnomaly()
    {
        var dir = CreateScratchDirectory();
        try
        {
            var config = new NistStRDStressHarness.StressConfig(1, 2, dir, MinimumExecutions: 1);
            static string WriteBundle(string runDirectory, int index, NistStRDStressHarness.CapturedFailure failure) =>
                throw new InvalidOperationException("A clean run must not produce bundles.");
            static string WriteSummary(
                NistStRDStressHarness.StressConfig cfg, string runDirectory, int targetCount, int processorCount,
                int maxDop, int executions, int logicFailures, int hostAnomalies,
                NistStRDStressHarness.StressDisposition disposition, bool meetsFloor, double elapsedSeconds,
                IReadOnlyList<string> bundlePaths) => throw new IOException("Required summary cannot be persisted.");

            var report = NistStRDStressHarness.Run(config, WriteBundle, WriteSummary);

            report.LogicFailures.Should().Be(0);
            report.Disposition.Should().Be(NistStRDStressHarness.StressDisposition.HostAnomalyObserved,
                "failed required evidence persistence must not acknowledge a clean stress pass");
            report.HostAnomalies.Should().Be(1);
            File.Exists(report.SummaryPath).Should().BeFalse();
        }
        finally
        {
            TryDeleteDirectory(dir);
        }
    }

    [Fact]
    public void InputHashes_MatchIndependentKnownAnswers()
    {
        // Python hashlib.sha256(struct.pack('<2d', 1, 2)) and '<6d', 1..6;
        // the harness intentionally serializes host-endian IEEE binary64 bytes.
        var doubles = typeof(NistStRDStressHarness).GetMethod("HashDoubles", BindingFlags.NonPublic | BindingFlags.Static)!;
        var matrix = typeof(NistStRDStressHarness).GetMethod("HashMatrixAndVector", BindingFlags.NonPublic | BindingFlags.Static)!;
        doubles.Invoke(null, [new double[][] { [1], [2] }]).Should().Be(BitConverter.IsLittleEndian
            ? "DC91CE9A50DDC828740AA26743716897FDB2BB64F1DB662FE263A59BE56145AE"
            : "F814737DA80B11B6D6E54C254B9D7E711669462C0E53585F776AFEA6EA073AFC");
        matrix.Invoke(null, [new double[,] { { 1, 2 }, { 3, 4 } }, new double[] { 5, 6 }]).Should().Be(BitConverter.IsLittleEndian
            ? "D73F023A3F852BF2E5C6D836CD36CD930D0091DCBA7F778161C707E1C58222B0"
            : "B11C9D9DF2CF55965D4FCDE61B66BDE30151087EF6CE0D8669F2FEE2082D0515");
    }

    private static string CreateScratchDirectory()
    {
        var dir = Path.Combine(
            Path.GetTempPath(),
            "boutquin-numerics-stress-" + Guid.NewGuid().ToString("N"));
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
            // Best-effort scratch cleanup; a leftover temp directory must not fail the test.
        }
    }
}
