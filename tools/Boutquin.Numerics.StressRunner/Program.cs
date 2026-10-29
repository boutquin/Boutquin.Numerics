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

using Boutquin.Numerics.Tests.Unit.Solvers.NistStRD;

// ── NIST/LM solver reproduce-or-bound stress runner ─────────────────────────
//
// Opt-in only. It runs the numerically-hard NIST solves under CPU oversubscription
// and records a three-state disposition. It is gated so it never runs incidentally:
//   • CI sets the environment variable BOUTQUIN_NUMERICS_STRESS=1, OR
//   • locally pass --stress.
//
// Usage:
//   dotnet run --project tools/Boutquin.Numerics.StressRunner -- --stress \
//       [--reps N] [--threads M] [--min K] [--out DIR] [--inject-failure KEY]
//
//   --inject-failure forces a captured failure for exactly one target so the capture
//   path can be verified. KEY must carry a family prefix: LM:<name> (nonlinear),
//   OLSD:<name> (OLS double), or OLSM:<name> (OLS decimal) — e.g. OLSD:Pontius. A key
//   that matches no target is rejected as a configuration error (exit 2), not run as a
//   vacuous clean pass.
//
// Exit codes:
//   0  BOUNDED-NON-REPRO   (>= min executions, 0 logic failures, 0 host anomalies)
//   2  CONFIG ERROR        (invalid CLI or run settings, or execution floor not met)
//   3  LOGIC-REPRODUCED    (>= 1 deterministic numeric assertion tripped)
//   4  HOST-ANOMALY-OBSERVED (0 logic failures, >= 1 non-assertion host anomaly)

Options options;
try
{
    options = ParseArgs(args);
}
catch (ArgumentException ex)
{
    Console.Error.WriteLine($"CONFIG-ERROR: {ex.Message}");
    return 2;
}

var gateEnv = Environment.GetEnvironmentVariable(NistStRDStressHarness.GateEnvironmentVariable);
var gated = string.Equals(gateEnv, "1", StringComparison.Ordinal) || options.Force;
if (!gated)
{
    Console.WriteLine(
        $"Stress runner is opt-in. Set {NistStRDStressHarness.GateEnvironmentVariable}=1 (CI) " +
        "or pass --stress (local) to execute. No stress pass run.");
    return 0;
}

var config = new NistStRDStressHarness.StressConfig(
    RepetitionsPerTarget: options.Repetitions,
    ThreadMultiplier: options.ThreadMultiplier,
    OutputDirectory: options.OutputDirectory,
    InjectFailureFor: options.InjectFailureFor,
    MinimumExecutions: options.MinimumExecutions);

Console.WriteLine("=== NIST/LM SOLVER STRESS RUN ===");
Console.WriteLine($"gate-env-variable    : {NistStRDStressHarness.GateEnvironmentVariable}");
Console.WriteLine($"reps-per-target      : {config.RepetitionsPerTarget}");
Console.WriteLine($"thread-multiplier    : {config.ThreadMultiplier}");
Console.WriteLine($"processor-count      : {Environment.ProcessorCount}");
if (config.InjectFailureFor is not null)
{
    Console.WriteLine($"inject-failure-for   : {config.InjectFailureFor}");
}

NistStRDStressHarness.StressReport report;
try
{
    report = NistStRDStressHarness.Run(config);
}
catch (ArgumentException ex)
{
    // Invalid run settings are configuration errors, not solver results.
    Console.Error.WriteLine($"CONFIG-ERROR: {ex.Message}");
    return 2;
}

Console.WriteLine("--- RESULT ---");
Console.WriteLine($"EXECUTION-COUNT      : {report.ExecutionCount}");
Console.WriteLine($"TARGET-COUNT         : {report.TargetCount}");
Console.WriteLine($"THREAD-MULTIPLIER    : {report.ThreadMultiplier}");
Console.WriteLine($"MAX-DEGREE-PARALLEL  : {report.MaxDegreeOfParallelism} (>= 2 x {report.ProcessorCount} = {2 * report.ProcessorCount})");
Console.WriteLine($"MEETS-EXECUTION-FLOOR: {report.MeetsExecutionFloor} (floor {config.MinimumExecutions})");
Console.WriteLine($"LOGIC-FAILURES       : {report.LogicFailures}");
Console.WriteLine($"HOST-ANOMALIES       : {report.HostAnomalies}");
Console.WriteLine($"DISPOSITION          : {NistStRDStressHarness.ToDispositionLabel(report.Disposition)}");
Console.WriteLine($"ELAPSED-SECONDS      : {report.ElapsedSeconds.ToString("F2", CultureInfo.InvariantCulture)}");
Console.WriteLine($"SUMMARY-PATH         : {report.SummaryPath}");
foreach (var bundle in report.BundlePaths)
{
    Console.WriteLine($"BUNDLE               : {bundle}");
}

if (!report.MeetsExecutionFloor)
{
    Console.WriteLine($"STRESS-INVARIANT-FAILED: executions={report.ExecutionCount} < floor={config.MinimumExecutions}");
    return 2;
}

return report.Disposition switch
{
    NistStRDStressHarness.StressDisposition.LogicReproduced => 3,
    NistStRDStressHarness.StressDisposition.HostAnomalyObserved => 4,
    _ => 0,
};

static Options ParseArgs(string[] args)
{
    var repetitions = 400;
    var threadMultiplier = 2;
    var minimumExecutions = 5000;
    var force = false;
    string? injectFailureFor = null;
    var outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), ".stress-artifacts");

    for (var i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--stress":
                force = true;
                break;
            case "--reps":
                repetitions = ReadInt(args, ref i);
                break;
            case "--threads":
                threadMultiplier = ReadInt(args, ref i);
                break;
            case "--min":
                minimumExecutions = ReadInt(args, ref i);
                break;
            case "--out":
                outputDirectory = ReadValue(args, ref i);
                break;
            case "--inject-failure":
                injectFailureFor = ReadValue(args, ref i);
                break;
            default:
                throw new ArgumentException($"Unknown argument: {args[i]}");
        }
    }

    return new Options(repetitions, threadMultiplier, minimumExecutions, force, injectFailureFor, outputDirectory);
}

static int ReadInt(string[] args, ref int index)
{
    var option = args[index];
    var value = ReadValue(args, ref index);
    if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
    {
        throw new ArgumentException($"{option} requires an integer value.");
    }

    return parsed;
}

static string ReadValue(string[] args, ref int index)
{
    var option = args[index];
    if (index + 1 >= args.Length || string.IsNullOrWhiteSpace(args[index + 1]) || args[index + 1].StartsWith("--", StringComparison.Ordinal))
    {
        throw new ArgumentException($"{option} requires a value.");
    }

    return args[++index];
}

internal sealed record Options(
    int Repetitions,
    int ThreadMultiplier,
    int MinimumExecutions,
    bool Force,
    string? InjectFailureFor,
    string OutputDirectory);
