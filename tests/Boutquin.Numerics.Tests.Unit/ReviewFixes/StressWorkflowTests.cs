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

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>Executes the Linux CI shell gate with controlled child-process outcomes.</summary>
public sealed class StressWorkflowTests
{
    [Theory]
    [InlineData(137, 0, 1, 3, 1)]
    [InlineData(134, 0, 1, 3, 1)]
    [InlineData(139, 0, 1, 3, 1)]
    [InlineData(4, 0, 1, 3, 1)]
    [InlineData(0, 0, 1, 3, 1)]
    [InlineData(137, 0, 0, 0, 2)]
    [InlineData(4, 137, 2, 3, 2)]
    public async Task CapturedLogic_OverridesExitCodeAndCannotBeRetried(
        int firstExit, int secondExit, int logicAttempt, int expectedExit, int expectedAttempts)
    {
        // The workflow runs on Ubuntu; use Git Bash when exercising it on Windows.
        var bash = OperatingSystem.IsWindows() ? "bash" : "/bin/bash";
        var root = new DirectoryInfo(AppContext.BaseDirectory);
        while (root is not null && !File.Exists(Path.Combine(root.FullName, "Boutquin.Numerics.slnx")))
        {
            root = root.Parent;
        }
        root.Should().NotBeNull();
        var workflow = File.ReadAllText(Path.Combine(root!.FullName, ".github", "workflows", "pr-verify.yml"));
        var step = workflow.Split("      - name: NIST/LM stress gate (reproduce-or-bound)\n", StringSplitOptions.None)[1];
        var block = step.Split("        run: |\n", StringSplitOptions.None)[1]
            .Split("\n      # always():", StringSplitOptions.None)[0];
        var body = string.Join('\n', block.Split('\n').Select(line => line.Length >= 10 ? line[10..] : line));
        var directory = Path.Combine(Path.GetTempPath(), "numerics-workflow-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            var script = Path.Combine(directory, "gate.sh");
            File.WriteAllText(script, """
                dotnet() {
                  if [ "$1" = test ]; then return 0; fi
                  count=0
                  if [ -f "$RUNNER_TEMP/count" ]; then count=$(cat "$RUNNER_TEMP/count"); fi
                  count=$((count + 1))
                  echo "$count" > "$RUNNER_TEMP/count"
                  if [ "$count" -eq "$LOGIC_ATTEMPT" ]; then
                    echo 'NIST-STRESS-CAPTURED index=0000 problem=Pontius kind=LOGIC' >&2
                  fi
                  if [ "$count" -eq 1 ]; then return "$FIRST_EXIT"; fi
                  return "$SECOND_EXIT"
                }
                # GNU sync FILE is unavailable on macOS; persistence is tested separately.
                sync() { return 0; }

                """ + body);
            var start = new ProcessStartInfo(bash)
            {
                WorkingDirectory = directory,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            start.ArgumentList.Add("-e");
            start.ArgumentList.Add(script);
            start.Environment["RUNNER_TEMP"] = directory;
            start.Environment["FIRST_EXIT"] = firstExit.ToString(System.Globalization.CultureInfo.InvariantCulture);
            start.Environment["SECOND_EXIT"] = secondExit.ToString(System.Globalization.CultureInfo.InvariantCulture);
            start.Environment["LOGIC_ATTEMPT"] = logicAttempt.ToString(System.Globalization.CultureInfo.InvariantCulture);
            using var process = Process.Start(start)!;
            var stdout = process.StandardOutput.ReadToEndAsync();
            var stderr = process.StandardError.ReadToEndAsync();
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
            await process.WaitForExitAsync(timeout.Token);
            var output = await stdout + await stderr;
            process.ExitCode.Should().Be(expectedExit, output);
            File.ReadAllText(Path.Combine(directory, "count")).Trim()
                .Should().Be(expectedAttempts.ToString(System.Globalization.CultureInfo.InvariantCulture), output);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
