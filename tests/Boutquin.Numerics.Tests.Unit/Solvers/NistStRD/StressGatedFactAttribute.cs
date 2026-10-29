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

namespace Boutquin.Numerics.Tests.Unit.Solvers.NistStRD;

/// <summary>
/// A <see cref="FactAttribute"/> that runs its fact only when the stress opt-in
/// environment variable (<see cref="NistStRDStressHarness.GateEnvironmentVariable"/>,
/// i.e. <c>BOUTQUIN_NUMERICS_STRESS=1</c>) is set; otherwise the fact is reported as
/// <em>skipped</em>, never executed.
/// </summary>
/// <remarks>
/// <para>
/// Facts that drive <see cref="NistStRDStressHarness.Run(NistStRDStressHarness.StressConfig)"/> manufacture CPU
/// oversubscription (the exact condition the stress harness exists to probe). Running
/// them in the default <c>dotnet test</c> invocation would re-import that oversubscription
/// into every ordinary suite run and add PR-time cost the harness is explicitly required
/// to keep out of the default suite. This attribute keeps that heavy work opt-in: the
/// default suite <em>discovers</em> the fact (so it stays visible as a skipped test with
/// its gating reason) but does not <em>execute</em> the harness.
/// </para>
/// <para>
/// The condition is evaluated in this attribute's constructor rather than as a
/// <c>[Fact(Skip = …)]</c> literal because <c>Skip</c> at the call site must be a
/// compile-time constant and so cannot key off an environment variable. Setting the
/// inherited <see cref="FactAttribute.Skip"/> property from the constructor body is the
/// runtime-conditional skip pattern for xunit v2: the reflection-based test discoverer
/// instantiates the attribute and then reads <see cref="FactAttribute.Skip"/>, so a
/// gated fact reports as skipped (not as a vacuously-passing early return).
/// </para>
/// <para>
/// CI keeps the regression coverage these facts provide by setting the gate variable in
/// the stress-gate step and running them explicitly there; see
/// <c>.github/workflows/pr-verify.yml</c>.
/// </para>
/// </remarks>
public sealed class StressGatedFactAttribute : FactAttribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="StressGatedFactAttribute"/> class,
    /// skipping the decorated fact unless
    /// <see cref="NistStRDStressHarness.GateEnvironmentVariable"/> is set to <c>"1"</c>.
    /// </summary>
    public StressGatedFactAttribute()
    {
        if (Environment.GetEnvironmentVariable(NistStRDStressHarness.GateEnvironmentVariable) != "1")
        {
            Skip = "Opt-in stress regression coverage — set "
                + NistStRDStressHarness.GateEnvironmentVariable
                + "=1 to run (drives NistStRDStressHarness.Run under CPU oversubscription; "
                + "kept out of the default dotnet test suite).";
        }
    }
}
