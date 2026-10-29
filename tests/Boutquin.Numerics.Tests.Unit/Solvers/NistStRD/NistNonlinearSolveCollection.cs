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
/// xunit collection definition that bounds cross-collection parallelism for the
/// heaviest, most convergence-sensitive NIST Levenberg&#8211;Marquardt solves.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="CollectionDefinitionAttribute.DisableParallelization"/> excludes this
/// single named collection from xunit's parallel bucket: while any test carrying
/// <c>[Collection(Name)]</c> runs, no other test collection in the assembly runs
/// concurrently with it. Every other collection in the assembly is unaffected and
/// keeps running in parallel with each other — the bound applies to exactly this one
/// collection, not to the suite as a whole. This is the deterministic parallelism
/// bound required for the heaviest nonlinear NIST solves (Lanczos1&#8211;3, Eckerle4,
/// Bennett5, MGH17 &#8212; 500&#8211;2000 LM iterations each), which are the exact
/// target set the 2026-07-08 flakiness observation implicated.
/// </para>
/// <para>
/// Members join by carrying <c>[Collection(Name)]</c>:
/// <see cref="Boutquin.Numerics.Tests.Unit.Solvers.NistStRD.NistStRDTests"/> (the
/// certified-value assertions) and
/// <see cref="Boutquin.Numerics.Tests.Unit.Solvers.NistStRD.NistStRDStressCaptureTests"/>
/// (the opt-in stress capture-path regressions, which drive the same heavy solves under
/// CPU oversubscription) — both share this bound so neither competes for CPU against the
/// rest of the suite.
/// </para>
/// </remarks>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class NistNonlinearSolveCollection
{
    /// <summary>The xunit collection name shared between this definition and every member test class.</summary>
    public const string Name = "NIST Nonlinear Solve Collection (bounded, non-parallel)";
}
