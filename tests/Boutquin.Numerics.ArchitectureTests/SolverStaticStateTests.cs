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
using FluentAssertions;

namespace Boutquin.Numerics.ArchitectureTests;

/// <summary>
/// Determinism-firewall regression guard (build-arena M-2, spec
/// <c>feature-solver-test-parallel-flakiness-hardening</c>): fails if any of the
/// four solver types the library's parallel-equivalence guarantee depends on gains
/// a non-<c>readonly</c> static field — the exact structural signature of shared
/// mutable state that could reintroduce a real cross-thread race the 2026-07-08
/// determinism audit ruled out for the current code.
/// </summary>
/// <remarks>
/// <para>
/// Companion to <c>Boutquin.Numerics.Tests.Unit.Solvers.SolverDeterminismTests</c>
/// (runtime/behavioral: solves under real concurrency and checks byte-identity).
/// This test is the structural half — it catches the defect at review/CI time,
/// before a race could ever manifest, by inspecting the type's own field
/// declarations rather than exercising it.
/// </para>
/// <para>
/// Types are resolved by full name via <see cref="Assembly.GetType(string, bool)"/>
/// rather than <c>typeof</c> because <c>DampedLinearSolve</c> /
/// <c>DampedLinearSolve&lt;T&gt;</c> are <see langword="internal"/> and this project
/// carries no <c>InternalsVisibleTo</c> grant from <c>Boutquin.Numerics</c> — the
/// guard works by pure reflection over the shipped assembly, the same contract any
/// external auditor could run. Resolution uses <c>throwOnError: true</c> so a
/// renamed or moved type fails the test setup loudly (a staleness self-check by
/// construction) instead of silently guarding nothing, matching the pattern
/// <see cref="DependencyTests"/> uses for its own forbidden-dependency list.
/// </para>
/// </remarks>
public sealed class SolverStaticStateTests
{
    private static readonly Assembly s_numericsAssembly = Assembly.Load("Boutquin.Numerics");

    /// <summary>
    /// Full CLR names (namespace + <c>Type.Name</c>, including the backtick-arity
    /// suffix for open generics) of the four solver types named in the spec's M-2
    /// design: the generic <c>LevenbergMarquardtSolver&lt;T&gt;</c> and
    /// <c>OrdinaryLeastSquares&lt;T&gt;</c>, both the generic and non-generic
    /// <c>DampedLinearSolve</c> forms, and the generic
    /// <c>LevenbergMarquardtBuffers&lt;T&gt;</c>. The legacy non-generic facades for
    /// the other three types (<c>LevenbergMarquardtSolver</c>,
    /// <c>OrdinaryLeastSquares</c>, <c>LevenbergMarquardtBuffers</c>) declare no
    /// static fields of their own — they hold only a
    /// <c>private readonly LevenbergMarquardtSolver&lt;double&gt;</c>-shaped inner
    /// instance — so guarding them would add theory rows with nothing to catch;
    /// the generic types are where the algorithm's own state lives.
    /// </summary>
    public static IEnumerable<object[]> GuardedTypeFullNames =>
    [
        ["Boutquin.Numerics.Solvers.LevenbergMarquardtSolver`1"],
        ["Boutquin.Numerics.Solvers.OrdinaryLeastSquares`1"],
        ["Boutquin.Numerics.Solvers.Internal.DampedLinearSolve`1"],
        ["Boutquin.Numerics.Solvers.Internal.DampedLinearSolve"],
        ["Boutquin.Numerics.Solvers.LevenbergMarquardtBuffers`1"],
    ];

    [Theory]
    [MemberData(nameof(GuardedTypeFullNames))]
    public void Type_ShouldNotDeclareNonReadonlyMutableStaticField(string fullName)
    {
        var guardedType = s_numericsAssembly.GetType(fullName, throwOnError: true)!;

        // Inspect every static field the type declares itself (not inherited),
        // public or private — a mutable static field is shared mutable state
        // regardless of its accessibility. `IsInitOnly` is the CLR's `readonly`;
        // `IsLiteral` is `const` (a compile-time value, never a runtime-mutable
        // slot) and is excluded because it cannot participate in a race.
        var offending = guardedType
            .GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.DeclaredOnly)
            .Where(f => !f.IsInitOnly && !f.IsLiteral)
            .Select(f => f.Name)
            .ToArray();

        offending.Should().BeEmpty(
            because: $"{guardedType.FullName} backs the parallel-equivalence determinism guarantee " +
                     "(see SolverDeterminismTests); a non-readonly static field is shared mutable " +
                     $"state a future edit could turn into a real cross-thread race — found: " +
                     $"[{string.Join(", ", offending)}]");
    }
}
