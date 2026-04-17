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

using System.Numerics;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.GenericParity;

/// <summary>
/// Parity-harness helpers for the generic-math migration. Each migrated type
/// gains test methods in Phase-specific files (e.g., <c>Solvers_ParityTests.cs</c>,
/// <c>LinearAlgebra_ParityTests.cs</c>) that call these helpers to assert the
/// generic version instantiated at the currently-shipped type produces
/// identical results to the pre-migration concrete version.
/// </summary>
/// <remarks>
/// <para>
/// <b>Purpose:</b> The parity harness is the gate that proves no behaviour
/// changed for existing consumers during the generic-math migration. Every
/// generic type must pass parity at the type it was previously hardcoded to
/// (<c>double</c> or <c>decimal</c>) before the migration phase ships.
/// </para>
/// <para>
/// <b>Scaffold (Phase 0):</b> This file lands empty-of-cases in Phase 0.
/// Phases 1-3 add per-type test methods in separate files within this
/// directory. The helpers below are shared across all phases.
/// </para>
/// </remarks>
internal static class ParityHarness
{
    /// <summary>
    /// Asserts that a generic invocation produces a result within
    /// <paramref name="tolerance"/> of the legacy invocation for a scalar value.
    /// </summary>
    /// <typeparam name="T">The numeric type being compared.</typeparam>
    /// <param name="legacyResult">Result from the pre-migration concrete-typed API.</param>
    /// <param name="genericResult">Result from the new generic API instantiated at <typeparamref name="T"/>.</param>
    /// <param name="tolerance">Maximum acceptable absolute difference.</param>
    /// <param name="context">
    /// Human-readable description of the comparison (e.g., "BrentSolver.Solve root value").
    /// Included in the assertion message on failure.
    /// </param>
    internal static void AssertScalarParity<T>(T legacyResult, T genericResult, T tolerance, string context)
        where T : IFloatingPoint<T>
    {
        var diff = T.Abs(genericResult - legacyResult);
        diff.Should().BeLessThanOrEqualTo(tolerance,
            because: $"generic {context} must match legacy to within {tolerance}");
    }

    /// <summary>
    /// Asserts that a generic invocation produces a result that is bit-identical
    /// to the legacy invocation for a scalar value.
    /// </summary>
    /// <typeparam name="T">The numeric type being compared.</typeparam>
    /// <param name="legacyResult">Result from the pre-migration concrete-typed API.</param>
    /// <param name="genericResult">Result from the new generic API instantiated at <typeparamref name="T"/>.</param>
    /// <param name="context">
    /// Human-readable description of the comparison. Included in the assertion message on failure.
    /// </param>
    internal static void AssertExactScalarParity<T>(T legacyResult, T genericResult, string context)
        where T : IFloatingPoint<T>
    {
        genericResult.Should().Be(legacyResult,
            because: $"generic {context} must be bit-identical to legacy");
    }

    /// <summary>
    /// Asserts element-wise parity between a legacy vector and a generic vector.
    /// </summary>
    /// <typeparam name="T">The numeric type being compared.</typeparam>
    /// <param name="legacyResult">Vector from the pre-migration concrete-typed API.</param>
    /// <param name="genericResult">Vector from the new generic API instantiated at <typeparamref name="T"/>.</param>
    /// <param name="tolerance">Maximum acceptable absolute difference per element.</param>
    /// <param name="context">
    /// Human-readable description of the comparison. Included in the assertion message on failure.
    /// </param>
    internal static void AssertVectorParity<T>(T[] legacyResult, T[] genericResult, T tolerance, string context)
        where T : IFloatingPoint<T>
    {
        genericResult.Should().HaveCount(legacyResult.Length,
            because: $"generic {context} vector length must match legacy");

        for (var i = 0; i < legacyResult.Length; i++)
        {
            var diff = T.Abs(genericResult[i] - legacyResult[i]);
            diff.Should().BeLessThanOrEqualTo(tolerance,
                because: $"generic {context}[{i}] must match legacy to within {tolerance}");
        }
    }

    /// <summary>
    /// Asserts element-wise parity between a legacy matrix and a generic matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type being compared.</typeparam>
    /// <param name="legacyResult">Matrix from the pre-migration concrete-typed API.</param>
    /// <param name="genericResult">Matrix from the new generic API instantiated at <typeparamref name="T"/>.</param>
    /// <param name="tolerance">Maximum acceptable absolute difference per element.</param>
    /// <param name="context">
    /// Human-readable description of the comparison. Included in the assertion message on failure.
    /// </param>
    internal static void AssertMatrixParity<T>(T[,] legacyResult, T[,] genericResult, T tolerance, string context)
        where T : IFloatingPoint<T>
    {
        var rows = legacyResult.GetLength(0);
        var cols = legacyResult.GetLength(1);

        genericResult.GetLength(0).Should().Be(rows,
            because: $"generic {context} row count must match legacy");
        genericResult.GetLength(1).Should().Be(cols,
            because: $"generic {context} column count must match legacy");

        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                var diff = T.Abs(genericResult[r, c] - legacyResult[r, c]);
                diff.Should().BeLessThanOrEqualTo(tolerance,
                    because: $"generic {context}[{r},{c}] must match legacy to within {tolerance}");
            }
        }
    }
}
