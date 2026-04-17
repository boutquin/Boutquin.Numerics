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

using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Tests.Verification;

/// <summary>
/// Python cross-checks for <see cref="ActiveSetQpSolver{T}"/>. The
/// reference generator
/// <c>tests/Verification/generate_qp_solver_vectors.py</c> uses
/// <c>scipy.optimize.minimize</c> with method <c>'SLSQP'</c> (Kraft 1988
/// Sequential Least Squares Programming) — a completely different
/// algorithm from the Cholesky-based active-set method in Boutquin.Numerics.
/// Both methods solve the same convex QP, so optimal weight vectors must agree
/// to solver precision.
/// </summary>
/// <remarks>
/// Five cases are checked: three MinVar (3-asset unconstrained,
/// 3-asset box-constrained, 5-asset moderate-correlation) and two MeanVar
/// (3-asset unconstrained, 5-asset box-constrained). Tolerance is 5e-6
/// per weight — the active-set solver converges to KKT tolerance 1e-10,
/// and SLSQP uses <c>ftol=1e-12</c> (function value, not parameter);
/// residual below 5e-6 is solver noise from two independent algorithms.
/// </remarks>
public sealed class QpSolverVerificationTests : CrossLanguageVerificationBase
{
    [Fact]
    public void ActiveSetQpSolver_MinVar_MatchesSlsqp()
    {
        using var doc = LoadVector("qp_solver");
        var cases = doc.RootElement.GetProperty("cases");

        foreach (var c in cases.EnumerateArray())
        {
            if (c.GetProperty("kind").GetString() != "minvar")
            {
                continue;
            }

            var name = c.GetProperty("name").GetString()!;
            var cov = GetDouble2D(c.GetProperty("covariance"));
            var lb = c.GetProperty("min_weight").GetDouble();
            var ub = c.GetProperty("max_weight").GetDouble();
            var expected = GetDoubleArray(c.GetProperty("weights"));

            var actual = ActiveSetQpSolver<double>.SolveMinVariance(cov, lb, ub);

            Assert.Equal(expected.Length, actual.Length);
            for (var i = 0; i < expected.Length; i++)
            {
                AssertScalarWithin(actual[i], expected[i], 5e-6, $"{name}[{i}]");
            }
        }
    }

    [Fact]
    public void ActiveSetQpSolver_MeanVar_MatchesSlsqp()
    {
        using var doc = LoadVector("qp_solver");
        var cases = doc.RootElement.GetProperty("cases");

        foreach (var c in cases.EnumerateArray())
        {
            if (c.GetProperty("kind").GetString() != "meanvar")
            {
                continue;
            }

            var name = c.GetProperty("name").GetString()!;
            var cov = GetDouble2D(c.GetProperty("covariance"));
            var means = GetDoubleArray(c.GetProperty("means"));
            var riskAversion = c.GetProperty("risk_aversion").GetDouble();
            var lb = c.GetProperty("min_weight").GetDouble();
            var ub = c.GetProperty("max_weight").GetDouble();
            var expected = GetDoubleArray(c.GetProperty("weights"));

            var actual = ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, riskAversion, lb, ub);

            Assert.Equal(expected.Length, actual.Length);
            for (var i = 0; i < expected.Length; i++)
            {
                AssertScalarWithin(actual[i], expected[i], 5e-6, $"{name}[{i}]");
            }
        }
    }
}
