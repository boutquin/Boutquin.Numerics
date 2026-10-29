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

namespace Boutquin.Numerics.Tests.Unit.Solvers;

public sealed class Arena20260928WriterContractTests
{
    [Theory]
    [InlineData(false, false, 1)]
    [InlineData(false, false, 2)]
    [InlineData(false, true, 1)]
    [InlineData(false, true, 2)]
    [InlineData(true, false, 1)]
    [InlineData(true, false, 2)]
    [InlineData(true, true, 1)]
    [InlineData(true, true, 2)]
    public void PartialWriter_RefusesUnwrittenInitialAndLaterEntries(bool pooled, bool partialJacobian, int failingCall)
    {
        var solver = new LevenbergMarquardtSolver<double>(20, 1e-12, 1e-12, 1e-12, 1e-3);
        var residualCalls = 0;
        var jacobianCalls = 0;
        void Residual(ReadOnlySpan<double> x, Span<double> r)
        {
            residualCalls++;
            r[0] = x[0] - 1;
            if (partialJacobian || residualCalls != failingCall)
            {
                r[1] = x[1] - 2;
            }
        }

        void Jacobian(ReadOnlySpan<double> x, Span<double> j)
        {
            jacobianCalls++;
            j[0] = 1;
            if (!partialJacobian || jacobianCalls != failingCall)
            {
                j[1] = 0;
                j[2] = 0;
                j[3] = 1;
            }
        }

        var buffers = new LevenbergMarquardtBuffers<double>(2, 2);
        var error = Assert.Throws<InvalidOperationException>(() => pooled
            ? solver.Solve(Residual, new[] { 0.0, 0.0 }, buffers, Jacobian)
            : solver.Solve(Residual, new[] { 0.0, 0.0 }, 2, Jacobian));
        Assert.Contains(partialJacobian ? "Jacobian" : "Residual", error.Message, StringComparison.Ordinal);
        Assert.Contains("non-finite", error.Message, StringComparison.Ordinal);
    }
}
