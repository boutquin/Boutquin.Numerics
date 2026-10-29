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

using System.Collections;
using System.Reflection;

namespace Boutquin.Numerics.Tests.Unit.Solvers.NistStRD;

public sealed class Arena20260928StressToleranceTests
{
    [Theory]
    [InlineData("Pontius", false, 1e-10)]
    [InlineData("Wampler4", false, 1e-10)]
    [InlineData("Wampler5", false, 1e-10)]
    [InlineData("Filip", false, 1e-6)]
    [InlineData("Pontius", true, 1e-11)]
    [InlineData("Wampler4", true, 1e-11)]
    [InlineData("Wampler5", true, 1e-11)]
    [InlineData("Filip", true, 1e-5)]
    public void ConfiguredTarget_RejectsCoefficientDriftBeyondCertifiedGate(string name, bool useDecimal, double drift)
    {
        var method = typeof(NistStRDStressHarness).GetMethod("BuildTargets", BindingFlags.NonPublic | BindingFlags.Static)!;
        var targets = (IEnumerable)method.Invoke(null, new object?[] { null, false })!;
        var label = name + (useDecimal ? " (OLS decimal)" : " (OLS double)");
        var target = targets.Cast<object>().Single(t => (string)t.GetType().GetProperty("Name")!.GetValue(t)! == label);
        var run = (Delegate)target.GetType().GetProperty("RunOnce")!.GetValue(target)!;
        Assert.Null(run.DynamicInvoke());

        // The target captures its own fresh NIST problem. Alter only that private
        // expected coefficient; the matrix, response, solver, and configured gate stay real.
        var closure = run.Target!;
        var field = closure.GetType().GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .Single(f => f.FieldType == typeof(NistLinearRegressionData.Problem));
        var problem = (NistLinearRegressionData.Problem)field.GetValue(closure)!;
        problem.Coefficients[0] *= 1 + drift;
        var failure = Assert.IsType<NistStRDStressHarness.CapturedFailure>(run.DynamicInvoke());
        Assert.Equal("LOGIC", failure.Kind);
        Assert.Equal("coefficient b0", failure.FailingField);
    }
}
