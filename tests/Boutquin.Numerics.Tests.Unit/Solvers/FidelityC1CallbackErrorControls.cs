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
using Boutquin.Numerics.Solvers.Internal;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

[Trait("Tier", "Unit")]
public sealed class FidelityC1CallbackErrorControls
{
    [Theory]
    [InlineData(false, 1.0)]
    [InlineData(true, 1.0)]
    [InlineData(false, 0.015625)]
    [InlineData(true, 0.015625)]
    public void CentralDifferenceRetainsCubicScaleAndCallbackErrorTerms(bool writerPath, double characteristicScale)
    {
        // Exact objective f(x)=x/s+(x/s)^3 has derivative 1/s and third derivative 6/s^3 at zero.
        // Bounded callback error eta*sign(x) independently contributes eta/h.
        // The central quotient is exactly 1/s+h^2/s^3+eta/h in real arithmetic.
        // Machine epsilon alone cannot bound callback error.
        const double h = 0.000006055454452393339060789892727936966935;
        const double eta = 1e-8;
        double Value(double x)
        {
            var u = x / characteristicScale;
            return u + u * u * u + Math.CopySign(eta, x);
        }
        double[] Residual(double[] x) => [Value(x[0])];
        void Writer(ReadOnlySpan<double> x, Span<double> r) => r[0] = Value(x[0]);
        var buffers = new LevenbergMarquardtBuffers<double>(1, 1);
        if (writerPath) { FiniteDifferenceJacobian<double>.EvaluateInto(Writer, [0], buffers); }
        else { FiniteDifferenceJacobian<double>.EvaluateInto(Residual, [0], buffers); }
        var exactDerivative = 1 / characteristicScale;
        var truncation = h * h / (characteristicScale * characteristicScale * characteristicScale);
        var evaluation = eta / h;
        var expected = exactDerivative + truncation + evaluation;
        // 32 relative ulps conservatively cover the short normal-arithmetic expression,
        // including the correctly rounded characteristic-step constant.
        var roundingBudget = 32 * Math.ScaleB(1.0, -52) * Math.Abs(expected);
        Assert.InRange(Math.Abs(buffers.Jacobian[0, 0] - expected), 0, roundingBudget);
        Assert.InRange(Math.Abs(buffers.Jacobian[0, 0] - exactDerivative),
            evaluation + truncation - roundingBudget, evaluation + truncation + roundingBudget);
    }
}
