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
public sealed class FidelityC1DifferenceScaleControls
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ZeroComponentUsesUnitCharacteristicScaleForCentralStencil(bool writerPath)
    {
        // Independently computed65-digit cube root of2^-52, rounded to double.
        // f=x+x³ has analytic derivative1 at zero and central truncation error h².
        const double h = 0.000006055454452393339060789892727936966935;
        var visits = new List<double>();
        double[] Residual(double[] x)
        {
            visits.Add(x[0]);
            return [x[0] + x[0] * x[0] * x[0]];
        }
        void Writer(ReadOnlySpan<double> x, Span<double> r)
        {
            visits.Add(x[0]);
            r[0] = x[0] + x[0] * x[0] * x[0];
        }
        var buffers = new LevenbergMarquardtBuffers<double>(1, 1);
        if (writerPath) { FiniteDifferenceJacobian<double>.EvaluateInto(Writer, [0], buffers); }
        else { FiniteDifferenceJacobian<double>.EvaluateInto(Residual, [0], buffers); }
        Assert.Equal(2, visits.Count);
        Assert.InRange(Math.Abs(visits[0] - h), 0, h * 2e-15);
        Assert.InRange(Math.Abs(visits[1] + h), 0, h * 2e-15);
        Assert.InRange(Math.Abs(buffers.Jacobian[0, 0] - (1 + h * h)), 0, 5e-16);
    }
}
