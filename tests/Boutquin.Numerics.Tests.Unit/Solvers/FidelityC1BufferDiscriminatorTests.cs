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

public sealed class FidelityC1BufferDiscriminatorTests
{
    [Fact]
    public void ShrinkAndRegrow_RetainsWorkCapacityAndExactCallbackShape()
    {
        var pool = new LevenbergMarquardtBuffers<double>(4, 7);
        // The public contract promises retained work storage, while callbacks must
        // receive the exact logical parameter length. Test those distinct obligations.
        var work = WorkArrays(pool);
        pool.Reset(2, 3);
        Assert.Equal(2, pool.ParameterCount);
        Assert.Equal(3, pool.ResidualCount);
        Assert.Equal(2, pool.CallbackParameters.Length);
        Assert.Equal(7, pool.Jacobian.GetLength(0));
        Assert.Equal(4, pool.Jacobian.GetLength(1));
        AssertSameWork(work, WorkArrays(pool));
        pool.Reset(4, 7);
        Assert.Equal(4, pool.CallbackParameters.Length);
        AssertSameWork(work, WorkArrays(pool));
    }

    private static Array[] WorkArrays(LevenbergMarquardtBuffers<double> pool) =>
    [
        pool.Jacobian, pool.NormalEquations, pool.Gradient, pool.Delta, pool.Rhs,
        pool.Y, pool.Theta, pool.TrialTheta, pool.EffectiveDelta, pool.PerturbedParameters,
        pool.PlusBuffer, pool.MinusBuffer, pool.Residual, pool.TrialResidual,
        pool.JacobianFlatScratch,
    ];

    private static void AssertSameWork(Array[] expected, Array[] actual)
    {
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Same(expected[i], actual[i]);
        }
    }
}
