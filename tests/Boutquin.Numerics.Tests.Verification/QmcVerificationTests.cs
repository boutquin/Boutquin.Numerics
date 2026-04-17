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

using Boutquin.Numerics.MonteCarlo;

namespace Boutquin.Numerics.Tests.Verification;

public sealed class QmcVerificationTests : CrossLanguageVerificationBase
{
    [Fact]
    public void SobolSequence_PerDimensionMean_ApproachesHalf()
    {
        using var doc = LoadVector("qmc");
        var dim = doc.RootElement.GetProperty("dimension").GetInt32();
        var count = doc.RootElement.GetProperty("num_points").GetInt32();

        var sobol = new SobolSequence(dim);
        var sums = new double[dim];
        for (var i = 0; i < count; i++)
        {
            var p = sobol.Next();
            for (var d = 0; d < dim; d++)
            {
                sums[d] += p[d];
            }
        }

        for (var d = 0; d < dim; d++)
        {
            // 16-point per-dim mean will not be exactly 0.5 — the envelope is
            // the discrepancy bound. Use a generous statistical tolerance.
            AssertScalarWithin(sums[d] / count, 0.5, 0.5, $"Sobol dim {d}");
        }
    }

    [Fact]
    public void HaltonSequence_PerDimensionMean_ApproachesHalf()
    {
        using var doc = LoadVector("qmc");
        var dim = doc.RootElement.GetProperty("dimension").GetInt32();
        var count = doc.RootElement.GetProperty("num_points").GetInt32();

        var halton = new HaltonSequence(dim);
        var sums = new double[dim];
        for (var i = 0; i < count; i++)
        {
            var p = halton.Next();
            for (var d = 0; d < dim; d++)
            {
                sums[d] += p[d];
            }
        }

        for (var d = 0; d < dim; d++)
        {
            AssertScalarWithin(sums[d] / count, 0.5, 0.5, $"Halton dim {d}");
        }
    }
}
