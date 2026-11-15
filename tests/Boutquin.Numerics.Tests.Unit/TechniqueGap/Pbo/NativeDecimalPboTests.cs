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

#if NET11_0_OR_GREATER
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.TechniqueGap.Pbo;

public sealed class NativeDecimalPboTests
{
    [Fact]
    public void NativeDecimalRanks_MatchIndependentRationalReference()
    {
        foreach (var panel in PboPanels.Design().Concat(PboPanels.Witnesses()))
        {
            if (!panel.Returns.Cast<double>().All(value => value == 0 || Math.Abs(value) is > 1e-12 and < 1e12))
            {
                continue;
            }

            var returns = PboPanels.To<decimal>(panel.Returns);
            var result = ProbabilityOfBacktestOverfitting<decimal>.Compute(returns, panel.SplitCount);
            var exact = PboExactReference.Logits(returns, panel.SplitCount);
            for (var i = 0; i < exact.Length; i++)
            {
                double.CreateChecked(result.LogitValues[i]).Should().BeApproximately(exact[i], 8e-15);
            }

            var ranks = PboExactReference.DoubledRanks(returns, panel.SplitCount);
            result.Pbo.Should().Be((decimal)ranks.Count(rank => rank <= returns.GetLength(0) + 1) / ranks.Length);
        }
    }
}
#endif
