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
using Boutquin.Numerics.Random;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

[Trait("Tier", "Unit")]
public sealed class C3AllocationRegressionTests
{
    [Fact]
    public void HarrellDavis_AboveStackCeiling_ReusesPooledBuffer()
    {
        var observations = Enumerable.Range(0, 513).Select(i => (double)i).ToArray();
        for (var i = 0; i < 5; i++)
        {
            _ = HarrellDavisPercentile.Compute<double>(observations, 0.5);
        }

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 10; i++)
        {
            _ = HarrellDavisPercentile.Compute<double>(observations, 0.5);
        }

        (GC.GetAllocatedBytesForCurrentThread() - before).Should().Be(0);
    }

    [Fact]
    public void RunMulti_UsesOneWorkingAndOneSnapshotRow()
    {
        var engine = new BootstrapMonteCarloEngine<double>(10000, new Pcg64RandomSource(19));
        double[] observations = [7];
        _ = engine.RunMulti(observations, 1, static (_, output) => output[0] = 7);
        var before = GC.GetAllocatedBytesForCurrentThread();
        var result = engine.RunMulti(observations, 1, static (_, output) => output[0] = 7);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        allocated.Should().BeLessThan(180000, "two 80000-byte rows plus metadata suffice");
        var returned = result.Statistics[0];
        returned[0] = -99;
        result.Statistics[0][0].Should().Be(7);
    }
}
