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

using System.Buffers;
using Boutquin.Numerics.Internal;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit;

public sealed class PooledScratchTests
{
    [Fact]
    public void Scope_Exception_ReturnsAndClearsRental()
    {
        var pool = new TrackingPool();
        Action act = () =>
        {
            using var scratch = new PooledScratch<int>(300, pool);
            scratch.Buffer![0] = 7;
            throw new InvalidOperationException("control");
        };
        act.Should().Throw<InvalidOperationException>().WithMessage("control");
        pool.Returns.Should().Be(1);
        pool.Values.Should().OnlyContain(value => value == 0);
    }

    [Fact]
    public void Scope_DisposedTwice_ReturnsOnce()
    {
        var pool = new TrackingPool();
        var scratch = new PooledScratch<int>(300, pool);
        scratch.Dispose();
        scratch.Dispose();
        pool.Returns.Should().Be(1);
    }

    private sealed class TrackingPool : ArrayPool<int>
    {
        internal int[] Values { get; } = new int[300];
        internal int Returns { get; private set; }
        public override int[] Rent(int minimumLength) => Values;
        public override void Return(int[] array, bool clearArray = false)
        {
            array.Should().BeSameAs(Values);
            Returns++;
            if (clearArray)
            {
                Array.Clear(array);
            }
        }
    }
}
