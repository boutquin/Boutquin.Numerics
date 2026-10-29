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

namespace Boutquin.Numerics.Internal;

/// <summary>Owns one optional scratch rental for a synchronous using scope.</summary>
internal ref struct PooledScratch<T>
{
    private readonly ArrayPool<T> _pool;
    internal T[]? Buffer { get; private set; }

    internal PooledScratch(int length, ArrayPool<T>? pool = null)
    {
        _pool = pool ?? ArrayPool<T>.Shared;
        Buffer = length > 0 ? _pool.Rent(length) : null;
    }

    public void Dispose()
    {
        if (Buffer is { } buffer)
        {
            Buffer = null;
            _pool.Return(buffer, clearArray: true);
        }
    }
}
