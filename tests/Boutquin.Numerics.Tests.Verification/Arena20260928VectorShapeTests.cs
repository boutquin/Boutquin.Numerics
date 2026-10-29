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

using System.Text.Json;

namespace Boutquin.Numerics.Tests.Verification;

public sealed class Arena20260928VectorShapeTests : CrossLanguageVerificationBase
{
    [Theory]
    [InlineData(false, "[[1,2],[3,4,999]]")]
    [InlineData(true, "[[1,2],[3,4,999]]")]
    [InlineData(false, "[[1,2],[3]]")]
    [InlineData(true, "[[1,2],[3]]")]
    public void RaggedMatrix_IsRejectedWithRowAndWidth(bool useDecimal, string json)
    {
        using var document = JsonDocument.Parse(json);
        var error = Assert.Throws<JsonException>(() =>
        {
            if (useDecimal) { GetDecimal2D(document.RootElement); }
            else { GetDouble2D(document.RootElement); }
        });
        Assert.Contains("row 1", error.Message);
        Assert.Contains("expected 2", error.Message);
    }

    [Fact]
    public void RectangularAndEmptyMatrices_RetainAllDimensionsAndValues()
    {
        using var rectangular = JsonDocument.Parse("[[1,2],[3,999]]");
        var doubles = GetDouble2D(rectangular.RootElement);
        var decimals = GetDecimal2D(rectangular.RootElement);
        Assert.Equal(999, doubles[1, 1]);
        Assert.Equal(999m, decimals[1, 1]);
        using var empty = JsonDocument.Parse("[]");
        Assert.Empty(GetDouble2D(empty.RootElement));
        Assert.Empty(GetDecimal2D(empty.RootElement));
    }
}
