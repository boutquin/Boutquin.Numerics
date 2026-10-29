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
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class C3ResamplingSobolAuthorTests
{
    [Fact]
    public void Sobol_AllDimensions_MatchNativeJoeKuoVectors()
    {
        // Native Joe-Kuo 21-Oct-2008 sobol.cc, unmodified, table new-6.21201.
        // Exact binary fractions below are native output multiplied by 512.
        // Includes every initial direction bit and two recurrence bits for maximum degree7.
        int[] indices = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256];
        int[][] numerators =
        [
            [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256],
            [384, 128, 128, 128, 384, 384, 128, 384, 384, 384, 384, 384, 128, 128, 384, 128, 384, 128, 384, 128, 128, 384, 128, 128, 128, 384, 128, 384, 128, 384, 128, 128],
            [128, 384, 384, 384, 128, 128, 384, 128, 128, 128, 128, 128, 384, 384, 128, 384, 128, 384, 128, 384, 384, 128, 384, 384, 384, 128, 384, 128, 384, 128, 384, 384],
            [192, 192, 320, 448, 192, 64, 192, 448, 448, 320, 448, 192, 192, 320, 192, 448, 192, 448, 448, 64, 64, 64, 192, 448, 448, 448, 192, 320, 64, 64, 192, 64],
            [64, 320, 192, 64, 64, 192, 320, 320, 320, 448, 320, 64, 320, 192, 64, 64, 64, 64, 320, 448, 448, 192, 320, 64, 64, 320, 320, 448, 448, 192, 320, 448],
            [96, 160, 480, 224, 288, 160, 224, 480, 480, 160, 352, 32, 480, 480, 416, 480, 416, 416, 480, 160, 96, 352, 96, 480, 224, 32, 288, 96, 352, 480, 224, 416],
            [32, 480, 288, 160, 352, 96, 416, 160, 160, 352, 32, 96, 160, 288, 480, 416, 480, 480, 160, 352, 416, 416, 288, 416, 160, 352, 96, 416, 160, 288, 416, 96],
            [48, 240, 240, 336, 144, 496, 272, 432, 240, 80, 48, 208, 336, 336, 176, 16, 400, 304, 400, 16, 368, 464, 304, 304, 272, 80, 80, 432, 112, 176, 208, 176],
            [16, 272, 464, 496, 496, 400, 176, 272, 80, 304, 16, 176, 496, 112, 336, 432, 112, 208, 304, 368, 208, 112, 16, 144, 432, 304, 48, 16, 208, 400, 368, 208],
            [24, 136, 360, 280, 72, 472, 408, 344, 504, 24, 200, 488, 232, 504, 504, 56, 88, 24, 216, 264, 168, 360, 216, 392, 88, 120, 88, 136, 72, 376, 376, 264],
            [8, 408, 184, 232, 440, 72, 296, 72, 424, 296, 216, 344, 280, 392, 168, 392, 40, 200, 488, 120, 120, 280, 200, 280, 488, 328, 104, 152, 152, 232, 8, 472],
            [12, 204, 420, 428, 332, 228, 340, 108, 380, 60, 484, 116, 220, 268, 412, 260, 452, 164, 180, 484, 364, 484, 28, 180, 404, 444, 380, 156, 116, 76, 212, 412],
            [4, 340, 284, 324, 244, 172, 124, 36, 212, 276, 316, 300, 452, 132, 308, 140, 492, 108, 348, 412, 276, 252, 212, 428, 124, 244, 276, 4, 236, 164, 220, 68],
            [6, 170, 150, 98, 462, 346, 442, 126, 54, 418, 378, 378, 418, 450, 14, 390, 290, 182, 42, 286, 438, 30, 454, 478, 342, 350, 46, 366, 258, 162, 498, 18],
            [2, 510, 394, 294, 314, 502, 454, 90, 226, 182, 70, 86, 102, 326, 314, 266, 206, 218, 374, 130, 162, 226, 274, 114, 298, 426, 314, 362, 494, 6, 302, 86],
            [3, 255, 347, 151, 399, 55, 187, 183, 147, 209, 233, 511, 121, 165, 263, 65, 465, 89, 185, 193, 497, 371, 89, 473, 53, 431, 501, 263, 387, 465, 203, 59]
        ];
        var sequence = new SobolSequence<double>(32, skip: 1);
        var selected = 0;
        for (var i = 1; i <= 256; i++)
        {
            var actual = sequence.Next();
            if (i != indices[selected])
            {
                continue;
            }
            for (var d = 0; d < 32; d++)
            {
                actual[d].Should().Be(numerators[selected][d] / 512.0, "author index {0}, dimension {1}", i, d + 1);
            }
            selected++;
        }
    }
}
