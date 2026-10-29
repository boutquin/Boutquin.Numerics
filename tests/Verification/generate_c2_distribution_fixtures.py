"""Emit independent fixture literals, never reading production implementation.

Usage: generate_c2_distribution_fixtures.py <source-dir>, where <source-dir> holds
laikov-2025-v1.c (the retained Laikov v1 ferf coefficients) and
acklam-2000-inverse-normal.txt (the Acklam citation transcription). Expected function
values use mpmath 1.3.0 at 400 digits.
The output retains the historical Laikov coefficient experiment and Acklam
reference literals. Production CDF accuracy is now tested against high-precision
Gaussian integrals by generate_distribution_vectors.py, not Laikov's coefficients.
"""
from pathlib import Path
import sys
import math
import re
import struct
import mpmath as mp

mp.mp.dps = 400
if len(sys.argv) != 2:
    sys.exit(__doc__)
source_dir = Path(sys.argv[1])
author = (source_dir / 'laikov-2025-v1.c').read_text()
ferf = author.split('double ferf (double x)')[1].split('double gerf')[0]
coefficients = re.findall(r'0x[0-9a-f.]+p[+-][0-9]+', ferf)
assert len(coefficients) == 17
for i, value in enumerate(coefficients):
    bits = struct.unpack('>Q', struct.pack('>d', float.fromhex(value)))[0]
    print(f'[InlineData("s_c{i:02}", 0x{bits:X}L)]')
acklam = (source_dir / 'acklam-2000-inverse-normal.txt').read_text()
for name, index, value in re.findall(r'^  ([abcd])\[([0-9])\] =\s*([+-]?[0-9.]+e[+-][0-9]+)', acklam, re.M):
    print(f'[InlineData("{name.upper()}{index}", {value})]')
for x in [0.01, 0.1, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
    print(f'[InlineData({x!r}, {float(mp.erf(mp.mpf(x)))!r})]')
for p in [math.nextafter(.02425, 0), .02425, math.nextafter(.02425, 1), .1, .3,
          math.nextafter(.97575, 0), .97575, math.nextafter(.97575, 1), 1e-20, 1e-8]:
    quantile = mp.sqrt(2) * mp.erfinv(2 * mp.mpf(p) - 1)
    print(f'[InlineData({p!r}, {float(quantile)!r})]')
print('CDF(-38) =', mp.nstr(mp.erfc(mp.mpf(38)/mp.sqrt(2))/2, 80))
print('Inverse(double.Epsilon) =', mp.nstr(mp.sqrt(2)*mp.erfinv(2*mp.mpf(float.fromhex('0x0.0000000000001p-1022'))-1), 80))
