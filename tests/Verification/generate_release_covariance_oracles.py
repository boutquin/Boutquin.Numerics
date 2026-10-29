"""Independent release covariance anchors: exact rational spectral projectors and 100-digit QIS.

Run with Python 3 and mpmath 1.3.0. No production library, covariance generator,
random stream, or eigensolver is imported. Output is printed for inspection.
"""

from fractions import Fraction as F
from itertools import combinations

import mpmath as mp


def nercome():
    raw = [(-248, -2874), (-863, -679), (-2934, -2090), (3708, 2261),
           (-802, 1494), (-1702, 2614), (687, 1887)]
    rows = [(F(a * 10**7), F(b, 10**11)) for a, b in raw]
    for size in [5, 1, 2, 4, 3]:
        subsets = list(combinations(range(7), size))
        estimate, residual = [F(0)] * 3, [F(0)] * 3
        for first in subsets:
            second = [k for k in range(7) if k not in first]

            def moment(indices):
                return [sum(rows[k][i] * rows[k][j] for k in indices) / len(indices)
                        for i, j in [(0, 0), (0, 1), (1, 1)]]

            a, b, d = moment(first)
            x, y, z = moment(second)
            h, q = a - d, 2 * b
            # Spectral projection B -> tr(B)/2 I + tr(HB)/(2*gap²) H.
            coefficient = (h * (x - z) + 2 * q * y) / (2 * (h * h + q * q))
            projected = [(x + z) / 2 + coefficient * h, coefficient * q,
                         (x + z) / 2 - coefficient * h]
            for j, entry in enumerate(projected):
                estimate[j] += entry / len(subsets)
                residual[j] += (entry - [x, y, z][j]) / len(subsets)
        score = residual[0]**2 + 2 * residual[1]**2 + residual[2]**2
        print("NERCOME", size, "score", float(score), "estimate", [float(v) for v in estimate])


def qis():
    mp.mp.dps = 100
    p, n = 3, 5
    c = mp.mpf(p) / n
    h = min(c**2, c**-2)**mp.mpf(".35") / mp.mpf(p)**mp.mpf(".35")
    eigenvalues = [mp.mpf("4e-15"), mp.mpf(".1"), mp.mpf(".4")]
    inverses = [1 / value for value in eigenvalues]
    delta = []
    for own in inverses:
        theta = sum(other * (other - own) / ((other - own)**2 + h**2 * other**2)
                    for other in inverses) / p
        conjugate = sum(h * other**2 / ((other - own)**2 + h**2 * other**2)
                        for other in inverses) / p
        delta.append(1 / (own * ((1 - c)**2 + 2*c*(1 - c)*theta
                                 + c**2*(theta**2 + conjugate**2))))
    print("QIS", [mp.nstr(value * sum(eigenvalues) / sum(delta), 80) for value in reversed(delta)])


if __name__ == "__main__":
    nercome()
    qis()
