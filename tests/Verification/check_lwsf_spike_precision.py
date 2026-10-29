"""Re-evaluate the retained single-factor spike observation with converged oracle precision.

No production estimator is imported. The published Ledoit-Wolf 2003 moment
formula (author covMarket.R at e3478ca1f32bdd8fb4c4d6bb7fc1ef4bdebb63c1)
is evaluated on the exact binary input values at four precisions. The panel
replays the retained investigator's Random(12) stream, including 120 preceding
nonzero integer draws. Requires mpmath 1.3.0.
"""

import math
import random
import sys

import mpmath as mp


def panel():
    random_stream = random.Random(12)

    def nonzero():
        value = 0
        while value == 0:
            value = random_stream.randint(-2**20, 2**20)
        return value

    for _ in range(120):
        nonzero()
    return [[math.ldexp(1, 490) if row == 4 and column == 0
             else math.ldexp(nonzero(), -480) for column in range(3)]
            for row in range(20)]


def reference(rows, digits):
    mp.mp.dps = digits
    count, dimension = len(rows), len(rows[0])
    exact = [[mp.mpf(value) for value in row] for row in rows]
    means = [mp.fsum(row[j] for row in exact) / count for j in range(dimension)]
    centered = [[row[j] - means[j] for j in range(dimension)] for row in exact]
    market = [mp.fsum(row) / dimension for row in centered]
    sample = [[mp.fsum(row[i] * row[j] for row in centered) / count
               for j in range(dimension)] for i in range(dimension)]
    market_variance = mp.fsum(value**2 for value in market) / count
    market_covariance = [mp.fsum(centered[k][i] * market[k] for k in range(count)) / count
                         for i in range(dimension)]
    beta = [value / market_variance for value in market_covariance]
    target = [[sample[i][i] if i == j else market_covariance[i] * market_covariance[j] / market_variance
               for j in range(dimension)] for i in range(dimension)]
    p_terms, r, dispersion = [], mp.mpf(0), mp.mpf(0)
    for i in range(dimension):
        for j in range(dimension):
            if i == j:
                continue  # p_ii-r_ii cancels exactly.
            p_terms.append(mp.fsum((row[i] * row[j] - sample[i][j])**2 for row in centered) / count)
            pair_r = mp.fsum((beta[j] * centered[k][i] + beta[i] * centered[k][j]
                          - beta[i] * beta[j] * market[k]) * market[k]
                         * centered[k][i] * centered[k][j] for k in range(count)) / count
            r += pair_r - target[i][j] * sample[i][j]
            dispersion += (target[i][j] - sample[i][j])**2
    p = mp.fsum(p_terms)
    intensity = max(mp.mpf(0), min(mp.mpf(1), (p - r) / dispersion / count))
    entry = (intensity * target[1][2] + (1 - intensity) * sample[1][2]) * count / (count - 1)
    return intensity, entry


def generate(write_vector=True):
    rows = panel()
    results = []
    for digits in [120, 350, 700, 1000]:
        intensity, entry = reference(rows, digits)
        print(f"digits={digits} intensity={mp.nstr(intensity, 30)} covariance12={mp.nstr(entry, 40)}")
        results.append((intensity, entry))
    # At 700 and 1000 digits both the selected intensity and the small output agree.
    assert results[-2][0] == results[-1][0] == 0
    assert abs(results[-2][1] / results[-1][1] - 1) < mp.mpf("1e-600")
    assert results[0][0] == 1  # The historical 120-digit oracle took the opposite branch.
    if write_vector:
        from conftest import save_vector

        output = {
            "source": "Ledoit-Wolf 2003 single-factor moment formula, converged precision check",
            "mpmath_version": mp.__version__,
            "oracle_digits": 1000,
            "returns": rows,
            "covariance_1_2": float(results[-1][1]),
        }
        save_vector("lwsf_spike_precision", output)


if __name__ == "__main__":
    generate(write_vector="--write-vector" in sys.argv)
