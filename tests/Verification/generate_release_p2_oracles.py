"""Exact-Fraction Box 1 replay for non-dyadic desired-position boundaries.

The percentile is the exact binary double supplied by the C# caller. Observations
are ascending integers, selected independently of production state. This extends
the paper-rational oracle to decisions missed by dyadic-percentile controls.
"""
from fractions import Fraction as F


def replay(p, count):
    p = F(p)
    q = [F(i) for i in range(1, 6)]
    n = [1, 2, 3, 4, 5]
    increments = [F(0), p / 2, p, (1 + p) / 2, F(1)]
    for seen in range(6, count + 1):
        observation = F(seen)
        if observation < q[0]:
            q[0], cell = observation, 0
        elif observation >= q[4]:
            q[4], cell = observation, 3
        else:
            cell = next(i for i in range(4) if q[i] <= observation < q[i + 1])
        for i in range(cell + 1, 5):
            n[i] += 1
        for i in range(1, 4):
            difference = 1 + (seen - 1) * increments[i] - n[i]
            if not (difference >= 1 and n[i + 1] - n[i] > 1
                    or difference <= -1 and n[i - 1] - n[i] < -1):
                continue
            d = 1 if difference > 0 else -1
            before, after = n[i] - n[i - 1], n[i + 1] - n[i]
            candidate = q[i] + F(d, before + after) * (
                (before + d) * (q[i + 1] - q[i]) / after
                + (after - d) * (q[i] - q[i - 1]) / before)
            if q[i - 1] < candidate < q[i + 1]:
                q[i] = candidate
            else:
                q[i] += F(d, n[i + d] - n[i]) * (q[i + d] - q[i])
            n[i] += d
    return n, q


if __name__ == '__main__':
    for p, count in [(0.1, 71), (0.05, 81), (0.9, 21), (0.3, 21), (.1, 1000), (.3, 1000)]:
        positions, heights = replay(p, count)
        print(p, count, positions, 'estimate=', float(heights[2]))
