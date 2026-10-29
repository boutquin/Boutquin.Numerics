"""Exact rational control for Jain–Chlamtac (1985), Table I / Box 1.

The explicit erratum sets the second initial observation to 0.15. Printed
intermediate arithmetic is imperfect; this evaluates the paper's equations
without rounding and prints the goldens used by PSquarePaperReplicationTests.
No C# implementation or binary floating-point recurrence is used.
"""

from fractions import Fraction


def main() -> None:
    observations = [
        Fraction(value)
        for value in (
            "0.02", "0.15", "0.74", "3.39", "0.83", "22.37", "10.15",
            "15.43", "38.62", "15.92", "34.60", "10.28", "1.47", "0.40",
            "0.05", "11.39", "0.27", "0.42", "0.09", "11.37",
        )
    ]
    heights = sorted(observations[:5])
    positions = [1, 2, 3, 4, 5]
    desired = [Fraction(value) for value in positions]
    increments = [Fraction(value, 4) for value in range(5)]
    medians = []
    for observation in observations[5:]:
        if observation < heights[0]:
            heights[0], cell = observation, 0
        elif observation >= heights[4]:
            heights[4], cell = observation, 3
        else:
            cell = next(i for i in range(4) if heights[i] <= observation < heights[i + 1])
        for i in range(cell + 1, 5):
            positions[i] += 1
        desired = [value + increment for value, increment in zip(desired, increments)]
        for i in range(1, 4):
            difference = desired[i] - positions[i]
            if not (
                difference >= 1 and positions[i + 1] - positions[i] > 1
                or difference <= -1 and positions[i - 1] - positions[i] < -1
            ):
                continue
            direction = 1 if difference > 0 else -1
            previous_gap = positions[i] - positions[i - 1]
            next_gap = positions[i + 1] - positions[i]
            candidate = heights[i] + Fraction(direction, previous_gap + next_gap) * (
                (previous_gap + direction) * (heights[i + 1] - heights[i]) / next_gap
                + (next_gap - direction) * (heights[i] - heights[i - 1]) / previous_gap
            )
            if heights[i - 1] < candidate < heights[i + 1]:
                heights[i] = candidate
            else:
                heights[i] += Fraction(direction, positions[i + direction] - positions[i]) * (
                    heights[i + direction] - heights[i]
                )
            positions[i] += direction
        medians.append(float(heights[2]))

    print("Medians:", medians)
    print("Positions:", positions)
    print("Heights:", [float(value) for value in heights])


if __name__ == "__main__":
    main()
