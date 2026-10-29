"""Lower-tail-dependence reference vectors."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from conftest import save_vector

SEED = 42
N = 1000
rng = np.random.default_rng(SEED)

def pseudo_obs(x):
    n = len(x)
    ranks = np.argsort(np.argsort(x)) + 1
    return ranks / (n + 1)

def lower_tail_lambda(x, y, q=0.05):
    u = pseudo_obs(x)
    v = pseudo_obs(y)
    joint = np.sum((u <= q) & (v <= q))
    marginal = np.sum(u <= q)
    return float(joint / marginal) if marginal > 0 else 0.0

x_mono = np.arange(1, N+1, dtype=float)
y_mono = np.arange(1, N+1, dtype=float)
lam_mono = lower_tail_lambda(x_mono, y_mono)

x_ind = rng.uniform(0, 1, N)
y_ind = rng.uniform(0, 1, N)
lam_ind = lower_tail_lambda(x_ind, y_ind)

def generate() -> None:
    save_vector("lower_tail", {
        "n": N,
        "seed": SEED,
        "monotone_lambda": float(lam_mono),
        "independent_lambda": float(lam_ind),
        "independent_x": x_ind.tolist(),
        "independent_y": y_ind.tolist(),
    })
    print(f"Monotone lambda: {lam_mono:.6f}")
    print(f"Independent lambda: {lam_ind:.6f}")


if __name__ == "__main__":
    generate()
