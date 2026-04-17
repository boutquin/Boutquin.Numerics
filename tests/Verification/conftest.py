"""
Shared fixtures and constants for Boutquin.Numerics cross-language
verification tests.

These tests validate C# numerical kernels against independent Python
reference implementations using numpy, scipy, statsmodels, and
scikit-learn.

Run ``python generate_vectors.py`` (or individual ``generate_*.py``
scripts) first to populate ``vectors/`` before executing ``pytest``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Precision tiers — match the tolerance to the calculation family
# ---------------------------------------------------------------------------
PRECISION_EXACT = 1e-10
PRECISION_NUMERIC = 1e-6
PRECISION_STATISTICAL = 1e-4


VECTORS_DIR = Path(__file__).parent / "vectors"


def save_vector(name: str, data: dict) -> None:
    """Serialize a vector dict as JSON into ``vectors/<name>.json``."""

    VECTORS_DIR.mkdir(exist_ok=True)
    path = VECTORS_DIR / f"{name}.json"

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        raise TypeError(f"Cannot serialize {type(obj)!r}")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=convert)


def load_vector(name: str) -> dict:
    path = VECTORS_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def distribution_vectors() -> dict:
    return load_vector("distributions")


@pytest.fixture(scope="session")
def solver_vectors() -> dict:
    return load_vector("solvers")


@pytest.fixture(scope="session")
def interpolation_vectors() -> dict:
    return load_vector("interpolation")


@pytest.fixture(scope="session")
def linalg_vectors() -> dict:
    return load_vector("linalg")


@pytest.fixture(scope="session")
def covariance_vectors() -> dict:
    return load_vector("covariance")


@pytest.fixture(scope="session")
def correlation_vectors() -> dict:
    return load_vector("correlation")


@pytest.fixture(scope="session")
def dsr_vectors() -> dict:
    return load_vector("dsr")


@pytest.fixture(scope="session")
def bootstrap_vectors() -> dict:
    return load_vector("bootstrap")


@pytest.fixture(scope="session")
def qmc_vectors() -> dict:
    return load_vector("qmc")


@pytest.fixture(scope="session")
def psd_vectors() -> dict:
    return load_vector("psd")


@pytest.fixture(scope="session")
def ols_vectors() -> dict:
    return load_vector("ols")


@pytest.fixture(scope="session")
def lm_vectors() -> dict:
    return load_vector("lm")


@pytest.fixture(scope="session")
def scalar_stats_vectors() -> dict:
    return load_vector("scalar_stats")


@pytest.fixture(scope="session")
def sample_moments_vectors() -> dict:
    return load_vector("sample_moments")


@pytest.fixture(scope="session")
def qp_solver_vectors() -> dict:
    return load_vector("qp_solver")
