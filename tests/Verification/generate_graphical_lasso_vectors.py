#!/usr/bin/env python3
# Copyright (c) 2026 Pierre G. Boutquin. All rights reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
"""
Generate cross-language verification vectors for GraphicalLassoSolver<T>.

Uses scikit-learn's GraphicalLasso to produce reference precision matrices
from known sparse-precision Gaussian models.  Output written to
tests/Verification/vectors/graphical_lasso.json.
"""

import numpy as np
from sklearn.covariance import graphical_lasso
import sklearn
import json
import sys

from conftest import VECTORS_DIR, save_vector

def all_entry_glasso(sample: np.ndarray, penalty: float) -> tuple[np.ndarray, np.ndarray]:
    """Independent sklearn off-diagonal solver for the equivalent all-entry objective.

    SPD precision has positive diagonal, so rho*sum(abs(diag Theta))=rho*tr(Theta).
    Absorb that term into tr((S+rhoI)*Theta), retaining off-diagonal alpha=rho.
    """
    return graphical_lasso(sample + penalty * np.eye(sample.shape[0]),
                           alpha=penalty, max_iter=500, tol=1e-10, enet_tol=1e-12)


def mat_to_list(m: np.ndarray) -> list:
    """Convert 2-D numpy array to a row-major list of lists."""
    return m.tolist()


def generate() -> None:
    rng = np.random.default_rng(seed=42)

    # ─────────────────────────────────────────────────────────────────────────
    #  Case 1: 3×3 sparse precision model
    #
    #  True precision: [[2, -0.5, 0], [-0.5, 2, 0], [0, 0, 2]]
    #  True covariance = inv(Theta)
    # ─────────────────────────────────────────────────────────────────────────
    theta_true_3 = np.array([
        [ 2.0, -0.5,  0.0],
        [-0.5,  2.0,  0.0],
        [ 0.0,  0.0,  2.0],
    ])
    sigma_true_3 = np.linalg.inv(theta_true_3)

    # Generate samples
    n_samples = 500
    samples_3 = rng.multivariate_normal(np.zeros(3), sigma_true_3, size=n_samples)

    # Sample covariance (MLE = biased)
    sample_cov_3 = np.cov(samples_3.T, bias=True)

    # Fit sklearn GraphicalLasso
    lambda_3 = 0.05
    sklearn_covariance_3, sklearn_precision_3 = all_entry_glasso(sample_cov_3, lambda_3)

    # ─────────────────────────────────────────────────────────────────────────
    #  Case 2: 5×5 block-diagonal sparse precision
    #
    #  Block 1 (indices 0,1): strongly connected
    #  Block 2 (indices 2,3,4): weakly connected
    #  Cross-block: zero precision
    # ─────────────────────────────────────────────────────────────────────────
    theta_true_5 = np.array([
        [2.0, -0.8,  0.0,  0.0,  0.0],
        [-0.8, 2.0,  0.0,  0.0,  0.0],
        [0.0,  0.0,  2.0, -0.3,  0.0],
        [0.0,  0.0, -0.3,  2.0, -0.3],
        [0.0,  0.0,  0.0, -0.3,  2.0],
    ])
    sigma_true_5 = np.linalg.inv(theta_true_5)

    n_samples_5 = 1000
    samples_5 = rng.multivariate_normal(np.zeros(5), sigma_true_5, size=n_samples_5)
    sample_cov_5 = np.cov(samples_5.T, bias=True)

    lambda_5 = 0.08
    sklearn_covariance_5, sklearn_precision_5 = all_entry_glasso(sample_cov_5, lambda_5)

    # ─────────────────────────────────────────────────────────────────────────
    #  Case 3: Large-λ diagonal test  (3×3 dense covariance)
    # ─────────────────────────────────────────────────────────────────────────
    sample_cov_dense_3 = np.array([
        [1.0, 0.8, 0.6],
        [0.8, 1.0, 0.7],
        [0.6, 0.7, 1.0],
    ])
    lambda_large = 10.0
    _, sklearn_precision_large = all_entry_glasso(sample_cov_dense_3, lambda_large)

    # ─────────────────────────────────────────────────────────────────────────
    #  Assemble JSON output
    # ─────────────────────────────────────────────────────────────────────────
    vectors = {
        "case_3x3_sparse": {
            "description": "3×3 sparse precision model (true theta has zero (0,2) and (1,2) entries)",
            "lambda": lambda_3,
            "n_samples": n_samples,
            "true_precision": mat_to_list(theta_true_3),
            "true_covariance": mat_to_list(sigma_true_3),
            "sample_covariance": mat_to_list(sample_cov_3),
            "sklearn_precision": mat_to_list(sklearn_precision_3),
            "sklearn_covariance": mat_to_list(sklearn_covariance_3),
        },
        "case_5x5_block_diagonal": {
            "description": "5×5 block-diagonal sparse precision model",
            "lambda": lambda_5,
            "n_samples": n_samples_5,
            "true_precision": mat_to_list(theta_true_5),
            "true_covariance": mat_to_list(sigma_true_5),
            "sample_covariance": mat_to_list(sample_cov_5),
            "sklearn_precision": mat_to_list(sklearn_precision_5),
            "sklearn_covariance": mat_to_list(sklearn_covariance_5),
        },
        "case_3x3_large_lambda": {
            "description": "3×3 dense covariance with very large λ → diagonal precision",
            "lambda": lambda_large,
            "sample_covariance": mat_to_list(sample_cov_dense_3),
            "sklearn_precision": mat_to_list(sklearn_precision_large),
        },
    }

    out_path = VECTORS_DIR / "graphical_lasso.json"
    refresh_existing_panels(vectors)

    print(f"Written {out_path}")


def refresh_existing_panels(vectors=None) -> None:
    """Keep every existing input covariance, update only all-entry expected solutions."""
    if vectors is None:
        vectors = json.loads((VECTORS_DIR / "graphical_lasso.json").read_text())
    for name, case in vectors.items():
        if not name.startswith("case_"):
            continue
        sample = np.asarray(case["sample_covariance"], dtype=float)
        covariance, precision = all_entry_glasso(sample, case["lambda"])
        case["sklearn_precision"] = precision.tolist()
        if "sklearn_covariance" in case:
            case["sklearn_covariance"] = covariance.tolist()
        if name == "case_3x3_sparse":
            grid = [case["lambda"], .1, .2]
            scores = []
            for penalty in grid:
                _, theta = all_entry_glasso(sample, penalty)
                threshold = np.max(np.abs(theta)) * 1e-10
                edges = np.count_nonzero(np.abs(theta - np.diag(np.diag(theta))) > threshold)
                # BIC evaluates the original empirical S, not the S+rhoI solver input.
                score = case["n_samples"] * (-np.linalg.slogdet(theta)[1] + np.sum(sample * theta)) + edges * np.log(case["n_samples"])
                scores.append(float(score))
            case["bic_grid"] = grid
            case["bic_scores_original_sample"] = scores
            case["bic_selected_lambda"] = grid[int(np.argmin(scores))]
    vectors["provenance"] = {
        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "objective": "all-entry L1 including diagonal; sklearn.graphical_lasso(S+rhoI,alpha=rho)",
        "equivalence": "For SPD Theta, diagonal abs sum equals traceTheta; tr((S+rhoI)Theta)+rho||Theta||1,off equals tr(STheta)+rho||Theta||1",
        "bic": "Original S; n-scaled Gaussian log likelihood, ordered offdiagonal edge count, first grid tie",
        "inputs": "Original checked-in sample covariance matrices retained; large-lambda case uses its exact covariance without synthetic resampling"
    }
    save_vector("graphical_lasso", vectors)


if __name__ == "__main__":
    if "--refresh-existing-panels" in sys.argv:
        refresh_existing_panels()
    else:
        generate()
