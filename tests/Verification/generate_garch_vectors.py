"""GARCH(1,1) reference vectors — pure-numpy DGP + scipy QMLE cross-check.

Convention: h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}
(h_t uses the PREVIOUS period's shock; both presample states use the
candidate residual second moment/n, Bollerslev1986 footnote5).
The reported QMLE objective is the mean per observation (equation18).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from conftest import save_vector

SEED = 42
N = 2000
TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA, TRUE_MU = 1e-5, 0.10, 0.85, 0.0

rng = np.random.default_rng(SEED)

# Simulate GARCH(1,1) using the same h_t = omega + alpha*eps_{t-1}^2 + beta*h_{t-1} convention
h_unc = TRUE_OMEGA / (1 - TRUE_ALPHA - TRUE_BETA)
h = h_unc
eps_prev = 0.0
returns = np.zeros(N)
h_vals = np.zeros(N)  # h_t is the variance for period t

for t in range(N):
    # h_t uses the previous eps
    h = TRUE_OMEGA + TRUE_ALPHA * eps_prev**2 + TRUE_BETA * h
    h_vals[t] = h
    z = rng.standard_normal()
    eps = np.sqrt(h) * z
    returns[t] = TRUE_MU + eps
    eps_prev = eps

# ── Fit via scipy Nelder-Mead (same reparameterisation as C#) ─────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def neg_qll(p):
    mu, ln_om, logit_ab, logit_af = p
    omega = np.exp(ln_om)
    ab    = sigmoid(logit_ab)
    alpha = ab * sigmoid(logit_af)
    beta  = ab - alpha
    h     = float(np.mean((returns - mu) ** 2))
    eps_squared_inner = h
    ll    = 0.0
    for t in range(N):
        h = omega + alpha * eps_squared_inner + beta * h
        if h <= 0:
            return 1e30
        eps = returns[t] - mu
        ll += 0.5 * np.log(h) + 0.5 * eps**2 / h
        eps_squared_inner = eps * eps
    return ll / N

alpha0, beta0 = 0.05, 0.90
sv = np.var(returns, ddof=0)
omega0 = (1 - alpha0 - beta0) * sv
x0 = [
    np.mean(returns),
    np.log(omega0),
    np.log((alpha0 + beta0) / (1 - alpha0 - beta0)),
    np.log((alpha0 / (alpha0 + beta0)) / (1 - alpha0 / (alpha0 + beta0))),
]

# Two-phase optimisation matching C# approach
res_coarse = minimize(neg_qll, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
res = minimize(neg_qll, res_coarse.x, method='Nelder-Mead',
               options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})


def validate_reference_fit(result, observations):
    """Reject invalid reference solves and check a local optimum independently.

    This validates the selected constant-mean, residual-second-moment/n
    presample convention for both variance and squared innovation. It does not reproduce Hull's spreadsheet seed or
    prove a global optimum of the nonconvex Gaussian QMLE likelihood.
    The validator evaluates the likelihood independently of ``neg_qll``.
    """
    if not result.success:
        raise AssertionError(f"GARCH reference solve failed: {result.message}")
    parameters = np.asarray(result.x, dtype=float)
    if parameters.shape != (4,) or not np.all(np.isfinite(parameters)):
        raise AssertionError("GARCH reference parameters must be four finite values")
    if not np.isfinite(result.fun):
        raise AssertionError("GARCH reference objective must be finite")
    data = np.asarray(observations, dtype=float)
    if data.ndim != 1 or len(data) < 2 or not np.all(np.isfinite(data)):
        raise AssertionError("GARCH reference observations must be finite")
    initial_variance = float(np.var(data, ddof=0))
    if not np.isfinite(initial_variance) or initial_variance <= 0:
        raise AssertionError("GARCH presample variance must be finite and positive")

    def independent_objective(point):
        mean, log_omega, persistence_logit, fraction_logit = point
        omega = float(np.exp(log_omega))
        persistence = float(expit(persistence_logit))
        alpha = persistence * float(expit(fraction_logit))
        beta = persistence - alpha
        if not (np.isfinite(omega) and omega > 0 and alpha >= 0 and beta >= 0
                and alpha + beta < 1):
            raise AssertionError("GARCH reference parameters violate the stationary domain")
        innovations = data - mean
        # Both presample states equal mean squared candidate innovations.
        filtered = np.empty(len(data), dtype=float)
        filtered[0] = omega + (alpha + beta) * float(np.mean(innovations ** 2))
        for index in range(1, len(data)):
            filtered[index] = (omega + alpha * innovations[index - 1] ** 2
                               + beta * filtered[index - 1])
        if not np.all(np.isfinite(filtered)) or not np.all(filtered > 0):
            raise AssertionError("GARCH reference filtered variances are not feasible")
        return float(np.mean((np.log(filtered) + innovations ** 2 / filtered) / 2))

    value = independent_objective(parameters)
    if not np.isfinite(value) or not np.isclose(value, result.fun,
                                                rtol=1e-10, atol=1e-8):
        raise AssertionError("Independent GARCH likelihood disagrees with solver objective")
    # Scale the mean coordinate by sample SD; the other three are dimensionless.
    coordinate_scale = np.array([np.sqrt(initial_variance), 1.0, 1.0, 1.0])
    max_improvement = 0.0
    max_scaled_gradient = 0.0
    for radius in (1e-4, 1e-3):
        for coordinate in range(4):
            offset = np.zeros(4)
            offset[coordinate] = radius * coordinate_scale[coordinate]
            plus = independent_objective(parameters + offset)
            minus = independent_objective(parameters - offset)
            max_improvement = max(max_improvement, value - plus, value - minus)
            max_scaled_gradient = max(max_scaled_gradient, abs(plus - minus) / (2 * radius))
    if max_improvement > 1e-7 or max_scaled_gradient > 1e-4:
        raise AssertionError("GARCH reference fit fails independent local optimality checks: "
                             f"improvement={max_improvement}, gradient={max_scaled_gradient}")
    return {
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "objective_mean": float(result.fun),
        "independent_objective_mean": value,
        "max_neighborhood_improvement": max_improvement,
        "max_scaled_central_gradient": max_scaled_gradient,
        "neighborhood_radii": [1e-4, 1e-3],
        "scope": "local optimum only; selected constant-mean and presample convention",
    }


fit_validation = validate_reference_fit(res, returns)

p = res.x
mu_fit    = p[0]
omega_fit = np.exp(p[1])
ab_fit    = sigmoid(p[2])
alpha_fit = ab_fit * sigmoid(p[3])
beta_fit  = ab_fit - alpha_fit

# ── Compute filtered h_t and standardised residuals ───────────────────────────
h = float(np.mean((returns - mu_fit) ** 2))
h_arr = np.zeros(N)
s_arr = np.zeros(N)
eps_squared_filt = h
for t in range(N):
    h = max(omega_fit + alpha_fit * eps_squared_filt + beta_fit * h, 1e-30)
    h_arr[t] = h
    eps = returns[t] - mu_fit
    s_arr[t] = eps / np.sqrt(h)
    eps_squared_filt = eps * eps

def generate() -> None:
    save_vector("garch", {
        "presample_contract": "candidate residual second moment/n for variance and squared innovation",
        "likelihood_scale": "mean per observation, Gaussian additive constant omitted",
        "seed": SEED,
        "n": N,
        "true_mu": TRUE_MU,
        "true_omega": TRUE_OMEGA,
        "true_alpha": TRUE_ALPHA,
        "true_beta": TRUE_BETA,
        "returns": returns.tolist(),
        "mu_fit": float(mu_fit),
        "omega_fit": float(omega_fit),
        "alpha_fit": float(alpha_fit),
        "beta_fit": float(beta_fit),
        "conditional_variances": h_arr.tolist(),
        "std_residuals": s_arr.tolist(),
        "fit_validation": fit_validation,
    })

    print(f"GARCH fit: mu={mu_fit:.6f}, omega={omega_fit:.2e}, alpha={alpha_fit:.4f}, beta={beta_fit:.4f}")
    print(f"alpha+beta={alpha_fit+beta_fit:.6f}")
    print("Saved to vectors/garch.json")


if __name__ == "__main__":
    generate()
