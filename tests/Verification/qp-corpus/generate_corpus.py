#!/usr/bin/env python3
"""Deterministic ActiveSetQpSolver regression corpus generator.

Writes tests/Verification/qp-corpus/corpus-double.json and corpus-decimal.json.

Every case is a long-only-or-boxed, fully-invested problem
    maximise  mu'w - (lambda/2) w'Sigma w     s.t.  1'w = 1,  lb <= w <= ub
(MinVariance is the special case mu = 0, lambda = 2).  The exact optimum comes from an
exact-rational (fractions.Fraction) active-set method whose result is then certified by the full
KKT conditions, which shares no code with the solver under test.  Every case records:

  pattern        active set of the optimum: '-' at lb, '+' at ub, '0' free
  optimum        the exact optimum, rounded to 60 significant digits (the C# unit test recomputes it
                 with a BigInteger rational oracle and compares)
  in_domain      all inputs in the operating domain (double: 1e-150 <= |x| <= 1e150 or 0;
                 decimal: |x| <= 1e14)
  representable  the exact optimum is representable in the working type
  cond1          exact 1-norm condition number of the free block Sigma_FF (unscaled; informational). The free
                 block F is the block that decides the optimum: the variables strictly inside the box at the exact
                 optimum (a variable left free by the oracle but sitting exactly on a bound is not in it)
  cond_eq        exact 1-norm condition number of the free block of the EQUILIBRATED covariance
                 D^-1 Sigma D^-1, D = diag(radix^e_i) with the exponents e_i of the covariance-only scaling
                 QpScaling<T>.EquilibrateCovariance, the block the solver's conditioning guard measures (the
                 radix power of sqrt(Sigma_ii), moved toward zero wherever a scaled covariance entry would not be
                 exact; the bounds and means take no part; see qp_scaling_exponents)
  qualified      in_domain and representable and 48 * n * u * cond_eq < 1   (n = number of assets;
                 u = 2^-53 double, 1e-28 decimal)
  sensitivity    one-ulp input sensitivity: max over single-entry +/-1 ulp perturbations of the
                 inputs (symmetric covariance pairs, bounds, means, lambda) of
                 max|w*(p) - w*| / max|w*|, evaluated exactly.  A decimal ulp is one unit of the
                 28th significant digit (at least 1e-28).  Perturbations that make the covariance
                 not positive definite are skipped and counted in sensitivity_skipped.

The seeded families are ports of the code-arena probe generators (R4-CH-SOLV-qp / R5-QP-families) and,
for nearq / nearo / degenb, of the held-out census harness (a block near the conditioning boundary built
exactly and rounded once into the working type; the same with its near-null direction orthogonal to the
budget row; a bound variable whose exact multiplier is about zero and whose addition makes the free block
near singular). The near-boundary families draw the block's relative curvature over a range that spans both
the old (4u.cond1) and the current (48.n.u.cond_eq) qualification boundaries; nearmo (the held-out family of
verification batch VB-M-3) is the nearo shape with that range moved to straddle the current margin, so that its
qualified cases sit near the top of the qualified band. The named witnesses keep the
ids they carry in the arena record; the held-out witnesses are explicit matrices as the census harness
dumped them.

usage: python3 generate_corpus.py [OUTDIR]        (default: the directory of this script)
"""
import itertools
import json
import math
import multiprocessing
import os
import random
import sys
from decimal import Context, Decimal
from fractions import Fraction as Fr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atomic_output import atomic_text

DMAX = Fr(1.7976931348623157e308)
DECMAX = Fr(79228162514264337593543950335)
U_DOUBLE = Fr(1, 2 ** 53)
U_DECIMAL = Fr(1, 10 ** 28)
CORPUS_CAP = 2300

# --------------------------------------------------------------------------------------------
# Exact oracle
# --------------------------------------------------------------------------------------------


def solve_lin(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c] != 0), None)
        if p is None:
            return None
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c] / piv
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def exact_pd(Sg):
    n = len(Sg)
    A = [row[:] for row in Sg]
    for k in range(n):
        if A[k][k] <= 0:
            return False
        for i in range(k + 1, n):
            f = A[i][k] / A[k][k]
            for j in range(k + 1, n):
                A[i][j] -= f * A[k][j]
    return True


def free_solve(Sg, mu, lam, lb, ub, status):
    """Equality-constrained solve on the free set; returns (w, nu) or None (no free variable / singular)."""
    n = len(mu)
    F = [i for i in range(n) if status[i] == 0]
    if not F:
        return None
    wB = {i: (lb if status[i] == -1 else ub) for i in range(n) if status[i] != 0}
    k = len(F)
    A = [[lam * Sg[F[r]][F[c]] for c in range(k)] + [Fr(1)] for r in range(k)] + [[Fr(1)] * k + [Fr(0)]]
    rhs = [mu[F[r]] - sum(lam * Sg[F[r]][j] * wj for j, wj in wB.items()) for r in range(k)] \
        + [Fr(1) - sum(wB.values())]
    x = solve_lin(A, rhs)
    if x is None:
        return None
    w = [None] * n
    for r, i in enumerate(F):
        w[i] = x[r]
    for i, v in wB.items():
        w[i] = v
    return w, x[k]


def kkt_for(Sg, mu, lam, lb, ub, status):
    """w if the pattern satisfies the full KKT conditions exactly, else None."""
    n = len(mu)
    if all(s != 0 for s in status):
        w = [lb if s == -1 else ub for s in status]
        if sum(w) != 1:
            return None
        g = [mu[i] - lam * sum(Sg[i][j] * w[j] for j in range(n)) for i in range(n)]
        lo = max([g[i] for i in range(n) if status[i] == -1], default=None)
        hi = min([g[i] for i in range(n) if status[i] == 1], default=None)
        if lo is not None and hi is not None and lo > hi:
            return None
        return w
    sol = free_solve(Sg, mu, lam, lb, ub, status)
    if sol is None:
        return None
    w, nu = sol
    for i in range(n):
        if status[i] == 0:
            if w[i] < lb or w[i] > ub:
                return None
            continue
        g = mu[i] - lam * sum(Sg[i][j] * w[j] for j in range(n)) - nu
        if status[i] == -1 and g > 0:
            return None
        if status[i] == 1 and g < 0:
            return None
    return w


def active_set(Sg, mu, lam, lb, ub, max_iter=None):
    """Exact primal active-set method (Bland's rule) from the feasible point w = 1/n.  Returns the
    pattern or None; the caller certifies the pattern with kkt_for."""
    n = len(mu)
    if n * lb > 1 or n * ub < 1:
        return None
    w = [Fr(1, n)] * n
    status = [0] * n
    for i in range(n):
        if w[i] == lb:
            status[i] = -1
        elif w[i] == ub:
            status[i] = 1
    max_iter = max_iter or 20 * (n + 5) ** 2
    for _ in range(max_iter):
        if all(s != 0 for s in status):
            tgt = [lb if s == -1 else ub for s in status]
            w = tgt
            g = [mu[i] - lam * sum(Sg[i][j] * w[j] for j in range(n)) for i in range(n)]
            lo = max([g[i] for i in range(n) if status[i] == -1], default=None)
            hi = min([g[i] for i in range(n) if status[i] == 1], default=None)
            # release the lowest-index violating variable (needs a multiplier nu in [lo, hi])
            if lo is not None and hi is not None and lo > hi:
                cand = [i for i in range(n) if (status[i] == -1 and g[i] > hi) or (status[i] == 1 and g[i] < lo)]
                if not cand:
                    return None
                status[min(cand)] = 0
                continue
            return status[:]
        sol = free_solve(Sg, mu, lam, lb, ub, status)
        if sol is None:
            return None
        wstar, nu = sol
        p = [wstar[i] - w[i] for i in range(n)]
        if all(x == 0 for x in p):
            viol = []
            for i in range(n):
                if status[i] == 0:
                    continue
                g = mu[i] - lam * sum(Sg[i][j] * w[j] for j in range(n)) - nu
                if (status[i] == -1 and g > 0) or (status[i] == 1 and g < 0):
                    viol.append(i)
            if not viol:
                return status[:]
            status[min(viol)] = 0
            continue
        alpha, block = Fr(1), None
        for i in range(n):
            if status[i] != 0 or p[i] == 0:
                continue
            a = ((lb - w[i]) if p[i] < 0 else (ub - w[i])) / p[i]
            if a < alpha:
                alpha, block = a, i
        w = [w[i] + alpha * p[i] for i in range(n)]
        if block is not None:
            status[block] = -1 if w[block] == lb else 1
            w[block] = lb if status[block] == -1 else ub
    return None


def oracle(Sg, mu, lam, lb, ub, hint=None):
    """(w, pattern) of the exact optimum, certified by exact KKT; None if infeasible."""
    n = len(mu)
    if hint is not None:
        w = kkt_for(Sg, mu, lam, lb, ub, hint)
        if w is not None:
            return w, tuple(hint)
    st = active_set(Sg, mu, lam, lb, ub)
    if st is not None:
        w = kkt_for(Sg, mu, lam, lb, ub, st)
        if w is not None:
            return w, tuple(st)
    if n <= 8:
        for st in sorted(itertools.product((0, -1, 1), repeat=n), key=lambda s: sum(1 for x in s if x)):
            w = kkt_for(Sg, mu, lam, lb, ub, st)
            if w is not None:
                return w, tuple(st)
    return None


# --------------------------------------------------------------------------------------------
# Case plumbing
# --------------------------------------------------------------------------------------------


def tofr(T, s):
    return Fr(s) if T == "decimal" else Fr(float(s))


def exact_inputs(cs):
    T = cs["T"]
    Sg = [[tofr(T, x) for x in row] for row in cs["cov"]]
    n = len(Sg)
    if cs["mode"] == "minvar":
        mu, lam = [Fr(0)] * n, Fr(2)
    else:
        mu = [tofr(T, x) for x in cs["mu"]]
        lam = tofr(T, cs["lambda"])
    return Sg, mu, lam, tofr(T, cs["lb"]), tofr(T, cs["ub"])


def in_domain(cs):
    T = cs["T"]
    vals = [x for row in cs["cov"] for x in row] + [cs["lb"], cs["ub"]]
    if cs["mode"] == "meanvar":
        vals += list(cs["mu"]) + [cs["lambda"]]
    for s in vals:
        v = abs(tofr(T, s))
        if v == 0:
            continue
        if T == "double" and not (Fr(1, 10 ** 150) <= v <= Fr(10 ** 150)):
            return False
        if T == "decimal" and v > Fr(10 ** 14):
            return False
    return True


def exact_in_type(cs):
    """Every decimal input must survive decimal.Parse unchanged: at most 28 fractional digits and a mantissa
    within decimal's 96 bits (the held-out census dumps carry 29-digit mantissas; every seeded family stays within
    28 significant digits)."""
    if cs["T"] != "decimal":
        return True
    vals = [x for row in cs["cov"] for x in row] + [cs["lb"], cs["ub"]]
    if cs["mode"] == "meanvar":
        vals += list(cs["mu"]) + [cs["lambda"]]
    for s in vals:
        d = Decimal(s)
        mant, scale = dec_parts(s)
        if d.as_tuple().exponent < -28 or mant > DEC_LIMIT or abs(Fr(d)) > DECMAX:
            return False
    return True


def lim_ok(T, wstar):
    lim = DMAX if T == "double" else DECMAX
    return all(abs(x) <= lim for x in wstar)


CTX60 = Context(prec=60, Emin=-999999, Emax=999999)


def frdec(x):
    if x == 0:
        return "0"
    return str(CTX60.divide(Decimal(x.numerator), Decimal(x.denominator)))


def one_norm(A):
    k = len(A)
    return max(sum(abs(A[i][j]) for i in range(k)) for j in range(k))


def cond1(Sf):
    k = len(Sf)
    if k == 0:
        return Fr(1)
    inv = []
    for c in range(k):
        e = [Fr(1) if r == c else Fr(0) for r in range(k)]
        col = solve_lin(Sf, e)
        inv.append(col)
    invm = [[inv[c][r] for c in range(k)] for r in range(k)]
    return one_norm(Sf) * one_norm(invm)


# --------------------------------------------------------------------------------------------
# Port of Internal/QpScaling<T>.EquilibrateCovariance's exponent choice (the scaling the qualification uses)
# --------------------------------------------------------------------------------------------

DEC_LIMIT = 2 ** 96 - 1
QUALIFICATION_C = 48


def dec_parts(s):
    """(mantissa, scale) of the decimal that decimal.Parse(s) produces (the scale keeps trailing zeros)."""
    sign, digits, exp = Decimal(s).as_tuple()
    mant = int("".join(map(str, digits))) if digits else 0
    if exp > 0:
        return mant * 10 ** exp, 0
    return mant, -exp


def shift_exact(T, s, k):
    """Whether value(s) * radix^k is formed without rounding (QpScaling.Shift(...).Exact)."""
    if k == 0:
        return True
    if T == "double":
        x = float(s)
        if x == 0.0:
            return True
        try:
            y = math.ldexp(x, k)
        except OverflowError:
            return False
        if math.isinf(y):
            return False
        return math.ldexp(y, -k) == x
    mant, scale = dec_parts(s)
    if mant == 0:
        return True
    target = scale - k
    if 0 <= target <= 28:
        return True
    if target < 0:
        return mant * 10 ** (-target) <= DEC_LIMIT
    while target > 28 and mant % 10 == 0:
        mant //= 10
        target -= 1
    return target <= 28


def sqrt_exponent(T, s):
    """ScaledArithmetic.ScaleExponent([Sqrt(x)]): the k with radix^(k-1) <= sqrt(x) < radix^k (0 unless x > 0)."""
    if T == "double":
        x = float(s)
        if not math.isfinite(x) or x <= 0:
            return 0
        return math.frexp(math.sqrt(x))[1]
    x = Fr(Decimal(s))
    if x <= 0:
        return 0
    k = 0
    while x >= Fr(100) ** k:
        k += 1
    while x < Fr(100) ** (k - 1):
        k -= 1
    return k


def qp_scaling_exponents(cs):
    """The radix exponents e_i of QpScaling<T>.EquilibrateCovariance(cov) (same loop, same order): the covariance-only
    scaling of design step 1 (spec amendment #11). Only the exactness of D^-1 Sigma D^-1 limits an exponent; the bounds
    and the means take no part."""
    T = cs["T"]
    cov = cs["cov"]
    n = len(cov)
    e = [sqrt_exponent(T, cov[i][i]) for i in range(n)]
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i, n):
                k = -(e[i] + e[j])
                if k != 0 and not (shift_exact(T, cov[i][j], k) and shift_exact(T, cov[j][i], k)):
                    if k < 0:
                        m = i if e[i] >= e[j] else j
                    else:
                        m = i if e[i] <= e[j] else j
                    e[m] += -1 if k < 0 else 1
                    changed = True
    return e


def cond_eq(cs, Sg, free):
    """Exact 1-norm condition number of the equilibrated free block D_F^-1 Sigma_FF D_F^-1."""
    e = qp_scaling_exponents(cs)
    radix = 2 if cs["T"] == "double" else 10
    return cond1([[Sg[i][j] / Fr(radix) ** (e[i] + e[j]) for j in free] for i in free])


def float_str(x):
    """repr of a Fraction as a float, tolerating magnitudes outside the float range."""
    if x == 0:
        return "0.0"
    try:
        return repr(float(x))
    except OverflowError:
        return "inf"


def ulp_perturbations(T, s):
    if T == "double":
        x = float(s)
        return [Fr(math.nextafter(x, math.inf)), Fr(math.nextafter(x, -math.inf))]
    d = Decimal(s)
    if d == 0:
        return [Fr(Decimal("1e-28")), Fr(Decimal("-1e-28"))]
    e = d.adjusted() - 27
    u = Fr(10) ** max(e, -28)
    # exact: a 29-digit input plus one unit does not fit the default 28-digit Decimal context
    return [Fr(d) + u, Fr(d) - u]


def enrich(cs):
    T = cs["T"]
    Sg, mu, lam, lb, ub = exact_inputs(cs)
    n = len(Sg)
    res = oracle(Sg, mu, lam, lb, ub)
    if res is None:
        raise RuntimeError("no exact optimum for " + cs["id"])
    w0, pat = res
    sc = max(abs(x) for x in w0)
    # condition number (exact) of the free block that decides the optimum: the variables strictly inside the box
    # (spec section 4 with design step 4's deciding block B = F*). A variable the oracle leaves free whose exact value
    # lands on a bound with a zero multiplier (degenerate at the bound) is not in it; at the exact optimum no variable
    # on a bound has a negative multiplier, so none joins.
    F = [i for i in range(n) if lb < w0[i] < ub]
    cond = cond1([[Sg[i][j] for j in F] for i in F])
    ceq = cond_eq(cs, Sg, F)
    u = U_DOUBLE if T == "double" else U_DECIMAL
    indom = in_domain(cs)
    rep = lim_ok(T, w0)
    qualified = bool(indom and rep and QUALIFICATION_C * n * u * ceq < 1)

    # one-ulp input sensitivity
    worst, skipped = Fr(0), 0

    def upd(S2, mu2, lam2, lb2, ub2):
        nonlocal worst, skipped
        if not exact_pd(S2):
            skipped += 1
            return
        r = oracle(S2, mu2, lam2, lb2, ub2, hint=pat)
        if r is None:
            skipped += 1
            return
        d = max(abs(a - b) for a, b in zip(r[0], w0)) / sc if sc else Fr(0)
        if d > worst:
            worst = d

    for i in range(n):
        for j in range(i, n):
            for nx in ulp_perturbations(T, cs["cov"][i][j]):
                S2 = [row[:] for row in Sg]
                S2[i][j] = S2[j][i] = nx
                upd(S2, mu, lam, lb, ub)
    for nx in ulp_perturbations(T, cs["lb"]):
        upd(Sg, mu, lam, nx, ub)
    for nx in ulp_perturbations(T, cs["ub"]):
        upd(Sg, mu, lam, lb, nx)
    if cs["mode"] == "meanvar":
        for i in range(n):
            for nx in ulp_perturbations(T, cs["mu"][i]):
                m2 = mu[:]
                m2[i] = nx
                upd(Sg, m2, lam, lb, ub)
        for nx in ulp_perturbations(T, cs["lambda"]):
            if nx > 0:
                upd(Sg, mu, nx, lb, ub)

    out = {
        "id": cs["id"], "family": cs["family"], "mode": cs["mode"],
        "cov": cs["cov"], "lb": cs["lb"], "ub": cs["ub"],
    }
    if cs["mode"] == "meanvar":
        out["mu"] = cs["mu"]
        out["lambda"] = cs["lambda"]
    out.update({
        "pattern": "".join("-" if s == -1 else "+" if s == 1 else "0" for s in pat),
        "optimum": [frdec(x) for x in w0],
        "in_domain": indom, "representable": rep,
        "cond1": float_str(cond), "cond_eq": float_str(ceq), "qualified": qualified,
        "sensitivity": float_str(worst), "sensitivity_skipped": skipped,
    })
    return out


# --------------------------------------------------------------------------------------------
# R4 families (ports of qp_family.py): Fraction-built inputs, ids <family>-<seed>-<idx>
# --------------------------------------------------------------------------------------------


def fmt(T, x):
    if T == "double":
        return repr(float(x))
    if x == 0:
        return "0"
    d = Decimal(x.numerator) / Decimal(x.denominator)
    d = Context(prec=26).plus(d)
    if d.as_tuple().exponent < -28:
        d = d.quantize(Decimal("1e-28"))
    return format(d, 'f')


def r4_make_case(rng, fam):
    if fam == "dec-diag-count":
        k = rng.randint(9, 12)
        n = k + 1
        v = Fr(rng.choice([1, 1, 2])) * Fr(10) ** -14
        base = Fr(rng.randint(8, 10)) * Fr(10) ** 13
        mu = [base] + [-base + Fr(rng.randint(-1000, 1000)) * Fr(10) ** 9 for _ in range(k)]
        if rng.random() < 0.5:
            mu = [-x for x in mu]
        lam = Fr(rng.choice([1, 2, 5])) * Fr(10) ** rng.choice([12, 13, 14])
        ub = Fr(rng.randint(1, 9)) * Fr(10) ** 13
        lb = -Fr(rng.randint(1, 9)) * Fr(10) ** 13
        cov = [[v if i == j else Fr(0) for j in range(n)] for i in range(n)]
        perm = list(range(n))
        rng.shuffle(perm)
        mu = [mu[p] for p in perm]
        return "decimal", "meanvar", cov, mu, lam, lb, ub
    if fam == "dec-corr":
        k = rng.randint(2, 5)
        n = k + 1
        v = Fr(rng.choice([1, 2, 4])) * Fr(10) ** -rng.choice([14, 12, 10])
        eps = Fr(1, 10 ** rng.randint(1, 4))
        rho = -(Fr(1) - eps) / (k - 1) if k > 1 else Fr(0)
        cov = [[Fr(0)] * n for _ in range(n)]
        cov[0][0] = Fr(rng.choice([1, 3])) * Fr(10) ** -rng.choice([12, 10, 8])
        for i in range(1, n):
            for j in range(1, n):
                cov[i][j] = v if i == j else v * rho
        base = Fr(rng.randint(1, 10)) * Fr(10) ** 13
        mu = [base] + [-base + Fr(rng.randint(-100, 100)) * Fr(10) ** 8 for _ in range(k)]
        lam = Fr(rng.choice([1, 2, 5])) * Fr(10) ** rng.choice([10, 12, 14])
        ub = Fr(rng.randint(1, 9)) * Fr(10) ** rng.choice([12, 13])
        lb = -Fr(rng.randint(1, 9)) * Fr(10) ** 13
        return "decimal", "meanvar", cov, mu, lam, lb, ub
    if fam == "dbl-corr":
        k = rng.randint(2, 4)
        n = k + 1
        v = Fr(float(rng.uniform(1, 9) * 10.0 ** rng.randint(-150, -140)))
        mubase = Fr(float(rng.uniform(1, 9) * 10.0 ** rng.randint(140, 149)))
        target = rng.uniform(0.6, 0.99) * 1.7976931348623157e308
        eps_f = float(mubase / v) / target
        if not (1e-12 < eps_f < 0.5):
            return None
        rho = float(-(1 - eps_f) / (k - 1))
        rhof = Fr(rho)
        cov = [[Fr(0)] * n for _ in range(n)]
        cov[0][0] = Fr(float(rng.uniform(1, 9) * 10.0 ** rng.randint(-150, 10)))
        for i in range(1, n):
            for j in range(1, n):
                cov[i][j] = v if i == j else Fr(float(v * rhof))
        sgn = rng.choice([1, -1])
        mu = [Fr(float(sgn * mubase))] + [Fr(float(-sgn * mubase * Fr(1 + rng.uniform(-1e-9, 1e-9))))
                                          for _ in range(k)]
        lam = Fr(float(rng.uniform(1, 9) * 10.0 ** rng.randint(-10, 150)))
        ub = Fr(float(rng.uniform(1, 9) * 10.0 ** rng.randint(100, 149)))
        lb = -Fr(float(rng.uniform(1, 9) * 10.0 ** rng.randint(100, 149)))
        perm = list(range(n))
        rng.shuffle(perm)
        cov = [[cov[perm[i]][perm[j]] for j in range(n)] for i in range(n)]
        mu = [mu[p] for p in perm]
        return "double", "meanvar", cov, mu, lam, lb, ub
    if fam == "dbl-minvar":
        k = rng.randint(2, 4)
        n = k + 1
        v = Fr(float(rng.uniform(1, 9) * 10.0 ** rng.randint(-150, -100)))
        eps = Fr(float(10.0 ** rng.uniform(-9, -2)))
        rho = Fr(float(-(1 - eps) / (k - 1)))
        V0 = Fr(float(rng.uniform(1, 9) * 10.0 ** rng.randint(100, 150)))
        cross_c = Fr(float(rng.uniform(0.01, 0.9))) * Fr(float((float(v) * float(V0) * float(eps) / k) ** 0.5))
        cov = [[Fr(0)] * n for _ in range(n)]
        cov[0][0] = V0
        for i in range(1, n):
            cov[0][i] = cov[i][0] = Fr(float(cross_c))
            for j in range(1, n):
                cov[i][j] = v if i == j else Fr(float(v * rho))
        ub = Fr(float(rng.uniform(1, 9) * 10.0 ** rng.randint(100, 150)))
        lb = -Fr(float(rng.uniform(1, 9) * 10.0 ** rng.randint(100, 150)))
        return "double", "minvar", cov, [Fr(0)] * n, Fr(2), lb, ub
    raise SystemExit("unknown R4 family " + fam)


def r4_family(fam, seed, count):
    rng = random.Random(seed)
    cases, tries = [], 0
    while len(cases) < count and tries < count * 50:
        tries += 1
        r = r4_make_case(rng, fam)
        if r is None:
            continue
        T, mode, cov, mu, lam, lb, ub = r
        n = len(mu)
        if not (lb * n <= 1 <= ub * n):
            continue
        cs = {"id": f"{fam}-{seed}-{len(cases)}", "family": fam, "T": T, "mode": mode,
              "cov": [[fmt(T, x) for x in row] for row in cov],
              "mu": [fmt(T, x) for x in mu], "lambda": fmt(T, lam), "lb": fmt(T, lb), "ub": fmt(T, ub)}
        cases.append(cs)
    return cases


# --------------------------------------------------------------------------------------------
# R5 families (ports of families_rng.py): ids <family>-<idx>, rng seeded from (seed, family)
# --------------------------------------------------------------------------------------------

D28 = Decimal("1e-28")
C28 = Context(prec=28)


def sgnpat(rng, k):
    s = [rng.choice([1, -1]) for _ in range(k)]
    if rng.random() < 0.25:
        s = [1] * k
    return s


def dstr(x):
    return format(x, 'f')


def block_double(rng, v, eps, s):
    k = len(s)
    return [[v * ((1.0 if i == j else 0.0) - (1 - eps) / k * s[i] * s[j]) for j in range(k)] for i in range(k)]


def block_decimal(v, eps, s):
    k = len(s)
    one = Decimal(1)
    off = C28.multiply(v, C28.divide(one - eps, Decimal(k)))
    out = [[None] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            val = C28.subtract(v, off) if i == j else C28.multiply(off, Decimal(-s[i] * s[j]))
            out[i][j] = val.quantize(D28) if val.as_tuple().exponent < -28 else val
    return out


def assemble(n, blocks):
    M = [[0 for _ in range(n)] for _ in range(n)]
    for (idx, B) in blocks:
        for a, i in enumerate(idx):
            for b, j in enumerate(idx):
                M[i][j] = B[a][b]
    return M


def tostr(T, M):
    if T == "double":
        return [[repr(float(x)) for x in row] for row in M]
    return [[dstr(Decimal(x)) if not isinstance(x, Decimal) else dstr(x) for x in row] for row in M]


def U(rng, a, b):
    return 10.0 ** rng.uniform(a, b)


def r5_make_base(rng, fam, idx):
    cid = f"{fam}-{idx}"
    if fam == "r5-rng-dbl-mean":
        k = rng.randint(2, 4)
        n = k + rng.randint(0, 1)
        eps = U(rng, -16, -5)
        s = sgnpat(rng, k)
        mb = U(rng, 120, 150)
        v = max(mb * 10.0 ** -rng.uniform(308.5, 316) / eps, 1e-150)
        blocks = [(list(range(k)), block_double(rng, v, eps, s))]
        if n > k:
            blocks.append(([k], [[U(rng, -150, 0)]]))
        M = assemble(n, blocks)
        mu = [mb * (s[i] if i < k else rng.choice([1, -1])) * (1 + rng.uniform(-1e-3, 1e-3)) for i in range(n)]
        lam = U(rng, 40, 150)
        lb = -U(rng, 100, 150)
        ub = U(rng, 100, 150)
        return {"id": cid, "T": "double", "mode": "meanvar", "cov": tostr("double", M),
                "mu": [repr(x) for x in mu], "lambda": repr(lam), "lb": repr(lb), "ub": repr(ub)}
    if fam in ("r5-rng-dbl-budget", "r5-rng-dbl-budget-min"):
        k = rng.randint(2, 3)
        n = k + 1
        sbig = U(rng, 120, 150)
        target = sbig * 10.0 ** -rng.uniform(308.5, 316)
        eps = U(rng, -16, -8)
        v = max(target / eps, 1e-150)
        s = sgnpat(rng, k)
        M = assemble(n, [([0], [[sbig]]), (list(range(1, n)), block_double(rng, v, eps, s))])
        c = {"id": cid, "T": "double", "mode": "minvar" if fam.endswith("min") else "meanvar",
             "cov": tostr("double", M), "lb": repr(-U(rng, 0, 140)), "ub": repr(U(rng, 0, 140))}
        if c["mode"] == "meanvar":
            ms = U(rng, -5, 5)
            c["mu"] = [repr(rng.gauss(0, 1) * ms) for _ in range(n)]
            c["lambda"] = repr(U(rng, -140, 5))
        return c
    if fam == "r5-rng-dbl-zc":
        k = rng.randint(2, 3)
        n = k + 1
        V0 = U(rng, 100, 150)
        eps = U(rng, -16, -10)
        v = U(rng, -150, -135)
        s = sgnpat(rng, k)
        B = block_double(rng, v, eps, s)
        lmin = v * eps
        cu = rng.uniform(0.3, 0.95) * math.sqrt(V0 * lmin)
        nrm = math.sqrt(k)
        cross = [cu * s[i] / nrm for i in range(k)]
        M = [[0.0] * n for _ in range(n)]
        M[0][0] = V0
        for i in range(k):
            M[0][i + 1] = M[i + 1][0] = cross[i]
            for j in range(k):
                M[i + 1][j + 1] = B[i][j]
        lb = rng.choice([rng.uniform(0.05, 1.0 / n), -U(rng, 100, 150)])
        ub = U(rng, 100, 150)
        return {"id": cid, "T": "double", "mode": "minvar", "cov": tostr("double", M), "lb": repr(lb), "ub": repr(ub)}
    if fam == "r5-dec-rng-mean":
        k = rng.choice([2, 4])
        n = k + rng.randint(0, 1)
        v = Decimal(rng.randint(1, 9)) * Decimal(10) ** -rng.randint(4, 10)
        eps = Decimal(rng.randint(1, 9)) * Decimal(10) ** -rng.randint(10, 16)
        s = sgnpat(rng, k)
        blocks = [(list(range(k)), block_decimal(v, eps, s))]
        if n > k:
            blocks.append(([k], [[Decimal(rng.randint(1, 9)) * Decimal(10) ** -rng.randint(0, 8)]]))
        M = assemble(n, blocks)
        mb = Decimal(rng.randint(1, 9)) * Decimal(10) ** rng.randint(11, 13)
        mu = [mb * (s[i] if i < k else rng.choice([1, -1])) + Decimal(rng.randint(-1000, 1000)) * Decimal(10) ** rng.randint(5, 9)
              for i in range(n)]
        lam = Decimal(rng.randint(1, 9)) * Decimal(10) ** rng.randint(0, 13)
        lb = -Decimal(rng.randint(1, 9)) * Decimal(10) ** rng.randint(8, 13)
        ub = Decimal(rng.randint(1, 9)) * Decimal(10) ** rng.randint(8, 13)
        return {"id": cid, "T": "decimal", "mode": "meanvar", "cov": tostr("decimal", M), "mu": [dstr(x) for x in mu],
                "lambda": dstr(lam), "lb": dstr(lb), "ub": dstr(ub)}
    if fam in ("r5-dec-rng-budget", "r5-dec-rng-budget-min"):
        k = rng.choice([2, 4])
        n = k + 1
        sbig = Decimal(rng.randint(1, 9)) * Decimal(10) ** rng.randint(10, 13)
        v = Decimal(rng.randint(1, 9)) * Decimal(10) ** -rng.randint(8, 14)
        eps = Decimal(rng.randint(1, 9)) * Decimal(10) ** -rng.randint(10, 14)
        s = sgnpat(rng, k)
        M = assemble(n, [([0], [[sbig]]), (list(range(1, n)), block_decimal(v, eps, s))])
        c = {"id": cid, "T": "decimal", "mode": "minvar" if fam.endswith("min") else "meanvar",
             "cov": tostr("decimal", M),
             "lb": dstr(-Decimal(rng.randint(1, 9)) * Decimal(10) ** rng.randint(0, 13)),
             "ub": dstr(Decimal(rng.randint(1, 9)) * Decimal(10) ** rng.randint(0, 13))}
        if c["mode"] == "meanvar":
            c["mu"] = [dstr(Decimal(rng.randint(-10 ** 6, 10 ** 6)) * Decimal(10) ** rng.randint(-6, 6)) for _ in range(n)]
            c["lambda"] = dstr(Decimal(rng.randint(1, 9)) * Decimal(10) ** rng.randint(-10, 6))
        return c
    if fam == "r5-dec-rng-zc":
        k = rng.choice([2, 4])
        n = k + 1
        V0 = Decimal(rng.randint(1, 9)) * Decimal(10) ** rng.randint(10, 13)
        v = Decimal(rng.randint(1, 9)) * Decimal(10) ** -rng.randint(8, 14)
        eps = Decimal(rng.randint(1, 9)) * Decimal(10) ** -rng.randint(10, 14)
        s = sgnpat(rng, k)
        B = block_decimal(v, eps, s)
        lmin = v * eps
        cu = Decimal(rng.uniform(0.3, 0.95)) * (V0 * lmin).sqrt(C28) / Decimal(k).sqrt(C28)
        M = [[Decimal(0)] * n for _ in range(n)]
        M[0][0] = V0
        for i in range(k):
            ci = C28.multiply(cu, Decimal(s[i])).quantize(D28)
            M[0][i + 1] = M[i + 1][0] = ci
            for j in range(k):
                M[i + 1][j + 1] = B[i][j]
        lb = rng.choice([Decimal(rng.randint(5, 30)) / Decimal(100),
                         -Decimal(rng.randint(1, 9)) * Decimal(10) ** rng.randint(8, 13)])
        ub = Decimal(rng.randint(1, 9)) * Decimal(10) ** rng.randint(8, 13)
        return {"id": cid, "T": "decimal", "mode": "minvar", "cov": tostr("decimal", M), "lb": dstr(lb), "ub": dstr(ub)}
    raise SystemExit("unknown R5 family " + fam)


def r5_make_disp(rng, fam, idx):
    """r5-dbl-disp-mv: one or two high-variance assets plus a generic low-variance block."""
    T = "decimal" if "-dec-" in fam else "double"
    mode = "minvar" if fam.endswith("min") else "meanvar"
    n = rng.randint(3, 5)
    h = 1 if n == 3 else rng.randint(1, 2)
    Q = [[rng.gauss(0, 1) for _ in range(n - h)] for _ in range(n - h)]
    m = n - h
    B = [[sum(Q[i][k] * Q[j][k] for k in range(m)) / m + (0.1 if i == j else 0.0) for j in range(m)] for i in range(m)]
    if T == "decimal":
        v = 10.0 ** -rng.uniform(4, 13)
        V = 10.0 ** rng.uniform(4, 13)
    else:
        v = 10.0 ** -rng.uniform(4, 60)
        V = 10.0 ** rng.uniform(4, 60)
    M = [[0.0] * n for _ in range(n)]
    for a in range(h):
        M[a][a] = V * rng.uniform(0.5, 2)
    for i in range(m):
        for j in range(m):
            M[h + i][h + j] = v * B[i][j]
    for a in range(h):
        for i in range(m):
            c = rng.uniform(-0.5, 0.5) * math.sqrt(M[a][a] * M[h + i][h + i]) / m
            M[a][h + i] = M[h + i][a] = c

    def q(x):
        if T == "double":
            return repr(float(x))
        d = Context(prec=12).create_decimal_from_float(x)
        return dstr(d.quantize(D28) if d.as_tuple().exponent < -28 else d)

    cov = [[q(M[i][j]) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i):
            cov[i][j] = cov[j][i]
    lb = rng.choice([0.0, -10.0 ** rng.uniform(-2, 1), rng.uniform(0, 1.0 / n)])
    ub = (1.0 / n) * (1 + 10.0 ** rng.uniform(-2, 0.7))
    c = {"id": f"{fam}-{idx}", "T": T, "mode": mode, "cov": cov, "lb": q(lb), "ub": q(ub)}
    if mode == "meanvar":
        ms = 10.0 ** rng.uniform(-6, 2)
        c["mu"] = [q(rng.gauss(0, 1) * ms) for _ in range(n)]
        lam = ms / v * 10.0 ** rng.uniform(-2, 2)
        lam = min(max(lam, 1e-6), 9e13) if T == "decimal" else lam
        c["lambda"] = q(lam)
    return c


def r5_make_zc2(rng, fam, idx):
    """r5-dec-zc2 (MinVar): two near-singular blocks coupled through cross covariances aligned with each
    block's near-null direction, wide bounds."""
    T = "decimal" if "-dec-" in fam else "double"
    k1, k2 = 2, rng.randint(1, 2)
    n = k1 + k2 + 1
    if T == "double":
        v1, v2 = U(rng, -150, -100), U(rng, -150, -100)
        e1, e2 = U(rng, -16, -10), U(rng, -16, -10)
        V0 = U(rng, 100, 150)
        big = U(rng, 100, 150)
    else:
        v1, v2 = U(rng, -12, -4), U(rng, -12, -4)
        e1, e2 = U(rng, -16, -10), U(rng, -16, -10)
        V0 = U(rng, 8, 14)
        big = U(rng, 8, 14)
    s1 = [1, -1]
    s2 = [1, -1][:k2]
    M = [[0.0] * n for _ in range(n)]
    B1 = block_double(rng, v1, e1, s1)
    for i in range(2):
        for j in range(2):
            M[i][j] = B1[i][j]
    if k2 == 2:
        B2 = block_double(rng, v2, e2, s2)
        for i in range(2):
            for j in range(2):
                M[2 + i][2 + j] = B2[i][j]
    else:
        M[2][2] = v2
    M[n - 1][n - 1] = V0
    lm1 = v1 * e1
    lm2 = (v2 * e2) if k2 == 2 else v2
    cpl = rng.uniform(0.2, 0.9) * math.sqrt(lm1 * lm2) / 2
    for i in range(2):
        for j in range(k2):
            M[i][2 + j] = M[2 + j][i] = cpl * s1[i] * s2[j]
    cz = rng.uniform(0.2, 0.9) * math.sqrt(V0 * lm1) / 2
    for i in range(2):
        M[i][n - 1] = M[n - 1][i] = cz * s1[i]

    def q(x):
        if T == "double":
            return repr(float(x))
        d = Context(prec=16).create_decimal_from_float(x)
        return dstr(d.quantize(D28) if d.as_tuple().exponent < -28 else d)

    cov = [[q(M[i][j]) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i):
            cov[i][j] = cov[j][i]
    return {"id": f"{fam}-{idx}", "T": T, "mode": "minvar", "cov": cov, "lb": q(-big), "ub": q(big * rng.uniform(0.5, 2))}


def r5_make(rng, fam, idx):
    if "-zc2" in fam:
        return r5_make_zc2(rng, fam, idx)
    if "-disp-" in fam:
        return r5_make_disp(rng, fam, idx)
    return r5_make_base(rng, fam, idx)


def r5_family(fam, seed, count):
    rng = random.Random(seed * 104729 + sum(map(ord, fam)))
    cases, tries = [], 0
    while len(cases) < count and tries < count * 40:
        tries += 1
        c = r5_make(rng, fam, len(cases))
        Sg, mu, lam, lb, ub = exact_inputs(c)
        n = len(Sg)
        if not exact_pd(Sg) or not (lb * n <= 1 <= ub * n) or not in_domain(c):
            continue
        c["family"] = fam
        cases.append(c)
    return cases


# --------------------------------------------------------------------------------------------
# Hand-built decimal cases (the high-disparity diagonal witnesses)
# --------------------------------------------------------------------------------------------


def hand_cases():
    out = []

    def add(cid, cov, lb, ub, mode="minvar", mu=None, lam=None):
        c = {"id": cid, "family": "hd-hand", "T": "decimal", "mode": mode, "cov": cov, "lb": lb, "ub": ub}
        if mode == "meanvar":
            c["mu"] = mu
            c["lambda"] = lam
        out.append(c)

    add("hd-diag-13", [["10000000000000", "0", "0"], ["0", "0.0000000000001", "0"], ["0", "0", "0.0000000000001"]], "0", "1")
    add("hd-diag-12", [["1000000000000", "0", "0"], ["0", "0.000000000001", "0"], ["0", "0", "0.000000000001"]], "0", "1")
    add("hd-diag-10", [["10000000000", "0", "0"], ["0", "0.0000000001", "0"], ["0", "0", "0.0000000001"]], "0", "1")
    add("hd-diag-8", [["100000000", "0", "0"], ["0", "0.00000001", "0"], ["0", "0", "0.00000001"]], "-1", "1")
    add("hd-diag-6", [["1000000", "0", "0"], ["0", "0.000001", "0"], ["0", "0", "0.000001"]], "-1", "1")
    add("hd-diag4-12", [["1000000000000", "0", "0", "0"], ["0", "0.000000000002", "0", "0"],
                        ["0", "0", "0.000000000003", "0"], ["0", "0", "0", "0.000000000004"]], "0", "1")
    add("hd-mv-12", [["1000000000000", "0", "0"], ["0", "0.000000000001", "0"], ["0", "0", "0.000000000001"]],
        "0", "1", "meanvar", ["0.00000000001", "0.00000000002", "0"], "1")
    add("hd-mv-10", [["10000000000", "0", "0"], ["0", "0.0000000001", "0"], ["0", "0", "0.0000000001"]],
        "-1", "1", "meanvar", ["0.000000001", "0.000000002", "0"], "1")
    return out


# --------------------------------------------------------------------------------------------
# Census families (ports of the held-out harness generators nearq / nearo / degenb in
# the exact-rational conditioning census): exact rational inputs rounded once into the working type
# --------------------------------------------------------------------------------------------


def round_to(T, x):
    """The working-type value nearest to the rational x, as the string the corpus stores. decimal: the nearest
    value with at most 28 significant digits and 28 decimal places (round half to even)."""
    if T == "double":
        return repr(float(x))
    if x == 0:
        return "0"
    a, b = abs(x.numerator), x.denominator
    for s in range(28, -1, -1):
        q, r = divmod(a * 10 ** s, b)
        if 2 * r > b or (2 * r == b and q % 2 == 1):
            q += 1
        if len(str(q)) <= 28:
            break
    else:
        raise ValueError("value beyond the decimal range")
    digits = str(q).rjust(s + 1, "0")
    text = digits if s == 0 else digits[:-s] + "." + digits[-s:]
    return ("-" if x < 0 else "") + text


def g16(T, x):
    """A double drawn by the harness as the working type (decimal: its 16-significant-digit decimal)."""
    if T == "double":
        return repr(float(x))
    return dstr(Decimal(format(x, ".16g")))


def census_signs(rng, k):
    s = [1 if rng.randrange(2) == 0 else -1 for _ in range(k)]
    if rng.randrange(4) == 0:
        s = [1] * k
    return s


def census_make(rng, fam, T):
    dec = T == "decimal"
    if fam in ("nearq", "nearo", "nearmo"):
        # nearmo (the VB-M-3 held-out family): the nearo shape with the eps band moved to straddle the
        # 48.n.u.cond_eq margin, so that qualified blocks sit near the top of the qualified band.
        oshape = fam in ("nearo", "nearmo")
        k = 2 * rng.randint(1, 6) if oshape else rng.randint(2, 12)
        with_v0 = rng.randrange(3) == 0 and k < 12
        n = k + (1 if with_v0 else 0)
        if oshape:
            s = [1] * (k // 2) + [-1] * (k // 2)
            rng.shuffle(s)
        else:
            s = census_signs(rng, k)
        if fam == "nearmo":
            le = rng.uniform(-28.0, -22.0) if dec else rng.uniform(-16.0, -10.0)
        else:
            le = rng.uniform(-29.5, -22.5) if dec else rng.uniform(-17.5, -11.0)
        fl = math.floor(le)
        eps = Fr(int(round(10 ** (le - fl) * 1e6)), 10 ** (6 - fl))
        vv = Fr(rng.randint(1, 999), 100) if dec else Fr(math.ldexp(rng.uniform(1, 2), rng.randint(-60, 60)))
        M = [[Fr(0)] * n for _ in range(n)]
        for i in range(k):
            for j in range(k):
                M[i][j] = vv * ((Fr(1) if i == j else Fr(0)) - (Fr(1) - eps) / k * s[i] * s[j])
        if with_v0:
            M[k][k] = vv * rng.randint(2, 999)
            for i in range(k):
                M[i][k] = M[k][i] = vv * s[i] * rng.randint(0, 2) / (7 * k)
        cov = [[round_to(T, M[i][j]) for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(i):
                cov[i][j] = cov[j][i]
        choice = rng.randrange(4)
        if choice == 0:
            lb, ub = "-1000", "1000"
        elif choice == 1:
            lb, ub = "0", "1"
        elif choice == 2:
            lb, ub = "-1", "2"
        else:
            lb = g16(T, -(10 ** rng.uniform(2, 13 if dec else 140)))
            ub = g16(T, 10 ** rng.uniform(2, 13 if dec else 140))
        return cov, lb, ub
    if fam == "degenb":
        k = rng.randint(1, 6)
        n = k + 1
        A = [[Fr(0)] * k for _ in range(k)]
        for i in range(k):
            for j in range(i + 1):
                v = Fr(rng.randint(8, 16)) if i == j else Fr(rng.randint(-1, 1))
                A[i][j] = A[j][i] = v
        den = 1 << 10
        a = [Fr(0)] * k
        acc = 0
        for i in range(k - 1):
            p = rng.randint(-200, 699)
            a[i] = Fr(p, den)
            acc += p
        a[k - 1] = Fr(den - acc, den)
        mode = rng.randrange(4)
        eta = Fr(0)
        if mode > 0:
            q = rng.randint(85, 97) if dec else rng.randint(46, 59)
            eta = Fr(1 if rng.randrange(2) == 0 else -1, 1 << q)
            if mode == 3:
                eta *= rng.randint(1, 8)
        a[k - 1] += eta
        scale = Fr(1) if dec else Fr(math.ldexp(1.0, rng.randint(-40, 40)))
        M = [[None] * n for _ in range(n)]
        for i in range(k):
            for j in range(k):
                M[i][j] = round_to(T, A[i][j] * scale)
        c = []
        for i in range(k):
            sacc = sum((A[i][j] * a[j] for j in range(k)), Fr(0))
            c.append(tofr(T, round_to(T, sacc * scale)) / scale)
        y = solve_lin(A, c)
        if y is None:
            return None
        dq = sum((c[i] * y[i] for i in range(k)), Fr(0))
        delta = dq / Fr(10) ** (rng.randint(20, 29) if dec else rng.randint(10, 17))
        for i in range(k):
            M[i][k] = M[k][i] = round_to(T, c[i] * scale)
        M[k][k] = round_to(T, (dq + delta) * scale)
        choice = rng.randrange(3)
        lb, ub = [("0", "1"), ("0", "2"), ("0.01", "1")][choice]
        perm = list(range(n))
        rng.shuffle(perm)
        cov = [[M[perm[i]][perm[j]] for j in range(n)] for i in range(n)]
        return cov, lb, ub
    raise SystemExit("unknown census family " + fam)


def census_family(fam, seed, count):
    """Family names carry the type (nearq-dbl, degenb-dec, ...); the generator is the census shape."""
    base, suffix = fam.rsplit("-", 1)
    T = "double" if suffix == "dbl" else "decimal"
    rng = random.Random(seed * 104729 + sum(map(ord, fam)))
    cases, tries = [], 0
    while len(cases) < count and tries < count * 60:
        tries += 1
        r = census_make(rng, base, T)
        if r is None:
            continue
        cov, lb, ub = r
        c = {"id": f"{fam}-{seed}-{len(cases)}", "family": fam, "T": T, "mode": "minvar", "cov": cov, "lb": lb, "ub": ub}
        Sg, mu, lam, lbf, ubf = exact_inputs(c)
        n = len(Sg)
        if not exact_pd(Sg) or not (lbf * n <= 1 <= ubf * n) or not in_domain(c) or not exact_in_type(c):
            continue
        cases.append(c)
    return cases


# --------------------------------------------------------------------------------------------
# Held-out witnesses: explicit matrices exactly as the held-out census harness dumped them
# (id, type, covariance, lb, ub). Each was a wrong vector or a refusal of a qualified case of an
# earlier core, or one of the extreme shapes the termination census found.
# --------------------------------------------------------------------------------------------

HELDOUT_WITNESSES = [
    # zero-correlation shape: a budget-orthogonal near-null direction (1, -1, -1, 1); refusal-permitted
    ("a1Y9. /1 0#1", "double",
     [["2.7385846168598383E+123", "-1.0480340939535158E-20", "1.0480340939535158E-20", "1.0480340939535158E-20", "-1.0480340939535158E-20"],
      ["-1.0480340939535158E-20", "2.1181073767866432E-148", "7.060357922622136E-149", "7.060357922622136E-149", "-7.060357922622136E-149"],
      ["1.0480340939535158E-20", "7.060357922622136E-149", "2.1181073767866432E-148", "-7.060357922622136E-149", "7.060357922622136E-149"],
      ["1.0480340939535158E-20", "7.060357922622136E-149", "-7.060357922622136E-149", "2.1181073767866432E-148", "7.060357922622136E-149"],
      ["-1.0480340939535158E-20", "-7.060357922622136E-149", "7.060357922622136E-149", "7.060357922622136E-149", "2.1181073767866432E-148"]],
     "-1.812747161683739E+112", "5.180313214203733E+137"),
    # extreme diagonal disparity at n = 2: diag(441029.36, 7.41e88), box [0, 1]
    ("v3probe-188#2", "double", [["441029.3620155419", "0"], ["0", "7.41199517350585E+88"]], "0", "1"),
    # repeated structure 3*G + 1e-8*I (n = 5, three groups), lb = 1/6
    ("degen-120#1", "double",
     [[repr(3.0 + 1e-8) if i == j else ("3.0" if i % 3 == j % 3 else "1.5") for j in range(5)] for i in range(5)],
     repr(1.0 / 6), "1"),
]


# The conditioning census dumps (VB_DUMP): the nine qualified double corr problems that the
# landed core refused with a certificate multiplier-sign failure (census batches tA* / tB*), the qualified decimal
# degenerate-at-bound problems that it refused (.0#3, v3r2degenb-66#3, v3r2degenb-129#2, vr2dgb-35#1) or answered
# with a wrong vector (tBd219#0, error 1.6x the bound; tBg113#0).
HELDOUT_DUMPS = [
    ('tAg1450#0', 'double', [['8.356458956383782E+134', '10618449.08823386', '10618449.08823386', '10618449.08823386', '10618449.08823386', '10618449.08823386'], ['10618449.08823386', '2.4254454047351216E-113', '-6.063607302110523E-114', '-6.063607302110523E-114', '-6.063607302110523E-114', '-6.063607302110523E-114'], ['10618449.08823386', '-6.063607302110523E-114', '2.4254454047351216E-113', '-6.063607302110523E-114', '-6.063607302110523E-114', '-6.063607302110523E-114'], ['10618449.08823386', '-6.063607302110523E-114', '-6.063607302110523E-114', '2.4254454047351216E-113', '-6.063607302110523E-114', '-6.063607302110523E-114'], ['10618449.08823386', '-6.063607302110523E-114', '-6.063607302110523E-114', '-6.063607302110523E-114', '2.4254454047351216E-113', '-6.063607302110523E-114'], ['10618449.08823386', '-6.063607302110523E-114', '-6.063607302110523E-114', '-6.063607302110523E-114', '-6.063607302110523E-114', '2.4254454047351216E-113']], '-0.25', '2'),
    ('tAg412#0', 'double', [['2.9785741560418515E+118', '0.0005175548769789013', '0.0005175548769789013', '0.0005175548769789013', '0.0005175548769789013', '0.0005175548769789013'], ['0.0005175548769789013', '1.854936131109619E-115', '-4.6373403119114927E-116', '-4.6373403119114927E-116', '-4.6373403119114927E-116', '-4.6373403119114927E-116'], ['0.0005175548769789013', '-4.6373403119114927E-116', '1.854936131109619E-115', '-4.6373403119114927E-116', '-4.6373403119114927E-116', '-4.6373403119114927E-116'], ['0.0005175548769789013', '-4.6373403119114927E-116', '-4.6373403119114927E-116', '1.854936131109619E-115', '-4.6373403119114927E-116', '-4.6373403119114927E-116'], ['0.0005175548769789013', '-4.6373403119114927E-116', '-4.6373403119114927E-116', '-4.6373403119114927E-116', '1.854936131109619E-115', '-4.6373403119114927E-116'], ['0.0005175548769789013', '-4.6373403119114927E-116', '-4.6373403119114927E-116', '-4.6373403119114927E-116', '-4.6373403119114927E-116', '1.854936131109619E-115']], '-0.25', '2'),
    ('tAg473#0', 'double', [['3.963619815804489E+111', '3.521659856946308', '3.521659856946308', '3.521659856946308', '3.521659856946308', '3.521659856946308'], ['3.521659856946308', '2.3658878939234342E-101', '-5.914719705274722E-102', '-5.914719705274722E-102', '-5.914719705274722E-102', '-5.914719705274722E-102'], ['3.521659856946308', '-5.914719705274722E-102', '2.3658878939234342E-101', '-5.914719705274722E-102', '-5.914719705274722E-102', '-5.914719705274722E-102'], ['3.521659856946308', '-5.914719705274722E-102', '-5.914719705274722E-102', '2.3658878939234342E-101', '-5.914719705274722E-102', '-5.914719705274722E-102'], ['3.521659856946308', '-5.914719705274722E-102', '-5.914719705274722E-102', '-5.914719705274722E-102', '2.3658878939234342E-101', '-5.914719705274722E-102'], ['3.521659856946308', '-5.914719705274722E-102', '-5.914719705274722E-102', '-5.914719705274722E-102', '-5.914719705274722E-102', '2.3658878939234342E-101']], '-0.25', '2'),
    ('tAg4367#2', 'double', [['5.222716018169612E+104', '5.983620027485086E-21', '5.983620027485086E-21', '5.983620027485086E-21', '5.983620027485086E-21', '5.983620027485086E-21'], ['5.983620027485086E-21', '6.315173903698513E-142', '-1.5686055670431006E-142', '-1.5686055670431006E-142', '-1.5686055670431006E-142', '-1.5686055670431006E-142'], ['5.983620027485086E-21', '-1.5686055670431006E-142', '6.315173903698513E-142', '-1.5686055670431006E-142', '-1.5686055670431006E-142', '-1.5686055670431006E-142'], ['5.983620027485086E-21', '-1.5686055670431006E-142', '-1.5686055670431006E-142', '6.315173903698513E-142', '-1.5686055670431006E-142', '-1.5686055670431006E-142'], ['5.983620027485086E-21', '-1.5686055670431006E-142', '-1.5686055670431006E-142', '-1.5686055670431006E-142', '6.315173903698513E-142', '-1.5686055670431006E-142'], ['5.983620027485086E-21', '-1.5686055670431006E-142', '-1.5686055670431006E-142', '-1.5686055670431006E-142', '-1.5686055670431006E-142', '6.315173903698513E-142']], '-0.25', '2'),
    ('tAg3269#0', 'double', [['4.668207575802289E+105', '3.937946020591067E-07', '3.937946020591067E-07', '3.937946020591067E-07', '3.937946020591067E-07', '3.937946020591067E-07'], ['3.937946020591067E-07', '1.9932808887653387E-113', '-4.982649111945542E-114', '-4.982649111945542E-114', '-4.982649111945542E-114', '-4.982649111945542E-114'], ['3.937946020591067E-07', '-4.982649111945542E-114', '1.9932808887653387E-113', '-4.982649111945542E-114', '-4.982649111945542E-114', '-4.982649111945542E-114'], ['3.937946020591067E-07', '-4.982649111945542E-114', '-4.982649111945542E-114', '1.9932808887653387E-113', '-4.982649111945542E-114', '-4.982649111945542E-114'], ['3.937946020591067E-07', '-4.982649111945542E-114', '-4.982649111945542E-114', '-4.982649111945542E-114', '1.9932808887653387E-113', '-4.982649111945542E-114'], ['3.937946020591067E-07', '-4.982649111945542E-114', '-4.982649111945542E-114', '-4.982649111945542E-114', '-4.982649111945542E-114', '1.9932808887653387E-113']], '-0.25', '2'),
    ('tAg2217#1', 'double', [['6.318075266883869E+107', '4.862492846981781E-17', '4.862492846981781E-17', '4.862492846981781E-17', '4.862492846981781E-17', '4.862492846981781E-17'], ['4.862492846981781E-17', '9.809573245543717E-132', '-2.4523932981651172E-132', '-2.4523932981651172E-132', '-2.4523932981651172E-132', '-2.4523932981651172E-132'], ['4.862492846981781E-17', '-2.4523932981651172E-132', '9.809573245543717E-132', '-2.4523932981651172E-132', '-2.4523932981651172E-132', '-2.4523932981651172E-132'], ['4.862492846981781E-17', '-2.4523932981651172E-132', '-2.4523932981651172E-132', '9.809573245543717E-132', '-2.4523932981651172E-132', '-2.4523932981651172E-132'], ['4.862492846981781E-17', '-2.4523932981651172E-132', '-2.4523932981651172E-132', '-2.4523932981651172E-132', '9.809573245543717E-132', '-2.4523932981651172E-132'], ['4.862492846981781E-17', '-2.4523932981651172E-132', '-2.4523932981651172E-132', '-2.4523932981651172E-132', '-2.4523932981651172E-132', '9.809573245543717E-132']], '-0.25', '2'),
    ('tBg1440#1', 'double', [['7.551711308476603E+138', '16780058.09555152', '16780058.09555152', '16780058.09555152', '16780058.09555152', '16780058.09555152'], ['16780058.09555152', '1.0317641906889877E-118', '-2.5791217596401777E-119', '-2.5791217596401777E-119', '-2.5791217596401777E-119', '-2.5791217596401777E-119'], ['16780058.09555152', '-2.5791217596401777E-119', '1.0317641906889877E-118', '-2.5791217596401777E-119', '-2.5791217596401777E-119', '-2.5791217596401777E-119'], ['16780058.09555152', '-2.5791217596401777E-119', '-2.5791217596401777E-119', '1.0317641906889877E-118', '-2.5791217596401777E-119', '-2.5791217596401777E-119'], ['16780058.09555152', '-2.5791217596401777E-119', '-2.5791217596401777E-119', '-2.5791217596401777E-119', '1.0317641906889877E-118', '-2.5791217596401777E-119'], ['16780058.09555152', '-2.5791217596401777E-119', '-2.5791217596401777E-119', '-2.5791217596401777E-119', '-2.5791217596401777E-119', '1.0317641906889877E-118']], '-0.25', '2'),
    ('tBg3395#2', 'double', [['1.6356607012025656E+109', '4.453579460291894E-24', '4.453579460291894E-24', '4.453579460291894E-24', '4.453579460291894E-24', '4.453579460291894E-24'], ['4.453579460291894E-24', '7.029582344582677E-150', '-1.7573913243772653E-150', '-1.7573913243772653E-150', '-1.7573913243772653E-150', '-1.7573913243772653E-150'], ['4.453579460291894E-24', '-1.7573913243772653E-150', '7.029582344582677E-150', '-1.7573913243772653E-150', '-1.7573913243772653E-150', '-1.7573913243772653E-150'], ['4.453579460291894E-24', '-1.7573913243772653E-150', '-1.7573913243772653E-150', '7.029582344582677E-150', '-1.7573913243772653E-150', '-1.7573913243772653E-150'], ['4.453579460291894E-24', '-1.7573913243772653E-150', '-1.7573913243772653E-150', '-1.7573913243772653E-150', '7.029582344582677E-150', '-1.7573913243772653E-150'], ['4.453579460291894E-24', '-1.7573913243772653E-150', '-1.7573913243772653E-150', '-1.7573913243772653E-150', '-1.7573913243772653E-150', '7.029582344582677E-150']], '-0.25', '2'),
    ('tBg2214#1', 'double', [['9.498218748047409E+109', '0.003996263047906717', '0.003996263047906717', '0.003996263047906717', '0.003996263047906717', '0.003996263047906717'], ['0.003996263047906717', '3.4875664766691516E-105', '-8.718916170273666E-106', '-8.718916170273666E-106', '-8.718916170273666E-106', '-8.718916170273666E-106'], ['0.003996263047906717', '-8.718916170273666E-106', '3.4875664766691516E-105', '-8.718916170273666E-106', '-8.718916170273666E-106', '-8.718916170273666E-106'], ['0.003996263047906717', '-8.718916170273666E-106', '-8.718916170273666E-106', '3.4875664766691516E-105', '-8.718916170273666E-106', '-8.718916170273666E-106'], ['0.003996263047906717', '-8.718916170273666E-106', '-8.718916170273666E-106', '-8.718916170273666E-106', '3.4875664766691516E-105', '-8.718916170273666E-106'], ['0.003996263047906717', '-8.718916170273666E-106', '-8.718916170273666E-106', '-8.718916170273666E-106', '-8.718916170273666E-106', '3.4875664766691516E-105']], '-0.25', '2'),
    ('.0#3', 'decimal', [['12.752745628356933593750000003', '-1.0263671874999999999999999999', '-1.6484374999999999999999999999', '11.207031250000000000000000001', '5.8925781250000000000000000001'], ['-1.0263671874999999999999999999', '16.000000000000000000000000000', '-1.0000000000000000000000000000', '1.0000000000000000000000000000', '1.0000000000000000000000000000'], ['-1.6484374999999999999999999999', '-1.0000000000000000000000000000', '16.000000000000000000000000000', '1.0000000000000000000000000000', '0'], ['11.207031250000000000000000001', '1.0000000000000000000000000000', '1.0000000000000000000000000000', '13.000000000000000000000000000', '1.0000000000000000000000000000'], ['5.8925781250000000000000000001', '1.0000000000000000000000000000', '0', '1.0000000000000000000000000000', '11.000000000000000000000000000']], '0', '1'),
    ('v3r2degenb-66#3', 'decimal', [['10.976614952087402343750000001', '-1.8681640625000000000000000000', '-1.2246093750000000000000000000', '11.276367187500000000000000000', '1.0351562500000000000000000000'], ['-1.8681640625000000000000000000', '11.000000000000000000000000000', '0', '-1.0000000000000000000000000000', '0'], ['-1.2246093750000000000000000000', '0', '11.000000000000000000000000000', '-1.0000000000000000000000000000', '-1.0000000000000000000000000000'], ['11.276367187500000000000000000', '-1.0000000000000000000000000000', '-1.0000000000000000000000000000', '12.000000000000000000000000000', '-1.0000000000000000000000000000'], ['1.0351562500000000000000000000', '0', '-1.0000000000000000000000000000', '-1.0000000000000000000000000000', '13.000000000000000000000000000']], '0.01', '1'),
    ('v3r2degenb-129#2', 'decimal', [['10.000000000000000000000000000', '-1.5214843750000000000000000000', '-1.0000000000000000000000000000', '0', '1.0000000000000000000000000000', '0'], ['-1.5214843750000000000000000000', '6.6589670181274414062500000067', '6.5244140625000000000000000000', '4.6738281250000000000000000000', '3.2412109375000000000000000001', '-0.1025390625000000000000000000'], ['-1.0000000000000000000000000000', '6.5244140625000000000000000000', '11.000000000000000000000000000', '1.0000000000000000000000000000', '0', '1.0000000000000000000000000000'], ['0', '4.6738281250000000000000000000', '1.0000000000000000000000000000', '9.000000000000000000000000000', '0', '1.0000000000000000000000000000'], ['1.0000000000000000000000000000', '3.2412109375000000000000000001', '0', '0', '16.000000000000000000000000000', '0'], ['0', '-0.1025390625000000000000000000', '1.0000000000000000000000000000', '1.0000000000000000000000000000', '0', '10.000000000000000000000000000']], '0.01', '1'),
    ('vr2dgb-35#1', 'decimal', [['11.000000000000000000000000000', '1.0000000000000000000000000000', '0', '0', '-0.5849609375000000000000000000', '1.0000000000000000000000000000'], ['1.0000000000000000000000000000', '9.000000000000000000000000000', '1.0000000000000000000000000000', '1.0000000000000000000000000000', '5.4687500000000000000000000000', '1.0000000000000000000000000000'], ['0', '1.0000000000000000000000000000', '11.000000000000000000000000000', '0', '-0.4130859375000000000000000000', '1.0000000000000000000000000000'], ['0', '1.0000000000000000000000000000', '0', '16.000000000000000000000000000', '4.9960937500000000000000000001', '0'], ['-0.5849609375000000000000000000', '5.4687500000000000000000000000', '-0.4130859375000000000000000000', '4.9960937500000000000000000001', '7.3581295013427734375000000001', '6.3847656250000000000000000000'], ['1.0000000000000000000000000000', '1.0000000000000000000000000000', '1.0000000000000000000000000000', '0', '6.3847656250000000000000000000', '14.000000000000000000000000000']], '0', '1'),
    ('tBd219#0', 'decimal', [['12.000000000000000000000000000', '1.0000000000000000000000000000', '-1.0000000000000000000000000000', '-1.0000000000000000000000000000', '6.9287109375000000000000000032', '-1.0000000000000000000000000000'], ['1.0000000000000000000000000000', '14.000000000000000000000000000', '1.0000000000000000000000000000', '1.0000000000000000000000000000', '9.022460937499999999999999997', '0'], ['-1.0000000000000000000000000000', '1.0000000000000000000000000000', '14.000000000000000000000000000', '1.0000000000000000000000000000', '-3.4423828125000000000000000452', '-1.0000000000000000000000000000'], ['-1.0000000000000000000000000000', '1.0000000000000000000000000000', '1.0000000000000000000000000000', '10.000000000000000000000000000', '-0.9482421875000000000000000032', '1.0000000000000000000000000000'], ['6.9287109375000000000000000032', '9.022460937499999999999999997', '-3.4423828125000000000000000452', '-0.9482421875000000000000000032', '10.593534469604492187605935367', '2.2978515625000000000000000032'], ['-1.0000000000000000000000000000', '0', '-1.0000000000000000000000000000', '1.0000000000000000000000000000', '2.2978515625000000000000000032', '14.000000000000000000000000000']], '0', '2'),
    ('tBg113#0', 'decimal', [['9.000000000000000000000000000', '0', '3.9111328125000000000000000000'], ['0', '9.000000000000000000000000000', '5.0888671874999999999999999927'], ['3.9111328125000000000000000000', '5.0888671874999999999999999927', '4.5770587921142578125000000375']], '0', '2'),
    # Counterexamples a fresh-seed held-out census (prefixes m3*) found against the rewritten core: degenerate-at-bound
    # problems refused or answered with a wrong vector at qualification, and near-boundary problems answered with a
    # wrong vector where a refusal was permitted.
    ('m3d1-106#1', 'decimal', [['13.000000000000000000000000000', '4.9882812500000000000000000162', '-1.0000000000000000000000000000'], ['4.9882812500000000000000000162', '4.5087852478027343749999998660', '4.1503906249999999999999998708'], ['-1.0000000000000000000000000000', '4.1503906249999999999999998708', '8.000000000000000000000000000']], '0', '2'),
    ('m3d1-110#1', 'decimal', [['4.0087890625000000000000000040', '4.1875000000000000000000000000', '3.8125000000000000000000000000'], ['4.1875000000000000000000000000', '8.000000000000000000000000000', '0'], ['3.8125000000000000000000000000', '0', '8.000000000000000000000000000']], '0', '2'),
    ('m3d1-156#2', 'decimal', [['11.000000000000000000000000000', '-1.0000000000000000000000000000', '0', '-1.0000000000000000000000000000', '1.0000000000000000000000000000', '0.9013671875000000000000000000', '1.0000000000000000000000000000'], ['-1.0000000000000000000000000000', '11.000000000000000000000000000', '-1.0000000000000000000000000000', '0', '0', '3.3886718750000000000000000032', '-1.0000000000000000000000000000'], ['0', '-1.0000000000000000000000000000', '11.000000000000000000000000000', '1.0000000000000000000000000000', '1.0000000000000000000000000000', '-5.3798828125000000000000000355', '0'], ['-1.0000000000000000000000000000', '0', '1.0000000000000000000000000000', '9.000000000000000000000000000', '1.0000000000000000000000000000', '3.4912109374999999999999999968', '0'], ['1.0000000000000000000000000000', '0', '1.0000000000000000000000000000', '1.0000000000000000000000000000', '16.000000000000000000000000000', '10.108398437499999999999999997', '1.0000000000000000000000000000'], ['0.9013671875000000000000000000', '3.3886718750000000000000000032', '-5.3798828125000000000000000355', '3.4912109374999999999999999968', '10.108398437499999999999999997', '12.171828269958496093750000036', '2.5292968750000000000000000000'], ['1.0000000000000000000000000000', '-1.0000000000000000000000000000', '0', '0', '1.0000000000000000000000000000', '2.5292968750000000000000000000', '11.000000000000000000000000000']], '0', '2'),
    ('m3d1-159#0', 'decimal', [['16.000000000000000000000000000', '9.468749999999999999999999897', '0'], ['9.468749999999999999999999897', '7.1032447814941406249999998788', '3.6738281250000000000000000000'], ['0', '3.6738281250000000000000000000', '9.000000000000000000000000000']], '0', '1'),
    ('m3d1-175#0', 'decimal', [['11.000000000000000000000000000', '5.6484374999999999999999999984', '1.0000000000000000000000000000'], ['5.6484374999999999999999999984', '7.4566802978515624999999999714', '9.027343749999999999999999974'], ['1.0000000000000000000000000000', '9.027343749999999999999999974', '16.000000000000000000000000000']], '0', '2'),
    ('m3d1-21#1', 'decimal', [['15.000000000000000000000000000', '1.0000000000000000000000000000', '1.0000000000000000000000000000', '17.447265624999999999999999998'], ['1.0000000000000000000000000000', '13.000000000000000000000000000', '0', '0.0322265624999999999999999999'], ['1.0000000000000000000000000000', '0', '11.000000000000000000000000000', '0.2187499999999999999999999999'], ['17.447265624999999999999999998', '0.0322265624999999999999999999', '0.2187499999999999999999999999', '20.475284576416015624999999997']], '0', '2'),
    ('m3d1-258#1', 'decimal', [['16.000000000000000000000000000', '-1.0000000000000000000000000000', '-1.0000000000000000000000000000', '5.3750000000000000000000000001'], ['-1.0000000000000000000000000000', '8.000000000000000000000000000', '1.0000000000000000000000000000', '1.3437499999999999999999999992'], ['-1.0000000000000000000000000000', '1.0000000000000000000000000000', '16.000000000000000000000000000', '7.2812499999999999999999999999'], ['5.3750000000000000000000000001', '1.3437499999999999999999999992', '7.2812499999999999999999999999', '5.6386718749999999999999999998']], '0.01', '1'),
    ('m3d1-38#0', 'decimal', [['16.000000000000000000000000000', '-1.0000000000000000000000000000', '0.4345703125000000000000000000', '-1.0000000000000000000000000000', '0'], ['-1.0000000000000000000000000000', '14.000000000000000000000000000', '-0.8740234374999999999999999998', '1.0000000000000000000000000000', '1.0000000000000000000000000000'], ['0.4345703125000000000000000000', '-0.8740234374999999999999999998', '6.4147586822509765625000000018', '6.0761718750000000000000000002', '5.4892578125000000000000000015'], ['-1.0000000000000000000000000000', '1.0000000000000000000000000000', '6.0761718750000000000000000002', '10.000000000000000000000000000', '1.0000000000000000000000000000'], ['0', '1.0000000000000000000000000000', '5.4892578125000000000000000015', '1.0000000000000000000000000000', '10.000000000000000000000000000']], '0', '2'),
    ('m3d1-8#2', 'decimal', [['5.3432025909423828125000000053', '-0.5439453125000000000000000000', '5.2080078125000000000000000000', '-1.7011718750000000000000000000', '4.2783203125000000000000000000', '4.1074218750000000000000000000'], ['-0.5439453125000000000000000000', '16.000000000000000000000000000', '-1.0000000000000000000000000000', '-1.0000000000000000000000000000', '-1.0000000000000000000000000000', '1.0000000000000000000000000000'], ['5.2080078125000000000000000000', '-1.0000000000000000000000000000', '12.000000000000000000000000000', '1.0000000000000000000000000000', '0', '1.0000000000000000000000000000'], ['-1.7011718750000000000000000000', '-1.0000000000000000000000000000', '1.0000000000000000000000000000', '16.000000000000000000000000000', '-1.0000000000000000000000000000', '0'], ['4.2783203125000000000000000000', '-1.0000000000000000000000000000', '0', '-1.0000000000000000000000000000', '13.000000000000000000000000000', '-1.0000000000000000000000000000'], ['4.1074218750000000000000000000', '1.0000000000000000000000000000', '1.0000000000000000000000000000', '0', '-1.0000000000000000000000000000', '11.000000000000000000000000000']], '0.01', '1'),
    ('m3g1-244#2', 'decimal', [['4.1475000000000000000000000002', '-0.5924999999999999999999999998', '-0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '-0.5924999999999999999999999998'], ['-0.5924999999999999999999999998', '4.1475000000000000000000000002', '-0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '-0.5924999999999999999999999998'], ['-0.5924999999999999999999999998', '-0.5924999999999999999999999998', '4.1475000000000000000000000002', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '-0.5924999999999999999999999998'], ['0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '4.1475000000000000000000000002', '-0.5924999999999999999999999998', '-0.5924999999999999999999999998', '-0.5924999999999999999999999998', '0.5924999999999999999999999998'], ['0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '-0.5924999999999999999999999998', '4.1475000000000000000000000002', '-0.5924999999999999999999999998', '-0.5924999999999999999999999998', '0.5924999999999999999999999998'], ['0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '-0.5924999999999999999999999998', '-0.5924999999999999999999999998', '4.1475000000000000000000000002', '-0.5924999999999999999999999998', '0.5924999999999999999999999998'], ['0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '-0.5924999999999999999999999998', '-0.5924999999999999999999999998', '-0.5924999999999999999999999998', '4.1475000000000000000000000002', '0.5924999999999999999999999998'], ['-0.5924999999999999999999999998', '-0.5924999999999999999999999998', '-0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '0.5924999999999999999999999998', '4.1475000000000000000000000002']], '0', '1'),
    ('m3g1-270#1', 'double', [['9.178054814637129E-11', '1.311150687805304E-11', '1.311150687805304E-11', '-1.311150687805304E-11', '-1.311150687805304E-11', '1.311150687805304E-11', '1.311150687805304E-11', '-1.311150687805304E-11'], ['1.311150687805304E-11', '9.178054814637129E-11', '-1.311150687805304E-11', '1.311150687805304E-11', '1.311150687805304E-11', '-1.311150687805304E-11', '-1.311150687805304E-11', '1.311150687805304E-11'], ['1.311150687805304E-11', '-1.311150687805304E-11', '9.178054814637129E-11', '1.311150687805304E-11', '1.311150687805304E-11', '-1.311150687805304E-11', '-1.311150687805304E-11', '1.311150687805304E-11'], ['-1.311150687805304E-11', '1.311150687805304E-11', '1.311150687805304E-11', '9.178054814637129E-11', '-1.311150687805304E-11', '1.311150687805304E-11', '1.311150687805304E-11', '-1.311150687805304E-11'], ['-1.311150687805304E-11', '1.311150687805304E-11', '1.311150687805304E-11', '-1.311150687805304E-11', '9.178054814637129E-11', '1.311150687805304E-11', '1.311150687805304E-11', '-1.311150687805304E-11'], ['1.311150687805304E-11', '-1.311150687805304E-11', '-1.311150687805304E-11', '1.311150687805304E-11', '1.311150687805304E-11', '9.178054814637129E-11', '-1.311150687805304E-11', '1.311150687805304E-11'], ['1.311150687805304E-11', '-1.311150687805304E-11', '-1.311150687805304E-11', '1.311150687805304E-11', '1.311150687805304E-11', '-1.311150687805304E-11', '9.178054814637129E-11', '1.311150687805304E-11'], ['-1.311150687805304E-11', '1.311150687805304E-11', '1.311150687805304E-11', '-1.311150687805304E-11', '-1.311150687805304E-11', '1.311150687805304E-11', '1.311150687805304E-11', '9.178054814637129E-11']], '-1000', '1000'),
    ('m3g1-77#1', 'double', [['1817532250.8463545', '363506450.1692709', '363506450.1692709', '-363506450.1692709', '-363506450.1692709', '363506450.1692709'], ['363506450.1692709', '1817532250.8463545', '-363506450.1692709', '363506450.1692709', '363506450.1692709', '-363506450.1692709'], ['363506450.1692709', '-363506450.1692709', '1817532250.8463545', '363506450.1692709', '363506450.1692709', '-363506450.1692709'], ['-363506450.1692709', '363506450.1692709', '363506450.1692709', '1817532250.8463545', '-363506450.1692709', '363506450.1692709'], ['-363506450.1692709', '363506450.1692709', '363506450.1692709', '-363506450.1692709', '1817532250.8463545', '363506450.1692709'], ['363506450.1692709', '-363506450.1692709', '-363506450.1692709', '363506450.1692709', '363506450.1692709', '1817532250.8463545']], '-1.1243346474838843E+88', '3.818090533826266E+89'),
    ('m3o1-109#1', 'decimal', [['5.7375000000000000000000000001', '1.9124999999999999999999999999', '-1.9124999999999999999999999999', '1.9124999999999999999999999999'], ['1.9124999999999999999999999999', '5.7375000000000000000000000001', '1.9124999999999999999999999999', '-1.9124999999999999999999999999'], ['-1.9124999999999999999999999999', '1.9124999999999999999999999999', '5.7375000000000000000000000001', '1.9124999999999999999999999999'], ['1.9124999999999999999999999999', '-1.9124999999999999999999999999', '1.9124999999999999999999999999', '5.7375000000000000000000000001']], '-1000', '1000'),
    ('m3o1-117#1', 'double', [['1.0888787212037332E-08', '2.177757442407466E-09', '-2.177757442407466E-09', '-2.177757442407466E-09', '2.177757442407466E-09', '2.177757442407466E-09'], ['2.177757442407466E-09', '1.0888787212037332E-08', '2.177757442407466E-09', '2.177757442407466E-09', '-2.177757442407466E-09', '-2.177757442407466E-09'], ['-2.177757442407466E-09', '2.177757442407466E-09', '1.0888787212037332E-08', '-2.177757442407466E-09', '2.177757442407466E-09', '2.177757442407466E-09'], ['-2.177757442407466E-09', '2.177757442407466E-09', '-2.177757442407466E-09', '1.0888787212037332E-08', '2.177757442407466E-09', '2.177757442407466E-09'], ['2.177757442407466E-09', '-2.177757442407466E-09', '2.177757442407466E-09', '2.177757442407466E-09', '1.0888787212037332E-08', '-2.177757442407466E-09'], ['2.177757442407466E-09', '-2.177757442407466E-09', '2.177757442407466E-09', '2.177757442407466E-09', '-2.177757442407466E-09', '1.0888787212037332E-08']], '-1', '2'),
    ('m3o1-17#0', 'decimal', [['5.4300000000000000000000000001', '1.8099999999999999999999999999', '1.8099999999999999999999999999', '-1.8099999999999999999999999999'], ['1.8099999999999999999999999999', '5.4300000000000000000000000001', '-1.8099999999999999999999999999', '1.8099999999999999999999999999'], ['1.8099999999999999999999999999', '-1.8099999999999999999999999999', '5.4300000000000000000000000001', '1.8099999999999999999999999999'], ['-1.8099999999999999999999999999', '1.8099999999999999999999999999', '1.8099999999999999999999999999', '5.4300000000000000000000000001']], '-1000', '1000'),
    ('m3o1-41#2', 'double', [['447645934868718', '89529186973743.6', '89529186973743.6', '-89529186973743.6', '89529186973743.6', '-89529186973743.6'], ['89529186973743.6', '447645934868718', '-89529186973743.6', '89529186973743.6', '-89529186973743.6', '89529186973743.6'], ['89529186973743.6', '-89529186973743.6', '447645934868718', '89529186973743.6', '-89529186973743.6', '89529186973743.6'], ['-89529186973743.6', '89529186973743.6', '89529186973743.6', '447645934868718', '89529186973743.6', '-89529186973743.6'], ['89529186973743.6', '-89529186973743.6', '-89529186973743.6', '89529186973743.6', '447645934868718', '89529186973743.6'], ['-89529186973743.6', '89529186973743.6', '89529186973743.6', '-89529186973743.6', '89529186973743.6', '447645934868718']], '-1', '2'),
    ('m3q1-105#0', 'double', [['0.0014076529383941169', '0.00028153058767882335', '-0.00028153058767882335', '0.00028153058767882335', '-0.00028153058767882335', '0.00028153058767882335'], ['0.00028153058767882335', '0.0014076529383941169', '0.00028153058767882335', '-0.00028153058767882335', '0.00028153058767882335', '-0.00028153058767882335'], ['-0.00028153058767882335', '0.00028153058767882335', '0.0014076529383941169', '0.00028153058767882335', '-0.00028153058767882335', '0.00028153058767882335'], ['0.00028153058767882335', '-0.00028153058767882335', '0.00028153058767882335', '0.0014076529383941169', '0.00028153058767882335', '-0.00028153058767882335'], ['-0.00028153058767882335', '0.00028153058767882335', '-0.00028153058767882335', '0.00028153058767882335', '0.0014076529383941169', '0.00028153058767882335'], ['0.00028153058767882335', '-0.00028153058767882335', '0.00028153058767882335', '-0.00028153058767882335', '0.00028153058767882335', '0.0014076529383941169']], '-1', '2'),
    ('m3q1-107#0', 'decimal', [['2.3400000000000000000000000002', '-0.7799999999999999999999999998', '0.7799999999999999999999999998', '0.7799999999999999999999999998'], ['-0.7799999999999999999999999998', '2.3400000000000000000000000002', '0.7799999999999999999999999998', '0.7799999999999999999999999998'], ['0.7799999999999999999999999998', '0.7799999999999999999999999998', '2.3400000000000000000000000002', '-0.7799999999999999999999999998'], ['0.7799999999999999999999999998', '0.7799999999999999999999999998', '-0.7799999999999999999999999998', '2.3400000000000000000000000002']], '-1000', '1000'),
    ('m3q1-129#0', 'decimal', [['1.4416666666666666666666666667', '0.2883333333333333333333333333', '-0.2883333333333333333333333333', '-0.2883333333333333333333333333', '0.2883333333333333333333333333', '0.2883333333333333333333333333'], ['0.2883333333333333333333333333', '1.4416666666666666666666666667', '0.2883333333333333333333333333', '0.2883333333333333333333333333', '-0.2883333333333333333333333333', '-0.2883333333333333333333333333'], ['-0.2883333333333333333333333333', '0.2883333333333333333333333333', '1.4416666666666666666666666667', '-0.2883333333333333333333333333', '0.2883333333333333333333333333', '0.2883333333333333333333333333'], ['-0.2883333333333333333333333333', '0.2883333333333333333333333333', '-0.2883333333333333333333333333', '1.4416666666666666666666666667', '0.2883333333333333333333333333', '0.2883333333333333333333333333'], ['0.2883333333333333333333333333', '-0.2883333333333333333333333333', '0.2883333333333333333333333333', '0.2883333333333333333333333333', '1.4416666666666666666666666667', '-0.2883333333333333333333333333'], ['0.2883333333333333333333333333', '-0.2883333333333333333333333333', '0.2883333333333333333333333333', '0.2883333333333333333333333333', '-0.2883333333333333333333333333', '1.4416666666666666666666666667']], '-1000', '1000'),
    ('m3q1-86#0', 'decimal', [['5.9675000000000000000000000001', '-0.8524999999999999999999999999', '0.8524999999999999999999999999', '0.8524999999999999999999999999', '0.8524999999999999999999999999', '-0.8524999999999999999999999999', '0.8524999999999999999999999999', '-0.8524999999999999999999999999'], ['-0.8524999999999999999999999999', '5.9675000000000000000000000001', '0.8524999999999999999999999999', '0.8524999999999999999999999999', '0.8524999999999999999999999999', '-0.8524999999999999999999999999', '0.8524999999999999999999999999', '-0.8524999999999999999999999999'], ['0.8524999999999999999999999999', '0.8524999999999999999999999999', '5.9675000000000000000000000001', '-0.8524999999999999999999999999', '-0.8524999999999999999999999999', '0.8524999999999999999999999999', '-0.8524999999999999999999999999', '0.8524999999999999999999999999'], ['0.8524999999999999999999999999', '0.8524999999999999999999999999', '-0.8524999999999999999999999999', '5.9675000000000000000000000001', '-0.8524999999999999999999999999', '0.8524999999999999999999999999', '-0.8524999999999999999999999999', '0.8524999999999999999999999999'], ['0.8524999999999999999999999999', '0.8524999999999999999999999999', '-0.8524999999999999999999999999', '-0.8524999999999999999999999999', '5.9675000000000000000000000001', '0.8524999999999999999999999999', '-0.8524999999999999999999999999', '0.8524999999999999999999999999'], ['-0.8524999999999999999999999999', '-0.8524999999999999999999999999', '0.8524999999999999999999999999', '0.8524999999999999999999999999', '0.8524999999999999999999999999', '5.9675000000000000000000000001', '0.8524999999999999999999999999', '-0.8524999999999999999999999999'], ['0.8524999999999999999999999999', '0.8524999999999999999999999999', '-0.8524999999999999999999999999', '-0.8524999999999999999999999999', '-0.8524999999999999999999999999', '0.8524999999999999999999999999', '5.9675000000000000000000000001', '0.8524999999999999999999999999'], ['-0.8524999999999999999999999999', '-0.8524999999999999999999999999', '0.8524999999999999999999999999', '0.8524999999999999999999999999', '0.8524999999999999999999999999', '-0.8524999999999999999999999999', '0.8524999999999999999999999999', '5.9675000000000000000000000001']], '-1000', '1000'),
    # Qualified degenerate-at-bound problems refused by the conditioning guard of the core with the exact subspace
    # minimiser (fresh-seed census prefixes r4a- / r4d-): the result is exact, but a free variable left exactly on its
    # bound made the deciding block numerically singular although the exact free block meets the margin.
    ('r4d-396#2', 'decimal', [['9.000000000000000000000000000', '-0.2275390625000000000000000000', '-1.0000000000000000000000000000', '0', '1.0000000000000000000000000000', '0', '-1.0000000000000000000000000000'], ['-0.2275390625000000000000000000', '7.0083780288696289062500000001', '5.3691406250000000000000000000', '-1.2949218750000000000000000000', '-2.4931640625000000000000000000', '6.9072265625000000000000000000', '3.1123046875000000000000000000'], ['-1.0000000000000000000000000000', '5.3691406250000000000000000000', '11.000000000000000000000000000', '0', '-1.0000000000000000000000000000', '0', '0'], ['0', '-1.2949218750000000000000000000', '0', '11.000000000000000000000000000', '1.0000000000000000000000000000', '0', '0'], ['1.0000000000000000000000000000', '-2.4931640625000000000000000000', '-1.0000000000000000000000000000', '1.0000000000000000000000000000', '10.000000000000000000000000000', '-1.0000000000000000000000000000', '1.0000000000000000000000000000'], ['0', '6.9072265625000000000000000000', '0', '0', '-1.0000000000000000000000000000', '16.000000000000000000000000000', '0'], ['-1.0000000000000000000000000000', '3.1123046875000000000000000000', '0', '0', '1.0000000000000000000000000000', '0', '11.000000000000000000000000000']], '0', '2'),
    ('r4d-628#0', 'double', [['2.75', '-0.0712890625', '-0.25', '-0.25'], ['-0.0712890625', '2.088701009750575', '0.29638671875', '2.279052734375'], ['-0.25', '0.29638671875', '3.25', '0.25'], ['-0.25', '2.279052734375', '0.25', '2.5']], '0', '2'),
    ('r4d-777#1', 'double', [['4194304', '-315392', '-524288', '524288'], ['-315392', '4243296.000000005', '3497984', '4210688'], ['-524288', '3497984', '6291456', '524288'], ['524288', '4210688', '524288', '6815744']], '0', '2'),
    ('r4a-435#1', 'double', [['12582912', '-151552', '-1048576', '0', '1048576'], ['-151552', '5492952.0000005495', '2166784', '6928384', '2449408'], ['-1048576', '2166784', '14680064', '0', '0'], ['0', '6928384', '0', '10485760', '-1048576'], ['1048576', '2449408', '0', '-1048576', '16777216']], '0', '2'),
    # Qualified decimal near-margin problems (VB-M-3 held-out family nearmo, prefixes v4p- / v4s-) answered with a wrong
    # vector, 1.06x and 2.02x the bound: the rounded reduced Hessian resolves the budget-orthogonal near-null direction
    # only coarsely, and the refinement stopped at its step limit short of the minimiser. tightub-10#0 is the same shape
    # built by hand (n = 10, v = 0.74, eps = 1.1e-25, alternating signs) with the upper bound 0.1003 just above the
    # interior optimum 0.1: a blocking bound decided on an unconverged refinement stopped on a vertex 3x the bound away.
    ('tightub-10#0', 'decimal', [['0.6660000000000000000000000081', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919'], ['0.0739999999999999999999999919', '0.6660000000000000000000000081', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919'], ['-0.0739999999999999999999999919', '0.0739999999999999999999999919', '0.6660000000000000000000000081', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919'], ['0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '0.6660000000000000000000000081', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919'], ['-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '0.6660000000000000000000000081', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919'], ['0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '0.6660000000000000000000000081', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919'], ['-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '0.6660000000000000000000000081', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919'], ['0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '0.6660000000000000000000000081', '0.0739999999999999999999999919', '-0.0739999999999999999999999919'], ['-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '0.6660000000000000000000000081', '0.0739999999999999999999999919'], ['0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '-0.0739999999999999999999999919', '0.0739999999999999999999999919', '0.6660000000000000000000000081']], '-1', '0.1003'),
    ('v4p-302#1', 'decimal', [['0.8066666666666666666666666784', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216'], ['-0.0733333333333333333333333216', '0.8066666666666666666666666784', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216'], ['-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.8066666666666666666666666784', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216'], ['0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.8066666666666666666666666784', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216'], ['0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.8066666666666666666666666784', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216'], ['0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.8066666666666666666666666784', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216'], ['0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.8066666666666666666666666784', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216'], ['-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.8066666666666666666666666784', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216'], ['0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.8066666666666666666666666784', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216'], ['-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.8066666666666666666666666784', '0.0733333333333333333333333216', '-0.0733333333333333333333333216'], ['0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.8066666666666666666666666784', '0.0733333333333333333333333216'], ['-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '-0.0733333333333333333333333216', '0.0733333333333333333333333216', '0.8066666666666666666666666784']], '0', '1'),
    ('v4s-149#0', 'decimal', [['0.6660000000000000000000000078', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922'], ['0.0739999999999999999999999922', '0.6660000000000000000000000078', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '-0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922'], ['-0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.6660000000000000000000000078', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922'], ['0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.6660000000000000000000000078', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '-0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922'], ['-0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.6660000000000000000000000078', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922'], ['-0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.6660000000000000000000000078', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922'], ['0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.6660000000000000000000000078', '-0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922'], ['0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.6660000000000000000000000078', '-0.0739999999999999999999999922', '0.0739999999999999999999999922'], ['0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.6660000000000000000000000078', '0.0739999999999999999999999922'], ['-0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '-0.0739999999999999999999999922', '-0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.0739999999999999999999999922', '0.6660000000000000000000000078']], '-1', '2'),
]


def heldout_cases():
    """The held-out witnesses as corpus cases, from HELDOUT_WITNESSES and the census dumps in heldout_dumps."""
    out = []
    for cid, T, cov, lb, ub in HELDOUT_WITNESSES + HELDOUT_DUMPS:
        out.append({"id": cid, "family": "heldout-dbl" if T == "double" else "heldout-dec", "T": T, "mode": "minvar",
                    "cov": cov, "lb": lb, "ub": ub})
    return out


# --------------------------------------------------------------------------------------------
# Corpus definition
# --------------------------------------------------------------------------------------------

# (generator, family, seed, count)
PLAN = [
    ("r4", "dbl-corr", 6, 60),
    ("r4", "dbl-corr", 8, 60),
    ("r4", "dbl-corr", 9, 60),
    ("r4", "dbl-minvar", 6, 60),
    ("r4", "dbl-minvar", 8, 60),
    ("r4", "dec-corr", 6, 60),
    ("r4", "dec-corr", 8, 60),
    ("r4", "dec-diag-count", 6, 40),
    ("r5", "r5-dbl-disp-mv", 21, 200),
    ("r5", "r5-dec-rng-mean", 11, 120),
    ("r5", "r5-dec-rng-budget", 11, 120),
    ("r5", "r5-dec-rng-budget-min", 11, 120),
    ("r5", "r5-dec-rng-zc", 11, 120),
    ("r5", "r5-dec-zc2", 11, 120),
    ("r5", "r5-rng-dbl-mean", 11, 80),
    ("r5", "r5-rng-dbl-budget", 11, 120),
    ("r5", "r5-rng-dbl-budget-min", 11, 120),
    ("r5", "r5-rng-dbl-zc", 11, 120),
    ("hand", "hd-hand", 0, 8),
    # added with the 48.n.u.cond_eq qualification: census-shape families and the held-out witnesses
    ("census", "nearq-dbl", 31, 60),
    ("census", "nearq-dec", 31, 60),
    ("census", "nearo-dbl", 31, 60),
    ("census", "nearo-dec", 31, 60),
    ("census", "degenb-dbl", 31, 120),
    ("census", "degenb-dec", 31, 120),
    # the VB-M-3 near-margin held-out family (its two qualified decimal wrong vectors are heldout-dec witnesses)
    ("census", "nearmo-dbl", 41, 24),
    ("census", "nearmo-dec", 41, 43),
    ("witness", "heldout-dbl", 0, 20),
    ("witness", "heldout-dec", 0, 25),
]

# Families of the M-1 corpus (the case set the pre-rewrite baseline counts and timing ratios were recorded on).
M1_FAMILIES = [
    "dbl-corr", "dbl-minvar", "dec-corr", "dec-diag-count", "r5-dbl-disp-mv", "r5-dec-rng-mean", "r5-dec-rng-budget",
    "r5-dec-rng-budget-min", "r5-dec-rng-zc", "r5-dec-zc2", "r5-rng-dbl-mean", "r5-rng-dbl-budget",
    "r5-rng-dbl-budget-min", "r5-rng-dbl-zc", "hd-hand",
]

WITNESSES = [
    "dbl-corr-6-28", "dbl-corr-8-49", "hd-diag4-12", "r5-dec-rng-zc-37", "r5-dec-rng-zc-47",
    "r5-dec-rng-mean-0", "r5-dbl-disp-mv-10",
] + [w[0] for w in HELDOUT_WITNESSES] + [w[0] for w in HELDOUT_DUMPS]


def build_cases():
    cases, fams = [], []
    for gen, fam, seed, count in PLAN:
        if gen == "r4":
            cs = r4_family(fam, seed, count)
        elif gen == "r5":
            cs = r5_family(fam, seed, count)
        elif gen == "census":
            cs = census_family(fam, seed, count)
        elif gen == "witness":
            cs = [c for c in heldout_cases() if c["family"] == fam]
        else:
            cs = hand_cases()
        if len(cs) != count:
            raise SystemExit(f"{fam}: generated {len(cs)} of {count}")
        for c in cs:
            if not exact_in_type(c):
                raise SystemExit("input not exactly representable in the working type: " + c["id"])
        entry = next((f for f in fams if f["family"] == fam), None)
        if entry is None:
            entry = {"family": fam, "generator": gen, "seeds": [], "count": 0}
            fams.append(entry)
        entry["seeds"].append(seed)
        entry["count"] += count
        cases += cs
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate case ids")
    missing = [w for w in WITNESSES if w not in ids]
    if missing:
        raise SystemExit("missing witnesses " + str(missing))
    if len(cases) > CORPUS_CAP:
        raise SystemExit("corpus exceeds the cap")
    return cases, fams


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    cases, fams = build_cases()
    with multiprocessing.Pool() as pool:
        rich = pool.map(enrich, cases, chunksize=4)
    byT = {"double": [], "decimal": []}
    for c, r in zip(cases, rich):
        byT[c["T"]].append(r)
    for T in ("double", "decimal"):
        doc_fams = [f for f in fams if any(c["family"] == f["family"] and c["T"] == T for c in cases)]
        path = os.path.join(outdir, f"corpus-{T}.json")
        with atomic_text(Path(path)) as fh:
            fh.write('{"schema_version":1,"type":"%s","qualification":"%s","families":%s,"witnesses":%s,\n"cases":[\n'
                     % (T, "in_domain and representable and %d*n*u*cond_eq < 1" % QUALIFICATION_C,
                        json.dumps(doc_fams, separators=(",", ":")),
                        json.dumps([w for w in WITNESSES if any(r["id"] == w for r in byT[T])], separators=(",", ":"))))
            fh.write(",\n".join(json.dumps(r, separators=(",", ":")) for r in byT[T]))
            fh.write("\n]}\n")
        print(path, len(byT[T]), "cases,", sum(1 for r in byT[T] if r["qualified"]), "qualified")


if __name__ == "__main__":
    main()
