"""Optimiser helpers for QAOA experiments: counted objective, shot sampling, CVaR, SPSA."""
import numpy as np
from scipy.optimize import minimize

SPSA_A, SPSA_C, SPSA_ALPHA, SPSA_GAMMA = 3.0, 0.2, 0.602, 0.101


class CountedObjective:
    """Wrap f so every call is counted and recorded; stop (StopIteration) at budget."""

    def __init__(self, f, budget=None):
        self.f, self.budget = f, budget
        self.count, self.values, self.points = 0, [], []

    def __call__(self, x):
        if self.budget is not None and self.count >= self.budget:
            raise StopIteration
        val = float(self.f(x))
        self.count += 1
        self.values.append(val)
        self.points.append(np.array(x, dtype=float))
        return val


def sample_energies(probs, energies, shots, rng):
    """Draw `shots` bitstrings (multinomial on probs); return their energies."""
    p = np.clip(probs, 0, None)
    idx = rng.choice(len(p), size=shots, p=p / p.sum())
    return energies[idx]


def cvar(samples, alpha):
    """Mean of the lowest alpha fraction of samples (alpha=1 is the plain mean)."""
    if not 0 < alpha <= 1:
        raise ValueError('alpha must be in (0, 1]')
    s = np.sort(np.asarray(samples, dtype=float))
    k = max(1, int(np.ceil(alpha * len(s))))
    return float(s[:k].mean())


def spsa(f, x0, maxiter=100, seed=0):
    """Simultaneous-perturbation SPSA: 2 evaluations per iteration. Returns (x, f(x))."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    for k in range(1, maxiter + 1):
        ak = SPSA_A / (k + 10) ** SPSA_ALPHA
        ck = SPSA_C / k ** SPSA_GAMMA
        delta = rng.choice([-1.0, 1.0], size=x.shape)
        grad = (f(x + ck * delta) - f(x - ck * delta)) / (2 * ck) * delta
        x = x - ak * grad
    return x, float(f(x))


def _scipy_options(method, budget):
    key = 'maxiter' if method == 'COBYLA' else 'maxfun' if method == 'L-BFGS-B' else 'maxfev'
    return {key: budget}


def run_optimizer(f, x0, method, budget, seed=0):
    """Run one optimiser under an evaluation budget.

    Returns dict(x, fun, nfev, values, points). x is the optimiser's answer, or the
    best-observed point when the budget cut the run short.
    """
    obj = CountedObjective(f, budget)
    try:
        if method == 'SPSA':
            x, _ = spsa(obj, x0, maxiter=budget // 2 - 1, seed=seed)
        else:
            x = minimize(obj, x0, method=method, options=_scipy_options(method, budget)).x
    except StopIteration:
        x = obj.points[int(np.argmin(obj.values))]
    return dict(x=np.asarray(x), fun=float(min(obj.values)), nfev=obj.count,
                values=obj.values, points=obj.points)
