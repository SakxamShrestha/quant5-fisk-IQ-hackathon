"""Shared optimisation and statevector analysis helpers."""
import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from .problem import all_feasible, bits_to_x, is_feasible


def brute_force_baselines(inst, Q, const):
    """Mean, best and worst QUBO energy over valid schedules."""
    vals = np.array([x @ Q @ x + const for x in all_feasible(inst)])
    return float(vals.mean()), float(vals.min()), float(vals.max())


def expectation_energy(circ, H):
    """Return f(theta) = <psi(theta)| H |psi(theta)> via exact statevector."""
    return lambda theta: Statevector(circ.assign_parameters(theta)).expectation_value(H).real


def optimise(f, n_params, method='COBYLA', restarts=3, maxiter=300, seed=0, x_init=None):
    """Best of several scipy.minimize runs from random (or a given) start."""
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        use_init = x_init is not None and r == 0
        x0 = np.asarray(x_init) if use_init else rng.uniform(0, np.pi, n_params)
        res = minimize(f, x0, method=method, options={'maxiter': maxiter})
        if best is None or res.fun < best.fun:
            best = res
    return best


def distribution_metrics(probs, inst, Q, const, E_best, E_worst):
    """Feasible rate, P(optimal) and approximation ratio of the feasible part of a distribution.

    probs: {bitstring: probability or count}. Counts are normalised.
    """
    total = sum(probs.values())
    feasible, p_opt, e_feasible = 0.0, 0.0, 0.0
    for key, w in probs.items():
        pr = w / total
        x = bits_to_x(key)
        if not is_feasible(inst, x):
            continue
        e = float(x @ Q @ x + const)
        feasible += pr
        e_feasible += pr * e
        if abs(e - E_best) < 1e-9:
            p_opt += pr
    mean_e = e_feasible / feasible if feasible else float('nan')
    approx = (E_worst - mean_e) / (E_worst - E_best) if feasible else float('nan')
    return dict(feasible_rate=feasible, p_optimal=p_opt,
                mean_feasible_energy=mean_e, approx_ratio=approx)
