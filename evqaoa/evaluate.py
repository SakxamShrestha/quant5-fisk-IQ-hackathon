"""Exact baselines, energy evaluation, angle optimisation and result scoring."""
from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from .problem import bitstring_to_schedule, is_valid, valid_schedules


@dataclass(frozen=True)
class Baselines:
    best: float     # optimum over valid schedules
    worst: float    # worst valid schedule
    mean: float     # average valid schedule = a random valid guess
    n_valid: int
    n_optimal: int  # valid schedules tied at the optimum


def exact_baselines(problem, model):
    """Brute force over valid schedules only; the score never sees the penalty."""
    vals = model.values(np.array(list(valid_schedules(problem))))
    best = float(vals.min())
    return Baselines(best, float(vals.max()), float(vals.mean()), len(vals),
                     int(np.isclose(vals, best, atol=1e-9).sum()))


def energy_function(circuit, hamiltonian):
    """theta -> <psi(theta)|H|psi(theta)>, exact statevector."""
    def energy(theta):
        return float(Statevector(circuit.assign_parameters(theta))
                     .expectation_value(hamiltonian).real)
    return energy


def output_distribution(circuit, theta):
    return Statevector(circuit.assign_parameters(theta)).probabilities_dict()


def minimise_energy(energy, n_params, method='COBYLA', restarts=3, maxiter=300,
                    seed=0, start=None):
    """Best scipy.minimize result over `restarts` runs; nfev_total sums all runs."""
    rng = np.random.default_rng(seed)
    best, total_evals = None, 0
    for r in range(restarts):
        use_start = start is not None and r == 0
        x0 = np.asarray(start, dtype=float) if use_start else rng.uniform(0, np.pi, n_params)
        res = minimize(energy, x0, method=method, options={'maxiter': maxiter})
        total_evals += int(res.nfev)
        if best is None or res.fun < best.fun:
            best = res
    best.nfev_total = total_evals
    return best


def score_distribution(dist, problem, model, baselines, tol=1e-9):
    """Score {bitstring: count or probability} against the exact baselines.

    valid_rate:      share of shots that obey every vehicle's demand.
    p_optimal:       share of shots that are an optimal schedule.
    p_optimal_valid: p_optimal after discarding invalid shots (post-selection).
    approx_ratio:    (worst - mean valid energy) / (worst - best), valid shots only.
    """
    total = float(sum(dist.values()))
    valid = optimal = weighted = 0.0
    for key, w in dist.items():
        x = bitstring_to_schedule(key)
        if not is_valid(problem, x):
            continue
        share, e = w / total, model.value(x)
        valid += share
        weighted += share * e
        optimal += share * (abs(e - baselines.best) < tol)
    if valid == 0:
        return dict(valid_rate=0.0, p_optimal=0.0, p_optimal_valid=0.0,
                    mean_valid_energy=float('nan'), approx_ratio=float('nan'))
    mean_e = weighted / valid
    span = baselines.worst - baselines.best
    return dict(valid_rate=float(valid), p_optimal=float(optimal),
                p_optimal_valid=float(optimal / valid), mean_valid_energy=float(mean_e),
                approx_ratio=float((baselines.worst - mean_e) / span) if span else 1.0)
