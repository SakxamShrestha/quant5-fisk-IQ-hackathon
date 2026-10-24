"""Weight sweep, Pareto-front recovery, hypervolume and random baselines."""
from itertools import product

import numpy as np
from scipy.optimize import minimize

from evqaoa import (bitstring_to_schedule, charging_model, energy_function, is_valid,
                    output_distribution, qaoa_circuit, schedule_metrics, to_ising,
                    valid_schedules)

COBYLA_TOL = 3e-3
MAXITER = 400


def weight_grid(denominator):
    """All (w1, w2, w3) = (i, j, k)/denominator with i+j+k = denominator."""
    d = denominator
    return [(i / d, j / d, (d - i - j) / d)
            for i, j in product(range(d + 1), repeat=2) if i + j <= d]


def metrics_or_none(problem, bitstring):
    """Metric triple of a measured bitstring, or None if it breaks a demand."""
    x = bitstring_to_schedule(bitstring)
    return schedule_metrics(problem, x) if is_valid(problem, x) else None


def sample_schedules(dist, shots, rng, problem):
    """Draw `shots` bitstrings from {bitstring: probability}; return metric triples."""
    keys = list(dist)
    probs = np.array([dist[k] for k in keys], dtype=float)
    picks = rng.choice(len(keys), size=shots, p=probs / probs.sum())
    cache = {}
    out = []
    for i in picks:
        if i not in cache:
            cache[i] = metrics_or_none(problem, keys[i])
        out.append(cache[i])
    return out


def _optimise(energy, n_params, start, maxiter, tol, rng):
    x0 = np.asarray(start, float) if start is not None else rng.uniform(0, np.pi, n_params)
    return minimize(energy, x0, method='COBYLA', tol=tol, options={'maxiter': maxiter})


def run_weighting(problem, weights, p=2, start=None, shots=256, seed=0,
                  maxiter=MAXITER, tol=COBYLA_TOL):
    """Optimise XY+Dicke QAOA for one weighting, then draw shot-sampled schedules.

    start=None uses a random angle guess (independent run); otherwise the given
    angles are the starting point (warm start). The optimiser stops at `tol`
    or after `maxiter` evaluations, whichever comes first.
    """
    model = charging_model(problem, weights)
    circ = qaoa_circuit(problem, to_ising(model), p, mixer='xy', start='dicke')
    rng = np.random.default_rng(seed)
    res = _optimise(energy_function(circ, to_ising(model)), 2 * p, start, maxiter, tol, rng)
    dist = output_distribution(circ, res.x)
    return dict(weights=tuple(weights), theta=[float(t) for t in res.x],
                energy=float(res.fun), nfev=int(res.nfev), hit_cap=bool(res.nfev >= maxiter),
                samples=sample_schedules(dist, shots, rng, problem))


def front_recovery(points, true_front):
    """Number of true Pareto points present among the given metric triples."""
    found = {tuple(float(v) for v in q) for q in points if q is not None}
    return len(found & {tuple(float(v) for v in q) for q in true_front})


def hypervolume(points, ref):
    """Volume dominated by points (minimisation) and bounded by ref, by grid enumeration."""
    pts = np.array([q for q in points if q is not None], dtype=float).reshape(-1, 3)
    ref = np.asarray(ref, dtype=float)
    pts = pts[(pts < ref).all(axis=1)]
    if len(pts) == 0:
        return 0.0
    axes = [np.unique(np.append(pts[:, k], ref[k])) for k in range(3)]
    gx, gy, gz = np.meshgrid(*[a[:-1] for a in axes], indexing='ij')
    cell = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    sizes = np.stack(np.meshgrid(*[np.diff(a) for a in axes], indexing='ij'), -1).reshape(-1, 3)
    dominated = (pts[None, :, :] <= cell[:, None, :]).all(axis=2).any(axis=1)
    return float(sizes.prod(axis=1)[dominated].sum())


def random_baseline(problem, shots, seed=0, valid_only=True):
    """Random schedules: uniform over valid ones, or uniform over all 2^n bitstrings."""
    rng = np.random.default_rng(seed)
    if valid_only:
        table = [schedule_metrics(problem, x) for x in valid_schedules(problem)]
        return [table[i] for i in rng.integers(0, len(table), size=shots)]
    n = problem.n_qubits
    return [metrics_or_none(problem, format(int(i), f'0{n}b'))
            for i in rng.integers(0, 2 ** n, size=shots)]


def cumulative_curves(sequence, true_front, ref, checkpoints):
    """Front points recovered and hypervolume after the first k shots, for each k."""
    recovered, volumes = [], []
    for k in checkpoints:
        head = sequence[:k]
        recovered.append(front_recovery(head, true_front))
        volumes.append(hypervolume(sorted({q for q in head if q}), ref))
    return recovered, volumes


def shots_to_full_front(sequence, true_front):
    """Index (1-based) of the shot at which every front point has been seen, else None."""
    target = {tuple(float(v) for v in q) for q in true_front}
    seen = set()
    for i, q in enumerate(sequence, 1):
        if q is not None and tuple(q) in target:
            seen.add(tuple(q))
            if seen == target:
                return i
    return None
