"""Weight sweep, Pareto-front recovery, hypervolume and random baselines."""
from itertools import product

import numpy as np
from qiskit.quantum_info import Statevector

from evqaoa import (all_feasible, bits_to_x, build_qaoa, build_qubo, is_feasible,
                    objectives, optimise, qubo_to_hamiltonian)


def weight_grid(denominator):
    """All (w1, w2, w3) = (i, j, k)/denominator with i+j+k = denominator."""
    d = denominator
    return [(i / d, j / d, (d - i - j) / d)
            for i, j in product(range(d + 1), repeat=2) if i + j <= d]


def sample_schedules(probs, shots, rng, inst):
    """Draw shots bitstring indices from probs; return objective triples (None if infeasible)."""
    idx = rng.choice(len(probs), size=shots, p=probs / probs.sum())
    out = []
    for i in idx:
        x = bits_to_x(format(int(i), f'0{inst.n}b'))
        out.append(objectives(inst, x) if is_feasible(inst, x) else None)
    return out


def run_weighting(inst, lam, scale, p=2, x_init=None, shots=256, seed=0, restarts=2):
    """Optimise QAOA for one weighting; return angles, energy and shot-sampled triples."""
    Q, const = build_qubo(inst, lam=lam, scale=scale)
    H = qubo_to_hamiltonian(Q, const)
    circ = build_qaoa(inst, H, p, mixer='xy', start='dicke')
    f = lambda th: Statevector(circ.assign_parameters(th)).expectation_value(H).real
    res = optimise(f, 2 * p, restarts=restarts, seed=seed, x_init=x_init)
    probs = Statevector(circ.assign_parameters(res.x)).probabilities()
    rng = np.random.default_rng(seed)
    return dict(lam=tuple(lam), theta=[float(t) for t in res.x], energy=float(res.fun),
                nfev=int(res.nfev), samples=sample_schedules(probs, shots, rng, inst))


def front_recovery(points, true_front):
    """Number of true Pareto points present among the given objective triples."""
    found = {tuple(int(v) for v in p) for p in points if p is not None}
    return len(found & {tuple(p) for p in true_front})


def hypervolume(points, ref):
    """Volume dominated by points (minimisation) and bounded by ref, by grid enumeration."""
    pts = np.array([p for p in points if p is not None], dtype=float).reshape(-1, 3)
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


def random_baseline(inst, shots, seed=0, feasible_only=True):
    """Uniform random schedules (or any of 2^n bitstrings, infeasible -> None)."""
    rng = np.random.default_rng(seed)
    if feasible_only:
        table = [objectives(inst, x) for x in all_feasible(inst)]
        return [table[i] for i in rng.integers(0, len(table), size=shots)]
    probs = np.ones(2 ** inst.n)
    return sample_schedules(probs, shots, rng, inst)


def cumulative_curves(sequence, true_front, ref, checkpoints):
    """Front points recovered and hypervolume after the first k shots, for each k."""
    recovered, volumes = [], []
    for k in checkpoints:
        head = sequence[:k]
        recovered.append(front_recovery(head, true_front))
        volumes.append(hypervolume(sorted({p for p in head if p}), ref))
    return recovered, volumes
