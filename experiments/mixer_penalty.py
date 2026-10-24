"""Penalty-weight sweep and mixer comparison on a QAOA EV-charging instance.

Hamiltonians used for optimisation include the penalty (X mixer only); all
metrics are scored on the penalty-free objective over feasible schedules.
"""
import time
from math import comb

import numpy as np
from qiskit.quantum_info import Statevector

from evqaoa import (brute_force_baselines, build_qaoa, build_qubo,
                    distribution_metrics, expectation_energy, optimise,
                    qubo_to_hamiltonian)


def random_baseline(inst):
    """Feasible fraction of the uniform distribution over all 2^n bitstrings."""
    count = 1
    for need in inst.need:
        count *= comb(inst.T, need)
    return count / 2 ** inst.n


def build_configs(penalties):
    """Config dicts: X+Hadamard per penalty, then XY+Dicke and XY+basis."""
    xs = [dict(name=f'X+H pen={pen:g}', mixer='x', start='hadamard', penalty=pen)
          for pen in penalties]
    xy = [dict(name='XY+Dicke', mixer='xy', start='dicke', penalty=0),
          dict(name='XY+basis', mixer='xy', start='basis', penalty=0)]
    return xs + xy


def make_problem(inst, lam, scale):
    """Penalty-free QUBO plus brute-force best/worst feasible energies."""
    Q, c = build_qubo(inst, lam, scale)
    _, e_best, e_worst = brute_force_baselines(inst, Q, c)
    return dict(Q=Q, c=c, E_best=e_best, E_worst=e_worst, lam=tuple(lam),
                scale=tuple(float(s) for s in scale))


def _circuit(inst, problem, cfg, p):
    """Circuit and optimisation Hamiltonian (with penalty when cfg asks)."""
    Qp, cp = build_qubo(inst, problem['lam'], problem['scale'], cfg['penalty'])
    H = qubo_to_hamiltonian(Qp, cp)
    start = 'dicke' if cfg['start'] == 'hadamard' else cfg['start']
    return build_qaoa(inst, H, p, mixer=cfg['mixer'], start=start), H


def _probabilities(circ, theta):
    sv = Statevector(circ.assign_parameters(theta))
    return sv.probabilities_dict()


def evaluate_angles(inst, problem, cfg, theta):
    """Penalty-free metrics of the QAOA state at the given angles."""
    p = len(theta) // 2
    circ, _ = _circuit(inst, problem, cfg, p)
    probs = _probabilities(circ, np.asarray(theta, dtype=float))
    return distribution_metrics(probs, inst, problem['Q'], problem['c'],
                                problem['E_best'], problem['E_worst'])


def _extend(theta, p_new):
    """Warm start: keep old betas/gammas, append a zero-angle layer."""
    p_old = len(theta) // 2
    betas, gammas = list(theta[:p_old]), list(theta[p_old:])
    return np.array(betas + [0.0] * (p_new - p_old) + gammas + [0.0] * (p_new - p_old))


def run_config(inst, problem, cfg, p_max=3, restarts=3, maxiter=200, seed=0):
    """Optimise one config for p = 1..p_max (warm-started); one dict per p."""
    rows, theta = [], None
    for p in range(1, p_max + 1):
        circ, H = _circuit(inst, problem, cfg, p)
        t0 = time.time()
        res = optimise(expectation_energy(circ, H), 2 * p, 'COBYLA',
                       restarts=restarts if theta is None else max(1, restarts - 1),
                       maxiter=maxiter, seed=seed + p,
                       x_init=None if theta is None else _extend(theta, p))
        theta = res.x
        metrics = evaluate_angles(inst, problem, cfg, theta)
        rows.append(dict(name=cfg['name'], mixer=cfg['mixer'], start=cfg['start'],
                         penalty=cfg['penalty'], p=p, seed=seed, energy=float(res.fun),
                         theta=[float(v) for v in theta], n_evals=int(res.nfev),
                         seconds=time.time() - t0,
                         **{k: float(v) for k, v in metrics.items()}))
    return rows


def run_sweep(inst, problem, penalties, p_max=3, restarts=3, maxiter=200, seeds=(0,)):
    """Run every config for every seed; flat list of per-(config, seed, p) dicts."""
    return [row for seed in seeds for cfg in build_configs(penalties)
            for row in run_config(inst, problem, cfg, p_max, restarts, maxiter, seed)]


def summarise(rows, metric):
    """{(name, p): (mean, std, best)} of a metric across seeds."""
    groups = {}
    for r in rows:
        groups.setdefault((r['name'], r['p']), []).append(r[metric])
    return {k: (float(np.mean(v)), float(np.std(v)), float(np.max(v)))
            for k, v in groups.items()}
