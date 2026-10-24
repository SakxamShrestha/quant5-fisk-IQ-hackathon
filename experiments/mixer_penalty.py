"""Penalty-weight sweep and mixer comparison on the depot charging problem.

The X-mixer circuits are optimised on a Hamiltonian that includes the demand
penalty; every result is scored on the penalty-free model over valid schedules.
"""
import time

import numpy as np

from evqaoa import (charging_model, count_valid, energy_function, exact_baselines,
                    minimise_energy, output_distribution, qaoa_circuit,
                    score_distribution, to_ising)

WEIGHTS = (0.5, 0.3, 0.2)


def random_baseline(problem):
    """Valid share of a uniformly random bitstring: valid schedules / 2^n."""
    return count_valid(problem) / 2 ** problem.n_qubits


def build_configs(penalties):
    """Config dicts: X+Hadamard per penalty, then XY+Dicke and XY+basis."""
    xs = [dict(name=f'X+H pen={pen:g}', mixer='x', start='hadamard', penalty=pen)
          for pen in penalties]
    xy = [dict(name='XY+Dicke', mixer='xy', start='dicke', penalty=0),
          dict(name='XY+basis', mixer='xy', start='basis', penalty=0)]
    return xs + xy


def make_study(problem, weights=WEIGHTS):
    """Penalty-free scoring model plus its exact baselines."""
    model = charging_model(problem, weights)
    return dict(problem=problem, weights=tuple(weights), model=model,
                baselines=exact_baselines(problem, model))


def _circuit(study, cfg, p):
    """Circuit and the Hamiltonian it is optimised against (penalised if asked)."""
    model = charging_model(study['problem'], study['weights'], penalty=cfg['penalty'])
    H = to_ising(model)
    return qaoa_circuit(study['problem'], H, p, mixer=cfg['mixer'], start=cfg['start']), H


def evaluate_angles(study, cfg, theta):
    """Penalty-free scores of the QAOA state at the given angles."""
    theta = np.asarray(theta, dtype=float)
    circ, _ = _circuit(study, cfg, len(theta) // 2)
    dist = output_distribution(circ, theta)
    return score_distribution(dist, study['problem'], study['model'], study['baselines'])


def _extend(theta, p_new):
    """Warm start: keep old betas/gammas, append a zero-angle layer."""
    p_old = len(theta) // 2
    pad = [0.0] * (p_new - p_old)
    return np.array(list(theta[:p_old]) + pad + list(theta[p_old:]) + pad)


def run_config(study, cfg, p_max=3, restarts=3, maxiter=200, seed=0):
    """Optimise one config for p = 1..p_max (warm-started); one dict per p."""
    rows, theta = [], None
    for p in range(1, p_max + 1):
        circ, H = _circuit(study, cfg, p)
        t0 = time.time()
        res = minimise_energy(energy_function(circ, H), 2 * p, 'COBYLA',
                              restarts=restarts if theta is None else max(1, restarts - 1),
                              maxiter=maxiter, seed=seed + p,
                              start=None if theta is None else _extend(theta, p))
        theta = res.x
        scores = evaluate_angles(study, cfg, theta)
        rows.append(dict(name=cfg['name'], mixer=cfg['mixer'], start=cfg['start'],
                         penalty=cfg['penalty'], p=p, seed=seed, energy=float(res.fun),
                         theta=[float(v) for v in theta], n_evals=int(res.nfev_total),
                         seconds=time.time() - t0,
                         **{k: float(v) for k, v in scores.items()}))
    return rows


def run_sweep(study, penalties, p_max=3, restarts=3, maxiter=200, seeds=(0,)):
    """Every config for every seed; flat list of per-(config, seed, p) dicts."""
    return [row for seed in seeds for cfg in build_configs(penalties)
            for row in run_config(study, cfg, p_max, restarts, maxiter, seed)]


def summarise(rows, metric):
    """{(name, p): (mean, std, best)} of a metric across seeds."""
    groups = {}
    for r in rows:
        groups.setdefault((r['name'], r['p']), []).append(r[metric])
    return {k: (float(np.mean(v)), float(np.std(v)), float(np.max(v)))
            for k, v in groups.items()}
