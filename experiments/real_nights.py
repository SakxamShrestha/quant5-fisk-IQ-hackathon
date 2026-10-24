"""QAOA on many real depot nights built from NLR delivery-truck schedules.

A night is three truck-days from one fleet charging in the depot's 10 pm - 2 am window
(12 qubits, like DEPOT). XY + Dicke QAOA is optimised on the exact valid-plan simulator of
experiments.scaling and compared, in dollars, kW and start/stops, with simple charging rules.

Start angles: beta from [0, pi] and gamma from [0, 2 pi / spread], where spread is the range
of the model's energies over valid plans, so the starts cover one full turn of the cost
layer (studies 01-07 still draw gamma from [0, pi]). Depth p also starts from an INTERP
stretch of the depth p-1 angles (Zhou et al. 2020).
"""
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from evqaoa import (ChargingProblem, charging_model, exact_baselines, qaoa_circuit,
                    to_ising, valid_schedules)
from evqaoa.real_data import (CHARGER_KW, PLACEHOLDER_TARIFF, eligible, load_truck_days,
                              make_night, sample_nights)
from experiments.hardware import TRANSPILE_SEED, submit_runs
from experiments.noise import transpile_for
from experiments.scaling import feasible_basis, make_simulator, metrics

OUT_DIR = Path(__file__).resolve().parents[1] / 'results' / '08_real_fleet'
FLEET = 'fleet2-warehouse-delivery'
WEIGHTS = (0.5, 0.3, 0.2)
N_NIGHTS = 50
REPS = (1, 2)
RESTARTS, MAXITER = 4, 3000
RULES = ('immediate', 'delayed')      # the NLR study's "as soon / as late as possible"
UNITS = ('dollars', 'peak_kw', 'start_stops')


def fleet_summary(fleet):
    """Truck-days, how many fit the window, and the charging hours of those that fit."""
    days = load_truck_days(fleet)
    fit = eligible(days)
    return dict(fleet=fleet, truck_days=len(days), eligible=len(fit),
                hours_needed=dict(sorted(Counter(d.hours_needed() for d in fit).items())))


def units(problem, X, kw=CHARGER_KW):
    """Dollars, highest depot load (kW) and start/stops of each plan in X, shape (k, n).

    Every charging hour draws the full `kw`, so an hour costs price (cents/kWh) * kw / 100 dollars.
    """
    grid = np.asarray(X).reshape(-1, problem.n_vehicles, problem.n_slots)
    return dict(dollars=(grid * np.asarray(problem.tariff)).sum(axis=(1, 2)) * kw / 100,
                peak_kw=grid.sum(axis=1).max(axis=1) * float(kw),
                start_stops=(grid[:, :, 1:] != grid[:, :, :-1]).sum(axis=(1, 2)).astype(float))


def _one(table):
    return {k: float(v[0]) for k, v in table.items()}


def rule_plan(problem, rule):
    """'immediate': every truck charges from the first hour; 'delayed': up to the last hour."""
    if rule not in RULES:
        raise ValueError(f'rule must be one of {RULES}')
    grid = np.zeros((problem.n_vehicles, problem.n_slots), dtype=int)
    for v, need in enumerate(problem.demand):
        if rule == 'immediate':
            grid[v, :need] = 1
        else:
            grid[v, problem.n_slots - need:] = 1
    return grid.ravel()


def interp_angles(theta, p):
    """INTERP warm start (Zhou et al. 2020): stretch depth p-1 betas and gammas over p layers."""
    old = np.asarray(theta, dtype=float).reshape(2, p - 1)
    grid_old, grid_new = np.linspace(0, 1, p - 1), np.linspace(0, 1, p)
    return np.concatenate([np.interp(grid_new, grid_old, row) for row in old])


def optimise(sim, p, prev=None, restarts=RESTARTS, maxiter=MAXITER, seed=0):
    """Best COBYLA run over random starts (plus INTERP of the depth p-1 angles `prev`).

    COBYLA works on u = (betas, gammas / scale) with scale = 2 / spread, so both halves have
    a period near pi; on raw gammas (period ~2 pi / spread) it needs ~20x more evaluations.
    """
    rng = np.random.default_rng(seed)
    scale = np.ones(2 * p)
    scale[p:] = 2 / (np.ptp(sim.energies) or 1.0)
    starts = [rng.uniform(0, np.pi, 2 * p) for _ in range(restarts)]
    if prev is not None:
        starts.insert(0, interp_angles(prev, p) / scale)
    runs = [minimize(lambda u: sim.energy(scale * u), u0, method='COBYLA',
                     options={'maxiter': maxiter}) for u0 in starts]
    best = min(runs, key=lambda r: r.fun)
    return dict(theta=[float(t) for t in scale * best.x], energy=float(best.fun),
                nfev=sum(int(r.nfev) for r in runs), converged=bool(best.success))


def run_night(trucks, tariff=PLACEHOLDER_TARIFF, reps=REPS, seed=0):
    """Optimise XY + Dicke QAOA for one night; score it against the rules and the best plan."""
    problem = make_night(trucks, tariff)
    model = charging_model(problem, WEIGHTS)
    base = exact_baselines(problem, model)
    sim = make_simulator(problem, model)
    plans = (feasible_basis(problem)[:, None] >> np.arange(problem.n_qubits)) & 1
    table = units(problem, plans)
    best_x = min(valid_schedules(problem), key=model.value)
    row = dict(trucks=[dict(day_id=d.day_id, miles=d.miles, kwh_needed=d.kwh_needed,
                            hours_needed=d.hours_needed(), leaves_at=d.leaves_at,
                            returns_at=d.returns_at) for d in trucks],
               demand=list(problem.demand), tariff=list(problem.tariff), n_valid=base.n_valid,
               n_optimal=base.n_optimal, p_opt_random=base.n_optimal / base.n_valid,
               ar_random=(base.worst - base.mean) / (base.worst - base.best),
               rules={rule: _one(units(problem, rule_plan(problem, rule)[None])) for rule in RULES},
               best=_one(units(problem, best_x[None])))
    prev = None
    for p in reps:
        opt = optimise(sim, p, prev if prev is not None and len(prev) == 2 * (p - 1) else None,
                       seed=seed + p)
        prev = opt['theta']
        probs, m = sim.probs(prev), metrics(sim, prev)
        top = int(np.argmax(probs))
        row[f'p{p}'] = dict(opt, approx_ratio=m['approx_ratio'], p_optimal=m['p_optimal'],
                            gain_vs_random=m['p_optimal'] / row['p_opt_random'],
                            expected={k: float(probs @ v) for k, v in table.items()},
                            most_likely=dict(_one({k: v[top:top + 1] for k, v in table.items()}),
                                             probability=float(probs[top]),
                                             is_optimal=bool(np.isclose(sim.energies[top], base.best,
                                                                        atol=1e-9))))
    return row


def run_nights(fleet=FLEET, n_nights=N_NIGHTS, tariff=PLACEHOLDER_TARIFF, reps=REPS, seed=0):
    """run_night on `n_nights` nights of three distinct truck-days drawn from `fleet`."""
    nights = sample_nights(eligible(load_truck_days(fleet)), n_nights, seed=seed)
    return [dict(night=i, fleet=fleet, **run_night(trucks, tariff, reps, seed=seed + 10 * i))
            for i, trucks in enumerate(nights)]


def source_units(rows, reps=REPS):
    """{source: {unit: mean over nights}} for the rules, QAOA (shot average) and the best plan."""
    sources = {rule: [r['rules'][rule] for r in rows] for rule in RULES}
    for p in reps:
        sources[f'qaoa_p{p}'] = [r[f'p{p}']['expected'] for r in rows]
    sources['best'] = [r['best'] for r in rows]
    return {src: {u: float(np.mean([v[u] for v in vals])) for u in UNITS}
            for src, vals in sources.items()}


def summarise(rows, reps=REPS):
    """Per depth: spread of QAOA quality over nights; plus mean real-unit results per source.

    distinct_problems counts nights that differ in more than truck order (the goals do not
    depend on which truck is listed first), i.e. by sorted demand and tariff.
    """
    out = dict(nights=len(rows),
               distinct_problems=len({(tuple(sorted(r['demand'])), tuple(r['tariff'])) for r in rows}),
               demand_mix={'-'.join(map(str, k)): v for k, v in
                           sorted(Counter(tuple(sorted(r['demand'])) for r in rows).items())},
               units=source_units(rows, reps))
    for p in reps:
        q = [r[f'p{p}'] for r in rows]
        gains = [x['gain_vs_random'] for x in q]
        out[f'p{p}'] = dict(approx_ratio_median=float(np.median([x['approx_ratio'] for x in q])),
                            p_optimal_median=float(np.median([x['p_optimal'] for x in q])),
                            gain_median=float(np.median(gains)), gain_min=float(np.min(gains)),
                            gain_max=float(np.max(gains)),
                            nights_beating_random=int(sum(g > 1 for g in gains)),
                            nights_most_likely_optimal=int(sum(x['most_likely']['is_optimal'] for x in q)),
                            nights_converged=int(sum(x['converged'] for x in q)))
    return out


# ---------------------------------------------------------------- hardware (study 06 format)

def typical_night(rows, p=REPS[-1]):
    """Index of the night with the median gain over a random guess (not the best night)."""
    gains = [r[f'p{p}']['gain_vs_random'] for r in rows]
    return int(np.argsort(gains)[len(gains) // 2])


def night_problem(row):
    return ChargingProblem(len(row['demand']), len(row['tariff']), tuple(row['tariff']),
                           tuple(row['demand']))


def run_name(p):
    return f'real_night_p{p}'


def hardware_runs(row, reps=REPS):
    """{name: (XY + Dicke circuit, optimised angles)} for one night, one entry per depth."""
    problem = night_problem(row)
    H = to_ising(charging_model(problem, WEIGHTS))
    return {run_name(p): (qaoa_circuit(problem, H, p, mixer='xy', start='dicke'),
                          np.asarray(row[f'p{p}']['theta']))
            for p in reps}


def compile_night(row, backend, reps=REPS):
    """{name: ISA circuit for backend}, transpiled like studies 05-07."""
    return {name: transpile_for(circ, theta, backend, seed=TRANSPILE_SEED)
            for name, (circ, theta) in hardware_runs(row, reps).items()}


def submit_night(row, backend, out_dir, shots=4096, compiled=None, reps=REPS):
    """One SamplerV2 job for the night's circuits plus readout calibration (writes jobs.json)."""
    compiled = compiled or compile_night(row, backend, reps)
    info = {run_name(p): dict(mixer='xy', start='dicke', reps=p, fleet=row['fleet'],
                              night=row['night'], truck_day_ids=[t['day_id'] for t in row['trucks']],
                              demand=row['demand'], tariff=row['tariff'], theta=row[f'p{p}']['theta'])
            for p in reps}
    return submit_runs(backend, list(info), shots=shots, out_dir=out_dir, compiled=compiled,
                       run_info=info)
