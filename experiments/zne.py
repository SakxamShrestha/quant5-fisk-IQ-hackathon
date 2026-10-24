"""Zero-noise extrapolation of the study-06 circuits on real IBM hardware (EstimatorV2).

Every quantity the Sampler study scored is diagonal in the computational basis, so it
is an expectation value of a Z-only observable and one measurement basis covers them all:

    p_valid       projector onto the 144 valid schedules
    p_optimal     projector onto the optimal schedule(s)
    energy        cost Hamiltonian <H> over all shots (valid or not)
    valid_energy  P_valid H, so mean valid energy = <P_valid H> / <P_valid>

The approximation ratio of experiments.noise.score is then
(worst - <P_valid H>/<P_valid>) / (worst - best), evaluated from the two estimates.
Diagonals are turned into SparsePauliOps with a Walsh-Hadamard transform
(Qiskit little-endian: bit i of the basis index is qubit i).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

from evqaoa import DEPOT, is_valid
from experiments.hardware import RUNS, TRANSPILE_SEED, build_run
from experiments.noise import circuit_stats, normalise, setup, transpile_for

OUT_DIR = 'results/07_hardware_extra/zne'
STUDY06_DIR = Path(__file__).resolve().parents[1] / 'results' / '06_hardware'
RUN_NAMES = ('xy_dicke_p2', 'xy_dicke_p1')
OBSERVABLES = ('p_valid', 'p_optimal', 'energy', 'valid_energy')

# Gate folding with fractional factors (a random subset of 2q gates is folded). p=2 already
# has 357 CZs and keeps only ~14% of its valid-share signal at factor 1, so factor 3 or 5
# would leave nothing above the noise floor to fit; 1.5 and 2 still carry signal.
NOISE_FACTORS = (1, 1.5, 2)
EXTRAPOLATORS = ('exponential', 'linear')
EXTRAPOLATED_FACTORS = tuple(float(x) for x in np.linspace(0, NOISE_FACTORS[-1], 9))

NUM_RANDOMIZATIONS = 64
SHOTS_PER_RANDOMIZATION = 128
SHOTS = NUM_RANDOMIZATIONS * SHOTS_PER_RANDOMIZATION      # per PUB per noise factor
TREX_RANDOMIZATIONS = 32

QPU_CAP_SECONDS = 60
# Study 06 on ibm_fez: 5 PUBs x 4096 shots (twirled, DD) charged 8 s -> 0.39 ms per shot.
SECONDS_PER_SHOT = 8 / (5 * 4096)
SAFETY = 1.5           # folded circuits are longer; estimator adds TREX / overhead
ZERO_TOL = 1e-12
PLOTTED = (('p_valid', 'P(valid schedule)'), ('p_optimal', 'P(optimal schedule)'),
           ('approx_ratio', 'approximation ratio'))
CURVE_STYLES = {'exponential': '-', 'linear': '-.'}
# bar-chart sources for notebook 07, in plotting order
ZNE_SOURCES = ('ideal simulator', 'FakeFez (noisy sim, study 05)', 'ibm_fez Sampler raw (study 06)',
               'ibm_fez Estimator, noise factor 1', 'ibm_fez ZNE (extrapolated to 0)')
ZNE_COLORS = dict(zip(ZNE_SOURCES, ('#9aa5b1', '#4c78a8', 'tab:orange', '#f2a65a', 'tab:red')))
BAR_METRICS = ('p_valid', 'p_optimal', 'approx_ratio')
OPTIMAL_TOL = 1e-9     # same tolerance as score_distribution


# ---------------------------------------------------------------- observables

def diagonal_observable(values):
    """SparsePauliOp of I/Z terms whose diagonal is `values` (index = little-endian basis state)."""
    values = np.asarray(values, dtype=float)
    n = int(np.log2(len(values))) if len(values) else 0
    if len(values) == 0 or 2 ** n != len(values):
        raise ValueError('need 2**n diagonal entries')
    coeffs = values
    for i in range(n):          # transform along bit i: s_i=0 -> a+b, s_i=1 -> a-b
        block = coeffs.reshape(2 ** (n - 1 - i), 2, 2 ** i)
        coeffs = np.concatenate([block[:, :1] + block[:, 1:], block[:, :1] - block[:, 1:]], axis=1)
    coeffs = coeffs.reshape(-1) / 2 ** n
    terms = [('Z' * bin(s).count('1'), [q for q in range(n) if s >> q & 1], c)
             for s, c in enumerate(coeffs) if abs(c) > ZERO_TOL]
    return SparsePauliOp.from_sparse_list(terms or [('', [], 0.0)], num_qubits=n).simplify()


def basis_schedules(n):
    """(2**n, n) array: row k is the schedule of basis state k (x_i = bit i)."""
    return (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1


def observables(problem, model, baselines):
    """{name: SparsePauliOp} for OBSERVABLES on the problem's qubits."""
    X = basis_schedules(problem.n_qubits)
    valid = np.array([is_valid(problem, x) for x in X], dtype=float)
    energy = model.values(X)
    optimal = valid * (np.abs(energy - baselines.best) < OPTIMAL_TOL)
    diagonals = dict(p_valid=valid, p_optimal=optimal, energy=energy, valid_energy=valid * energy)
    return {name: diagonal_observable(diagonals[name]) for name in OBSERVABLES}


def expectations(state, ops):
    return {name: float(state.expectation_value(op).real) for name, op in ops.items()}


def counts_expectations(counts, ops):
    """Same expectations from a Sampler histogram (bitstring keys, qubit 0 last)."""
    probs = normalise(counts)
    out = {}
    for name, op in ops.items():
        diag = np.real(op.to_matrix(sparse=True).diagonal())
        out[name] = float(sum(p * diag[int(k, 2)] for k, p in probs.items()))
    return out


def approx_ratio(p_valid, valid_energy, baselines, p_std=0.0, e_std=0.0):
    """(ratio, std) as in score: (worst - <PH>/<P>) / (worst - best).

    The std treats the two estimates as independent (they are positively correlated,
    so this overstates the error a little).
    """
    if p_valid <= 0:
        return float('nan'), float('nan')
    span = baselines.worst - baselines.best
    mean_e = valid_energy / p_valid
    mean_std = np.hypot(e_std / p_valid, valid_energy * p_std / p_valid ** 2)
    return float((baselines.worst - mean_e) / span), float(mean_std / span)


# ---------------------------------------------------------------- circuits

def ideal_values(circ, theta, ops, baselines):
    vals = expectations(Statevector(circ.assign_parameters(theta)), ops)
    return dict(vals, approx_ratio=approx_ratio(vals['p_valid'], vals['valid_energy'], baselines)[0])


def isa_pub(tc, ops):
    """(ISA circuit without final measurements, observables mapped onto its physical qubits)."""
    isa = tc.remove_final_measurements(inplace=False)
    return isa, [ops[name].apply_layout(tc.layout) for name in OBSERVABLES]


# ---------------------------------------------------------------- options / budget

def estimator_options():
    return {
        'resilience_level': 2,
        'default_shots': SHOTS,
        'max_execution_time': QPU_CAP_SECONDS,
        'resilience': {
            'measure_mitigation': True,
            'measure_noise_learning': {'num_randomizations': TREX_RANDOMIZATIONS},
            'zne_mitigation': True,
            'zne': {'amplifier': 'gate_folding', 'noise_factors': list(NOISE_FACTORS),
                    'extrapolator': list(EXTRAPOLATORS),
                    'extrapolated_noise_factors': list(EXTRAPOLATED_FACTORS)},
        },
        'dynamical_decoupling': {'enable': True, 'sequence_type': 'XpXm'},
        'twirling': {'enable_gates': True, 'enable_measure': True,
                     'num_randomizations': NUM_RANDOMIZATIONS,
                     'shots_per_randomization': SHOTS_PER_RANDOMIZATION},
    }


def estimate_qpu_seconds(n_pubs):
    """Reasoned QPU-time estimate from study 06's measured cost per shot."""
    circuit_shots = n_pubs * len(NOISE_FACTORS) * SHOTS
    trex_shots = n_pubs * TREX_RANDOMIZATIONS * SHOTS_PER_RANDOMIZATION
    total = circuit_shots + trex_shots
    return dict(shots_total=total, circuit_shots=circuit_shots, trex_shots=trex_shots,
                seconds_per_shot=SECONDS_PER_SHOT, safety=SAFETY,
                quantum_seconds=float(total * SECONDS_PER_SHOT * SAFETY))


# ---------------------------------------------------------------- submit / collect

def submit_zne(backend, run_names=RUN_NAMES, out_dir=OUT_DIR):
    """One EstimatorV2 job, one PUB per run (all observables share it). Writes zne_job.json."""
    model, _, baselines = setup()
    ops = observables(DEPOT, model, baselines)
    pubs, stats, ideal, physical = [], {}, {}, {}
    for name in run_names:
        circ, theta = build_run(name)
        tc = transpile_for(circ, theta, backend, seed=TRANSPILE_SEED)
        pubs.append(isa_pub(tc, ops))
        stats[name] = circuit_stats(tc)
        physical[name] = [int(q) for q in tc.layout.final_index_layout()]
        ideal[name] = ideal_values(circ, theta, ops, baselines)
    budget = estimate_qpu_seconds(len(pubs))
    if budget['quantum_seconds'] > QPU_CAP_SECONDS:
        raise ValueError(f"estimated {budget['quantum_seconds']:.0f} s exceeds the QPU cap")
    options = estimator_options()
    job = EstimatorV2(mode=backend, options=options).run(pubs)
    meta = dict(backend=backend.name, job_id=job.job_id(), run_names=list(run_names),
                observables=list(OBSERVABLES), observable_terms={k: len(v) for k, v in ops.items()},
                noise_factors=list(NOISE_FACTORS), extrapolators=list(EXTRAPOLATORS),
                options=options, budget_estimate=budget, service_estimate=_estimate(job),
                circuit_stats=stats, physical_qubits=physical, ideal=ideal,
                runs={name: RUNS[name] for name in run_names},
                timestamp_utc=datetime.now(timezone.utc).isoformat())
    _write(out_dir, 'zne_job.json', meta)
    return job


def parse_pub(pub, baselines):
    """Per-observable unmitigated (factor 1), per-factor and ZNE values, plus approx ratio."""
    d = pub.data
    evs_nf, stds_nf = np.asarray(d.evs_noise_factors), np.asarray(d.stds_noise_factors)
    ens_nf = np.asarray(getattr(d, 'ensemble_stds_noise_factors', stds_nf))
    ev_x, sd_x = np.asarray(d.evs_extrapolated), np.asarray(d.stds_extrapolated)
    rows = {}
    for i, name in enumerate(OBSERVABLES):
        rows[name] = dict(
            unmitigated=float(evs_nf[i, 0]), unmitigated_std=float(stds_nf[i, 0]),
            per_factor=[dict(noise_factor=f, value=float(evs_nf[i, j]), std=float(stds_nf[i, j]),
                             ensemble_std=float(ens_nf[i, j])) for j, f in enumerate(NOISE_FACTORS)],
            zne=float(np.asarray(d.evs)[i]), zne_std=float(np.asarray(d.stds)[i]),
            extrapolated={ex: dict(noise_factors=list(EXTRAPOLATED_FACTORS),
                                   values=ev_x[i, k].tolist(), stds=sd_x[i, k].tolist())
                          for k, ex in enumerate(EXTRAPOLATORS)})
    return dict(observables=rows, approx_ratio=_ratios(rows, baselines),
                metadata=_jsonable(getattr(pub, 'metadata', {})))


def _ratios(rows, baselines):
    p, e = rows['p_valid'], rows['valid_energy']

    def ratio(pv, ps, ev, es):
        value, std = approx_ratio(pv, ev, baselines, ps, es)
        return dict(value=value, std=std)

    return dict(unmitigated=ratio(p['unmitigated'], p['unmitigated_std'],
                                  e['unmitigated'], e['unmitigated_std']),
                zne=ratio(p['zne'], p['zne_std'], e['zne'], e['zne_std']),
                per_factor=[dict(noise_factor=a['noise_factor'],
                                 **ratio(a['value'], a['std'], b['value'], b['std']))
                            for a, b in zip(p['per_factor'], e['per_factor'])])


def study06_reference(name, ops, baselines, study06_dir=STUDY06_DIR):
    """Sampler numbers of the same circuit: raw counts re-scored here, FakeFez from study 05."""
    counts_path, results_path = Path(study06_dir) / 'hw_counts.json', Path(study06_dir) / 'hardware_results.json'
    if not counts_path.exists():
        return {}, {}
    vals = counts_expectations(json.loads(counts_path.read_text())[name]['counts'], ops)
    raw = dict(vals, approx_ratio=approx_ratio(vals['p_valid'], vals['valid_energy'], baselines)[0])
    fakefez = {}
    if results_path.exists():
        fakefez = json.loads(results_path.read_text())['runs'][name].get('study05_fakefez', {})
    return raw, fakefez


def collect_zne(out_dir=OUT_DIR, service=None):
    """Fetch the job named in zne_job.json; write zne_results.json and zne_usage.json."""
    meta = json.loads((Path(out_dir) / 'zne_job.json').read_text())
    job = (service or QiskitRuntimeService()).job(meta['job_id'])
    model, _, baselines = setup()
    ops = observables(DEPOT, model, baselines)
    runs = {}
    for name, pub in zip(meta['run_names'], job.result()):
        parsed = parse_pub(pub, baselines)
        ideal = meta['ideal'][name]
        for obs, row in parsed['observables'].items():
            row['ideal'] = ideal[obs]
        parsed['approx_ratio']['ideal'] = ideal['approx_ratio']
        raw, fakefez = study06_reference(name, ops, baselines)
        runs[name] = dict(parsed, study06_sampler_raw=raw, study05_fakefez=fakefez,
                          circuit_stats=meta['circuit_stats'][name])
    usage = _usage(job)
    out = dict(backend=meta['backend'], job_id=meta['job_id'], noise_factors=meta['noise_factors'],
               extrapolators=meta['extrapolators'], shots_per_factor=SHOTS, usage=usage, runs=runs,
               note='approx_ratio = (worst - <P_valid H>/<P_valid>)/(worst - best); ZNE ratio uses '
                    'the ZNE estimates of both; std assumes independent errors.')
    _write(out_dir, 'zne_results.json', out)
    _write(out_dir, 'zne_usage.json', usage)
    return out


def _series(run, key, baselines):
    """(per-factor points, zne point, {extrapolator: (xs, ys)}, ideal, study-06 raw) for a plot."""
    raw = run.get('study06_sampler_raw', {}).get(key)
    if key == 'approx_ratio':
        r = run['approx_ratio']
        p, e = run['observables']['p_valid']['extrapolated'], run['observables']['valid_energy']['extrapolated']
        curves = {}
        for ex in p:
            xs = p[ex]['noise_factors']
            ratios = [approx_ratio(pv, ev, baselines)[0] for pv, ev in zip(p[ex]['values'], e[ex]['values'])]
            curves[ex] = (xs, ratios)
        return r['per_factor'], r['zne'], curves, r['ideal'], raw
    row = run['observables'][key]
    curves = {ex: (c['noise_factors'], c['values']) for ex, c in row['extrapolated'].items()}
    return row['per_factor'], dict(value=row['zne'], std=row['zne_std']), curves, row['ideal'], raw


def zne_bar_table(results):
    """({run: {source: {metric: value}}}, matching stds) for hardware_analysis.plot_grouped_bars."""
    ideal_src, fake_src, raw_src, unmit_src, zne_src = ZNE_SOURCES
    table, errors = {}, {}
    for name, run in results['runs'].items():
        obs, ratio = run['observables'], run['approx_ratio']
        row = {ideal_src: dict(p_valid=obs['p_valid']['ideal'], p_optimal=obs['p_optimal']['ideal'],
                               approx_ratio=ratio['ideal'])}
        fake = run.get('study05_fakefez') or {}
        if fake:
            row[fake_src] = dict(p_valid=fake['valid_rate'], p_optimal=fake['p_optimal'],
                                 approx_ratio=fake['approx_ratio'])
        raw = run.get('study06_sampler_raw') or {}
        if raw:
            row[raw_src] = {k: raw[k] for k in BAR_METRICS}
        row[unmit_src] = dict(p_valid=obs['p_valid']['unmitigated'],
                              p_optimal=obs['p_optimal']['unmitigated'],
                              approx_ratio=ratio['unmitigated']['value'])
        row[zne_src] = dict(p_valid=obs['p_valid']['zne'], p_optimal=obs['p_optimal']['zne'],
                            approx_ratio=ratio['zne']['value'])
        table[name] = row
        errors[name] = {
            unmit_src: dict(p_valid=obs['p_valid']['unmitigated_std'],
                            p_optimal=obs['p_optimal']['unmitigated_std'],
                            approx_ratio=ratio['unmitigated']['std']),
            zne_src: dict(p_valid=obs['p_valid']['zne_std'], p_optimal=obs['p_optimal']['zne_std'],
                          approx_ratio=ratio['zne']['std'])}
    return table, errors


def plot_extrapolation(results, path, metrics=PLOTTED):
    """Per run and metric: chip values at each noise factor, fitted curves, ZNE and ideal.

    Fits the server could not evaluate (all NaN) are left out instead of drawn empty.
    """
    baselines = setup()[2]
    names = list(results['runs'])
    backend = results.get('backend', 'chip')
    fig, axes = plt.subplots(len(names), len(metrics), figsize=(5.4 * len(metrics), 4.2 * len(names)),
                             squeeze=False)
    for row_axes, name in zip(axes, names):
        for ax, (key, label) in zip(row_axes, metrics):
            points, zero, curves, ideal, raw = _series(results['runs'][name], key, baselines)
            for ex, (xs, ys) in curves.items():
                if np.all(np.isnan(np.asarray(ys, dtype=float))):
                    continue
                ax.plot(xs, ys, CURVE_STYLES.get(ex, ':'), color='#4c78a8', alpha=0.8, label=f'{ex} fit')
            ax.errorbar([pt['noise_factor'] for pt in points], [pt['value'] for pt in points],
                        [pt['std'] for pt in points], fmt='o', color='tab:orange', capsize=3,
                        label=f'{backend} (Estimator)')
            ax.errorbar([0], [zero['value']], [zero['std']], fmt='s', color='tab:red', capsize=3,
                        label='ZNE estimate (server best fit)')
            ax.axhline(ideal, ls='--', color='grey', label='ideal (noise-free)')
            if raw is not None:
                ax.plot([1.05], [raw], 'x', color='k', label='study 06 Sampler raw')
            ax.set_xlabel('noise factor (1 = circuit as run, 0 = zero-noise limit)')
            ax.set_ylabel(label)
            ax.set_title(f'{name}: {label}')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    fig.suptitle('Zero-noise extrapolation on ' + backend)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def _estimate(job):
    try:
        return dict(job.usage_estimation)
    except Exception as err:            # estimate is informational only
        return {'error': str(err)}


def _usage(job):
    """Job usage metrics (QPU seconds) if the service reports them."""
    try:
        return job.metrics().get('usage', {})
    except Exception as err:            # metrics are optional; keep the results
        return {'error': str(err)}


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


def _write(out_dir, filename, data):
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / filename).write_text(json.dumps(data, indent=2))
