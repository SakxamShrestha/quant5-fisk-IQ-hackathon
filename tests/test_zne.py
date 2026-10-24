"""Checks for the ZNE (EstimatorV2) study: projector observables, options, faked job round trip."""
import json

import numpy as np
import pytest
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit_ibm_runtime.options import EstimatorOptions

import experiments.hardware as hw
import experiments.zne as zne
from evqaoa import DEPOT, qaoa_circuit
from experiments.noise import score, setup, transpile_for


def diagonal(op):
    return np.real(op.to_matrix(sparse=True).diagonal())


@pytest.fixture(scope='module')
def scoring():
    model, H, baselines = setup()
    return model, H, baselines


@pytest.fixture(scope='module')
def observables(scoring):
    model, _, baselines = scoring
    return zne.observables(DEPOT, model, baselines)


@pytest.fixture(scope='module')
def leaky_state(scoring):
    """X-mixer state that leaks out of the valid subspace, so every observable is non-trivial."""
    _, H, _ = scoring
    circ = qaoa_circuit(DEPOT, H, reps=1, mixer='x', start='hadamard')
    return circ, Statevector(circ.assign_parameters([0.4, 0.7]))


@pytest.fixture
def fast(monkeypatch):
    """Skip the angle optimisation and the network-backed estimator."""
    monkeypatch.setattr(hw, 'best_angles', lambda circ, H, seed: np.full(circ.num_parameters, 0.3))
    monkeypatch.setattr(zne, 'EstimatorV2', FakeEstimator)
    FakeEstimator.instances.clear()


# ---------------------------------------------------------------- observables

def test_diagonal_observable_reproduces_any_diagonal():
    values = np.random.default_rng(0).normal(size=8)
    op = zne.diagonal_observable(values)
    assert isinstance(op, SparsePauliOp) and op.num_qubits == 3
    assert set(''.join(op.paulis.to_labels())) <= {'I', 'Z'}
    assert diagonal(op) == pytest.approx(values)


def test_diagonal_observable_is_little_endian():
    op = zne.diagonal_observable(np.array([0, 1, 0, 0]))     # |01>: qubit 0 set
    sv = Statevector.from_label('01')
    assert sv.expectation_value(op).real == pytest.approx(1.0)
    assert Statevector.from_label('10').expectation_value(op).real == pytest.approx(0.0)


def test_diagonal_observable_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        zne.diagonal_observable(np.ones(6))


def test_observable_set_and_sizes(observables, scoring):
    _, _, baselines = scoring
    assert tuple(observables) == zne.OBSERVABLES
    for op in observables.values():
        assert op.num_qubits == DEPOT.n_qubits
        assert set(''.join(op.paulis.to_labels())) <= {'I', 'Z'}
        assert len(op) <= 2 ** DEPOT.n_qubits
    p_valid, p_opt = diagonal(observables['p_valid']), diagonal(observables['p_optimal'])
    assert sorted(set(np.round(p_valid, 9))) == [0.0, 1.0]
    assert p_valid.sum() == pytest.approx(baselines.n_valid)
    assert p_opt.sum() == pytest.approx(baselines.n_optimal)
    assert np.all(p_opt <= p_valid)


def test_projectors_match_score_on_basis_states(observables, scoring):
    model, _, baselines = scoring
    p_valid, p_opt = diagonal(observables['p_valid']), diagonal(observables['p_optimal'])
    for idx in (0, 0b000011001100, int('110000011100', 2), 4095):
        key = format(idx, '012b')
        s = score({key: 1}, model, baselines)
        assert p_valid[idx] == pytest.approx(s['valid_rate'])
        assert p_opt[idx] == pytest.approx(s['p_optimal'])


def test_energy_observable_is_the_cost_hamiltonian(observables, scoring):
    _, H, _ = scoring
    assert diagonal(observables['energy']) == pytest.approx(np.real(H.to_matrix(sparse=True).diagonal()))


def test_statevector_expectations_equal_exact_scores(observables, scoring, leaky_state):
    model, H, baselines = scoring
    _, sv = leaky_state
    s = score(sv.probabilities_dict(), model, baselines)
    vals = zne.expectations(sv, observables)
    assert 0.0 < vals['p_valid'] < 1.0
    assert vals['p_valid'] == pytest.approx(s['valid_rate'], abs=1e-9)
    assert vals['p_optimal'] == pytest.approx(s['p_optimal'], abs=1e-9)
    assert vals['energy'] == pytest.approx(sv.expectation_value(H).real, abs=1e-9)
    ratio, _ = zne.approx_ratio(vals['p_valid'], vals['valid_energy'], baselines)
    assert ratio == pytest.approx(s['approx_ratio'], abs=1e-9)


def test_xy_dicke_ideal_values_are_fully_valid(fast, observables, scoring):
    _, _, baselines = scoring
    circ, theta = zne.build_run('xy_dicke_p1')
    ideal = zne.ideal_values(circ, theta, observables, baselines)
    assert ideal['p_valid'] == pytest.approx(1.0, abs=1e-9)
    assert ideal['valid_energy'] == pytest.approx(ideal['energy'], abs=1e-9)
    assert 0 < ideal['approx_ratio'] <= 1


def test_approx_ratio_propagates_errors(scoring):
    _, _, b = scoring
    mean_e = (b.best + b.worst) / 2
    ratio, std = zne.approx_ratio(0.5, 0.5 * mean_e, b, p_std=0.0, e_std=0.0)
    assert ratio == pytest.approx(0.5) and std == pytest.approx(0.0)
    _, std = zne.approx_ratio(0.5, 0.5 * mean_e, b, p_std=0.01, e_std=0.01)
    assert std > 0
    assert np.isnan(zne.approx_ratio(0.0, 0.1, b)[0])


def test_counts_expectations_match_score(observables, scoring):
    model, _, baselines = scoring
    counts = {'110000011100': 30, '000000000000': 70}
    vals = zne.counts_expectations(counts, observables)
    s = score(counts, model, baselines)
    assert vals['p_valid'] == pytest.approx(s['valid_rate'])
    assert vals['p_optimal'] == pytest.approx(s['p_optimal'])
    ratio, _ = zne.approx_ratio(vals['p_valid'], vals['valid_energy'], baselines)
    assert ratio == pytest.approx(s['approx_ratio'])


# ---------------------------------------------------------------- options / budget

def test_estimator_options_enable_zne_dd_twirling_trex():
    opts = zne.estimator_options()
    res = opts['resilience']
    assert res['zne_mitigation'] is True and res['measure_mitigation'] is True
    assert tuple(res['zne']['noise_factors']) == zne.NOISE_FACTORS
    assert res['zne']['extrapolator'] == list(zne.EXTRAPOLATORS)
    assert res['zne']['extrapolated_noise_factors'][0] == 0
    assert opts['dynamical_decoupling'] == {'enable': True, 'sequence_type': 'XpXm'}
    assert opts['twirling']['enable_gates'] and opts['twirling']['enable_measure']
    assert opts['max_execution_time'] <= zne.QPU_CAP_SECONDS
    EstimatorOptions(**opts)                     # the runtime accepts it


def test_noise_factors_start_at_one_and_increase():
    assert zne.NOISE_FACTORS[0] == 1
    assert list(zne.NOISE_FACTORS) == sorted(set(zne.NOISE_FACTORS))


def test_budget_estimate_fits_the_cap():
    est = zne.estimate_qpu_seconds(n_pubs=len(zne.RUN_NAMES))
    assert 0 < est['quantum_seconds'] < zne.QPU_CAP_SECONDS
    assert est['shots_total'] == (len(zne.RUN_NAMES) * len(zne.NOISE_FACTORS) * zne.SHOTS
                                  + est['trex_shots'])
    assert zne.estimate_qpu_seconds(n_pubs=4)['quantum_seconds'] > est['quantum_seconds']


# ---------------------------------------------------------------- layout

def test_isa_pub_strips_measurements_and_lays_out_observables(fast, observables):
    backend = GenericBackendV2(num_qubits=14, seed=5)
    circ, theta = zne.build_run('xy_dicke_p1')
    tc = transpile_for(circ, theta, backend, seed=3)
    isa, ops = zne.isa_pub(tc, observables)
    assert 'measure' not in isa.count_ops()
    assert all(op.num_qubits == backend.num_qubits for op in ops)
    sv = Statevector(isa)
    logical = zne.expectations(Statevector(circ.assign_parameters(theta)), observables)
    for name, op in zip(zne.OBSERVABLES, ops):
        assert sv.expectation_value(op).real == pytest.approx(logical[name], abs=1e-6)


# ---------------------------------------------------------------- fake runtime

class FakeData:
    def __init__(self, n_obs, n_factors, n_extrap, n_eval, base=0.2):
        f = np.arange(1, n_factors + 1)
        self.evs_noise_factors = base / f[None, :] * np.ones((n_obs, 1))
        self.stds_noise_factors = np.full((n_obs, n_factors), 0.01)
        self.ensemble_stds_noise_factors = np.full((n_obs, n_factors), 0.008)
        self.evs = np.full(n_obs, 2 * base)
        self.stds = np.full(n_obs, 0.05)
        self.evs_extrapolated = np.full((n_obs, n_extrap, n_eval), base)
        self.stds_extrapolated = np.full((n_obs, n_extrap, n_eval), 0.04)


class FakePubResult:
    def __init__(self):
        n_eval = len(zne.estimator_options()['resilience']['zne']['extrapolated_noise_factors'])
        self.data = FakeData(len(zne.OBSERVABLES), len(zne.NOISE_FACTORS), len(zne.EXTRAPOLATORS),
                             n_eval)
        self.metadata = {'shots': zne.SHOTS}


class FakeJob:
    def __init__(self, n_pubs, job_id='zne-123'):
        self._id, self.n_pubs = job_id, n_pubs

    def job_id(self):
        return self._id

    def result(self):
        return [FakePubResult() for _ in range(self.n_pubs)]

    @property
    def usage_estimation(self):
        return {'quantum_seconds': 21.0}

    def metrics(self):
        return {'usage': {'quantum_seconds': 19, 'seconds': 19}}


class FakeEstimator:
    instances = []

    def __init__(self, mode=None, options=None):
        self.mode, self.options, self.pubs = mode, options, None
        FakeEstimator.instances.append(self)

    def run(self, pubs):
        self.pubs = list(pubs)
        return FakeJob(len(self.pubs))


class FakeService:
    def __init__(self, n_pubs):
        self.n_pubs, self.asked = n_pubs, None

    def job(self, job_id):
        self.asked = job_id
        return FakeJob(self.n_pubs, job_id)


def test_submit_writes_job_json_with_one_two_pub_job(fast, tmp_path):
    names = list(zne.RUN_NAMES)
    job = zne.submit_zne(FakeFez(), names, out_dir=tmp_path)
    meta = json.loads((tmp_path / 'zne_job.json').read_text())
    est = FakeEstimator.instances[-1]
    assert len(FakeEstimator.instances) == 1 and len(est.pubs) == len(names)
    assert job.job_id() == meta['job_id'] == 'zne-123'
    assert meta['backend'] == 'fake_fez' and meta['run_names'] == names
    assert meta['noise_factors'] == list(zne.NOISE_FACTORS)
    assert meta['options'] == est.options
    assert meta['observables'] == list(zne.OBSERVABLES)
    assert meta['circuit_stats']['xy_dicke_p2']['two_qubit_gates'] > 0
    assert meta['budget_estimate']['quantum_seconds'] < zne.QPU_CAP_SECONDS
    assert set(meta['ideal']['xy_dicke_p1']) >= set(zne.OBSERVABLES) | {'approx_ratio'}
    assert meta['timestamp_utc'].endswith('+00:00')
    circuit, ops = est.pubs[0]
    assert len(ops) == len(zne.OBSERVABLES) and ops[0].num_qubits == circuit.num_qubits


def test_parse_pub_reads_every_noise_factor(scoring):
    _, _, baselines = scoring
    parsed = zne.parse_pub(FakePubResult(), baselines)
    p = parsed['observables']['p_valid']
    assert p['unmitigated'] == pytest.approx(0.2) and p['unmitigated_std'] == pytest.approx(0.01)
    assert [r['noise_factor'] for r in p['per_factor']] == list(zne.NOISE_FACTORS)
    assert p['per_factor'][1]['value'] == pytest.approx(0.1)
    assert p['zne'] == pytest.approx(0.4) and p['zne_std'] == pytest.approx(0.05)
    assert set(p['extrapolated']) == set(zne.EXTRAPOLATORS)
    assert len(p['extrapolated']['linear']['values']) == len(p['extrapolated']['linear']['noise_factors'])
    assert set(parsed['approx_ratio']) >= {'unmitigated', 'zne', 'per_factor'}


def test_collect_writes_results_with_comparisons(fast, tmp_path):
    names = list(zne.RUN_NAMES)
    zne.submit_zne(FakeFez(), names, out_dir=tmp_path)
    service = FakeService(n_pubs=len(names))
    out = zne.collect_zne(out_dir=tmp_path, service=service)
    saved = json.loads((tmp_path / 'zne_results.json').read_text())
    assert service.asked == 'zne-123'
    assert saved == json.loads(json.dumps(out))
    assert saved['usage']['quantum_seconds'] == 19
    assert json.loads((tmp_path / 'zne_usage.json').read_text())['quantum_seconds'] == 19
    entry = saved['runs']['xy_dicke_p2']
    for name in zne.OBSERVABLES:
        row = entry['observables'][name]
        assert {'ideal', 'unmitigated', 'per_factor', 'zne', 'zne_std'} <= set(row)
    assert entry['observables']['p_valid']['ideal'] == pytest.approx(1.0, abs=1e-9)
    assert 'study06_sampler_raw' in entry and 'p_valid' in entry['study06_sampler_raw']


def test_plot_extrapolation_writes_png(fast, tmp_path):
    zne.submit_zne(FakeFez(), list(zne.RUN_NAMES), out_dir=tmp_path)
    out = zne.collect_zne(out_dir=tmp_path, service=FakeService(n_pubs=len(zne.RUN_NAMES)))
    fig = zne.plot_extrapolation(out, tmp_path / 'zne.png')
    assert (tmp_path / 'zne.png').stat().st_size > 0
    assert len(fig.axes) == len(zne.RUN_NAMES) * len(zne.PLOTTED)


# ---------------------------------------------------------------- notebook 07 summary + plots

def synthetic_row(values, ideal, zne_value):
    nan = [float('nan')] * len(zne.EXTRAPOLATED_FACTORS)
    return dict(unmitigated=values[0], unmitigated_std=0.01, ideal=ideal, zne=zne_value, zne_std=0.04,
                per_factor=[dict(noise_factor=f, value=v, std=0.01, ensemble_std=0.01)
                            for f, v in zip(zne.NOISE_FACTORS, values)],
                extrapolated=dict(exponential=dict(noise_factors=list(zne.EXTRAPOLATED_FACTORS),
                                                   values=nan, stds=nan),
                                  linear=dict(noise_factors=list(zne.EXTRAPOLATED_FACTORS),
                                              values=list(np.linspace(0.3, 0.05, len(nan))),
                                              stds=[0.01] * len(nan))))


def synthetic_results():
    ratio = dict(value=0.75, std=0.04)
    run = dict(observables=dict(p_valid=synthetic_row([0.16, 0.08, 0.06], 1.0, 0.39),
                                p_optimal=synthetic_row([0.011, 0.006, 0.004], 0.10, 0.035),
                                energy=synthetic_row([0.7, 0.8, 0.9], 0.2, 0.5),
                                valid_energy=synthetic_row([0.11, 0.06, 0.05], 0.15, 0.3)),
               approx_ratio=dict(unmitigated=ratio, zne=dict(value=0.78, std=0.26), ideal=0.81,
                                 per_factor=[dict(noise_factor=f, **ratio) for f in zne.NOISE_FACTORS]),
               study06_sampler_raw=dict(p_valid=0.169, p_optimal=0.0117, approx_ratio=0.757),
               study05_fakefez=dict(valid_rate=0.34, p_optimal=0.028, approx_ratio=0.78))
    return dict(backend='ibm_fez', runs={'xy_dicke_p2': run, 'xy_dicke_p1': run})


def test_zne_bar_table_lists_every_source_per_run():
    table, errors = zne.zne_bar_table(synthetic_results())
    row = table['xy_dicke_p2']
    assert list(row) == list(zne.ZNE_SOURCES)
    ideal, fake, raw, unmit, mitig = (row[s] for s in zne.ZNE_SOURCES)
    assert ideal == {'p_valid': 1.0, 'p_optimal': 0.10, 'approx_ratio': 0.81}
    assert fake == {'p_valid': 0.34, 'p_optimal': 0.028, 'approx_ratio': 0.78}
    assert raw == {'p_valid': 0.169, 'p_optimal': 0.0117, 'approx_ratio': 0.757}
    assert unmit['p_valid'] == 0.16 and unmit['approx_ratio'] == 0.75
    assert mitig == {'p_valid': 0.39, 'p_optimal': 0.035, 'approx_ratio': 0.78}
    err = errors['xy_dicke_p2']
    assert err[zne.ZNE_SOURCES[-1]] == {'p_valid': 0.04, 'p_optimal': 0.04, 'approx_ratio': 0.26}
    assert err[zne.ZNE_SOURCES[-2]]['p_valid'] == 0.01
    assert zne.ZNE_SOURCES[0] not in err


def test_zne_bar_table_skips_missing_references():
    results = synthetic_results()
    run = dict(results['runs']['xy_dicke_p2'], study06_sampler_raw={}, study05_fakefez={})
    table, _ = zne.zne_bar_table(dict(results, runs={'xy_dicke_p2': run}))
    assert zne.ZNE_SOURCES[1] not in table['xy_dicke_p2']
    assert zne.ZNE_SOURCES[2] not in table['xy_dicke_p2']


def test_plot_extrapolation_metric_subset_labels_and_legends(tmp_path):
    metrics = zne.PLOTTED[:2]
    fig = zne.plot_extrapolation(synthetic_results(), tmp_path / 'z.png', metrics=metrics)
    assert (tmp_path / 'z.png').stat().st_size > 0
    assert len(fig.axes) == 2 * len(metrics)
    for ax in fig.axes:
        assert ax.get_xlabel() and ax.get_ylabel() and ax.get_title()
        labels = ax.get_legend_handles_labels()[1]
        assert ax.get_legend() is not None
        assert 'linear fit' in labels and 'exponential fit' not in labels    # all-NaN fit skipped
        ideal = [ln for ln in ax.get_lines() if ln.get_label() == 'ideal (noise-free)']
        assert ideal and ideal[0].get_linestyle() == '--'
