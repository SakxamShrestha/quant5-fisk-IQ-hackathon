"""Checks for the real-hardware helpers (IBM service, backend and job are all faked)."""
import json

import numpy as np
import pytest
from qiskit_ibm_runtime.fake_provider import FakeFez

import experiments.hardware as hw
from evqaoa import DEPOT
from experiments.noise import correct_readout


class FakeBitArray:
    def __init__(self, counts):
        self._counts = counts

    def get_counts(self):
        return dict(self._counts)


class FakePubResult:
    def __init__(self, counts):
        self.data = type('Data', (), {'meas': FakeBitArray(counts)})()


class FakeJob:
    def __init__(self, n_pubs, job_id='job-123'):
        self._id, self.n_pubs = job_id, n_pubs

    def job_id(self):
        return self._id

    def status(self):
        return 'DONE'

    def result(self):
        # calibration PUBs come last: all-0 then all-1
        return [FakePubResult({'0' * 12: 10, '1' * 12: 2})] * self.n_pubs

    def metrics(self):
        return {'usage': {'quantum_seconds': 4, 'seconds': 4}}


class FakeSampler:
    instances = []

    def __init__(self, mode=None, options=None):
        self.mode, self.options, self.pubs = mode, options, None
        FakeSampler.instances.append(self)

    def run(self, pubs, shots=None):
        self.pubs, self.shots = list(pubs), shots
        return FakeJob(len(self.pubs))


class FakeService:
    def __init__(self, n_pubs):
        self.n_pubs, self.asked = n_pubs, None

    def job(self, job_id):
        self.asked = job_id
        return FakeJob(self.n_pubs, job_id)


@pytest.fixture
def fast(monkeypatch):
    """Skip the angle optimisation and the network-backed sampler."""
    monkeypatch.setattr(hw, 'best_angles', lambda circ, H, seed: np.full(circ.num_parameters, 0.3))
    monkeypatch.setattr(hw, 'SamplerV2', FakeSampler)
    FakeSampler.instances.clear()


def test_runs_name_study_05_configurations():
    assert 'xy_dicke_p2' in hw.RUNS
    assert len(hw.RUNS) <= 3
    for cfg in hw.RUNS.values():
        assert set(cfg) >= {'mixer', 'start', 'reps', 'penalty', 'angle_seed', 'fakefez_config'}
        assert cfg['angle_seed'] == cfg['reps']        # study 05 used seed = p


def test_build_run_returns_circuit_and_matching_angles(fast):
    circ, theta = hw.build_run('xy_dicke_p2')
    assert circ.num_qubits == DEPOT.n_qubits
    assert len(theta) == circ.num_parameters == 4


def test_build_run_rejects_unknown_name():
    with pytest.raises(KeyError):
        hw.build_run('nope')


def test_submit_writes_jobs_json_with_one_job(fast, tmp_path):
    names = ['xy_dicke_p2', 'xy_dicke_p1']
    job = hw.submit_runs(FakeFez(), names, shots=128, out_dir=tmp_path)
    meta = json.loads((tmp_path / 'jobs.json').read_text())
    sampler = FakeSampler.instances[-1]
    assert job.job_id() == meta['job_id'] == 'job-123'
    assert meta['backend'] == 'fake_fez' and meta['shots'] == 128
    assert meta['run_names'] == names
    assert len(sampler.pubs) == len(names) + 2          # + all-0 / all-1 calibration
    assert set(meta['circuit_stats']) == set(names)
    assert meta['circuit_stats']['xy_dicke_p2']['two_qubit_gates'] > 0
    assert len(meta['physical_qubits']['xy_dicke_p2']) == DEPOT.n_qubits
    assert meta['timestamp_utc'].endswith('+00:00')


def test_submit_enables_dd_and_twirling(fast, tmp_path):
    hw.submit_runs(FakeFez(), ['xy_dicke_p1'], shots=64, out_dir=tmp_path)
    opts = FakeSampler.instances[-1].options
    assert opts['dynamical_decoupling'] == {'enable': True, 'sequence_type': 'XpXm'}
    assert opts['twirling']['enable_gates'] and opts['twirling']['enable_measure']
    assert json.loads((tmp_path / 'jobs.json').read_text())['options'] == opts


def test_calibration_circuits_cover_every_measured_qubit(fast, tmp_path):
    hw.submit_runs(FakeFez(), ['xy_dicke_p2', 'xy_dicke_p1'], shots=64, out_dir=tmp_path)
    meta = json.loads((tmp_path / 'jobs.json').read_text())
    union = meta['readout_calibration']['physical_qubits']
    for qubits in meta['physical_qubits'].values():
        assert set(qubits) <= set(union)
    all_ones = FakeSampler.instances[-1].pubs[-1]
    assert all_ones.count_ops().get('x', 0) == len(union)


def test_collect_writes_counts_per_run(fast, tmp_path):
    names = ['xy_dicke_p1']
    hw.submit_runs(FakeFez(), names, shots=64, out_dir=tmp_path)
    service = FakeService(n_pubs=3)
    out = hw.collect(out_dir=tmp_path, service=service)
    saved = json.loads((tmp_path / 'hw_counts.json').read_text())
    assert service.asked == 'job-123'
    assert saved == out
    entry = saved['xy_dicke_p1']
    assert entry['counts'] == {'0' * 12: 10, '1' * 12: 2}
    assert entry['backend'] == 'fake_fez' and entry['job_id'] == 'job-123'
    assert {'shots', 'options', 'circuit_stats', 'physical_qubits'} <= set(entry)
    assert saved['usage']['quantum_seconds'] == 4
    assert 'readout_calibration' in saved


def test_readout_rates_from_calibration_counts():
    cal = {'physical_qubits': [5, 9],
           'counts_all0': {'00': 90, '01': 10},        # bit 0 (qubit 5) flips 10%
           'counts_all1': {'11': 80, '10': 20}}        # bit 0 (qubit 5) drops 20%
    rates = hw.readout_rates(cal, [9, 5])
    assert rates[0] == pytest.approx((0.0, 0.0))
    assert rates[1] == pytest.approx((0.1, 0.2))
    fixed = correct_readout({'00': 0.9, '10': 0.1}, rates)   # bit 1 = qubit 5 now
    assert fixed['00'] == pytest.approx(1.0, abs=1e-9)
