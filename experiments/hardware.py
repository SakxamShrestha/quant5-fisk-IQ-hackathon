"""Real IBM hardware runs of the study-05 circuits (SamplerV2, DD + twirling).

Each run has a FakeFez twin in results/05_noise (same mixer, start, p, angle seed
and transpile seed), so hardware and simulator numbers compare one to one.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from evqaoa import DEPOT, qaoa_circuit
from experiments.noise import (best_angles, circuit_stats, flip_rates,
                               measured_physical_qubits, setup, transpile_for)

OUT_DIR = 'results/06_hardware'
SHOTS = 4096
PENALTY = 1.5          # study 05's penalty for the X-mixer baseline
TRANSPILE_SEED = 7     # transpile_for default, as in study 05
CAL_NAMES = ('readout_all0', 'readout_all1')

RUNS = {
    'xy_dicke_p2': dict(mixer='xy', start='dicke', reps=2, penalty=0.0, angle_seed=2,
                        fakefez_config='XY + Dicke'),
    'xy_dicke_p1': dict(mixer='xy', start='dicke', reps=1, penalty=0.0, angle_seed=1,
                        fakefez_config='XY + Dicke'),
    'x_penalty_p2': dict(mixer='x', start='hadamard', reps=2, penalty=PENALTY, angle_seed=2,
                         fakefez_config=f'X + penalty {PENALTY}'),
}

SAMPLER_OPTIONS = {
    'dynamical_decoupling': {'enable': True, 'sequence_type': 'XpXm'},
    'twirling': {'enable_gates': True, 'enable_measure': True},
}


def build_run(name):
    """(parameterised QAOA circuit, optimised angles) exactly as study 05 built them."""
    cfg = RUNS[name]
    _, H, _ = setup(penalty=cfg['penalty'])
    circ = qaoa_circuit(DEPOT, H, reps=cfg['reps'], mixer=cfg['mixer'], start=cfg['start'])
    return circ, best_angles(circ, H, seed=cfg['angle_seed'])


def compile_runs(backend, run_names):
    """{name: ISA circuit for backend}."""
    return {name: transpile_for(*build_run(name), backend, seed=TRANSPILE_SEED)
            for name in run_names}


def calibration_circuits(backend, physical):
    """All-0 and all-1 circuits measured on `physical` (bit i <- physical[i])."""
    circuits = []
    for label in (0, 1):
        qc = QuantumCircuit(len(physical))
        if label:
            qc.x(range(len(physical)))
        qc.measure_all()
        circuits.append(transpile(qc, backend, initial_layout=physical, optimization_level=0))
    return circuits


def submit_runs(backend, run_names, shots=SHOTS, out_dir=OUT_DIR, compiled=None):
    """One SamplerV2 job: every run, then the two readout-calibration PUBs. Writes jobs.json."""
    compiled = compiled or compile_runs(backend, run_names)
    physical = {name: measured_physical_qubits(compiled[name]) for name in run_names}
    union = sorted({q for qs in physical.values() for q in qs})
    pubs = [compiled[name] for name in run_names] + calibration_circuits(backend, union)
    job = SamplerV2(mode=backend, options=SAMPLER_OPTIONS).run(pubs, shots=shots)
    meta = dict(backend=backend.name, job_id=job.job_id(), run_names=list(run_names),
                pub_order=list(run_names) + list(CAL_NAMES), shots=shots,
                options=SAMPLER_OPTIONS,
                circuit_stats={name: circuit_stats(compiled[name]) for name in run_names},
                physical_qubits=physical, readout_calibration={'physical_qubits': union},
                runs={name: RUNS[name] for name in run_names},
                timestamp_utc=datetime.now(timezone.utc).isoformat())
    _write(out_dir, 'jobs.json', meta)
    return job


def collect(out_dir=OUT_DIR, service=None):
    """Fetch the finished job named in jobs.json and write hw_counts.json."""
    meta = json.loads((Path(out_dir) / 'jobs.json').read_text())
    job = (service or QiskitRuntimeService()).job(meta['job_id'])
    counts = [pub.data.meas.get_counts() for pub in job.result()]
    by_pub = dict(zip(meta['pub_order'], counts))
    out = {name: dict(backend=meta['backend'], job_id=meta['job_id'], shots=meta['shots'],
                      options=meta['options'], circuit_stats=meta['circuit_stats'][name],
                      physical_qubits=meta['physical_qubits'][name], counts=by_pub[name])
           for name in meta['run_names']}
    out['readout_calibration'] = dict(meta['readout_calibration'],
                                      counts_all0=by_pub[CAL_NAMES[0]],
                                      counts_all1=by_pub[CAL_NAMES[1]])
    out['usage'] = _usage(job)
    _write(out_dir, 'hw_counts.json', out)
    return out


def readout_rates(calibration, physical):
    """Flip rates for a run's measured qubits, in its bit order, for correct_readout."""
    rates = flip_rates(calibration['counts_all0'], calibration['counts_all1'])
    by_qubit = dict(zip(calibration['physical_qubits'], rates))
    return [by_qubit[q] for q in physical]


def _usage(job):
    """Job usage metrics (QPU seconds) if the service reports them."""
    try:
        return job.metrics().get('usage', {})
    except Exception as err:            # metrics are optional; keep the counts
        return {'error': str(err)}


def _write(out_dir, filename, data):
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / filename).write_text(json.dumps(data, indent=2))
