from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from evqaoa import EXAMPLE_3x4, build_qaoa, is_feasible, bits_to_x
from experiments.noise import (feasible_fraction, hellinger, load_counts,
                               postselect, local_readout_mitigate, setup_problem)

HW = Path(__file__).resolve().parents[1] / 'upstream/results/hw_counts_run2_mitigated.json'


def test_loader_parses_hardware_json():
    counts = load_counts(HW)
    assert sum(counts.values()) == 4096
    assert all(len(k) == 12 and set(k) <= {'0', '1'} for k in counts)
    assert all(isinstance(v, int) for v in counts.values())


def test_hellinger_identical_is_one():
    p = {'00': 0.3, '11': 0.7}
    assert abs(hellinger(p, dict(p)) - 1.0) < 1e-12
    assert hellinger({'00': 1}, {'11': 1}) == 0.0


def test_noiseless_xy_sampling_is_fully_feasible():
    inst, H, *_ = setup_problem()
    qc = build_qaoa(inst, H, 1, mixer='xy', start='basis')
    bound = qc.assign_parameters([0.3, 0.5])
    bound.measure_all()
    sim = AerSimulator()
    counts = sim.run(transpile(bound, sim), shots=512, seed_simulator=1).result().get_counts()
    assert feasible_fraction(counts, inst) == 1.0


def test_postselect_keeps_only_feasible():
    inst = EXAMPLE_3x4
    good = '001100110011'
    kept = postselect({good: 5, '0' * 12: 3}, inst)
    assert kept == {good: 5}
    assert is_feasible(inst, bits_to_x(good))


def test_readout_mitigation_inverts_known_flip():
    eps = [(0.1, 0.1)] * 2
    noisy = {'00': 0.81, '01': 0.09, '10': 0.09, '11': 0.01}
    fixed = local_readout_mitigate(noisy, eps)
    assert abs(fixed['00'] - 1.0) < 1e-9
