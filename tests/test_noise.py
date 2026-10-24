"""Checks for the noisy-simulation helpers."""
import numpy as np
import pytest
from qiskit_aer import AerSimulator

from evqaoa import (DEPOT, bitstring_to_schedule, is_valid, qaoa_circuit,
                    schedule_to_bitstring, valid_schedules)
from experiments.noise import (correct_readout, hellinger_fidelity, keep_valid,
                               setup, transpile_for)


def test_hellinger_of_identical_distributions_is_one():
    p = {'00': 3, '11': 1}
    assert hellinger_fidelity(p, p) == pytest.approx(1.0)
    assert hellinger_fidelity({'00': 1}, {'11': 1}) == pytest.approx(0.0)


def test_noiseless_xy_sampling_gives_only_valid_schedules():
    _, H, _ = setup()
    circ = qaoa_circuit(DEPOT, H, reps=1, mixer='xy', start='basis')
    sim = AerSimulator()
    tc = transpile_for(circ, [0.4, 0.7], sim)
    counts = sim.run(tc, shots=512, seed_simulator=1).result().get_counts()
    assert all(is_valid(DEPOT, bitstring_to_schedule(k)) for k in counts)


def test_keep_valid_drops_rule_breaking_shots():
    valid_key = schedule_to_bitstring(next(valid_schedules(DEPOT)))
    assert keep_valid({valid_key: 7, '0' * 12: 3}) == {valid_key: 7}


def test_penalty_changes_the_hamiltonian_but_not_the_score():
    _, H0, base0 = setup(penalty=0.0)
    _, H1, base1 = setup(penalty=1.5)
    assert base0 == base1
    assert not np.isclose(H0.coeffs.real.sum(), H1.coeffs.real.sum())


def test_readout_correction_inverts_a_known_flip():
    fixed = correct_readout({'0': 0.9, '1': 0.1}, [(0.1, 0.0)])  # 10% of 0s read as 1
    assert fixed.get('0', 0) == pytest.approx(1.0, abs=1e-9)
