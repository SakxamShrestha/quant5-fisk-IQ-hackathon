import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from evqaoa import (DEPOT, charging_model, qaoa_circuit, to_ising, valid_schedules)
from experiments import scaling as sc

WEIGHTS = (0.5, 0.3, 0.2)


def _model(problem):
    return charging_model(problem, WEIGHTS)


def test_family_sizes_and_feasible_counts():
    assert sc.make_problem(3, 4) == DEPOT
    assert sc.make_problem(4, 4).n_qubits == 16
    assert sc.make_problem(4, 5).n_qubits == 20
    assert sc.make_problem(4, 6).n_qubits == 24
    assert len(sc.feasible_basis(sc.make_problem(4, 4))) == 6 ** 4
    assert len(sc.feasible_basis(sc.make_problem(4, 5))) == 10 ** 4


def test_tariff_length_matches_slots_and_four_slot_is_depot():
    for slots in (4, 5, 6):
        assert len(sc.make_problem(4, slots).tariff) == slots
    assert sc.TARIFF[4] == DEPOT.tariff
    with pytest.raises(ValueError):
        sc.make_problem(3, 7)


def test_basis_matches_bruteforce_enumeration():
    expected = {sum(int(b) << i for i, b in enumerate(x)) for x in valid_schedules(DEPOT)}
    assert set(sc.feasible_basis(DEPOT).tolist()) == expected


@pytest.mark.parametrize('problem_args', [(3, 4), (2, 5)])
def test_subspace_energy_matches_qiskit_statevector(problem_args):
    problem = sc.make_problem(*problem_args)
    model = _model(problem)
    H = to_ising(model)
    theta = np.random.default_rng(1).uniform(0, np.pi, 4)
    circ = qaoa_circuit(problem, H, 2, mixer='xy', start='dicke').assign_parameters(theta)
    ref = Statevector(circ).expectation_value(H).real
    sim = sc.make_simulator(problem, model)
    assert sim.energy(theta) == pytest.approx(ref, abs=1e-9)


def test_xy_mixer_on_4x4_stays_feasible_in_real_circuit():
    problem = sc.make_problem(4, 4)
    H = to_ising(_model(problem))
    theta = np.random.default_rng(2).uniform(0, np.pi, 2)
    probs = Statevector(qaoa_circuit(problem, H, 1).assign_parameters(theta)).probabilities()
    mass = sum(probs[i] for i in sc.feasible_basis(problem).tolist())
    assert mass == pytest.approx(1.0, abs=1e-9)


def test_transfer_keeps_parameter_count_and_copies():
    theta = np.array([0.1, 0.2, 0.3, 0.4])
    out = sc.transfer_angles(theta, p=2)
    assert out.shape == theta.shape and out is not theta
    with pytest.raises(ValueError):
        sc.transfer_angles(theta, p=3)


def test_metrics_bounds_and_optimiser_trace():
    sim = sc.make_simulator(DEPOT, _model(DEPOT))
    m = sc.metrics(sim, np.array([0.3, 0.3]))
    assert 0 <= m['approx_ratio'] <= 1 and 0 <= m['p_optimal'] <= 1
    run = sc.run_cobyla(sim.energy, np.array([0.3, 0.3]), maxiter=20)
    assert run['nfev'] == len(run['trace'])
    assert all(a >= b for a, b in zip(run['trace'], run['trace'][1:]))


def test_zero_angles_give_uniform_valid_energy():
    sim = sc.make_simulator(DEPOT, _model(DEPOT))
    assert sim.energy(np.zeros(2)) == pytest.approx(float(sim.energies.mean()))


def test_evals_to_reach():
    assert sc.evals_to_reach([5, 4, 3, 3], 3.5) == 3
    assert sc.evals_to_reach([5, 4], 1) is None
