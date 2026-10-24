import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from evqaoa import (EXAMPLE_3x4, all_feasible, build_qaoa, build_qubo,
                    objective_scales, qubo_to_hamiltonian)
from experiments import scaling as sc

LAM = (0.5, 0.3, 0.2)


def _qubo(inst):
    return build_qubo(inst, LAM, objective_scales(inst))


def test_family_sizes_and_feasible_counts():
    assert sc.make_instance(3, 4) == EXAMPLE_3x4
    assert sc.make_instance(4, 4).n == 16
    assert sc.make_instance(4, 5).n == 20
    assert sc.make_instance(4, 6).n == 24
    assert len(sc.feasible_basis(sc.make_instance(4, 4))) == 6 ** 4
    assert len(sc.feasible_basis(sc.make_instance(4, 5))) == 10 ** 4


def test_tariff_length_matches_slots():
    for T in (4, 5, 6):
        assert len(sc.make_instance(4, T).prices) == T


def test_basis_matches_bruteforce_enumeration():
    inst = sc.make_instance(3, 4)
    expected = {sum(int(b) << i for i, b in enumerate(x)) for x in all_feasible(inst)}
    assert set(sc.feasible_basis(inst).tolist()) == expected


def test_subspace_energy_matches_qiskit_statevector():
    inst = EXAMPLE_3x4
    Q, const = _qubo(inst)
    H = qubo_to_hamiltonian(Q, const)
    theta = np.random.default_rng(1).uniform(0, np.pi, 4)
    circ = build_qaoa(inst, H, 2).assign_parameters(theta)
    ref = Statevector(circ).expectation_value(H).real
    sim = sc.make_simulator(inst, Q, const, p=2)
    assert sim.energy(theta) == pytest.approx(ref, abs=1e-9)


def test_xy_mixer_on_4x4_stays_feasible_in_real_circuit():
    inst = sc.make_instance(4, 4)
    Q, const = _qubo(inst)
    H = qubo_to_hamiltonian(Q, const)
    theta = np.random.default_rng(2).uniform(0, np.pi, 2)
    sv = Statevector(build_qaoa(inst, H, 1).assign_parameters(theta))
    probs = sv.probabilities()
    feasible = set(sc.feasible_basis(inst).tolist())
    mass = sum(probs[i] for i in feasible)
    assert mass == pytest.approx(1.0, abs=1e-9)


def test_transfer_keeps_parameter_count_and_copies():
    theta = np.array([0.1, 0.2, 0.3, 0.4])
    out = sc.transfer_angles(theta, p=2)
    assert out.shape == theta.shape and out is not theta
    with pytest.raises(ValueError):
        sc.transfer_angles(theta, p=3)


def test_metrics_bounds_and_optimiser_trace():
    inst = sc.make_instance(3, 4)
    Q, const = _qubo(inst)
    sim = sc.make_simulator(inst, Q, const, p=1)
    m = sc.metrics(sim, np.array([0.3, 0.3]))
    assert 0 <= m['approx_ratio'] <= 1 and 0 <= m['p_optimal'] <= 1
    run = sc.run_cobyla(sim.energy, np.array([0.3, 0.3]), maxiter=20)
    assert run['nfev'] == len(run['trace'])
    assert all(a >= b for a, b in zip(run['trace'], run['trace'][1:]))


def test_evals_to_reach():
    assert sc.evals_to_reach([5, 4, 3, 3], 3.5) == 3
    assert sc.evals_to_reach([5, 4], 1) is None
