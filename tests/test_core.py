"""Upstream validation tests, ported to pytest."""
import numpy as np
from qiskit.quantum_info import Statevector

from evqaoa import (EXAMPLE_3x4, all_feasible, brute_force_baselines,
                    build_qaoa, build_qubo, bits_to_x, distribution_metrics,
                    is_feasible, objective_scales, objectives, pareto_front,
                    qubo_to_hamiltonian)

LAM = (0.5, 0.3, 0.2)


def _example_hamiltonian():
    Q, c = build_qubo(EXAMPLE_3x4, LAM, objective_scales(EXAMPLE_3x4))
    return Q, c, qubo_to_hamiltonian(Q, c)


def test_feasible_count_and_pareto_front():
    scheds = list(all_feasible(EXAMPLE_3x4))
    assert len(scheds) == 216
    front = pareto_front([objectives(EXAMPLE_3x4, x) for x in scheds])
    assert len(front) == 7
    assert (6, 18, 6) in front and (11, 10, 4) in front


def test_qubo_matches_weighted_objectives():
    scale = objective_scales(EXAMPLE_3x4)
    Q, c = build_qubo(EXAMPLE_3x4, LAM, scale)
    for x in all_feasible(EXAMPLE_3x4):
        expected = sum(l * f / s for l, f, s in zip(LAM, objectives(EXAMPLE_3x4, x), scale))
        assert abs(x @ Q @ x + c - expected) < 1e-9


def test_penalty_is_zero_on_feasible_and_positive_off():
    Q0, c0 = build_qubo(EXAMPLE_3x4)
    Qp, cp = build_qubo(EXAMPLE_3x4, penalty=5.0)
    x = next(all_feasible(EXAMPLE_3x4))
    assert abs((x @ Qp @ x + cp) - (x @ Q0 @ x + c0)) < 1e-9
    zeros = np.zeros(EXAMPLE_3x4.n)
    assert (zeros @ Qp @ zeros + cp) - (zeros @ Q0 @ zeros + c0) == 5.0 * 12


def test_hamiltonian_diagonal_matches_qubo():
    Q, c, H = _example_hamiltonian()
    diag = np.real(H.to_matrix(sparse=True).diagonal())
    for idx in [0, 5, 1234, 4095]:
        x = bits_to_x(format(idx, '012b'))
        assert abs(diag[idx] - (x @ Q @ x + c)) < 1e-7


def test_xy_mixer_stays_in_feasible_subspace():
    _, _, H = _example_hamiltonian()
    circ = build_qaoa(EXAMPLE_3x4, H, p=2)
    theta = np.random.default_rng(1).uniform(0, 3, 4)
    probs = Statevector(circ.assign_parameters(theta)).probabilities_dict()
    for key, pr in probs.items():
        if pr > 1e-12:
            assert is_feasible(EXAMPLE_3x4, bits_to_x(key))


def test_zero_angles_give_mean_valid_energy():
    Q, c, H = _example_hamiltonian()
    circ = build_qaoa(EXAMPLE_3x4, H, p=2)
    e0 = Statevector(circ.assign_parameters(np.zeros(4))).expectation_value(H).real
    mean, _, _ = brute_force_baselines(EXAMPLE_3x4, Q, c)
    assert abs(e0 - mean) < 1e-9


def test_distribution_metrics_on_point_mass():
    Q, c, _ = _example_hamiltonian()
    mean, best, worst = brute_force_baselines(EXAMPLE_3x4, Q, c)
    best_x = min(all_feasible(EXAMPLE_3x4), key=lambda x: x @ Q @ x + c)
    key = ''.join(str(b) for b in best_x[::-1])
    m = distribution_metrics({key: 10, '0' * 12: 10}, EXAMPLE_3x4, Q, c, best, worst)
    assert m['feasible_rate'] == 0.5 and m['p_optimal'] == 0.5
    assert abs(m['approx_ratio'] - 1.0) < 1e-9
