"""Core model checks: brute force, cost model, Hamiltonian, circuits, scoring."""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from evqaoa import (DEPOT, ChargingProblem, bitstring_to_schedule, charging_model,
                    count_valid, exact_baselines, is_valid, metric_ranges,
                    non_dominated, qaoa_circuit, schedule_metrics,
                    schedule_to_bitstring, score_distribution, to_ising,
                    valid_schedules)

WEIGHTS = (0.5, 0.3, 0.2)


def test_valid_schedule_count_matches_formula():
    # C(4,2) * C(4,1) * C(4,2) = 6 * 4 * 6
    assert count_valid(DEPOT) == 144
    assert len(list(valid_schedules(DEPOT))) == 144


def test_problem_rejects_bad_shapes():
    with pytest.raises(ValueError):
        ChargingProblem(2, 3, tariff=(1, 2), demand=(1, 1))
    with pytest.raises(ValueError):
        ChargingProblem(2, 3, tariff=(1, 2, 3), demand=(1, 4))


def test_metrics_on_hand_example():
    # vehicle 0: slots 1,2 | vehicle 1: slot 2 | vehicle 2: slots 2,3
    x = np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1])
    cost, peak, switches = schedule_metrics(DEPOT, x)
    assert cost == (14 + 9) + 9 + (9 + 21)
    assert peak == 0 + 1 + 9 + 1
    assert switches == 2 + 2 + 1


def test_model_equals_weighted_metrics_on_every_valid_schedule():
    model = charging_model(DEPOT, WEIGHTS)
    scales = metric_ranges(DEPOT)
    for x in valid_schedules(DEPOT):
        expected = sum(w * m / s for w, m, s in zip(WEIGHTS, schedule_metrics(DEPOT, x), scales))
        assert model.value(x) == pytest.approx(expected, abs=1e-9)


def test_penalty_vanishes_on_valid_and_counts_violations():
    plain, pen = charging_model(DEPOT, WEIGHTS), charging_model(DEPOT, WEIGHTS, penalty=2.0)
    x = next(valid_schedules(DEPOT))
    assert pen.value(x) == pytest.approx(plain.value(x))
    empty = np.zeros(DEPOT.n_qubits)
    assert pen.value(empty) - plain.value(empty) == pytest.approx(2.0 * (4 + 1 + 4))


def test_ising_diagonal_matches_model_on_all_bitstrings():
    model = charging_model(DEPOT, WEIGHTS, penalty=1.0)
    diag = np.real(to_ising(model).to_matrix(sparse=True).diagonal())
    xs = (np.arange(2 ** DEPOT.n_qubits)[:, None] >> np.arange(DEPOT.n_qubits)) & 1
    assert np.allclose(diag, model.values(xs), atol=1e-9)


def test_bitstring_round_trip_and_order():
    x = np.array([1] + [0] * 11)
    assert schedule_to_bitstring(x).endswith('1')
    assert np.array_equal(bitstring_to_schedule(schedule_to_bitstring(x)), x)


@pytest.mark.parametrize('start', ['dicke', 'basis'])
def test_xy_mixer_never_leaves_valid_schedules(start):
    H = to_ising(charging_model(DEPOT, WEIGHTS))
    qc = qaoa_circuit(DEPOT, H, reps=2, mixer='xy', start=start)
    theta = np.random.default_rng(7).uniform(0, 3, qc.num_parameters)
    for key, p in Statevector(qc.assign_parameters(theta)).probabilities_dict().items():
        if p > 1e-12:
            assert is_valid(DEPOT, bitstring_to_schedule(key))


def test_dicke_start_at_zero_angles_is_random_valid_guess():
    model = charging_model(DEPOT, WEIGHTS)
    H = to_ising(model)
    qc = qaoa_circuit(DEPOT, H, reps=1, mixer='xy', start='dicke')
    e0 = Statevector(qc.assign_parameters([0.0, 0.0])).expectation_value(H).real
    assert e0 == pytest.approx(exact_baselines(DEPOT, model).mean, abs=1e-9)


def test_x_mixer_circuit_shape_and_bad_options():
    H = to_ising(charging_model(DEPOT, WEIGHTS))
    qc = qaoa_circuit(DEPOT, H, reps=3, mixer='x')
    assert qc.num_qubits == 12 and qc.num_parameters == 6
    assert [p.name for p in qc.parameters][:3] == ['beta[0]', 'beta[1]', 'beta[2]']
    with pytest.raises(ValueError):
        qaoa_circuit(DEPOT, H, reps=1, mixer='xy', start='hadamard')


def test_pareto_front_is_non_dominated_and_nonempty():
    front = non_dominated([schedule_metrics(DEPOT, x) for x in valid_schedules(DEPOT)])
    assert len(front) >= 2
    for p in front:
        assert not any(q != p and all(a <= b for a, b in zip(q, p)) for q in front)


def test_score_distribution_half_optimal_half_invalid():
    model = charging_model(DEPOT, WEIGHTS)
    base = exact_baselines(DEPOT, model)
    best_x = min(valid_schedules(DEPOT), key=model.value)
    s = score_distribution({schedule_to_bitstring(best_x): 5, '0' * 12: 5}, DEPOT, model, base)
    assert s['valid_rate'] == 0.5 and s['p_optimal'] == 0.5 and s['p_optimal_valid'] == 1.0
    assert s['approx_ratio'] == pytest.approx(1.0)
