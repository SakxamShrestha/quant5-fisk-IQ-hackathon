"""Tests for experiments.optimizers (fast, no circuits needed except one smoke test)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.optimizers import (CountedObjective, cvar, run_optimizer,  # noqa: E402
                                    sample_energies, spsa)


def test_cvar_alpha_one_equals_mean():
    s = np.array([5.0, 1.0, 3.0, 9.0])
    assert cvar(s, 1.0) == pytest.approx(s.mean())


def test_cvar_small_alpha_equals_min():
    s = np.array([5.0, 1.0, 3.0, 9.0])
    assert cvar(s, 0.1) == 1.0


def test_cvar_half_averages_lowest_half():
    assert cvar(np.array([4.0, 1.0, 2.0, 8.0]), 0.5) == pytest.approx(1.5)


def test_cvar_rejects_bad_alpha():
    with pytest.raises(ValueError):
        cvar(np.array([1.0]), 0.0)


def test_counter_counts_calls_and_records():
    obj = CountedObjective(lambda x: float(np.sum(x ** 2)))
    obj(np.array([1.0, 2.0]))
    obj(np.array([0.0, 0.0]))
    assert obj.count == 2
    assert obj.values == [5.0, 0.0]
    assert len(obj.points) == 2


def test_counter_stops_at_budget():
    obj = CountedObjective(lambda x: 0.0, budget=3)
    with pytest.raises(StopIteration):
        for _ in range(10):
            obj(np.zeros(1))
    assert obj.count == 3


def test_spsa_decreases_quadratic():
    f = lambda x: float(np.sum((x - 1.0) ** 2))
    x0 = np.array([3.0, -2.0, 0.5])
    x, fun = spsa(f, x0, maxiter=200, seed=1)
    assert fun < 0.2 * f(x0)


def test_sampled_energy_converges_to_exact():
    probs = np.array([0.2, 0.5, 0.3])
    energies = np.array([1.0, 2.0, 4.0])
    samples = sample_energies(probs, energies, 200000, np.random.default_rng(0))
    assert samples.mean() == pytest.approx(probs @ energies, abs=0.02)


def test_run_optimizer_all_methods_on_quadratic():
    f = lambda x: float(np.sum((x - 0.5) ** 2))
    for method in ('COBYLA', 'Nelder-Mead', 'Powell', 'L-BFGS-B', 'SPSA'):
        res = run_optimizer(f, np.array([2.0, -1.0]), method, budget=150, seed=0)
        assert res['nfev'] <= 150
        assert res['fun'] < 0.5, method


# ---- helpers that tie the optimisers to the charging problem -------------------
from evqaoa import (DEPOT, charging_model, exact_baselines, output_distribution,  # noqa: E402
                    qaoa_circuit, score_distribution, to_ising)
from experiments.optimizers import (all_bitstring_energies, score_probs,  # noqa: E402
                                    shot_objective)


@pytest.fixture(scope='module')
def setup():
    model = charging_model(DEPOT)
    circ = qaoa_circuit(DEPOT, to_ising(model), 1, mixer='xy', start='dicke')
    return model, circ, exact_baselines(DEPOT, model)


def test_energy_table_matches_model_value(setup):
    model, _, _ = setup
    energies = all_bitstring_energies(model)
    assert len(energies) == 2 ** DEPOT.n_qubits
    x = np.array([(37 >> i) & 1 for i in range(DEPOT.n_qubits)])
    assert energies[37] == pytest.approx(model.value(x))


def test_score_probs_matches_core_score_distribution(setup):
    model, circ, base = setup
    theta = np.array([0.4, 0.9])
    dist = output_distribution(circ, theta)
    probs = np.array([dist.get(format(i, f'0{DEPOT.n_qubits}b'), 0.0)
                      for i in range(2 ** DEPOT.n_qubits)])
    ours = score_probs(probs, DEPOT, model, base)
    core = score_distribution(dist, DEPOT, model, base)
    for key in ('valid_rate', 'p_optimal', 'approx_ratio'):
        assert ours[key] == pytest.approx(core[key], abs=1e-9)


def test_shot_objective_alpha_one_is_noisy_mean_and_small_alpha_is_lower():
    probs, energies = np.array([0.5, 0.5]), np.array([1.0, 3.0])
    mean_obj = shot_objective(lambda th: probs, energies, 5000, np.random.default_rng(0), 1.0)
    low_obj = shot_objective(lambda th: probs, energies, 5000, np.random.default_rng(0), 0.1)
    assert mean_obj(None) == pytest.approx(2.0, abs=0.1)
    assert low_obj(None) == pytest.approx(1.0)


def test_fast_probabilities_match_full_statevector(setup):
    from qiskit.quantum_info import Statevector
    from experiments.optimizers import fast_probabilities
    _, circ, _ = setup
    theta = np.array([0.3, 1.1])
    probs_of = fast_probabilities(circ)
    reference = Statevector(circ.assign_parameters(theta)).probabilities()
    assert np.allclose(probs_of(theta), reference, atol=1e-12)
