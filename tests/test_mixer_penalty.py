import numpy as np
import pytest

from evqaoa import EXAMPLE_3x4, objective_scales
from experiments.mixer_penalty import (build_configs, evaluate_angles,
                                       make_problem, random_baseline, run_sweep, summarise,
                                       run_config)

INST = EXAMPLE_3x4
LAM = (0.5, 0.3, 0.2)


@pytest.fixture(scope='module')
def problem():
    return make_problem(INST, LAM, objective_scales(INST))


def test_random_baseline_is_feasible_fraction():
    assert random_baseline(INST) == pytest.approx(216 / 4096)


def test_build_configs_names_and_count():
    cfgs = build_configs((0, 1))
    assert [c['name'] for c in cfgs] == ['X+H pen=0', 'X+H pen=1', 'XY+Dicke', 'XY+basis']
    assert cfgs[2]['penalty'] == 0 and cfgs[2]['mixer'] == 'xy'


def test_problem_energy_bounds_ordered(problem):
    assert problem['E_best'] < problem['E_worst']
    assert problem['Q'].shape == (12, 12)


def test_x_mixer_zero_angles_matches_uniform_state(problem):
    cfg = build_configs((0,))[0]
    m = evaluate_angles(INST, problem, cfg, np.zeros(2))
    assert m['feasible_rate'] == pytest.approx(216 / 4096)


@pytest.mark.parametrize('idx', [1, 2])
def test_xy_stays_feasible_at_random_angles(problem, idx):
    cfg = build_configs((0,))[idx]
    theta = np.random.default_rng(1).uniform(0, np.pi, 4)
    m = evaluate_angles(INST, problem, cfg, theta)
    assert m['feasible_rate'] == pytest.approx(1.0, abs=1e-8)


def test_run_config_schema_and_warm_start(problem):
    cfg = build_configs((0,))[2]
    rows = run_config(INST, problem, cfg, p_max=2, restarts=1, maxiter=15, seed=0)
    assert [r['p'] for r in rows] == [1, 2]
    for r in rows:
        assert {'name', 'p', 'energy', 'feasible_rate', 'p_optimal',
                'approx_ratio', 'theta', 'n_evals'} <= set(r)
        assert len(r['theta']) == 2 * r['p']
        assert 0 <= r['p_optimal'] <= 1


def test_sweep_and_summarise_over_seeds(problem):
    rows = run_sweep(INST, problem, (0,), p_max=1, restarts=1, maxiter=8, seeds=(0, 1))
    assert len(rows) == 3 * 2
    summary = summarise(rows, 'feasible_rate')
    assert set(summary) == {('X+H pen=0', 1), ('XY+Dicke', 1), ('XY+basis', 1)}
    mean, std, best = summary[('XY+Dicke', 1)]
    assert mean == pytest.approx(1.0) and best >= mean and std >= 0
