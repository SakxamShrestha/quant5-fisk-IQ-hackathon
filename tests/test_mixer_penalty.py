import numpy as np
import pytest

from evqaoa import DEPOT
from experiments.mixer_penalty import (build_configs, evaluate_angles, make_study,
                                       random_baseline, run_config, run_sweep, summarise)


@pytest.fixture(scope='module')
def study():
    return make_study(DEPOT)


def test_random_baseline_is_valid_fraction():
    assert random_baseline(DEPOT) == pytest.approx(144 / 4096)


def test_build_configs_names_and_count():
    cfgs = build_configs((0, 1))
    assert [c['name'] for c in cfgs] == ['X+H pen=0', 'X+H pen=1', 'XY+Dicke', 'XY+basis']
    assert cfgs[2]['penalty'] == 0 and cfgs[2]['mixer'] == 'xy'


def test_study_baselines_ordered(study):
    base = study['baselines']
    assert base.best < base.mean < base.worst
    assert base.n_valid == 144 and base.n_optimal == 4


def test_x_mixer_zero_angles_matches_uniform_state(study):
    m = evaluate_angles(study, build_configs((0,))[0], np.zeros(2))
    assert m['valid_rate'] == pytest.approx(144 / 4096)


@pytest.mark.parametrize('idx', [1, 2])
def test_xy_stays_valid_at_random_angles(study, idx):
    theta = np.random.default_rng(1).uniform(0, np.pi, 4)
    m = evaluate_angles(study, build_configs((0,))[idx], theta)
    assert m['valid_rate'] == pytest.approx(1.0, abs=1e-8)


def test_run_config_schema_and_warm_start(study):
    cfg = build_configs((0,))[2]
    rows = run_config(study, cfg, p_max=2, restarts=1, maxiter=15, seed=0)
    assert [r['p'] for r in rows] == [1, 2]
    for r in rows:
        assert {'name', 'p', 'energy', 'valid_rate', 'p_optimal',
                'approx_ratio', 'theta', 'n_evals'} <= set(r)
        assert len(r['theta']) == 2 * r['p']
        assert 0 <= r['p_optimal'] <= 1


def test_sweep_and_summarise_over_seeds(study):
    rows = run_sweep(study, (0,), p_max=1, restarts=1, maxiter=8, seeds=(0, 1))
    assert len(rows) == 3 * 2
    summary = summarise(rows, 'valid_rate')
    assert set(summary) == {('X+H pen=0', 1), ('XY+Dicke', 1), ('XY+basis', 1)}
    mean, std, best = summary[('XY+Dicke', 1)]
    assert mean == pytest.approx(1.0) and best >= mean and std >= 0
