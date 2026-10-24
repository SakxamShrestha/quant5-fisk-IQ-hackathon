import numpy as np

from evqaoa import EXAMPLE_3x4, all_feasible, objectives, pareto_front
from experiments.pareto import (cumulative_curves, front_recovery, hypervolume,
                                random_baseline, weight_grid)

INST = EXAMPLE_3x4
TABLE = [objectives(INST, x) for x in all_feasible(INST)]
TRUE_FRONT = pareto_front(TABLE)


def test_weight_grid_sums_to_one_and_non_negative():
    grid = weight_grid(8)
    assert len(grid) == 45
    assert all(min(w) >= 0 and abs(sum(w) - 1) < 1e-12 for w in grid)


def test_hypervolume_single_point_is_box_volume():
    assert hypervolume([(1, 2, 3)], ref=(4, 6, 10)) == 3 * 4 * 7


def test_hypervolume_two_points_union():
    hv = hypervolume([(0, 0, 0), (1, 1, 1)], ref=(2, 2, 2))
    assert hv == 8


def test_hypervolume_empty_is_zero():
    assert hypervolume([], ref=(1, 1, 1)) == 0.0


def test_brute_force_front_has_seven_known_points():
    assert len(TRUE_FRONT) == 7
    assert (6, 18, 6) in TRUE_FRONT and (11, 10, 4) in TRUE_FRONT
    assert front_recovery(TABLE, TRUE_FRONT) == 7


def test_random_baseline_reproducible_with_seed():
    a = random_baseline(INST, 50, seed=3, feasible_only=True)
    b = random_baseline(INST, 50, seed=3, feasible_only=True)
    c = random_baseline(INST, 50, seed=4, feasible_only=True)
    assert a == b and a != c and len(a) == 50


def test_random_all_bitstrings_marks_infeasible_as_none():
    seq = random_baseline(INST, 300, seed=0, feasible_only=False)
    assert any(s is None for s in seq)


def test_cumulative_curves_monotone():
    seq = random_baseline(INST, 200, seed=1, feasible_only=True)
    ref = tuple(np.max(TABLE, axis=0) + 1)
    rec, hv = cumulative_curves(seq, TRUE_FRONT, ref, [10, 100, 200])
    assert rec == sorted(rec) and hv == sorted(hv)
