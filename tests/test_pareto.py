import numpy as np

from evqaoa import DEPOT, is_valid, non_dominated, schedule_metrics, valid_schedules
from experiments.pareto import (cumulative_curves, front_recovery, hypervolume,
                                random_baseline, run_weighting, shots_to_full_front,
                                weight_grid)

TABLE = [schedule_metrics(DEPOT, x) for x in valid_schedules(DEPOT)]
TRUE_FRONT = non_dominated(TABLE)


def test_weight_grid_sums_to_one_and_non_negative():
    grid = weight_grid(8)
    assert len(grid) == 45
    assert all(min(w) >= 0 and abs(sum(w) - 1) < 1e-12 for w in grid)


def test_hypervolume_single_point_is_box_volume():
    assert hypervolume([(1, 2, 3)], ref=(4, 6, 10)) == 3 * 4 * 7


def test_hypervolume_two_points_union():
    assert hypervolume([(0, 0, 0), (1, 1, 1)], ref=(2, 2, 2)) == 8


def test_hypervolume_empty_and_outside_reference_is_zero():
    assert hypervolume([], ref=(1, 1, 1)) == 0.0
    assert hypervolume([None, (5, 5, 5)], ref=(1, 1, 1)) == 0.0


def test_brute_force_front_has_nine_points_from_144_schedules():
    assert len(TABLE) == 144
    assert len(TRUE_FRONT) == 9
    assert (55.0, 13.0, 6.0) in TRUE_FRONT and (97.0, 7.0, 3.0) in TRUE_FRONT
    assert front_recovery(TABLE, TRUE_FRONT) == 9


def test_front_recovery_ignores_none_and_off_front_points():
    assert front_recovery([None, (999.0, 999.0, 999.0), TRUE_FRONT[0]], TRUE_FRONT) == 1


def test_random_valid_baseline_reproducible_and_always_valid():
    a = random_baseline(DEPOT, 50, seed=3, valid_only=True)
    b = random_baseline(DEPOT, 50, seed=3, valid_only=True)
    c = random_baseline(DEPOT, 50, seed=4, valid_only=True)
    assert a == b and a != c and len(a) == 50 and None not in a


def test_random_bitstring_baseline_marks_invalid_as_none_mostly():
    seq = random_baseline(DEPOT, 2000, seed=0, valid_only=False)
    valid_share = np.mean([s is not None for s in seq])
    assert 0.01 < valid_share < 0.08   # 144 / 4096 = 3.5%


def test_cumulative_curves_monotone():
    seq = random_baseline(DEPOT, 200, seed=1, valid_only=True)
    ref = tuple(np.max(TABLE, axis=0) + 1)
    rec, hv = cumulative_curves(seq, TRUE_FRONT, ref, [10, 100, 200])
    assert rec == sorted(rec) and hv == sorted(hv)


def test_shots_to_full_front():
    seq = [None, TRUE_FRONT[0], TRUE_FRONT[0]] + TRUE_FRONT[1:]
    assert shots_to_full_front(seq, TRUE_FRONT) == len(seq)
    assert shots_to_full_front(seq[:-1], TRUE_FRONT) is None


def test_run_weighting_samples_only_valid_schedules_with_xy_dicke():
    out = run_weighting(DEPOT, (0.5, 0.3, 0.2), p=1, shots=64, seed=0, maxiter=20)
    assert len(out['samples']) == 64
    assert all(s is not None for s in out['samples'])
    assert len(out['theta']) == 2 and out['nfev'] <= 20
