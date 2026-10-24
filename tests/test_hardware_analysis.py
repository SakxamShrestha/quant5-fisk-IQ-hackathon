"""Checks for the real-hardware analysis helpers (pure functions, no QPU)."""
import matplotlib
import pytest

matplotlib.use('Agg')

from evqaoa import DEPOT, schedule_to_bitstring, valid_schedules
from experiments.hardware_analysis import (SOURCES, analyse_run, options_summary,
                                           plot_fidelity, plot_metric_bars,
                                           plot_top_schedules, record_table, run_names,
                                           study05_row, top_schedules)
from experiments.noise import setup

VALID = [schedule_to_bitstring(x) for x, _ in zip(valid_schedules(DEPOT), range(3))]
INVALID = '0' * DEPOT.n_qubits
OPTIONS = {'dynamical_decoupling': {'enable': True, 'sequence_type': 'XpXm'},
           'twirling': {'enable_gates': True, 'enable_measure': True}}


def entry(counts, job='job-1'):
    return dict(backend='ibm_fez', job_id=job, shots=sum(counts.values()), options=OPTIONS,
                circuit_stats={'depth': 300, 'two_qubit_gates': 200},
                physical_qubits=list(range(DEPOT.n_qubits)), counts=counts)


HW = {'run_a': entry({VALID[0]: 6, INVALID: 4}), 'run_b': entry({VALID[1]: 10}),
      'usage': {'quantum_seconds': 4},
      'readout_calibration': {'physical_qubits': [0], 'counts_all0': {}, 'counts_all1': {}}}


def test_run_names_skips_metadata_keys():
    assert run_names(HW) == ['run_a', 'run_b']


def test_options_summary_names_dd_and_twirling():
    text = options_summary(OPTIONS)
    assert 'XpXm' in text and 'gate' in text and 'measure' in text
    assert options_summary({}) == 'none'


def test_record_table_has_one_row_per_run():
    rows = record_table(HW)
    assert [r['run'] for r in rows] == ['run_a', 'run_b']
    assert rows[0]['two_qubit_gates'] == 200 and rows[0]['depth'] == 300
    assert rows[0]['backend'] == 'ibm_fez' and rows[0]['job_id'] == 'job-1'
    assert rows[0]['shots'] == 10


def test_analyse_run_scores_every_source_and_fidelities():
    model, _, base = setup()
    dists = {'ideal': {VALID[0]: 1.0}, 'fakefez': {VALID[0]: 8, INVALID: 2},
             'hardware': {VALID[0]: 6, INVALID: 4}}
    out = analyse_run(dists, model, base)
    assert set(out['scores']) == {'ideal', 'fakefez', 'hardware', 'hardware_valid'}
    assert out['scores']['hardware']['valid_rate'] == pytest.approx(0.6)
    assert out['scores']['hardware_valid']['valid_rate'] == pytest.approx(1.0)
    assert out['hellinger']['hardware_vs_ideal'] == pytest.approx(0.6)
    assert out['hellinger']['hardware_vs_fakefez'] == pytest.approx(
        (0.6 ** 0.5 * 0.8 ** 0.5 + 0.4 ** 0.5 * 0.2 ** 0.5) ** 2)
    assert out['hellinger']['fakefez_vs_ideal'] == pytest.approx(0.8)


def test_analyse_run_includes_readout_corrected_when_given():
    model, _, base = setup()
    dists = {'ideal': {VALID[0]: 1}, 'fakefez': {VALID[0]: 1}, 'hardware': {INVALID: 1},
             'hardware_readout': {VALID[0]: 1}}
    out = analyse_run(dists, model, base)
    assert out['scores']['hardware_readout']['valid_rate'] == pytest.approx(1.0)
    assert out['scores']['hardware_valid']['valid_rate'] == 0.0


def test_analyse_run_does_not_mutate_inputs():
    model, _, base = setup()
    hw_counts = {VALID[0]: 6, INVALID: 4}
    analyse_run({'ideal': {VALID[0]: 1}, 'fakefez': {VALID[0]: 1}, 'hardware': hw_counts},
                model, base)
    assert hw_counts == {VALID[0]: 6, INVALID: 4}


def test_top_schedules_ranks_by_reference_and_reads_other_sources():
    ideal = {VALID[0]: 0.5, VALID[1]: 0.3, VALID[2]: 0.2}
    rows = top_schedules(ideal, {'hardware': {VALID[1]: 3, INVALID: 1}}, k=2)
    assert [r['bitstring'] for r in rows] == [VALID[0], VALID[1]]
    assert rows[0]['ideal'] == pytest.approx(0.5)
    assert rows[0]['hardware'] == 0.0 and rows[1]['hardware'] == pytest.approx(0.75)
    assert rows[0]['valid']


def test_study05_row_matches_label_and_depth():
    rows = [{'config': 'XY + Dicke', 'p': 1}, {'config': 'XY + Dicke', 'p': 2}]
    assert study05_row(rows, 'XY + Dicke', 2) == rows[1]
    assert study05_row(rows, 'XY + basis', 2) is None


def test_sources_order_starts_with_ideal():
    assert SOURCES[0] == 'ideal' and 'hardware' in SOURCES


def test_plots_write_png_files(tmp_path):
    model, _, base = setup()
    dists = {'ideal': {VALID[0]: 0.7, VALID[1]: 0.3}, 'fakefez': {VALID[0]: 8, INVALID: 2},
             'hardware': {VALID[0]: 6, INVALID: 4}}
    results = {'run_a': analyse_run(dists, model, base)}
    tops = {'run_a': top_schedules(dists['ideal'], dists, k=10)}
    for fn, arg, name in [(plot_metric_bars, results, 'bars.png'),
                          (plot_fidelity, results, 'fid.png'),
                          (plot_top_schedules, tops, 'top.png')]:
        fn(arg, tmp_path / name)
        assert (tmp_path / name).stat().st_size > 0


def test_metric_bars_draw_random_guess_reference(tmp_path):
    model, _, base = setup()
    dists = {'ideal': {VALID[0]: 1}, 'fakefez': {VALID[0]: 1}, 'hardware': {VALID[0]: 1}}
    fig = plot_metric_bars({'run_a': analyse_run(dists, model, base)}, tmp_path / 'b.png',
                           reference={'valid_rate': 0.035})
    lines = [ln for ax in fig.axes for ln in ax.get_lines()]
    assert len(lines) == 1 and lines[0].get_ydata()[0] == pytest.approx(0.035)
