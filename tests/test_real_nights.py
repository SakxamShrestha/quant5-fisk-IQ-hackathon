"""Checks for the real-night study: units, charging rules, optimiser and hardware wiring."""
import json

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeFez

import experiments.hardware as hw
import experiments.real_nights as rn
from evqaoa import (ChargingProblem, charging_model, is_valid, qaoa_circuit, to_ising)
from evqaoa.real_data import TruckDay, make_night
from experiments.scaling import make_simulator
from test_hardware import FakeSampler, FakeService

TARIFF = (32, 14, 9, 21)


def truck(miles, day_id):
    return TruckDay('test', day_id, miles, ((8.0, 17.0),))


@pytest.fixture(scope='module')
def rows():
    """Two small real-style nights (2 trucks, 8 qubits) run through the whole pipeline."""
    nights = [(truck(100, 1), truck(50, 2)), (truck(150, 3), truck(100, 4))]
    return [dict(night=i, fleet='test', **rn.run_night(t, TARIFF, seed=i)) for i, t in enumerate(nights)]


def test_units_of_a_hand_made_plan():
    problem = ChargingProblem(2, 4, TARIFF, (2, 1))
    x = np.array([1, 1, 0, 0,     # truck 0 charges 10 pm and 11 pm
                  0, 1, 0, 0])    # truck 1 charges 11 pm
    u = rn.units(problem, x[None])
    assert u['dollars'][0] == pytest.approx(32 + 14 + 14)    # cents/kWh * 100 kWh / 100
    assert u['peak_kw'][0] == 200
    assert u['start_stops'][0] == 1 + 2


@pytest.mark.parametrize('rule, expected', [('immediate', [1, 1, 0, 0, 1, 0, 0, 0]),
                                            ('delayed', [0, 0, 1, 1, 0, 0, 0, 1])])
def test_rule_plans_are_valid_blocks(rule, expected):
    problem = ChargingProblem(2, 4, TARIFF, (2, 1))
    x = rn.rule_plan(problem, rule)
    assert x.tolist() == expected and is_valid(problem, x)
    with pytest.raises(ValueError):
        rn.rule_plan(problem, 'random')


def test_interp_stretches_the_schedule():
    assert rn.interp_angles([0.4, 1.5], 2).tolist() == [0.4, 0.4, 1.5, 1.5]
    out = rn.interp_angles([0.2, 0.6, 1.0, 3.0], 3)     # betas (0.2, 0.6), gammas (1, 3)
    assert out == pytest.approx([0.2, 0.4, 0.6, 1.0, 2.0, 3.0])


def test_subspace_simulator_matches_the_circuit_on_a_real_night():
    problem = make_night([truck(150, 1), truck(50, 2), truck(100, 3)], TARIFF)
    model = charging_model(problem, rn.WEIGHTS)
    H = to_ising(model)
    theta = np.random.default_rng(4).uniform(0, 6, 4)
    circ = qaoa_circuit(problem, H, 2, mixer='xy', start='dicke').assign_parameters(theta)
    ref = Statevector(circ).expectation_value(H).real
    assert make_simulator(problem, model).energy(theta) == pytest.approx(ref, abs=1e-9)


def test_optimise_finds_the_depth_one_optimum():
    problem = make_night([truck(100, 1), truck(50, 2)], TARIFF)
    sim = make_simulator(problem, charging_model(problem, rn.WEIGHTS))
    opt = rn.optimise(sim, 1, seed=0)
    grid = min(sim.energy([b, g]) for b in np.linspace(0, np.pi, 61)
               for g in np.linspace(0, 4 * np.pi / np.ptp(sim.energies), 241))
    assert opt['converged'] and opt['energy'] <= grid + 1e-6


def test_run_night_schema_and_sanity(rows):
    for r in rows:
        assert {'trucks', 'demand', 'tariff', 'rules', 'best', 'p1', 'p2'} <= set(r)
        assert r['p2']['energy'] <= r['p1']['energy'] + 1e-9           # INTERP start keeps it monotone here
        for p in ('p1', 'p2'):
            q = r[p]
            assert 0 < q['p_optimal'] <= 1 and len(q['theta']) == 2 * int(p[1])
            assert q['gain_vs_random'] == pytest.approx(q['p_optimal'] / r['p_opt_random'])
            assert set(q['expected']) == set(rn.UNITS)
        json.dumps(r)                                                  # results must be JSON-ready


def test_summarise_counts_distinct_problems_and_sources(rows):
    s = rn.summarise(rows + [dict(rows[0], night=2)])
    assert s['nights'] == 3 and s['distinct_problems'] == 2
    assert set(s['units']) == {'immediate', 'delayed', 'qaoa_p1', 'qaoa_p2', 'best'}
    assert s['p2']['gain_min'] <= s['p2']['gain_median'] <= s['p2']['gain_max']


def test_typical_night_is_the_median_not_the_best():
    rows = [{'p2': {'gain_vs_random': g}} for g in (5.0, 1.0, 2.0)]
    assert rn.typical_night(rows) == 2


def _hw_row():
    return dict(night=7, fleet='test', demand=[2, 1, 2], tariff=list(TARIFF),
                trucks=[{'day_id': i} for i in (11, 12, 13)],
                p1={'theta': [0.3, 0.4]}, p2={'theta': [0.3, 0.2, 0.4, 0.5]})


def test_submit_night_writes_study06_style_jobs_json(monkeypatch, tmp_path):
    monkeypatch.setattr(hw, 'SamplerV2', FakeSampler)
    FakeSampler.instances.clear()
    row = _hw_row()
    rn.submit_night(row, FakeFez(), tmp_path, shots=64)
    meta = json.loads((tmp_path / 'jobs.json').read_text())
    assert meta['run_names'] == ['real_night_p1', 'real_night_p2']
    assert meta['runs']['real_night_p2']['truck_day_ids'] == [11, 12, 13]
    assert meta['runs']['real_night_p2']['theta'] == row['p2']['theta']
    assert len(FakeSampler.instances[-1].pubs) == 2 + 2               # + readout calibration
    saved = hw.collect(out_dir=tmp_path, service=FakeService(n_pubs=4))
    assert {name for name in saved if name.startswith('real_night')} == set(meta['run_names'])
