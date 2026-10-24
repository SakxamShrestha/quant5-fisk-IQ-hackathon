"""Checks for the NLR truck-schedule loader and the real-night problems built from it."""
import pytest

from evqaoa import ChargingProblem
from evqaoa.real_data import (FLEETS, TruckDay, eligible, load_truck_days, make_night,
                              sample_nights, window_slots)


def truck(miles, *shifts, day_id=1):
    return TruckDay('test', day_id, miles, tuple(shifts))


def test_every_fleet_loads_one_entry_per_truck_day():
    counts = {f: len(load_truck_days(f)) for f in FLEETS}
    assert counts == {'fleet1-beverage-delivery': 76, 'fleet2-warehouse-delivery': 100,
                      'fleet3-food-delivery': 236}


def test_known_truck_day_from_the_files():
    # beverage truck-day 2: leaves 07:43:09, back 17:34:56 after 106.5 miles
    day = {d.day_id: d for d in load_truck_days(FLEETS[0])}[2]
    assert day.miles == pytest.approx(106.5329, abs=1e-4)
    assert day.leaves_at == pytest.approx(7 + 43 / 60 + 9 / 3600)
    assert day.returns_at == pytest.approx(17 + 34 / 60 + 56 / 3600)
    assert day.kwh_needed == pytest.approx(1.8 * day.miles)
    assert day.hours_needed() == 2                 # 191.8 kWh at 100 kW
    assert day.parked_during((22, 2))


def test_hours_needed_rounds_up_at_100_kw():
    assert truck(50, (8, 17)).hours_needed() == 1    # 90 kWh
    assert truck(100, (8, 17)).hours_needed() == 2   # 180 kWh
    assert truck(100, (8, 17)).hours_needed(kw=200) == 1


@pytest.mark.parametrize('shift, parked', [((8, 17), True), ((8, 22), True), ((8, 22.5), False),
                                           ((1, 3), False), ((2, 9), True), ((21, 23), False)])
def test_parked_during_a_window_past_midnight(shift, parked):
    assert truck(50, shift).parked_during((22, 2)) is parked


def test_parked_during_a_same_day_window():
    assert not truck(50, (8, 19)).parked_during((18, 22))
    assert truck(50, (8, 17)).parked_during((18, 22))
    assert window_slots((18, 22)) == 4 and window_slots((22, 2)) == 4


def test_eligible_counts_per_fleet():
    counts = [len(eligible(load_truck_days(f))) for f in FLEETS]
    assert counts == [76, 98, 150]


def test_sample_nights_is_seeded_and_uses_distinct_truck_days():
    days = eligible(load_truck_days(FLEETS[1]))
    a, b = sample_nights(days, 5, seed=3), sample_nights(days, 5, seed=3)
    assert a == b and len(a) == 5
    for night in a:
        assert len({d.day_id for d in night}) == 3


def test_make_night_turns_miles_into_demand():
    problem = make_night([truck(50, (8, 17)), truck(100, (8, 17)), truck(150, (8, 17))],
                         tariff=(32, 14, 9, 21))
    assert problem == ChargingProblem(3, 4, (32, 14, 9, 21), (1, 2, 3))
    assert problem.n_qubits == 12


def test_make_night_rejects_what_the_model_cannot_represent():
    ok = truck(50, (8, 17))
    with pytest.raises(ValueError, match='one price per hour'):
        make_night([ok], tariff=(1, 2, 3))
    with pytest.raises(ValueError, match='negative prices'):
        make_night([ok], tariff=(5, -1, 3, 4))
    with pytest.raises(ValueError, match='not parked'):
        make_night([truck(50, (23, 23.5))])
    with pytest.raises(ValueError, match='needs 5 h'):
        make_night([truck(270, (8, 17))])          # 486 kWh -> 5 h > 4 slots
