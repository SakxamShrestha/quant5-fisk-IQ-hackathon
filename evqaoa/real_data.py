"""Real delivery-truck schedules (NLR Fleet DNA, Borlaug et al. 2021) as depot charging nights.

One truck-day (one truck on one day) becomes one vehicle of a ChargingProblem: its miles
set how many one-hour slots it must charge, and it can only be scheduled if it is parked
at the depot for the whole charging window. See data/nlr_fleet_2021/SOURCE.md.
"""
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .problem import DEPOT, ChargingProblem

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'nlr_fleet_2021'
FLEETS = ('fleet1-beverage-delivery', 'fleet2-warehouse-delivery', 'fleet3-food-delivery')
KWH_PER_MILE = 1.8      # the NLR study's average truck consumption
CHARGER_KW = 100        # the NLR study's constant charging power
WINDOW = (22, 2)        # 10 pm to 2 am: the depot's four one-hour slots
# Stand-in until real prices are added: the depot's own night tariff (cents/kWh).
PLACEHOLDER_TARIFF = DEPOT.tariff


def _hours(clock):
    h, m, s = map(int, clock.split(':'))
    return h + m / 60 + s / 3600


@dataclass(frozen=True)
class TruckDay:
    fleet: str
    day_id: int
    miles: float
    shifts: tuple   # ((start_hour, end_hour), ...) on the road, hours since midnight

    @property
    def kwh_needed(self):
        return KWH_PER_MILE * self.miles

    def hours_needed(self, kw=CHARGER_KW):
        return math.ceil(self.kwh_needed / kw)

    @property
    def leaves_at(self):
        return min((s for s, _ in self.shifts), default=None)

    @property
    def returns_at(self):
        return max((e for _, e in self.shifts), default=None)

    def parked_during(self, window=WINDOW):
        """Parked for all of [start, end) hours; a window past midnight continues next morning.

        The next morning repeats this day's schedule, as the NLR code assumes.
        """
        start, end = window
        pieces = [(start, end)] if start < end else [(start, 24), (0, end)]
        return all(e <= a or s >= b for a, b in pieces for s, e in self.shifts)


def load_truck_days(fleet, data_dir=DATA_DIR):
    """Every truck-day of one fleet, in file order."""
    folder = Path(data_dir) / fleet
    shifts = {}
    with open(folder / 'veh_schedules.csv', newline='') as f:
        for row in csv.DictReader(f):
            day = shifts.setdefault(int(row['veh_op_day_id']), [])
            if row['on_shift'] == '1':
                day.append((_hours(row['start_time']), _hours(row['end_time'])))
    with open(folder / 'veh_op_days.csv', newline='') as f:
        return [TruckDay(fleet, int(row['veh_op_day_id']), float(row['vmt']),
                         tuple(shifts.get(int(row['veh_op_day_id']), ())))
                for row in csv.DictReader(f)]


def window_slots(window=WINDOW):
    start, end = window
    return (end - start) % 24


def eligible(truck_days, window=WINDOW):
    """Truck-days parked for the whole window that need 1..n_slots hours of charging."""
    n_slots = window_slots(window)
    return [d for d in truck_days
            if d.parked_during(window) and 1 <= d.hours_needed() <= n_slots]


def sample_nights(truck_days, n_nights, n_trucks=3, seed=0):
    """`n_nights` groups of `n_trucks` distinct truck-days, drawn uniformly with a fixed seed."""
    rng = np.random.default_rng(seed)
    return [tuple(truck_days[i] for i in rng.choice(len(truck_days), n_trucks, replace=False))
            for _ in range(n_nights)]


def make_night(trucks, tariff=PLACEHOLDER_TARIFF, window=WINDOW):
    """ChargingProblem for these truck-days charging in `window` at `tariff` (one price per hour)."""
    n_slots = window_slots(window)
    if len(tariff) != n_slots:
        raise ValueError(f'tariff needs one price per hour of the window ({n_slots})')
    if min(tariff) < 0:
        # metric_ranges scales each goal by its largest value, which breaks for negative
        # costs; switch it to (largest - smallest) before using such prices.
        raise ValueError('negative prices need range-based normalisation in metric_ranges')
    for d in trucks:
        if not d.parked_during(window):
            raise ValueError(f'truck-day {d.day_id} is not parked for the whole window')
        if not 1 <= d.hours_needed() <= n_slots:
            raise ValueError(f'truck-day {d.day_id} needs {d.hours_needed()} h, '
                             f'the window has {n_slots}')
    return ChargingProblem(len(trucks), n_slots, tuple(tariff),
                           tuple(d.hours_needed() for d in trucks))
