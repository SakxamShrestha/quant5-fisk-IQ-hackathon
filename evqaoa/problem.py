"""Overnight depot charging: the classical problem definition.

A depot has `n_vehicles` electric vehicles and `n_slots` one-hour charging
slots. Vehicle v must charge in exactly `demand[v]` slots. A schedule is a
binary vector x with x[qubit(v, t)] = 1 when vehicle v charges in slot t.
"""
from dataclasses import dataclass
from itertools import combinations, product
from math import comb

import numpy as np


@dataclass(frozen=True)
class ChargingProblem:
    n_vehicles: int
    n_slots: int
    tariff: tuple   # electricity price per slot, cents/kWh
    demand: tuple   # slots of charging each vehicle needs

    def __post_init__(self):
        if len(self.tariff) != self.n_slots:
            raise ValueError('tariff needs one price per slot')
        if len(self.demand) != self.n_vehicles:
            raise ValueError('demand needs one entry per vehicle')
        if any(not 0 <= d <= self.n_slots for d in self.demand):
            raise ValueError('each demand must be between 0 and n_slots')

    @property
    def n_qubits(self):
        return self.n_vehicles * self.n_slots

    def qubit(self, vehicle, slot):
        """Vehicle-major layout: each vehicle owns a contiguous block of qubits."""
        return vehicle * self.n_slots + slot

    def as_grid(self, x):
        return np.asarray(x).reshape(self.n_vehicles, self.n_slots)


def schedule_metrics(problem, x):
    """(energy_cost, peak_load, switches) for schedule x. All are minimised.

    energy_cost: total tariff paid.
    peak_load:   sum over slots of (vehicles charging)^2, large when charging bunches up.
    switches:    on/off transitions per vehicle, a proxy for battery wear.
    """
    grid = problem.as_grid(x)
    energy_cost = float((grid * np.asarray(problem.tariff)).sum())
    peak_load = float((grid.sum(axis=0) ** 2).sum())
    switches = float((grid[:, 1:] != grid[:, :-1]).sum())
    return energy_cost, peak_load, switches


def is_valid(problem, x):
    return bool(np.array_equal(problem.as_grid(x).sum(axis=1), problem.demand))


def valid_schedules(problem):
    """Yield every schedule that meets each vehicle's demand exactly."""
    rows_per_vehicle = []
    for need in problem.demand:
        rows = []
        for chosen in combinations(range(problem.n_slots), need):
            row = np.zeros(problem.n_slots, dtype=int)
            row[list(chosen)] = 1
            rows.append(row)
        rows_per_vehicle.append(rows)
    for rows in product(*rows_per_vehicle):
        yield np.concatenate(rows)


def count_valid(problem):
    return int(np.prod([comb(problem.n_slots, d) for d in problem.demand]))


def metric_ranges(problem):
    """Largest value of each metric over valid schedules (for normalising weights)."""
    values = np.array([schedule_metrics(problem, x) for x in valid_schedules(problem)])
    return tuple(float(m) if m > 0 else 1.0 for m in values.max(axis=0))


def non_dominated(points):
    """Pareto-optimal subset of metric tuples (all minimised)."""
    unique = sorted(set(points))
    return [p for p in unique
            if not any(q != p and all(a <= b for a, b in zip(q, p)) for q in unique)]


def bitstring_to_schedule(bitstring):
    """Qiskit bitstrings list qubit 0 last, so reverse them."""
    return np.array([int(b) for b in bitstring.replace(' ', '')[::-1]], dtype=int)


def schedule_to_bitstring(x):
    return ''.join(str(int(b)) for b in np.asarray(x)[::-1])


# Reference depot: three vehicles, 10 pm to 2 am, a time-of-use night tariff
# (cents/kWh). The delivery van (vehicle 1) needs only one hour of charge.
DEPOT = ChargingProblem(n_vehicles=3, n_slots=4, tariff=(32, 14, 9, 21), demand=(2, 1, 2))
