"""EV depot charging instance, objectives and brute-force helpers.

Adapted from Yash Khamkar's ev-qaoa (MIT), https://github.com/Yankee4567/ev-qaoa
"""
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np


@dataclass(frozen=True)
class Instance:
    V: int          # number of vehicles
    T: int          # number of time slots
    prices: tuple   # electricity price per slot (length T)
    need: tuple     # charging slots needed per vehicle (length V)

    @property
    def n(self):
        return self.V * self.T  # one qubit per (vehicle, slot), index v * T + t


def objectives(inst, x):
    """Return (cost, peak, wear) for a binary schedule x of length V*T."""
    X = np.asarray(x).reshape(inst.V, inst.T)
    cost = int((X * np.array(inst.prices)).sum())
    peak = int((X.sum(axis=0) ** 2).sum())
    wear = int(np.abs(np.diff(X, axis=1)).sum())
    return cost, peak, wear


def all_feasible(inst):
    """Yield every schedule where vehicle v charges in exactly need[v] slots."""
    per_vehicle = []
    for v in range(inst.V):
        rows = []
        for slots in combinations(range(inst.T), inst.need[v]):
            r = np.zeros(inst.T, dtype=int)
            r[list(slots)] = 1
            rows.append(r)
        per_vehicle.append(rows)
    for combo in product(*per_vehicle):
        yield np.concatenate(combo)


def is_feasible(inst, x):
    X = np.asarray(x).reshape(inst.V, inst.T)
    return all(X[v].sum() == inst.need[v] for v in range(inst.V))


def pareto_front(points):
    """Non-dominated objective triples (all objectives minimised)."""
    pts = sorted(set(points))
    return [p for p in pts
            if not any(q != p and all(a <= b for a, b in zip(q, p)) for q in pts)]


def objective_scales(inst):
    """Worst-case value of each objective over valid schedules, used for normalisation."""
    F = np.array([objectives(inst, x) for x in all_feasible(inst)])
    return F.max(axis=0)


def bits_to_x(bitstring):
    """Qiskit prints qubit 0 on the right, so reverse before reading."""
    return np.array([int(b) for b in bitstring.replace(' ', '')[::-1]])


# The reference instance used throughout the upstream project.
EXAMPLE_3x4 = Instance(V=3, T=4, prices=(4, 1, 1, 3), need=(2, 2, 2))
