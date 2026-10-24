"""Turn the weighted charging objective into a cost function and an Ising Hamiltonian.

The objective is quadratic in the binary variables, so we store it as
f(x) = offset + sum_i h[i] x_i + sum_{i<j} J[i, j] x_i x_j
and map it to Pauli-Z terms with x_i = (1 - Z_i) / 2, building the operator in
the sparse-list style used by the hackathon's QAOA starter notebook.
"""
from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .problem import metric_ranges


@dataclass(frozen=True)
class QuadraticModel:
    offset: float
    linear: np.ndarray      # shape (n,)
    quadratic: np.ndarray   # shape (n, n), strictly upper triangular

    def value(self, x):
        x = np.asarray(x, dtype=float)
        return float(self.offset + self.linear @ x + x @ self.quadratic @ x)

    def values(self, X):
        """Vectorised value() for a (k, n) array of schedules."""
        X = np.asarray(X, dtype=float)
        return self.offset + X @ self.linear + np.einsum('ki,ij,kj->k', X, self.quadratic, X)


def _add_pair(J, i, j, w):
    a, b = min(i, j), max(i, j)
    J[a, b] += w


def charging_model(problem, weights=(0.5, 0.3, 0.2), normalise=True, penalty=0.0):
    """Quadratic model of w_cost*cost + w_peak*peak + w_switch*switches (+ demand penalty).

    normalise divides each metric by its largest valid value so weights are comparable.
    penalty adds penalty * sum_v (sum_t x[v,t] - demand[v])^2, zero on valid schedules.
    """
    n, P = problem.n_qubits, problem
    scales = metric_ranges(P) if normalise else (1.0, 1.0, 1.0)
    w_cost, w_peak, w_switch = (w / s for w, s in zip(weights, scales))
    h, J, offset = np.zeros(n), np.zeros((n, n)), 0.0

    for v in range(P.n_vehicles):
        for t in range(P.n_slots):
            h[P.qubit(v, t)] += w_cost * P.tariff[t]

    for t in range(P.n_slots):            # (sum_v x_vt)^2 = sum_v x_vt + 2 sum_{v<u} x_vt x_ut
        column = [P.qubit(v, t) for v in range(P.n_vehicles)]
        for k, i in enumerate(column):
            h[i] += w_peak
            for j in column[k + 1:]:
                _add_pair(J, i, j, 2 * w_peak)

    for v in range(P.n_vehicles):         # (x_a - x_b)^2 = x_a + x_b - 2 x_a x_b
        for t in range(P.n_slots - 1):
            a, b = P.qubit(v, t), P.qubit(v, t + 1)
            h[a] += w_switch
            h[b] += w_switch
            _add_pair(J, a, b, -2 * w_switch)

    if penalty:
        for v, need in enumerate(P.demand):
            row = [P.qubit(v, t) for t in range(P.n_slots)]
            offset += penalty * need ** 2
            for k, i in enumerate(row):
                h[i] += penalty * (1 - 2 * need)
                for j in row[k + 1:]:
                    _add_pair(J, i, j, 2 * penalty)

    return QuadraticModel(offset, h, J)


def to_ising(model):
    """SparsePauliOp whose diagonal equals model.value on every basis state."""
    n = len(model.linear)
    terms, identity = [], model.offset
    for i, hi in enumerate(model.linear):
        if hi:
            identity += hi / 2
            terms.append(('Z', [i], -hi / 2))
    for i, j in zip(*np.nonzero(model.quadratic)):
        w = model.quadratic[i, j]
        identity += w / 4
        terms += [('Z', [i], -w / 4), ('Z', [j], -w / 4), ('ZZ', [i, j], w / 4)]
    terms.append(('', [], identity))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()
