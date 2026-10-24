"""QUBO construction and QUBO -> Ising Hamiltonian conversion.

Adapted from Yash Khamkar's ev-qaoa (MIT), https://github.com/Yankee4567/ev-qaoa
"""
import numpy as np
from qiskit.quantum_info import SparsePauliOp


def build_qubo(inst, lam=(1, 1, 1), scale=(1, 1, 1), penalty=0.0):
    """Return (Q, const) with f(x) = x @ Q @ x + const, Q upper-triangular.

    lam weights (cost, peak, wear); scale normalises each objective;
    penalty > 0 adds penalty * sum_v (sum_t x[v,t] - need[v])^2.
    """
    V, T, n = inst.V, inst.T, inst.n
    l1, l2, l3 = (l / s for l, s in zip(lam, scale))

    def q(v, t):
        return v * T + t

    Q = np.zeros((n, n))
    const = 0.0

    for v in range(V):
        for t in range(T):
            Q[q(v, t), q(v, t)] += l1 * inst.prices[t]

    for t in range(T):
        for v in range(V):
            Q[q(v, t), q(v, t)] += l2
            for w in range(v + 1, V):
                Q[q(v, t), q(w, t)] += 2 * l2

    for v in range(V):
        for t in range(T - 1):
            Q[q(v, t), q(v, t)] += l3
            Q[q(v, t + 1), q(v, t + 1)] += l3
            Q[q(v, t), q(v, t + 1)] += -2 * l3

    if penalty:
        for v in range(V):
            k = inst.need[v]
            for t in range(T):
                Q[q(v, t), q(v, t)] += penalty * (1 - 2 * k)
                for s in range(t + 1, T):
                    Q[q(v, t), q(v, s)] += 2 * penalty
            const += penalty * k * k

    return Q, const


def qubo_to_hamiltonian(Q, const=0.0):
    """Map QUBO to a SparsePauliOp via x_i = (I - Z_i) / 2 (little-endian)."""
    n = Q.shape[0]
    terms = []
    offset = const

    def z_label(*qubits):
        s = ['I'] * n
        for i in qubits:
            s[n - 1 - i] = 'Z'
        return ''.join(s)

    for i in range(n):
        if Q[i, i] != 0:
            offset += Q[i, i] / 2.0
            terms.append((z_label(i), -Q[i, i] / 2.0))

    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                offset += Q[i, j] / 4.0
                terms.append((z_label(i), -Q[i, j] / 4.0))
                terms.append((z_label(j), -Q[i, j] / 4.0))
                terms.append((z_label(i, j), Q[i, j] / 4.0))

    terms.append(('I' * n, offset))
    return SparsePauliOp.from_list(terms).simplify()


def qubo_energy(Q, const, x):
    x = np.asarray(x)
    return float(x @ Q @ x + const)
