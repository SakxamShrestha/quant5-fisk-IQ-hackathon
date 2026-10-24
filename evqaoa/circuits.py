"""QAOA circuits: X mixer or constraint-preserving XY mixer with Dicke / basis start.

Adapted from Yash Khamkar's ev-qaoa (MIT), https://github.com/Yankee4567/ev-qaoa
Parameter order of the returned circuit is (betas..., gammas...), alphabetical.
"""
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation, XXPlusYYGate


def dicke_vector(T, k):
    """Equal superposition of all T-bit strings with exactly k ones."""
    vec = np.zeros(2 ** T)
    for slots in combinations(range(T), k):
        vec[sum(1 << t for t in slots)] = 1.0
    return vec / np.linalg.norm(vec)


def start_state(qc, inst, kind='dicke'):
    for v in range(inst.V):
        reg = [v * inst.T + t for t in range(inst.T)]
        if kind == 'dicke':
            qc.append(StatePreparation(dicke_vector(inst.T, inst.need[v])), reg)
        else:  # 'basis': X gates only, shallow and hardware-friendly
            for t in range(inst.need[v]):
                qc.x(reg[t])


def cost_layer(qc, H_cost, gamma):
    """exp(-i gamma H) as RZ / RZZ gates (H has only Z and ZZ terms)."""
    for label, coeff in H_cost.to_list():
        zs = [i for i, ch in enumerate(reversed(label)) if ch == 'Z']
        w = 2 * coeff.real * gamma
        if len(zs) == 1:
            qc.rz(w, zs[0])
        elif len(zs) == 2:
            qc.rzz(w, zs[0], zs[1])


def xy_mixer(qc, inst, beta):
    """Swap a 1 and a 0 between neighbouring slots of the same vehicle only."""
    for v in range(inst.V):
        base = v * inst.T
        for first in (0, 1):
            for t in range(first, inst.T - 1, 2):
                qc.append(XXPlusYYGate(2 * beta), [base + t, base + t + 1])


def x_mixer(qc, inst, beta):
    for q in range(inst.n):
        qc.rx(2 * beta, q)


def build_qaoa(inst, H_cost, p, mixer='xy', start='dicke'):
    gam, bet = ParameterVector('gamma', p), ParameterVector('beta', p)
    qc = QuantumCircuit(inst.n)
    if mixer == 'xy':
        start_state(qc, inst, start)
    else:
        qc.h(range(inst.n))
    for layer in range(p):
        cost_layer(qc, H_cost, gam[layer])
        (xy_mixer if mixer == 'xy' else x_mixer)(qc, inst, bet[layer])
    return qc
