"""QAOA circuits for the charging problem.

Two mixers:
  'x'  : transverse field (RX on every qubit) from the uniform superposition.
         Explores all 2^n bitstrings, so rule-breaking schedules appear.
  'xy' : nearest-neighbour XY mixer inside each vehicle's block
         (Hadfield et al. 2019; Wang et al. 2020). It conserves the number of
         charging slots per vehicle, so a valid start state stays valid.
Start states for the XY mixer:
  'dicke' : equal superposition of all valid rows for each vehicle (exact, deeper).
  'basis' : one fixed valid schedule via X gates (shallow, hardware-friendly).
Parameter order of the circuit is (beta..., gamma...), as Qiskit sorts by name.
"""
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation, XXPlusYYGate

MIXERS = ('x', 'xy')
STARTS = ('hadamard', 'dicke', 'basis')


def dicke_amplitudes(n_slots, k):
    """Statevector over n_slots qubits with equal weight on every k-of-n pattern."""
    amps = np.zeros(2 ** n_slots)
    for chosen in combinations(range(n_slots), k):
        amps[sum(1 << s for s in chosen)] = 1.0
    return amps / np.sqrt(amps.sum())


def _prepare(qc, problem, start):
    if start == 'hadamard':
        qc.h(range(problem.n_qubits))
        return
    for v, need in enumerate(problem.demand):
        block = [problem.qubit(v, t) for t in range(problem.n_slots)]
        if start == 'dicke':
            qc.append(StatePreparation(dicke_amplitudes(problem.n_slots, need)), block)
        else:
            qc.x(block[:need])


def _cost_unitary(qc, hamiltonian, gamma):
    """exp(-i gamma H) for an H made only of Z and ZZ terms."""
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        support = np.flatnonzero(pauli.z)
        angle = 2 * float(np.real(coeff)) * gamma
        if len(support) == 1:
            qc.rz(angle, int(support[0]))
        elif len(support) == 2:
            qc.rzz(angle, int(support[0]), int(support[1]))


def _xy_layer(qc, problem, beta):
    """Even pairs then odd pairs, so gates in each half act on disjoint qubits."""
    for v in range(problem.n_vehicles):
        for parity in (0, 1):
            for t in range(parity, problem.n_slots - 1, 2):
                qc.append(XXPlusYYGate(2 * beta),
                          [problem.qubit(v, t), problem.qubit(v, t + 1)])


def qaoa_circuit(problem, hamiltonian, reps, mixer='xy', start=None):
    """Parameterised QAOA circuit with `reps` cost+mixer layers."""
    if mixer not in MIXERS:
        raise ValueError(f'mixer must be one of {MIXERS}')
    start = start or ('dicke' if mixer == 'xy' else 'hadamard')
    if start not in STARTS:
        raise ValueError(f'start must be one of {STARTS}')
    if mixer == 'xy' and start == 'hadamard':
        raise ValueError('the XY mixer needs a valid start state (dicke or basis)')

    gammas, betas = ParameterVector('gamma', reps), ParameterVector('beta', reps)
    qc = QuantumCircuit(problem.n_qubits)
    _prepare(qc, problem, start)
    for layer in range(reps):
        _cost_unitary(qc, hamiltonian, gammas[layer])
        if mixer == 'xy':
            _xy_layer(qc, problem, betas[layer])
        else:
            qc.rx(2 * betas[layer], range(problem.n_qubits))
    return qc
