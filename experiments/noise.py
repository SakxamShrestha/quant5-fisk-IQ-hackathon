"""Helpers for the noisy-hardware study (offline: FakeFez / Aer only)."""
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import Statevector

from evqaoa import (EXAMPLE_3x4, brute_force_baselines, build_qaoa, build_qubo,
                    distribution_metrics, expectation_energy, is_feasible,
                    bits_to_x, objective_scales, optimise, qubo_to_hamiltonian)

LAM = (0.5, 0.3, 0.2)
TWO_QUBIT_NAMES = ('cz', 'cx', 'ecr', 'rzz', 'xx_plus_yy')


def load_counts(path):
    """Read a counts JSON into {12-bit string: int}; tolerates wrapped or spaced formats."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and isinstance(data.get('counts'), dict):
        data = data['counts']
    if not isinstance(data, dict) or not data:
        raise ValueError(f'{path}: expected a non-empty {{bitstring: count}} mapping')
    counts = {}
    for key, val in data.items():
        bits = str(key).replace(' ', '')
        if not bits or set(bits) - {'0', '1'}:
            raise ValueError(f'{path}: bad bitstring key {key!r}')
        counts[bits] = counts.get(bits, 0) + int(round(val))
    return counts


def normalise(dist):
    total = float(sum(dist.values()))
    return {k: v / total for k, v in dist.items()}


def hellinger(p, q):
    """Hellinger fidelity (sum sqrt(p q))^2, same definition as qiskit."""
    p, q = normalise(p), normalise(q)
    return float(sum(np.sqrt(p[k] * q[k]) for k in p.keys() & q.keys()) ** 2)


def feasible_fraction(dist, inst):
    d = normalise(dist)
    return float(sum(v for k, v in d.items() if is_feasible(inst, bits_to_x(k))))


def postselect(counts, inst):
    return {k: v for k, v in counts.items() if is_feasible(inst, bits_to_x(k))}


def setup_problem(inst=EXAMPLE_3x4, penalty=0.0):
    """Return (inst, H, Q_free, c_free, E_best, E_worst) with the penalty-free score QUBO."""
    scale = objective_scales(inst)
    Q0, c0 = build_qubo(inst, LAM, scale)
    _, e_best, e_worst = brute_force_baselines(inst, Q0, c0)
    Q, c = build_qubo(inst, LAM, scale, penalty=penalty)
    return inst, qubo_to_hamiltonian(Q, c), Q0, c0, e_best, e_worst


def optimise_angles(inst, H, p, mixer, start, seed=0):
    circ = build_qaoa(inst, H, p, mixer=mixer, start=start)
    res = optimise(expectation_energy(circ, H), 2 * p, restarts=3, maxiter=200, seed=seed)
    return circ, np.asarray(res.x)


def ideal_distribution(circ, theta):
    probs = Statevector(circ.assign_parameters(theta)).probabilities_dict()
    return {k: v for k, v in probs.items() if v > 1e-12}


def transpile_for(circ, theta, backend, seed=7):
    bound = circ.assign_parameters(theta)
    bound.measure_all()
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=seed)
    return pm.run(bound)


def circuit_stats(tc):
    ops = tc.count_ops()
    return dict(depth=tc.depth(),
                two_qubit_gates=int(sum(v for k, v in ops.items() if k in TWO_QUBIT_NAMES)))


def run_counts(tc, backend_or_sim, shots=4096, seed=11):
    return backend_or_sim.run(tc, shots=shots, seed_simulator=seed).result().get_counts()


def depolarizing_model(p2q, basis=('cz', 'rz', 'sx', 'x')):
    nm = NoiseModel(basis_gates=list(basis))
    nm.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), ['cz'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2q / 10, 1), ['sx', 'x'])
    return nm


def generic_transpile(circ, theta, seed=7):
    bound = circ.assign_parameters(theta)
    bound.measure_all()
    return transpile(bound, basis_gates=['cz', 'rz', 'sx', 'x'], optimization_level=3,
                     seed_transpiler=seed)


def measured_physical_qubits(tc):
    """Physical qubit read for each virtual qubit (clbit i <- virtual qubit i)."""
    layout = tc.layout
    final = layout.final_index_layout() if layout is not None else list(range(tc.num_qubits))
    return [int(q) for q in final[:len(tc.clbits)]]


def calibrate_readout(backend, sim, physical, shots=4096, seed=3):
    """Per-bit (P(read1|0), P(read0|1)) from all-0 / all-1 calibration circuits on the same qubits."""
    n, rates, out = len(physical), [], {}
    for label in (0, 1):
        qc = QuantumCircuit(n)
        if label:
            qc.x(range(n))
        qc.measure_all()
        tc = transpile(qc, backend, initial_layout=physical, optimization_level=0)
        out[label] = normalise(run_counts(tc, sim, shots, seed + label))
    for i in range(n):  # clbit i is character n-1-i
        ch = n - 1 - i
        p10 = sum(v for k, v in out[0].items() if k[ch] == '1')
        p01 = sum(v for k, v in out[1].items() if k[ch] == '0')
        rates.append((p10, p01))
    return rates


def local_readout_mitigate(dist, rates):
    """Invert independent per-bit flip matrices; rates[i] = (P(1|0), P(0|1)) for clbit i."""
    n = len(rates)
    vec = np.zeros(2 ** n)
    for k, v in normalise(dist).items():
        vec[int(k, 2)] = v
    tensor = vec.reshape([2] * n)  # axis a <-> clbit n-1-a
    for i, (p10, p01) in enumerate(rates):
        inv = np.linalg.inv(np.array([[1 - p10, p01], [p10, 1 - p01]]))
        tensor = np.moveaxis(np.tensordot(inv, tensor, axes=([1], [n - 1 - i])), 0, n - 1 - i)
    vec = np.clip(tensor.reshape(-1), 0, None)
    vec = vec / vec.sum()
    return {format(i, f'0{n}b'): float(v) for i, v in enumerate(vec) if v > 1e-12}


def score(dist, problem):
    inst, _, Q0, c0, e_best, e_worst = problem
    return distribution_metrics(normalise(dist), inst, Q0, c0, e_best, e_worst)
