"""Noisy-hardware study helpers: FakeFez noise, depolarizing sweep, readout correction."""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel, depolarizing_error

from evqaoa import (DEPOT, bitstring_to_schedule, charging_model, energy_function,
                    exact_baselines, is_valid, minimise_energy, score_distribution,
                    to_ising)

TWO_QUBIT_NAMES = {'cz', 'cx', 'ecr', 'rzz', 'xx_plus_yy', 'swap'}
WEIGHTS = (0.5, 0.3, 0.2)


def setup(problem=DEPOT, penalty=0.0):
    """(scoring model, Hamiltonian to optimise, exact baselines). Scoring never includes the penalty."""
    score_model = charging_model(problem, WEIGHTS)
    opt_model = charging_model(problem, WEIGHTS, penalty=penalty) if penalty else score_model
    return score_model, to_ising(opt_model), exact_baselines(problem, score_model)


def normalise(dist):
    total = float(sum(dist.values()))
    return {k.replace(' ', ''): v / total for k, v in dist.items()}


def hellinger_fidelity(p, q):
    p, q = normalise(p), normalise(q)
    bc = sum(np.sqrt(p[k] * q.get(k, 0.0)) for k in p)
    return float(bc ** 2)


def keep_valid(dist, problem=DEPOT):
    """Post-selection: drop every shot that breaks a vehicle's demand."""
    return {k: v for k, v in dist.items() if is_valid(problem, bitstring_to_schedule(k))}


def best_angles(circ, H, seed=0, restarts=3, maxiter=300):
    res = minimise_energy(energy_function(circ, H), circ.num_parameters,
                          restarts=restarts, maxiter=maxiter, seed=seed)
    return np.asarray(res.x)


def transpile_for(circ, theta, backend, seed=7):
    bound = circ.assign_parameters(theta)
    bound.measure_all()
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=seed)
    return pm.run(bound)


def circuit_stats(tc):
    ops = tc.count_ops()
    return dict(depth=int(tc.depth()),
                two_qubit_gates=int(sum(v for k, v in ops.items() if k in TWO_QUBIT_NAMES)))


def sample(tc, simulator, shots=4096, seed=11):
    return simulator.run(tc, shots=shots, seed_simulator=seed).result().get_counts()


def depolarizing_model(p2q, basis=('cz', 'rz', 'sx', 'x')):
    """Generic device: two-qubit depolarizing p2q, single-qubit p2q/10."""
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
    """Physical qubit read into each classical bit (bit i <- virtual qubit i)."""
    layout = tc.layout
    final = layout.final_index_layout() if layout is not None else list(range(tc.num_qubits))
    return [int(q) for q in final[:len(tc.clbits)]]


def calibrate_readout(backend, simulator, physical, shots=4096, seed=3):
    """Per-bit (P(read 1 | 0), P(read 0 | 1)) from all-0 and all-1 calibration circuits."""
    n, prepared = len(physical), {}
    for label in (0, 1):
        qc = QuantumCircuit(n)
        if label:
            qc.x(range(n))
        qc.measure_all()
        tc = transpile(qc, backend, initial_layout=physical, optimization_level=0)
        prepared[label] = normalise(sample(tc, simulator, shots, seed + label))
    rates = []
    for i in range(n):                       # bit i is character n-1-i
        ch = n - 1 - i
        p10 = sum(v for k, v in prepared[0].items() if k[ch] == '1')
        p01 = sum(v for k, v in prepared[1].items() if k[ch] == '0')
        rates.append((p10, p01))
    return rates


def correct_readout(dist, rates):
    """Undo independent per-bit flips by inverting each 2x2 confusion matrix."""
    n = len(rates)
    vec = np.zeros(2 ** n)
    for k, v in normalise(dist).items():
        vec[int(k, 2)] = v
    tensor = vec.reshape([2] * n)            # axis a <-> bit n-1-a
    for i, (p10, p01) in enumerate(rates):
        inv = np.linalg.inv(np.array([[1 - p10, p01], [p10, 1 - p01]]))
        axis = n - 1 - i
        tensor = np.moveaxis(np.tensordot(inv, tensor, axes=([1], [axis])), 0, axis)
    vec = np.clip(tensor.reshape(-1), 0, None)
    vec /= vec.sum()
    return {format(i, f'0{n}b'): float(v) for i, v in enumerate(vec) if v > 1e-12}


def score(dist, model, baselines, problem=DEPOT):
    return score_distribution(normalise(dist), problem, model, baselines)
