"""Scaling and parameter-transfer helpers for the EV-charging QAOA.

Fast exact simulation: the XY mixer + Dicke start never leave the feasible
subspace (dimension prod_v C(T, need_v), e.g. 50625 for 4x6 instead of 2^24).
The cost layer is diagonal; each XXPlusYY gate only swaps amplitudes of basis
states that differ by moving one charge between neighbouring slots.
"""
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from evqaoa import DEPOT, ChargingProblem

# Night time-of-use tariffs in cents/kWh: an expensive evening peak, a cheap
# overnight trough, then a rise toward the morning commute. The 4-slot tariff is
# the depot's; the 5- and 6-slot ones are our own plausible stretches of it
# (earlier slots step down gradually into the same trough and morning price).
TARIFF = {4: DEPOT.tariff,
          5: (32, 18, 11, 9, 21),
          6: (32, 20, 13, 9, 10, 21)}
DEFAULT_DEMAND = 2  # slots of charging per vehicle, except the depot's own (2, 1, 2)


def make_problem(n_vehicles, n_slots, demand=None):
    """ChargingProblem on the family tariff; DEPOT's demand for 3x4, else 2 per vehicle."""
    if n_slots not in TARIFF:
        raise ValueError(f'no tariff defined for {n_slots} slots')
    if demand is None:
        is_depot = (n_vehicles, n_slots) == (DEPOT.n_vehicles, DEPOT.n_slots)
        demand = DEPOT.demand if is_depot else (DEFAULT_DEMAND,) * n_vehicles
    return ChargingProblem(n_vehicles, n_slots, TARIFF[n_slots], tuple(demand))


def feasible_basis(problem):
    """Sorted array of basis-state integers (qubit i = bit i) meeting every demand."""
    basis = np.zeros(1, dtype=np.int64)
    for v in range(problem.n_vehicles):
        masks = [sum(1 << problem.qubit(v, t) for t in slots)
                 for slots in combinations(range(problem.n_slots), problem.demand[v])]
        basis = np.add.outer(basis, np.array(masks, dtype=np.int64)).ravel()
    return np.sort(basis)


def _bit_matrix(basis, n):
    return (basis[:, None] >> np.arange(n)) & 1


def subspace_energies(problem, model, basis):
    """Model energy of every feasible basis state."""
    return model.values(_bit_matrix(basis, problem.n_qubits))


def _mixer_pairs(problem):
    """Neighbouring-slot qubit pairs in the gate order of evqaoa.ansatz (even, then odd)."""
    return [(problem.qubit(v, t), problem.qubit(v, t + 1))
            for v in range(problem.n_vehicles)
            for parity in (0, 1)
            for t in range(parity, problem.n_slots - 1, 2)]


def _pair_tables(basis, pairs):
    tables = []
    for a, b in pairs:
        differ = (((basis >> a) ^ (basis >> b)) & 1).astype(bool)
        partner = np.arange(len(basis))
        partner[differ] = np.searchsorted(basis, basis[differ] ^ ((1 << a) | (1 << b)))
        tables.append((differ.astype(float), partner))
    return tables


@dataclass(frozen=True)
class Simulator:
    energies: np.ndarray
    tables: tuple

    def state(self, theta):
        theta = np.asarray(theta, dtype=float)
        p = len(theta) // 2
        betas, gammas = theta[:p], theta[p:]
        psi = np.full(len(self.energies), 1 / np.sqrt(len(self.energies)), complex)
        for beta, gamma in zip(betas, gammas):
            psi = psi * np.exp(-1j * gamma * self.energies)
            c, s = np.cos(beta), np.sin(beta)
            for differ, partner in self.tables:
                psi = psi * (1 - differ) + differ * (c * psi - 1j * s * psi[partner])
        return psi

    def probs(self, theta):
        return np.abs(self.state(theta)) ** 2

    def energy(self, theta):
        return float(self.probs(theta) @ self.energies)


def make_simulator(problem, model):
    """Depth is inferred from len(theta), so one simulator serves every p."""
    basis = feasible_basis(problem)
    energies = subspace_energies(problem, model, basis)
    tables = tuple(_pair_tables(basis, _mixer_pairs(problem)))
    return Simulator(energies=energies, tables=tables)


def metrics(sim, theta):
    """Approx ratio and P(optimal) over the feasible space (feasible rate is 1)."""
    probs, e = sim.probs(theta), sim.energies
    best, worst = e.min(), e.max()
    mean = float(probs @ e)
    p_opt = float(probs[np.isclose(e, best, atol=1e-9)].sum())
    return dict(energy=mean, approx_ratio=float((worst - mean) / (worst - best)),
                p_optimal=p_opt, e_best=float(best), e_worst=float(worst),
                e_mean_uniform=float(e.mean()))


def transfer_angles(theta, p):
    """Copy angles (betas then gammas) for reuse on another instance."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2 * p,):
        raise ValueError(f'expected {2 * p} angles for p={p}, got {theta.shape}')
    return theta.copy()


def run_cobyla(f, x0, maxiter=200):
    """COBYLA run recording the best-so-far energy after every evaluation."""
    trace = []

    def wrapped(x):
        val = f(x)
        trace.append(min(val, trace[-1]) if trace else val)
        return val

    res = minimize(wrapped, np.asarray(x0, float), method='COBYLA',
                   options={'maxiter': maxiter})
    return dict(x=res.x, fun=float(res.fun), nfev=len(trace), trace=trace)


def random_restarts(f, n_params, restarts, maxiter, seed):
    """Several random-start COBYLA runs; returns best angles and a joint trace."""
    rng = np.random.default_rng(seed)
    best_x, best_fun, joint = None, np.inf, []
    for _ in range(restarts):
        run = run_cobyla(f, rng.uniform(0, np.pi, n_params), maxiter)
        joint += [min(v, best_fun) for v in run['trace']]
        if run['fun'] < best_fun:
            best_x, best_fun = run['x'], run['fun']
    return dict(x=best_x, fun=best_fun, nfev=len(joint), trace=joint)


def evals_to_reach(trace, target_energy):
    """1-based number of evaluations until best-so-far <= target, else None."""
    for i, val in enumerate(trace, start=1):
        if val <= target_energy:
            return i
    return None


def time_call(fn, repeats=3):
    """Median wall-clock seconds of fn()."""
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        times.append(perf_counter() - t0)
    return float(np.median(times))
