"""Scaling and parameter-transfer helpers for the EV-charging QAOA.

Fast exact simulation: the XY mixer + Dicke start never leave the feasible
subspace (dimension prod_v C(T, need_v), e.g. 50625 for 4x6 instead of 2^24).
Cost layer is diagonal; each XXPlusYY gate only swaps amplitudes of basis
states that differ by moving one charge between neighbouring slots.
"""
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from evqaoa import Instance

# Overnight tariff (arbitrary price units per slot): expensive evening peak,
# cheap night trough, rising again toward morning.
TARIFF = {4: (4, 1, 1, 3), 5: (4, 2, 1, 1, 3), 6: (4, 3, 1, 1, 2, 3)}
DEFAULT_NEED = 2  # every vehicle needs 2 charging slots


def make_instance(V, T, need=DEFAULT_NEED):
    """Instance with V vehicles, T slots, tariff from TARIFF, equal needs."""
    if T not in TARIFF:
        raise ValueError(f'no tariff defined for T={T}')
    return Instance(V=V, T=T, prices=TARIFF[T], need=(need,) * V)


def feasible_basis(inst):
    """Sorted array of basis-state integers (qubit i = bit i) with exact needs."""
    basis = np.zeros(1, dtype=np.int64)
    for v in range(inst.V):
        masks = [sum(1 << (v * inst.T + t) for t in slots)
                 for slots in combinations(range(inst.T), inst.need[v])]
        basis = np.add.outer(basis, np.array(masks, dtype=np.int64)).ravel()
    return np.sort(basis)


def _bit_matrix(basis, n):
    return (basis[:, None] >> np.arange(n)) & 1


def subspace_energies(inst, Q, const, basis):
    X = _bit_matrix(basis, inst.n).astype(float)
    return np.einsum('ij,jk,ik->i', X, Q, X) + const


def _mixer_pairs(inst):
    """Neighbouring-slot qubit pairs in the same order as circuits.xy_mixer."""
    pairs = []
    for v in range(inst.V):
        base = v * inst.T
        for first in (0, 1):
            pairs += [(base + t, base + t + 1) for t in range(first, inst.T - 1, 2)]
    return pairs


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
    p: int
    energies: np.ndarray
    tables: tuple

    def state(self, theta):
        betas, gammas = np.asarray(theta[:self.p]), np.asarray(theta[self.p:])
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


def make_simulator(inst, Q, const, p):
    basis = feasible_basis(inst)
    energies = subspace_energies(inst, Q, const, basis)
    tables = tuple(_pair_tables(basis, _mixer_pairs(inst)))
    return Simulator(p=p, energies=energies, tables=tables)


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
