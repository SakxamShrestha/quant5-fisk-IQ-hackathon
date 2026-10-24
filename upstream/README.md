# EV Depot Charging with QAOA

Scheduling electric-vehicle charging at a depot as a three-objective optimisation problem, encoded as a QUBO / Ising Hamiltonian and solved with QAOA in Qiskit (Aer simulation + IBM QPU run). Qiskit Fall Fest, Track 2: Quantum Application Engineering.

## Problem statement

A depot charges V electric vehicles over T overnight time slots. Decision variable `x\[v,t] = 1` if vehicle v charges in slot t (qubit index = `v\*T + t`). Each vehicle needs exactly `need\[v]` charging slots. Three objectives, all minimised:

|Objective|Formula|Meaning|
|-|-|-|
|Cost|`sum\_{v,t} price\[t] \* x\[v,t]`|electricity bill|
|Peak proxy|`sum\_t (sum\_v x\[v,t])^2`|penalises many vehicles charging together|
|Wear|`sum\_{v,t} (x\[v,t] - x\[v,t+1])^2`|start/stop switches|

Worked example: 3 vehicles, 4 slots, prices (4,1,1,3), each needs 2 slots -> 216 feasible schedules and a 7-point Pareto front (verified by brute force and a unit test).

## Architecture

1. `Instance`, `objectives`, brute-force enumeration, Pareto filter, objective normalisation
2. QUBO builder (`build\_qubo`, with optional constraint penalty)
3. QUBO -> Ising Hamiltonian (`SparsePauliOp`, `x = (1 - Z)/2`)
4. QAOA ansatz (`QAOAAnsatz`, p = 1, 12 qubits)
5. Parameter optimisation with COBYLA on an exact statevector (Part A, local simulation)
6. Decoding / analysing bitstrings into 3x4 charging matrices
7. Part B: transpilation to IBM `ibm\_fez`, `SamplerV2` with dynamical decoupling (XpXm), gate twirling and measurement twirling, 4096 shots; comparison with Aer

Tests (in the notebook) check: brute-force counts and Pareto front, QUBO == weighted objectives on all 216 schedules, Hamiltonian energy == QUBO energy, circuit has 12 qubits and 2 parameters.

## How to reproduce

```
pip install -r requirements.txt
jupyter notebook notebooks/dead\_and\_alive\_orignal.ipynb
```

The notebook loads the saved hardware counts from `results/` (`RUN\_ON\_HARDWARE = False`), so no QPU or IBM account is needed. Set it to `True` with your own IBM token to resubmit.

## Results

* Validation tests all pass (see notebook).
* Hardware run: backend `ibm\_fez`, job ID `db17t3megvvc73bh96m0`, 4096 shots, transpiled depth 31 -> 103.
* Hardware output was strongly noisy: 1516 unique bitstrings, 28 of them valid schedules.
* Best valid schedules found (both in simulation and on hardware) reach weighted score 0.5175, which equals the brute-force optimum among valid schedules.
* Plots: charging schedule/heatmap, QPU histogram, Aer-vs-QPU distribution (see notebook; add PNGs from `results/`).

## Limitations (honest)

* **No quantum speedup is claimed.** At 12 qubits, brute force over 216 schedules is instant.
* The ansatz uses the standard X mixer without the penalty term, so the constraint "exactly `need\[v]` slots per vehicle" is **not enforced during optimisation**. The optimiser lowers the energy by moving probability towards low-charging (all-zero) states, so only a small fraction of samples are valid schedules, and valid schedules are filtered by post-selection afterwards. Valid schedules therefore appear with low probability.
* COBYLA was limited to 30 evaluations and did not fully converge; p = 1 only.
* Hardware result is a single p = 1 run with error suppression only (DD + twirling); no ZNE.
* Only one weighting (0.5, 0.3, 0.2) was run, so the full Pareto front was not reconstructed by QAOA.

## Future work

Constraint-preserving XY mixer with Dicke/basis start state (removes invalid outputs and gives hardware error detection), penalty-weight sweep, multiple weightings to rebuild the Pareto front with a random baseline, parameter transfer to larger instances, CVaR objective, larger instances (4x4, 4x5, 4x6).

## References

* Farhi, Goldstone, Gutmann (2014). A Quantum Approximate Optimization Algorithm. arXiv:1411.4028
* Hadfield et al. (2019). From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz. Algorithms 12(2):34
* Wang, Rubin, Dominy, Rieffel (2020). XY mixers. Phys. Rev. A 101, 012320 (future work)
* Qiskit and qiskit-ibm-runtime documentation, https://docs.quantum.ibm.com
* Qiskit Fall Fest, "General Prompt: Project Tracks \& Guidelines"

## License

MIT, see `LICENSE`. Authors: <Yash Khamkar>

