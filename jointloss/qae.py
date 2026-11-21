"""Quantum amplitude estimation (QAE) for the five focus targets.

Circuit layout (9 qubits):

* qubits 0-7: state register for the capped term state x = (a, b, r1, r2), with
  a and b on 3 qubits each and r1, r2 on 1 qubit each (x = r2 + 2 r1 + 4 b + 32 a);
* qubit 8: objective qubit.

The operator A prepares  sum_x sqrt(p(x)) |x> ( sqrt(1 - g(x)) |0> + sqrt(g(x)) |1> )
with g(x) = f(x) / f_max, so the probability of reading 1 on the objective qubit
is the amplitude  a = E_p[f] / f_max.  The Grover operator
Q = A S_0 A^dagger S_chi  rotates by 2 theta, with a = sin^2(theta), so after m
applications the probability of reading 1 is sin^2((2m + 1) theta).

The estimator is maximum-likelihood QAE without phase estimation
(Suzuki et al. 2020), which needs only the circuits Q^m A for a schedule of m.

Version 1 of the prototype: the term distribution p(x) is computed classically
and loaded exactly ("oracle-loaded" benchmark). It measures query complexity;
it does not yet make the circuit perform the three-year convolution itself.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, UCRYGate
from qiskit.quantum_info import Statevector

from .model import SHAPE, Scenario, payoff_grids, term_distribution

N_STATE_QUBITS = 8
OBJECTIVE = 8
N_QUBITS = 9


def register_index() -> np.ndarray:
    """Register value x of every grid state, as an array of shape SHAPE."""
    a, b, r1, r2 = np.indices(SHAPE)
    return r2 + 2 * r1 + 4 * b + 32 * a


def encode(sc: Scenario, target) -> tuple[np.ndarray, np.ndarray, float]:
    """Probabilities p(x) and scaled payoffs g(x) on the 256 register values.

    `target` is the name of a focus target, or a payoff grid of shape SHAPE.
    Returns (p, g, f_max). Register values that are not valid states get p = 0.
    """
    f = payoff_grids(sc)[target] if isinstance(target, str) else np.asarray(target, float)
    f_max = float(f.max())
    idx = register_index().ravel()
    p = np.zeros(2 ** N_STATE_QUBITS)
    g = np.zeros(2 ** N_STATE_QUBITS)
    p[idx] = term_distribution(sc).ravel()
    g[idx] = f.ravel() / f_max
    return p, g, f_max


def state_preparation(sc: Scenario, target) -> QuantumCircuit:
    """The operator A: load p(x), then rotate the objective qubit by the payoff."""
    p, g, _ = encode(sc, target)
    qc = QuantumCircuit(N_QUBITS, name="A")
    qc.append(StatePreparation(np.sqrt(p)), range(N_STATE_QUBITS))
    angles = 2.0 * np.arcsin(np.sqrt(np.clip(g, 0.0, 1.0)))
    qc.append(UCRYGate(list(angles)), [OBJECTIVE] + list(range(N_STATE_QUBITS)))
    return qc


def grover_operator(a_circuit: QuantumCircuit) -> QuantumCircuit:
    """Q = A S_0 A^dagger S_chi (up to a global phase)."""
    qc = QuantumCircuit(N_QUBITS, name="Q")
    qc.z(OBJECTIVE)                                   # S_chi: flip the sign of "good" states
    qc.compose(a_circuit.inverse(), inplace=True)     # A^dagger
    qc.x(range(N_QUBITS))                             # S_0: flip the sign of |0...0>
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.compose(a_circuit, inplace=True)               # A
    return qc


def qae_circuit(sc: Scenario, target, power: int) -> QuantumCircuit:
    """The circuit Q^power A, before measurement of the objective qubit."""
    a_circuit = state_preparation(sc, target)
    qc = QuantumCircuit(N_QUBITS)
    qc.compose(a_circuit, inplace=True)
    if power:
        q = grover_operator(a_circuit)
        for _ in range(power):
            qc.compose(q, inplace=True)
    return qc


def objective_probability(qc: QuantumCircuit) -> float:
    """Exact probability of measuring 1 on the objective qubit (statevector)."""
    return float(Statevector(qc).probabilities([OBJECTIVE])[1])


def amplitude(sc: Scenario, target) -> tuple[float, float]:
    """Exact amplitude a = E[f] / f_max and the scale f_max, from the model."""
    p, g, f_max = encode(sc, target)
    return float((p * g).sum()), f_max


# --- Maximum-likelihood amplitude estimation --------------------------------
def schedule(m_max: int, growth: float = 1.5) -> np.ndarray:
    """Geometric schedule of Grover powers: 0, 1, then growing by `growth` up to m_max.

    The usual doubling schedule (growth = 2) left the likelihood with competing
    peaks on the rare targets (occasional errors of 40%). Growth 1.5 removes them
    at a small extra query cost; see results/schedule_study.csv.
    """
    powers, m = [0], 1.0
    while round(m) <= m_max:
        if round(m) != powers[-1]:
            powers.append(int(round(m)))
        m *= growth
    return np.array(powers)


def n_queries(powers: np.ndarray, shots: int) -> int:
    """Number of applications of A (or A^dagger): shots * sum(2 m + 1)."""
    return int(shots * (2 * powers + 1).sum())


def ml_estimate(hits: np.ndarray, powers: np.ndarray, shots: int) -> float:
    """Maximum-likelihood estimate of the amplitude a from the hit counts.

    Global grid search over theta in [0, pi/2] fine enough to resolve the fastest
    oscillation, followed by a local refinement of the best grid cell.
    """
    k = 2 * powers + 1

    def loglik(theta: np.ndarray) -> np.ndarray:
        s = np.sin(np.outer(theta, k)) ** 2
        s = np.clip(s, 1e-300, 1.0 - 1e-16)
        return (hits * np.log(s) + (shots - hits) * np.log1p(-s)).sum(axis=1)

    n_grid = max(2000, 40 * int(k.max()))
    lo, hi = 0.0, np.pi / 2
    for _ in range(4):                                # grid, then three zooms
        grid = np.linspace(lo, hi, n_grid)
        best = int(np.argmax(loglik(grid)))
        lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, n_grid - 1)]
        n_grid = 200
    return float(np.sin(0.5 * (lo + hi)) ** 2)


def simulate_mlqae(a: float, powers: np.ndarray, shots: int, rng: np.random.Generator,
                   runs: int = 1) -> np.ndarray:
    """Run maximum-likelihood QAE `runs` times on an ideal (noise-free) device.

    Measurement outcomes of the circuit Q^m A are Bernoulli with probability
    sin^2((2m + 1) theta). The circuit tests confirm this to machine precision,
    so shots are drawn from that law directly, which is exactly what an ideal
    shot-based simulation of the circuits returns.
    """
    theta = np.arcsin(np.sqrt(a))
    prob = np.sin((2 * powers + 1) * theta) ** 2
    hits = rng.binomial(shots, prob, size=(runs, len(powers)))
    return np.array([ml_estimate(h, powers, shots) for h in hits])


# --- Resource counts ---------------------------------------------------------
BASIS = ["cx", "rz", "sx", "x"]


def gate_counts(qc: QuantumCircuit) -> dict[str, int]:
    """Transpile to a CX + single-qubit basis (all-to-all) and count resources."""
    tqc = transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=7)
    ops = tqc.count_ops()
    return {
        "qubits": tqc.num_qubits,
        "two_qubit_gates": int(ops.get("cx", 0)),
        "single_qubit_gates": int(sum(v for g, v in ops.items() if g != "cx")),
        "depth": int(tqc.depth()),
        "two_qubit_depth": int(tqc.depth(lambda inst: inst.operation.num_qubits == 2)),
    }
