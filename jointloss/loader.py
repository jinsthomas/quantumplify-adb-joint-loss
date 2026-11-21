"""Shared-shock loader: a state preparation that scales with the number of countries.

The exact loader in `qae.py` writes the whole joint distribution into the circuit,
so its cost grows with the 10^n state space. The ADB model has more structure:

* earthquakes: country-only events plus shared events that hit every country;
  given the NUMBER k of shared events in the term, the countries are independent;
* precipitation: a shared trigger event plus country-only triggers; given whether
  the shared event occurred (s), the countries are independent.

Because all events are Poisson, three years with caps equal one period of three
years, so the term distribution can be loaded directly (no in-circuit adder).

Circuit: a 2-qubit register for k (0, 1, 2, 3+), 1 qubit for s, then for every
country 3 qubits for its earthquake steps and 1 qubit for its rain trigger, each
prepared by rotations controlled on k or s. Qubits: 3 + 4n. The circuit is built
from n small per-country tables (4 x 5 numbers each), never from the joint table.

This is the conditional-independence construction of Egger et al. (2021) for
credit risk, applied to a common-shock catastrophe model.

For more than two countries the brief defines no model. Here every country shares
ONE regional shock source and the two calibrated countries are reused in turn;
this is an assumption made to study scaling, not an ADB portfolio.
"""
from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, UCRYGate
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from .model import (LAMBDA_KYRGYZ, LAMBDA_TAJIK, N_STEPS, N_YEARS, RAIN_Q, Scenario, shared_eq_rate)

K_MAX = 3                      # shared-event counts 0, 1, 2 and "3 or more"
BASIS = ["cx", "rz", "sx", "x"]
CALIBRATIONS = (LAMBDA_KYRGYZ, LAMBDA_TAJIK)


def country_rates(n: int) -> list[np.ndarray]:
    """Event rates of n countries (the two calibrated countries, reused in turn)."""
    return [np.array(CALIBRATIONS[i % 2]) for i in range(n)]


# --- Classical pieces (per country, O(n) in total) ---------------------------
def _capped_add(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    out = np.zeros(N_STEPS + 1)
    for i, pi in enumerate(p):
        for j, qj in enumerate(q):
            out[min(i + j, N_STEPS)] += pi * qj
    return out


def _own_events(rate: float, sizes: np.ndarray) -> np.ndarray:
    """Capped steps from a country's own events over the term (5-state chain)."""
    g = np.zeros((N_STEPS + 1, N_STEPS + 1))
    for a in range(N_STEPS + 1):
        for k in range(1, N_STEPS + 1):
            g[a, min(a + k, N_STEPS)] += rate * sizes[k - 1]
    np.fill_diagonal(g, 0.0)
    np.fill_diagonal(g, -g.sum(axis=1))
    return expm(g)[0]


def shared_count_pmf(sc: Scenario, years: int = N_YEARS) -> np.ndarray:
    """P(k shared earthquake events in the term), k = 0, 1, 2 and 3 or more."""
    c = shared_eq_rate(sc) * years
    pk = np.array([math.exp(-c) * c ** k / math.factorial(k) for k in range(K_MAX)])
    return np.append(pk, 1.0 - pk.sum())


def country_conditionals(sc: Scenario, lam: np.ndarray, years: int = N_YEARS) -> np.ndarray:
    """P(steps of one country | k shared events), shape (4, 5).

    Row 3 stands for "3 or more" shared events and uses exactly 3; the mass
    involved is about 1e-10 for the term.
    """
    c = shared_eq_rate(sc)
    total = lam.sum()
    sizes = lam / total
    current = _own_events((total - c) * years, sizes)
    mark = np.concatenate([[0.0], sizes])
    rows = []
    for _ in range(K_MAX + 1):
        rows.append(current)
        current = _capped_add(current, mark)
    return np.array(rows)


def rain_parameters(sc: Scenario, years: int = N_YEARS) -> tuple[float, float]:
    """(P(shared rain event in the term), P(country-only trigger in the term))."""
    mu = -math.log(1.0 - RAIN_Q) * sc.rain_rate_multiplier
    shared = sc.rain_shared_fraction * mu
    return 1.0 - math.exp(-shared * years), 1.0 - math.exp(-(mu - shared) * years)


# --- Qubit layout -------------------------------------------------------------
def k_qubits() -> list[int]:
    return [0, 1]


S_QUBIT = 2


def a_qubits(i: int) -> list[int]:
    return [3 + 4 * i, 4 + 4 * i, 5 + 4 * i]


def r_qubit(i: int) -> int:
    return 6 + 4 * i


def n_loader_qubits(n: int) -> int:
    return 3 + 4 * n


def _angle(p_one: float) -> float:
    return 2.0 * math.asin(math.sqrt(min(max(p_one, 0.0), 1.0)))


def _multiplexed_ry(qc: QuantumCircuit, target: int, controls: list[int], prob_one) -> None:
    """Rotate `target` so that P(1) = prob_one(control bits), for every control value."""
    n = len(controls)
    angles = []
    for idx in range(2 ** n):
        bits = tuple((idx >> j) & 1 for j in range(n))
        angles.append(_angle(prob_one(bits)))
    qc.append(UCRYGate(angles), [target] + controls)


def _ratio(num: float, den: float) -> float:
    return 0.0 if den <= 0 else num / den


def shared_shock_loader(sc: Scenario, n: int = 2, years: int = N_YEARS) -> QuantumCircuit:
    """State preparation of the n-country term distribution on 3 + 4n qubits."""
    qc = QuantumCircuit(n_loader_qubits(n), name="loader")
    qc.append(StatePreparation(np.sqrt(shared_count_pmf(sc, years))), k_qubits())
    p_shared_rain, p_own_rain = rain_parameters(sc, years)
    qc.ry(_angle(p_shared_rain), S_QUBIT)
    k0, k1 = k_qubits()
    for i, lam in enumerate(country_rates(n)):
        pmf = country_conditionals(sc, lam, years)          # (4, 5)
        a0, a1, a2 = a_qubits(i)
        row = lambda b: pmf[b[0] + 2 * b[1]]
        # Binary tree over the value a = a0 + 2 a1 + 4 a2, in 0..4.
        _multiplexed_ry(qc, a2, [k0, k1], lambda b: row(b)[4])
        _multiplexed_ry(qc, a1, [k0, k1, a2],
                        lambda b: 0.0 if b[2] else _ratio(row(b)[2] + row(b)[3], row(b)[:4].sum()))
        _multiplexed_ry(qc, a0, [k0, k1, a1, a2],
                        lambda b: 0.0 if b[3] else _ratio(row(b)[2 * b[2] + 1],
                                                          row(b)[2 * b[2]] + row(b)[2 * b[2] + 1]))
        _multiplexed_ry(qc, r_qubit(i), [S_QUBIT], lambda b: 1.0 if b[0] else p_own_rain)
    return qc


def loaded_distribution(qc: QuantumCircuit, n: int) -> np.ndarray:
    """Distribution over (a_1..a_n, r_1..r_n) encoded by the loader (statevector)."""
    qubits = [q for i in range(n) for q in a_qubits(i)] + [r_qubit(i) for i in range(n)]
    probs = Statevector(qc).probabilities(qubits)            # qubits[0] is the least significant bit
    out = np.zeros((N_STEPS + 1,) * n + (2,) * n)
    for idx in np.flatnonzero(probs > 0):
        a = [(idx >> (3 * i)) & 7 for i in range(n)]
        r = [(idx >> (3 * n + i)) & 1 for i in range(n)]
        if max(a) > N_STEPS:
            assert probs[idx] < 1e-18
            continue
        out[tuple(a + r)] += probs[idx]
    return out


# --- Exact n-country references ----------------------------------------------
def exact_distribution_brute_force(sc: Scenario, n: int, years: int = N_YEARS) -> np.ndarray:
    """Term distribution from the full 5^n-state chain (no use of the mixture).

    Independent check of the loader; feasible for n up to about 4.
    """
    c = shared_eq_rate(sc)
    rates = country_rates(n)
    size = (N_STEPS + 1) ** n
    states = np.array(np.unravel_index(np.arange(size), (N_STEPS + 1,) * n)).T
    index = {tuple(s): j for j, s in enumerate(states)}
    g = np.zeros((size, size))
    marks = [np.concatenate([[0.0], lam / lam.sum()]) for lam in rates]
    for j, s in enumerate(states):
        for i, lam in enumerate(rates):                       # country-only events
            for k in range(1, N_STEPS + 1):
                t = s.copy()
                t[i] = min(t[i] + k, N_STEPS)
                g[j, index[tuple(t)]] += (lam.sum() - c) * marks[i][k]
        if c > 0:                                             # shared events hit every country
            for ks in np.ndindex(*(N_STEPS,) * n):
                weight = c * np.prod([marks[i][k + 1] for i, k in enumerate(ks)])
                t = np.minimum(s + np.array(ks) + 1, N_STEPS)
                g[j, index[tuple(t)]] += weight
    np.fill_diagonal(g, 0.0)
    np.fill_diagonal(g, -g.sum(axis=1))
    eq = expm(g * years)[0].reshape((N_STEPS + 1,) * n)
    p_shared, p_own = rain_parameters(sc, years)
    rain = np.zeros((2,) * n)
    for r in np.ndindex(*(2,) * n):
        ones = sum(r)
        given_no_shared = p_own ** ones * (1 - p_own) ** (n - ones)
        rain[r] = (1 - p_shared) * given_no_shared + (p_shared if ones == n else 0.0)
    return np.multiply.outer(eq, rain)


def exact_product_event(sc: Scenario, n: int, country_event, years: int = N_YEARS) -> float:
    """P(event holds in every country), for an event defined per country.

    `country_event(a, r)` is True or False. Uses the shared-shock mixture, so the
    cost is linear in n: this is the exact CLASSICAL method that the same
    structure makes available.
    """
    pk = shared_count_pmf(sc, years)
    p_shared, p_own = rain_parameters(sc, years)
    total = 0.0
    for k, weight_k in enumerate(pk):
        for s, weight_s in ((0, 1 - p_shared), (1, p_shared)):
            prod = 1.0
            for lam in country_rates(n):
                pmf = country_conditionals(sc, lam, years)[k]
                p_rain = 1.0 if s else p_own
                prod *= sum(pmf[a] * (p_rain if r else 1 - p_rain)
                            for a in range(N_STEPS + 1) for r in (0, 1) if country_event(a, r))
            total += weight_k * weight_s * prod
    return float(total)


# --- Amplitude-estimation circuits on top of the loader -----------------------
EVENTS = {
    "all_bonds_pay": lambda a, r: a > 0 or r == 1,
    "all_bonds_exhausted": lambda a, r: a == N_STEPS and r == 1,
}


def flag_state_preparation(sc: Scenario, n: int, event: str, years: int = N_YEARS):
    """A = loader followed by a flag on the objective qubit. Returns (circuit, objective)."""
    base = n_loader_qubits(n)
    if event == "all_bonds_exhausted":
        qc = QuantumCircuit(base + 1, name="A")
        qc.compose(shared_shock_loader(sc, n, years), range(base), inplace=True)
        controls = [q for i in range(n) for q in (a_qubits(i)[2], r_qubit(i))]
        qc.mcx(controls, base)
        return qc, base
    if event == "all_bonds_pay":
        qc = QuantumCircuit(base + n + 1, name="A")           # one "pays" qubit per country
        qc.compose(shared_shock_loader(sc, n, years), range(base), inplace=True)
        for i in range(n):
            bits = a_qubits(i) + [r_qubit(i)]
            qc.x(bits)                                        # pays = NOT(all four bits zero)
            qc.mcx(bits, base + i)
            qc.x(bits)
            qc.x(base + i)
        qc.mcx(list(range(base, base + n)), base + n)
        return qc, base + n
    raise ValueError(event)


def grover_operator(a_circuit: QuantumCircuit, objective: int) -> QuantumCircuit:
    """Q = A S_0 A^dagger S_chi for any A (up to a global phase)."""
    n = a_circuit.num_qubits
    qc = QuantumCircuit(n, name="Q")
    qc.z(objective)
    qc.compose(a_circuit.inverse(), inplace=True)
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.compose(a_circuit, inplace=True)
    return qc


def objective_probability(a_circuit: QuantumCircuit, objective: int, power: int) -> float:
    """P(objective = 1) after Q^power A, by statevector simulation."""
    qc = a_circuit.copy()
    if power:
        q = grover_operator(a_circuit, objective)
        for _ in range(power):
            qc.compose(q, inplace=True)
    return float(Statevector(qc).probabilities([objective])[1])


def gate_counts(qc: QuantumCircuit) -> dict[str, int]:
    tqc = transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=7)
    ops = tqc.count_ops()
    return {"qubits": tqc.num_qubits, "two_qubit_gates": int(ops.get("cx", 0)),
            "two_qubit_depth": int(tqc.depth(lambda inst: inst.operation.num_qubits == 2))}
