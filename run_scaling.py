"""Scaling study of the shared-shock loader with the number of countries.

Writes results/scaling.csv: qubits and two-qubit gates of the loader, of the
state preparation A and of one Grover iterate Q, the exact event probability
(shared-shock mixture, linear cost in n), and whether the circuit was verified
on a statevector. Countries beyond two reuse the two calibrated countries and
share one regional shock source (an assumption for the scaling study).
"""
import csv
import time
from pathlib import Path

import numpy as np

from jointloss import loader as L
from jointloss.model import SCENARIOS

OUT = Path(__file__).parent / "results" / "scaling.csv"
VERIFY_UP_TO = {"all_bonds_exhausted": 4, "all_bonds_pay": 3}     # statevector size limit


def main():
    sc = SCENARIOS["adb-base"]
    rows = []
    for n in (2, 3, 4, 5, 6, 8, 10):
        loader = L.gate_counts(L.shared_shock_loader(sc, n))
        for event in L.EVENTS:
            t0 = time.time()
            a_circuit, objective = L.flag_state_preparation(sc, n, event)
            a_counts = L.gate_counts(a_circuit)
            q_counts = L.gate_counts(L.grover_operator(a_circuit, objective))
            exact = L.exact_product_event(sc, n, L.EVENTS[event])
            verified, max_dev = "no", ""
            if n <= VERIFY_UP_TO[event]:
                theta = np.arcsin(np.sqrt(exact))
                dev = max(abs(L.objective_probability(a_circuit, objective, m) - np.sin((2 * m + 1) * theta) ** 2)
                          for m in (0, 1, 2))
                verified, max_dev = "yes", f"{dev:.1e}"
            rows.append([n, 10 ** n, 16 ** n, event, exact, loader["qubits"], loader["two_qubit_gates"],
                         a_counts["qubits"], a_counts["two_qubit_gates"], q_counts["two_qubit_gates"],
                         q_counts["two_qubit_depth"], verified, max_dev])
            print(n, event, f"{exact:.3e}", a_counts, q_counts["two_qubit_gates"], verified, max_dev,
                  f"{time.time() - t0:.0f}s")
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["countries", "annual_states", "amplitudes_for_exact_loading", "event", "exact_probability",
                    "loader_qubits", "loader_two_qubit_gates", "A_qubits", "A_two_qubit_gates",
                    "Q_two_qubit_gates", "Q_two_qubit_depth", "statevector_verified", "max_deviation"])
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
