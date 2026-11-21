"""Fault-tolerant resource estimate and noise limits for the rarest target.

Two circuit designs for "both bonds exhausted" (two countries, base scenario):
  exact loading       9 qubits, joint table loaded amplitude by amplitude
  shared-shock loader 12 qubits, conditional-independence loader + flag

For each, counts the non-Clifford rotations of A and Q, then runs the Microsoft
Azure Quantum Resource Estimator (package `qdk`, offline) on the deepest circuit
and on the whole measurement campaign of two schedules:
  schedule(256)   largest Grover power 195, relative RMSE 6.8% (10% class)
  schedule(2048)  largest Grover power 1478, relative RMSE 0.85% (1% class)

Assumptions of the estimator: surface code, 15-to-1 T factories, error budget 1%
per shot, rotation synthesis as in Beverland et al. (2022). Qubit models:
  ns_e3  50 ns gates, physical error 1e-3 (superconducting-like)
  ns_e4  50 ns gates, physical error 1e-4
  us_e3  100 microsecond gates, physical error 1e-3 (trapped-ion-like)

Writes results/fault_tolerant.csv and results/noise_limits.csv.
Needs: pip install "qdk[qre,qiskit]"   (optional dependency)
"""
import csv
import math
import warnings
from pathlib import Path

import numpy as np
from qiskit import transpile

from jointloss import loader as L
from jointloss import qae
from jointloss.model import BASE, TARGETS

warnings.filterwarnings("ignore")
RES = Path(__file__).parent / "results"
SHOTS = 32
QUBIT_MODELS = {"ns_e3": "qubit_gate_ns_e3", "ns_e4": "qubit_gate_ns_e4", "us_e3": "qubit_gate_us_e3"}
SCHEDULES = {"10% class (RMSE 6.8%)": 256, "1% class (RMSE 0.85%)": 2048}


def non_clifford(circuit):
    """(arbitrary-angle rotations, T-like rotations, CX gates) after transpilation."""
    t = transpile(circuit, basis_gates=qae.BASIS, optimization_level=3, seed_transpiler=7)
    rot = t_like = 0
    for inst in t.data:
        if inst.operation.name == "rz":
            k = float(inst.operation.params[0]) / (np.pi / 4)
            if abs(k - round(k)) < 1e-9:
                t_like += int(round(k)) % 2
            else:
                rot += 1
    return rot, t_like, int(t.count_ops().get("cx", 0))


def designs():
    a1 = qae.state_preparation(BASE, TARGETS[4])
    a2, obj = L.flag_state_preparation(BASE, 2, "all_bonds_exhausted")
    return {
        "exact loading": (a1.num_qubits, non_clifford(a1), non_clifford(qae.grover_operator(a1))),
        "shared-shock loader": (a2.num_qubits, non_clifford(a2), non_clifford(L.grover_operator(a2, obj))),
    }


def estimate(n_qubits, rotations, t_count, model):
    from qdk.estimator import EstimatorParams, LogicalCounts
    p = EstimatorParams()
    p.qubit_params.name = QUBIT_MODELS[model]
    p.error_budget = 0.01
    r = LogicalCounts({"numQubits": n_qubits, "tCount": t_count, "rotationCount": rotations,
                       "rotationDepth": rotations}).estimate(params=p)
    pc, bd = r["physicalCounts"], r["physicalCounts"]["breakdown"]
    return {"physical_qubits": pc["physicalQubits"], "runtime_s": pc["runtime"] / 1e9,
            "code_distance": r["logicalQubit"]["codeDistance"], "t_states": bd["numTstates"],
            "t_per_rotation": bd["numTsPerRotation"], "factories": bd["numTfactories"],
            "logical_qubits": bd["algorithmicLogicalQubits"]}


def main():
    rows, noise = [], []
    for name, (n_qubits, (rot_a, t_a, cx_a), (rot_q, t_q, cx_q)) in designs().items():
        print(f"{name}: {n_qubits} qubits; A: {rot_a} rotations, {cx_a} CX; Q: {rot_q} rotations, {cx_q} CX")
        for label, m_max in SCHEDULES.items():
            powers = qae.schedule(m_max)
            deepest = int(powers[-1])
            for model in QUBIT_MODELS:
                e = estimate(n_qubits, rot_a + deepest * rot_q, t_a + deepest * t_q, model)
                campaign_s = sum(SHOTS * estimate(n_qubits, rot_a + int(m) * rot_q, t_a + int(m) * t_q, model)["runtime_s"]
                                 for m in powers)
                campaign_t = sum(SHOTS * estimate(n_qubits, rot_a + int(m) * rot_q, t_a + int(m) * t_q, model)["t_states"]
                                 for m in powers)
                rows.append([name, label, deepest, cx_a + deepest * cx_q, model, e["logical_qubits"],
                             e["code_distance"], e["t_per_rotation"], e["t_states"], e["factories"],
                             e["physical_qubits"], e["runtime_s"], campaign_t, campaign_s])
        # Noise (Tanaka et al. 2021): the signal contrast decays as exp(-kappa m), kappa = CX(Q) x error.
        theta = math.asin(math.sqrt(3.1633e-5))
        first_peak = (math.pi / (2 * theta) - 1) / 2
        for err in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
            kappa = cx_q * err
            noise.append([name, err, math.exp(-cx_a * err), 1 / kappa, math.exp(-kappa * first_peak),
                          math.exp(-kappa * 195), math.exp(-kappa * 1478)])
    with open(RES / "fault_tolerant.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["design", "accuracy_class", "largest_grover_power", "two_qubit_gates_deepest_circuit",
                    "qubit_model", "logical_qubits_with_layout", "code_distance", "t_per_rotation",
                    "t_states_deepest_circuit", "t_factories", "physical_qubits", "seconds_per_shot_deepest",
                    "t_states_campaign", "seconds_campaign"])
        w.writerows(rows)
    with open(RES / "noise_limits.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["design", "two_qubit_error", "fidelity_of_A", "useful_grover_power_1_over_kappa",
                    "contrast_at_first_signal_peak_m139", "contrast_at_m195", "contrast_at_m1478"])
        w.writerows(noise)
    for r in rows:
        print(r[0][:12], r[1][:9], r[4], "d", r[6], "T/rot", r[7], f"T {r[8]:.2e}", "fact", r[9],
              f"phys {r[10]:,}", f"shot {r[11]:.3g}s", f"campaign T {r[12]:.2e} {r[13]/3600:.3g} h")
    for r in noise:
        print(r[0][:12], f"err {r[1]:g} F(A) {r[2]:.3f} useful m {r[3]:.1f} contrast@139 {r[4]:.2e} @195 {r[5]:.2e} @1478 {r[6]:.2e}")


if __name__ == "__main__":
    main()
