"""Run the benchmark: exact values, four classical baselines and QAE.

Writes CSV files to results/:
  exact_targets.csv     the five targets for the five scenarios (Table 5)
  error_curves.csv      relative RMSE against query budget, every method
  queries_to_target.csv queries each method needs for 10% and 1% relative RMSE
  schedule_study.csv    effect of the Grover schedule growth factor
  resources.csv         transpiled gate counts of A and Q per target

Usage: python run_experiments.py [--runs 200] [--quick]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from jointloss import qae
from jointloss.baselines import BASELINES, Sampler
from jointloss.model import SCENARIOS, TARGETS, exact_targets

OUT = Path(__file__).parent / "results"
SHOTS = 32
QAE_NAME = "QAE (maximum likelihood)"


def write(name, header, rows):
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote results/{name} ({len(rows)} rows)")


def rel_rmse(estimates, exact):
    return np.sqrt(((estimates - exact) ** 2).mean(axis=0)) / exact


def interpolate_queries(queries, errors, level):
    """Queries needed to reach a relative RMSE `level` (log-log interpolation)."""
    q, e = np.log(np.asarray(queries, float)), np.log(np.asarray(errors, float))
    target = np.log(level)
    below = np.where(e <= target)[0]
    if len(below) == 0:
        slope, intercept = np.polyfit(q[-4:], e[-4:], 1)      # extrapolate the tail
        return float(np.exp((target - intercept) / slope)), True
    i = below[0]
    if i == 0:
        return float(np.exp(q[0])), False
    frac = (target - e[i - 1]) / (e[i] - e[i - 1])
    return float(np.exp(q[i - 1] + frac * (q[i] - q[i - 1]))), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    runs = 40 if args.quick else args.runs
    budgets = [2 ** k for k in range(7, 16 if args.quick else 19)]      # 128 .. 262,144 paths
    m_maxes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    rng = np.random.default_rng(20261121)
    t0 = time.time()

    # 1. Exact reference values.
    write("exact_targets.csv", ["scenario"] + list(TARGETS),
          [[n] + [exact_targets(sc)[t] for t in TARGETS] for n, sc in SCENARIOS.items()])

    # 2. Error against query budget.
    curves, summary = [], []
    for name, sc in SCENARIOS.items():
        exact = np.array([exact_targets(sc)[t] for t in TARGETS])
        sampler = Sampler(sc)
        series = {}
        for method, fn in BASELINES.items():
            errs = []
            for n in budgets:
                est = np.array([fn(sampler, n, rng) for _ in range(runs)])
                errs.append(rel_rmse(est, exact))
            series[method] = (budgets, np.array(errs))
        qs, errs = [], []
        for m_max in m_maxes:
            powers = qae.schedule(m_max)
            row = []
            for target, ex in zip(TARGETS, exact):
                a, f_max = qae.amplitude(sc, target)
                est = qae.simulate_mlqae(a, powers, SHOTS, rng, runs=runs) * f_max
                row.append(rel_rmse(est, ex))
            qs.append(qae.n_queries(powers, SHOTS))
            errs.append(row)
        series[QAE_NAME] = (qs, np.array(errs))
        for method, (q, e) in series.items():
            for qi, ei in zip(q, e):
                for target, v in zip(TARGETS, ei):
                    curves.append([name, method, target, qi, v])
            for j, target in enumerate(TARGETS):
                for level in (0.10, 0.01):
                    need, extrapolated = interpolate_queries(q, e[:, j], level)
                    summary.append([name, method, target, level, round(need), extrapolated])
        print(f"{name}: done at {time.time() - t0:.0f}s")
    write("error_curves.csv", ["scenario", "method", "target", "queries", "relative_rmse"], curves)
    write("queries_to_target.csv",
          ["scenario", "method", "target", "relative_rmse_level", "queries", "extrapolated"], summary)

    # 3. Schedule study on the rarest target (base scenario).
    base = SCENARIOS["adb-base"]
    a, _ = qae.amplitude(base, TARGETS[4])
    rows = []
    for growth in (2.0, 1.5, 1.25):
        for m_max in (128, 512, 2048):
            powers = qae.schedule(m_max, growth)
            err = np.abs(qae.simulate_mlqae(a, powers, SHOTS, rng, runs=runs) - a) / a
            rows.append([growth, m_max, len(powers), qae.n_queries(powers, SHOTS),
                         np.sqrt((err ** 2).mean()), np.median(err), err.max()])
    write("schedule_study.csv", ["growth", "max_power", "circuits", "queries",
                                 "relative_rmse", "median_relative_error", "max_relative_error"], rows)

    # 4. Gate counts.
    rows = []
    for target in TARGETS:
        a_circ = qae.state_preparation(base, target)
        for label, circ in (("A (state preparation)", a_circ), ("Q (one Grover iterate)", qae.grover_operator(a_circ))):
            c = qae.gate_counts(circ)
            rows.append([target, label, c["qubits"], c["two_qubit_gates"], c["single_qubit_gates"],
                         c["depth"], c["two_qubit_depth"]])
    write("resources.csv", ["target", "circuit", "qubits", "two_qubit_gates", "single_qubit_gates",
                            "depth", "two_qubit_depth"], rows)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
