"""Basis-risk add-on: exact values, circuit check and queries needed.

Writes results/basis_risk.csv. Usage: python run_basis_risk.py [--runs 200]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from jointloss import qae
from jointloss.baselines import Sampler, importance, monte_carlo, sobol
from jointloss.basis_risk import BASIS_TARGETS, basis_payoff_grids
from jointloss.model import SCENARIOS, term_distribution
from run_experiments import SHOTS, interpolate_queries, rel_rmse

METHODS = {"Monte Carlo": monte_carlo, "Scrambled Sobol": sobol, "Importance sampling": importance}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    runs = ap.parse_args().runs
    rng = np.random.default_rng(20261122)
    budgets = [2 ** k for k in range(7, 17)]
    m_maxes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    rows = []
    for name, sc in SCENARIOS.items():
        grids = basis_payoff_grids(sc)
        term = term_distribution(sc)
        exact = np.array([(term * grids[t]).sum() for t in BASIS_TARGETS])
        sampler = Sampler(sc, payoff_table=grids)
        need = {}
        for method, fn in METHODS.items():
            errs = np.array([rel_rmse(np.array([fn(sampler, n, rng) for _ in range(runs)]), exact) for n in budgets])
            need[method] = [interpolate_queries(budgets, errs[:, j], 0.01)[0] for j in range(len(BASIS_TARGETS))]
        qs, errs = [], []
        for m_max in m_maxes:
            powers = qae.schedule(m_max)
            row = []
            for t, ex in zip(BASIS_TARGETS, exact):
                a, f_max = qae.amplitude(sc, grids[t])
                row.append(rel_rmse(qae.simulate_mlqae(a, powers, SHOTS, rng, runs=runs) * f_max, ex))
            qs.append(qae.n_queries(powers, SHOTS))
            errs.append(row)
        errs = np.array(errs)
        need["QAE (maximum likelihood)"] = [interpolate_queries(qs, errs[:, j], 0.01)[0] for j in range(len(BASIS_TARGETS))]
        for j, t in enumerate(BASIS_TARGETS):
            # Circuit check on the statevector for one Grover power.
            a, _ = qae.amplitude(sc, grids[t])
            got = qae.objective_probability(qae.qae_circuit(sc, grids[t], 2))
            assert abs(got - np.sin(5 * np.arcsin(np.sqrt(a))) ** 2) < 1e-10
            for method, vals in need.items():
                rows.append([name, t, exact[j], method, round(vals[j])])
        print(name, dict(zip(BASIS_TARGETS, np.round(exact, 5))))
    out = Path(__file__).parent / "results" / "basis_risk.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "target", "exact_value", "method", "queries_for_1pct_relative_rmse"])
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
