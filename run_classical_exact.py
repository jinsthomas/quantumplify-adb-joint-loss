"""Exact classical requirements and the crossover with QAE.

For every scenario and target, computes the exact number of paths needed for 10%
and 1% relative error by plain Monte Carlo, the brief's importance sampler and a
target-tuned importance sampler, and the accuracy below which the measured QAE
curve needs fewer queries than tuned importance sampling.

Writes results/classical_exact.csv. Needs results/error_curves.csv.
"""
import csv
from pathlib import Path

import numpy as np

from jointloss.model import SCENARIOS, TARGETS, annual_distribution, payoff_grids
from jointloss.tuned_is import brief_proposal, optimise_proposal, path_payoffs, relative_variance

RES = Path(__file__).parent / "results"


def qae_curve(scenario, target):
    q, e = [], []
    with open(RES / "error_curves.csv") as fh:
        for r in csv.DictReader(fh):
            if r["scenario"] == scenario and r["target"] == target and r["method"].startswith("QAE"):
                q.append(float(r["queries"]))
                e.append(float(r["relative_rmse"]))
    return np.array(q), np.array(e)


def crossover(queries, errors, variance):
    """Relative error below which QAE needs fewer queries than variance / e^2.

    Fits queries = C * error^(-beta) to the last six QAE points (log-log).
    """
    beta, log_c = np.polyfit(-np.log(errors[-6:]), np.log(queries[-6:]), 1)
    if beta >= 2:
        return float("nan"), beta
    # variance / e^2 = C * e^(-beta)  =>  e = (variance / C)^(1 / (2 - beta))
    return float(np.exp((np.log(variance) - log_c) / (2.0 - beta))), beta


def main():
    rows = []
    for name, sc in SCENARIOS.items():
        p = annual_distribution(sc).ravel()
        q_brief = brief_proposal(sc)
        for target in TARGETS:
            f = path_payoffs(sc, payoff_grids(sc)[target])
            v_mc = relative_variance(p, p, f)
            v_brief = relative_variance(p, q_brief, f)
            v_tuned = relative_variance(p, optimise_proposal(p, f, q_brief), f)
            queries, errors = qae_curve(name, target)
            eps, beta = crossover(queries, errors, v_tuned)
            rows.append([name, target, round(v_mc / 0.01), round(v_brief / 0.01), round(v_tuned / 0.01),
                         round(v_mc / 1e-4), round(v_brief / 1e-4), round(v_tuned / 1e-4),
                         eps, beta])
        print(name, "done")
    with open(RES / "classical_exact.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "target", "mc_paths_10pct", "brief_is_paths_10pct", "tuned_is_paths_10pct",
                    "mc_paths_1pct", "brief_is_paths_1pct", "tuned_is_paths_1pct",
                    "qae_crossover_relative_error", "qae_fitted_exponent"])
        w.writerows(rows)
    print("wrote", RES / "classical_exact.csv")


if __name__ == "__main__":
    main()
