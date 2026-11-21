"""Check against the organisers' evaluation conventions (evaluation-v1).

The public documentation of the organisers' benchmark package sets a default
budget of 8,192 samples and pass tolerances on the RMSE across repeats:
  expected principal loss        1%   (floor 0.0001 x combined principal)
  joint attachment, exceedance   5%   (floor 0.0001 probability)
  stop-loss                      5%   (floor 0.00001 x combined principal)
  both bonds exhausted          10%   (floor 0.000001 probability)
This script reports which methods pass at that budget and writes result files
in the organisers' JSON layout. It does not use the organisers' package, so the
files have not been validated with `qjlb validate`.

Writes results/evaluation_v1_check.csv and results/submission/*.json.
Usage: python run_evaluation.py [--runs 200]
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from jointloss import qae
from jointloss.baselines import BASELINES, Sampler
from jointloss.model import SCENARIOS, TARGETS, annual_distribution, exact_targets, payoff_grids
from jointloss.tuned_is import brief_proposal, optimise_proposal, path_payoffs, sample_estimate

RES = Path(__file__).parent / "results"
BUDGET = 8192
SHOTS = 32
# problem_id of adb-base as printed in the organisers' sample submission; the other
# scenarios' ids come from their problem.json files, which we do not have.
PROBLEM_IDS = {"adb-base": "1e05d3e5593e9f38955e8ff4271a15d65e538ed6555955aff2f84e2d3f8e5fda"}


def tolerances(sc):
    p = 2 * sc.principal
    rel = [0.01, 0.05, 0.05, 0.05, 0.10]
    floor = [1e-4 * p, 1e-4, 1e-4, 1e-5 * p, 1e-6]
    return np.array(rel), np.array(floor)


def largest_schedule(budget):
    best = qae.schedule(1)
    for m in range(1, 4097):
        powers = qae.schedule(m)
        if qae.n_queries(powers, SHOTS) <= budget:
            best = powers
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    runs = ap.parse_args().runs
    rng = np.random.default_rng(20261122)
    powers_budget = largest_schedule(BUDGET)
    powers_full = qae.schedule(2048)
    rows = []
    (RES / "submission").mkdir(exist_ok=True)
    for name, sc in SCENARIOS.items():
        exact = np.array([exact_targets(sc)[t] for t in TARGETS])
        rel, floor = tolerances(sc)
        limit = np.maximum(rel * np.abs(exact), floor)
        sampler = Sampler(sc)
        rmse = {}
        for method, fn in BASELINES.items():
            est = np.array([fn(sampler, BUDGET, rng) for _ in range(runs)])
            rmse[f"{method} (as in the brief)"] = np.sqrt(((est - exact) ** 2).mean(axis=0))
        # Tuned importance sampling: one mixture proposal serves all five targets.
        p = annual_distribution(sc).ravel()
        f_paths = [path_payoffs(sc, payoff_grids(sc)[t]) for t in TARGETS]
        q_brief = brief_proposal(sc)
        q_mix = (p + sum(optimise_proposal(p, f, q_brief) for f in f_paths)) / (len(TARGETS) + 1)
        idx = rng.choice(len(p), size=(runs, BUDGET, 3), p=q_mix)
        ratio = (p / q_mix)[idx].prod(axis=2)
        est = np.stack([(f[idx[..., 0], idx[..., 1], idx[..., 2]] * ratio).mean(axis=1) for f in f_paths], axis=1)
        rmse["Importance sampling, tuned mixture"] = np.sqrt(((est - exact) ** 2).mean(axis=0))
        for label, powers in ((f"QAE, {qae.n_queries(powers_budget, SHOTS):,} queries per target", powers_budget),
                              (f"QAE, {qae.n_queries(powers_full, SHOTS):,} queries per target", powers_full)):
            col = []
            for target, ex in zip(TARGETS, exact):
                a, f_max = qae.amplitude(sc, target)
                e = qae.simulate_mlqae(a, powers, SHOTS, rng, runs=runs) * f_max
                col.append(np.sqrt(((e - ex) ** 2).mean()))
            rmse[label] = np.array(col)
        for method, values in rmse.items():
            passed = values <= limit
            rows.append([name, method] + [f"{v / abs(ex):.4f}" for v, ex in zip(values, exact)]
                        + ["pass" if ok else "fail" for ok in passed] + ["PASS" if passed.all() else "FAIL"])
        # Result files in the organisers' layout: one QAE run and the exact solver.
        t0 = time.perf_counter()
        prep = time.perf_counter()
        amplitudes = [qae.amplitude(sc, t) for t in TARGETS]
        prep = time.perf_counter() - prep
        estimates = {t: float(qae.simulate_mlqae(a, powers_full, SHOTS, rng)[0] * f_max)
                     for t, (a, f_max) in zip(TARGETS, amplitudes)}
        elapsed = time.perf_counter() - t0
        queries = qae.n_queries(powers_full, SHOTS) * len(TARGETS)
        submission = {
            "problem_id": PROBLEM_IDS.get(name), "scenario": name,
            "method": "maximum_likelihood_qae_exact_loading", "execution": "quantum_simulator",
            "estimates": estimates,
            "resources": {"elapsed_seconds": elapsed, "preparation_seconds": prep,
                          "shots": SHOTS * len(powers_full) * len(TARGETS), "qubits": qae.N_QUBITS,
                          "model_evaluations": queries},
            "notes": "Ideal statevector-verified simulation; measurement shots drawn from the verified "
                     "outcome law. Elapsed time is simulator time on a laptop CPU, not quantum run time.",
        }
        (RES / "submission" / f"{name}.json").write_text(json.dumps(submission, indent=2))
        t0 = time.perf_counter()
        exact_now = exact_targets(sc)
        (RES / "submission" / f"{name}.exact.json").write_text(json.dumps({
            "problem_id": PROBLEM_IDS.get(name), "scenario": name, "method": "exact_enumeration",
            "execution": "classical_cpu", "estimates": exact_now,
            "resources": {"elapsed_seconds": time.perf_counter() - t0, "preparation_seconds": 0.0,
                          "shots": 0, "qubits": 0, "model_evaluations": 0}}, indent=2))
        print(name, "done")
    with open(RES / "evaluation_v1_check.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "method"] + [f"rel_rmse_{t}" for t in TARGETS] + [f"pass_{t}" for t in TARGETS]
                   + ["all_targets"])
        w.writerows(rows)
    print("wrote", RES / "evaluation_v1_check.csv")


if __name__ == "__main__":
    main()
