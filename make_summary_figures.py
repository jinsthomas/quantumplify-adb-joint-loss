"""Two summary charts for the README: the crossover on the rarest target and loader scaling.

Reads results/error_curves.csv, results/classical_exact.csv and results/scaling.csv.
Writes results/figures/crossover_rare_target.png and results/figures/loader_scaling.png.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).parent / "results"
FIG = RES / "figures"
TARGET = "term_both_bonds_exhausted_probability"
GREY, ORANGE, GREEN, RED, BLUE = "#7f7f7f", "#ff7f0e", "#2ca02c", "#d62728", "#1f77b4"


def crossover_chart():
    row = next(r for r in csv.DictReader(open(RES / "classical_exact.csv"))
               if r["scenario"] == "adb-base" and r["target"] == TARGET)
    variance = {"Monte Carlo": float(row["mc_paths_10pct"]) * 0.01,
                "Importance sampling (brief)": float(row["brief_is_paths_10pct"]) * 0.01,
                "Importance sampling (tuned)": float(row["tuned_is_paths_10pct"]) * 0.01}
    q, e = [], []
    for r in csv.DictReader(open(RES / "error_curves.csv")):
        if r["scenario"] == "adb-base" and r["target"] == TARGET and r["method"].startswith("QAE"):
            q.append(float(r["queries"]))
            e.append(float(r["relative_rmse"]))
    q, e = np.array(q), np.array(e)
    n = np.logspace(2, 7, 200)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for (label, v), color in zip(variance.items(), (GREY, ORANGE, GREEN)):
        ax.loglog(n, np.sqrt(v / n), color=color, lw=2, label=f"{label} (exact)")
    ax.loglog(q, e, color=RED, marker="*", ms=8, lw=1.5, label="QAE (simulated)")
    beta, log_c = np.polyfit(-np.log(e[-6:]), np.log(q[-6:]), 1)
    q_fit = np.logspace(np.log10(q[-1]), 7, 50)
    ax.loglog(q_fit, np.exp((log_c - np.log(q_fit)) / beta), color=RED, lw=1.5, ls="--", label="QAE (fitted trend)")
    cross = float(row["qae_crossover_relative_error"])
    ax.axhline(cross, color="k", lw=0.7, ls=":")
    ax.text(130, cross * 1.15, f"QAE overtakes tuned sampling below {cross:.1%} error", fontsize=9)
    ax.set_ylim(1e-3, 30)
    ax.set_xlabel("Queries (simulated term paths, or applications of the circuit A)")
    ax.set_ylabel("Relative error")
    ax.set_title("Both bonds exhausted (about 3 in 100,000): error against queries", fontsize=11)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "crossover_rare_target.png", dpi=170)
    plt.close(fig)


def scaling_chart():
    rows = [r for r in csv.DictReader(open(RES / "scaling.csv")) if r["event"] == "all_bonds_exhausted"]
    n = np.array([int(r["countries"]) for r in rows])
    gates = np.array([int(r["A_two_qubit_gates"]) for r in rows])
    amplitudes = np.array([float(r["amplitudes_for_exact_loading"]) for r in rows])
    verified = np.array([r["statevector_verified"] == "yes" for r in rows])
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.semilogy(n, amplitudes, color=GREY, marker="o", lw=2, label="Exact loading: amplitudes to load (16^n)")
    ax.semilogy(n, gates, color=BLUE, marker="s", lw=2, label="Shared-shock loader: two-qubit gates in A")
    ax.semilogy(n[verified], gates[verified], color=BLUE, marker="s", ms=12, mfc="none", lw=0,
                label="Verified on a statevector")
    for x, y in zip(n, gates):
        ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8)
    ax.set_xlabel("Number of countries (synthetic portfolio, one regional shock)")
    ax.set_ylabel("Size of the state preparation")
    ax.set_title("State preparation cost against portfolio size", fontsize=11)
    ax.set_xticks(n)
    ax.set_ylim(25, 1e13)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "loader_scaling.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    crossover_chart()
    scaling_chart()
    print("figures written to", FIG)
