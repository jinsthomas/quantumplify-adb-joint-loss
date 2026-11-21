"""Draw the result figures from results/*.csv into results/figures/.

Usage: python make_figures.py [results_dir]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TARGET_LABELS = {
    "term_expected_principal_loss_usd": "Expected principal loss",
    "term_joint_attachment_probability": "Both bonds pay",
    "term_portfolio_loss_exceedance_probability": "Portfolio loss above USD 80m",
    "term_portfolio_stop_loss_usd": "Expected loss above USD 80m",
    "term_both_bonds_exhausted_probability": "Both bonds exhausted",
}
STYLE = {
    "Monte Carlo": ("#7f7f7f", "o", "-"),
    "Scrambled Sobol": ("#1f77b4", "s", "-"),
    "Stratified": ("#2ca02c", "^", "-"),
    "Importance sampling": ("#ff7f0e", "D", "-"),
    "QAE (maximum likelihood)": ("#d62728", "*", "-"),
}


def load_curves():
    data = defaultdict(lambda: ([], []))
    with open(RES / "error_curves.csv") as fh:
        for r in csv.DictReader(fh):
            q, e = data[(r["scenario"], r["target"], r["method"])]
            q.append(float(r["queries"]))
            e.append(float(r["relative_rmse"]))
    return data


def error_figure(data, scenario, fname):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharex=True)
    for ax, (target, label) in zip(axes, TARGET_LABELS.items()):
        for method, (color, marker, ls) in STYLE.items():
            q, e = data[(scenario, target, method)]
            ax.loglog(q, e, color=color, marker=marker, ls=ls, ms=5, lw=1.4, label=method)
        ax.axhline(0.10, color="k", lw=0.6, ls=":")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Queries (term paths or applications of A)")
        ax.grid(True, which="major", alpha=0.3)
    axes[0].set_ylabel("Relative RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.suptitle(f"Estimation error against query budget, scenario {scenario}", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)


def rare_tail_figure(data):
    target = "term_both_bonds_exhausted_probability"
    scenarios = sorted({k[0] for k in data})
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4 * len(scenarios), 4.2), sharey=True)
    for ax, scenario in zip(axes, scenarios):
        for method, (color, marker, ls) in STYLE.items():
            q, e = data[(scenario, target, method)]
            ax.loglog(q, e, color=color, marker=marker, ls=ls, ms=5, lw=1.4, label=method)
        ax.axhline(0.10, color="k", lw=0.6, ls=":")
        ax.set_title(scenario, fontsize=11)
        ax.set_xlabel("Queries")
        ax.grid(True, which="major", alpha=0.3)
    axes[0].set_ylabel("Relative RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.suptitle("Both bonds exhausted (the rarest target), all five scenarios", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(FIG / "rare_tail_all_scenarios.png", dpi=160)
    plt.close(fig)


def grid_figure(data, scenario, fname):
    """Five targets on a 2 x 3 grid (portrait page), legend in the sixth cell."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    cells = axes.ravel()
    for ax, (target, label) in zip(cells, TARGET_LABELS.items()):
        for method, (color, marker, ls) in STYLE.items():
            q, e = data[(scenario, target, method)]
            ax.loglog(q, e, color=color, marker=marker, ls=ls, ms=4, lw=1.3, label=method)
        ax.axhline(0.10, color="k", lw=0.6, ls=":")
        ax.axhline(0.01, color="k", lw=0.6, ls=":")
        ax.set_title(label, fontsize=10.5)
        ax.set_xlabel("Queries", fontsize=9)
        ax.set_ylabel("Relative RMSE", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="major", alpha=0.3)
    handles, labels = cells[0].get_legend_handles_labels()
    cells[5].axis("off")
    cells[5].legend(handles, labels, loc="center", frameon=False, fontsize=10,
                    title="Dotted lines: 10% and 1% error")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    curves = load_curves()
    for sc in sorted({k[0] for k in curves}):
        error_figure(curves, sc, f"error_vs_queries_{sc}.png")
    rare_tail_figure(curves)
    grid_figure(curves, "adb-base", "error_grid_adb-base.png")
    print("figures written to", FIG)
