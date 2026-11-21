"""Exact joint-loss model for the ADB Disaster Relief Bonds benchmark.

Rebuilds the model described in the challenge brief ("Quantum Approaches to
Joint-Loss Modelling for Natural Catastrophe Risk", Section 3b):

* two bonds (Kyrgyz Republic, Tajikistan), each with an earthquake sub-limit paid
  in quarter steps and an extreme-precipitation sub-limit paid at most once;
* earthquake: compound Poisson per country with a shared-event rate, giving a
  25-state continuous-time Markov chain on cumulative quarter steps (a, b);
* precipitation: one trigger per country with a shared rate;
* annual state (a, b, r1, r2): 5 x 5 x 2 x 2 = 100 states;
* term = three independent years, summed with caps (4, 4, 1, 1).

Everything here is exact enumeration (no sampling). It is the ground truth for
the classical baselines and for the quantum estimator.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm

# --- Calibration from the brief --------------------------------------------
# Table 1: annual earthquake exceedance probabilities per country, for payouts of
# at least 25%, 50%, 75% and 100% of the earthquake sub-limit.
EXCEEDANCE_KYRGYZ = (0.03883, 0.02331, 0.01790, 0.01184)
EXCEEDANCE_TAJIK = (0.03925, 0.01743, 0.01332, 0.01043)


def implied_rates(exceedance: tuple[float, float, float, float]) -> tuple[float, ...]:
    """Poisson rates lambda_k of events paying exactly k quarter steps (k = 1..4).

    Inverts the compound Poisson model: with total rate Lambda,
    P(a = 0) = exp(-Lambda), P(a = 1) = exp(-Lambda) * l1,
    P(a = 2) = exp(-Lambda) * (l2 + l1^2 / 2),
    P(a = 3) = exp(-Lambda) * (l3 + l1 * l2 + l1^3 / 6).
    Reproduces Table 2 of the brief to its printed precision.
    """
    p1, p2, p3, p4 = exceedance
    total = -np.log(1.0 - p1)
    e = np.exp(total)
    l1 = (p1 - p2) * e
    l2 = (p2 - p3) * e - l1 ** 2 / 2
    l3 = (p3 - p4) * e - l1 * l2 - l1 ** 3 / 6
    return (float(l1), float(l2), float(l3), float(total - l1 - l2 - l3))


LAMBDA_KYRGYZ = implied_rates(EXCEEDANCE_KYRGYZ)   # Table 2: 0.016147, 0.005498, 0.006215, 0.011743
LAMBDA_TAJIK = implied_rates(EXCEEDANCE_TAJIK)     # Table 2: 0.022711, 0.004020, 0.002915, 0.010395
# Cross-trigger conditional probabilities: Tajikistan given Kyrgyz, and reverse.
CROSS_TJ_GIVEN_KG = 0.075
CROSS_KG_GIVEN_TJ = 0.074
# Precipitation: annual trigger probability per country.
RAIN_Q = 0.02
# Shared precipitation rate as a fraction of the smaller country rate.
RAIN_SHARED_FRACTION = 0.25

N_STEPS = 4          # earthquake quarter steps (0..4)
N_YEARS = 3          # bond term in years
SHAPE = (N_STEPS + 1, N_STEPS + 1, 2, 2)   # (a, b, r1, r2)
N_STATES = int(np.prod(SHAPE))             # 100


@dataclass(frozen=True)
class Scenario:
    """One benchmark scenario. Defaults are the base case ("adb-base")."""
    name: str = "adb-base"
    shared_multiplier: float = 1.0        # m in the brief (0 = independent)
    rain_rate_multiplier: float = 1.0     # 2 = rain-stress
    rain_shared_fraction: float = RAIN_SHARED_FRACTION
    eq_limit: float = 70e6                # earthquake sub-limit per bond (USD)
    rain_limit: float = 10e6              # precipitation sub-limit per bond (USD)
    threshold: float = 80e6               # T for the portfolio targets (USD)

    @property
    def principal(self) -> float:
        return self.eq_limit + self.rain_limit


BASE = Scenario()
SCENARIOS = {
    "adb-base": BASE,
    "adb-independent": replace(BASE, name="adb-independent",
                               shared_multiplier=0.0, rain_shared_fraction=0.0),
    "adb-common-shock": replace(BASE, name="adb-common-shock", shared_multiplier=2.0),
    "adb-rain-stress": replace(BASE, name="adb-rain-stress", rain_rate_multiplier=2.0),
    "adb-rain-cover": replace(BASE, name="adb-rain-cover", eq_limit=60e6, rain_limit=20e6),
}


# --- Earthquake -------------------------------------------------------------
def shared_eq_rate(sc: Scenario) -> float:
    """Shared-event rate c = m * 1/2 * (0.075 * Lambda_1 + 0.074 * Lambda_2)."""
    big1, big2 = sum(LAMBDA_KYRGYZ), sum(LAMBDA_TAJIK)
    return sc.shared_multiplier * 0.5 * (CROSS_TJ_GIVEN_KG * big1 + CROSS_KG_GIVEN_TJ * big2)


def eq_generator(sc: Scenario) -> np.ndarray:
    """25 x 25 generator of the chain on capped cumulative steps (a, b).

    Country-only events arrive at rate Lambda_i - c; shared events at rate c hit
    both countries, with sizes drawn independently from w_i,k = lambda_i,k / Lambda_i.
    """
    lam1, lam2 = np.array(LAMBDA_KYRGYZ), np.array(LAMBDA_TAJIK)
    big1, big2 = lam1.sum(), lam2.sum()
    w1, w2 = lam1 / big1, lam2 / big2
    c = shared_eq_rate(sc)
    n = N_STEPS + 1
    g = np.zeros((n * n, n * n))
    idx = lambda a, b: a * n + b
    for a in range(n):
        for b in range(n):
            i = idx(a, b)
            for k in range(1, N_STEPS + 1):
                g[i, idx(min(a + k, N_STEPS), b)] += (big1 - c) * w1[k - 1]
                g[i, idx(a, min(b + k, N_STEPS))] += (big2 - c) * w2[k - 1]
                for l in range(1, N_STEPS + 1):
                    g[i, idx(min(a + k, N_STEPS), min(b + l, N_STEPS))] += c * w1[k - 1] * w2[l - 1]
    # Jumps that do not change the state (already capped) sit on the diagonal;
    # set the diagonal so every row sums to zero.
    np.fill_diagonal(g, 0.0)
    np.fill_diagonal(g, -g.sum(axis=1))
    return g


def eq_annual(sc: Scenario) -> np.ndarray:
    """Annual joint distribution of earthquake steps, shape (5, 5)."""
    n = N_STEPS + 1
    return expm(eq_generator(sc))[0].reshape(n, n)


# --- Precipitation ----------------------------------------------------------
def rain_annual(sc: Scenario) -> np.ndarray:
    """Annual joint distribution of precipitation triggers, shape (2, 2)."""
    mu = -np.log(1.0 - RAIN_Q) * sc.rain_rate_multiplier
    c_r = sc.rain_shared_fraction * mu          # both country rates are equal
    p00 = np.exp(-(2.0 * mu - c_r))
    p_single = np.exp(-mu) - p00                # one country only
    p11 = 1.0 - p00 - 2.0 * p_single
    return np.array([[p00, p_single], [p_single, p11]])


# --- Annual and term distributions -----------------------------------------
def annual_distribution(sc: Scenario) -> np.ndarray:
    """Annual distribution over (a, b, r1, r2), shape (5, 5, 2, 2)."""
    return np.einsum("ab,rs->abrs", eq_annual(sc), rain_annual(sc))


def capped_convolve(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Distribution of the capped sum of two independent states."""
    out = np.zeros(SHAPE)
    caps = np.array(SHAPE) - 1
    for i in np.argwhere(p > 0):
        pi = p[tuple(i)]
        for j in np.argwhere(q > 0):
            k = np.minimum(i + j, caps)
            out[tuple(k)] += pi * q[tuple(j)]
    return out


def term_distribution(sc: Scenario, years: int = N_YEARS) -> np.ndarray:
    """Distribution of the capped term state after `years` independent years."""
    annual = annual_distribution(sc)
    term = annual
    for _ in range(years - 1):
        term = capped_convolve(term, annual)
    return term


# --- Losses and targets -----------------------------------------------------
def loss_grids(sc: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Principal loss of each bond on the state grid: L_i = EQ * a/4 + RAIN * r_i."""
    a, b, r1, r2 = np.indices(SHAPE)
    l1 = sc.eq_limit * a / N_STEPS + sc.rain_limit * r1
    l2 = sc.eq_limit * b / N_STEPS + sc.rain_limit * r2
    return l1, l2


TARGETS = (
    "term_expected_principal_loss_usd",
    "term_joint_attachment_probability",
    "term_portfolio_loss_exceedance_probability",
    "term_portfolio_stop_loss_usd",
    "term_both_bonds_exhausted_probability",
)


def payoff_grids(sc: Scenario) -> dict[str, np.ndarray]:
    """The function f of each target, evaluated on the state grid."""
    l1, l2 = loss_grids(sc)
    total = l1 + l2
    return {
        TARGETS[0]: total,
        TARGETS[1]: ((l1 > 0) & (l2 > 0)).astype(float),
        TARGETS[2]: (total > sc.threshold).astype(float),
        TARGETS[3]: np.maximum(total - sc.threshold, 0.0),
        TARGETS[4]: ((l1 >= sc.principal) & (l2 >= sc.principal)).astype(float),
    }


def exact_targets(sc: Scenario) -> dict[str, float]:
    """Exact value of the five focus targets for a scenario."""
    term = term_distribution(sc)
    return {name: float((term * f).sum()) for name, f in payoff_grids(sc).items()}


if __name__ == "__main__":
    for sc in SCENARIOS.values():
        t = exact_targets(sc)
        print(f"{sc.name:18s} E[L]={t[TARGETS[0]]/1e6:7.3f}m  both pay={t[TARGETS[1]]:.4%}  "
              f"L>T={t[TARGETS[2]]:.4%}  stop-loss={t[TARGETS[3]]:11,.0f}  "
              f"both exhausted={t[TARGETS[4]]:.5%}")
