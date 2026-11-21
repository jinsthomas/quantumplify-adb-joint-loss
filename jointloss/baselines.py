"""The four classical sampling baselines named in the challenge brief.

All four estimate the five focus targets from simulated three-year paths.
One "query" is one simulated term path (three annual draws).

* Monte Carlo        inverse-CDF sampling, three uniforms per path.
* Scrambled Sobol    3-D Sobol points, eight independent scrambles averaged.
* Stratified         strata = year of first payout, weights p0^(k-1) (1 - p0).
* Importance         annual proposal = 1/2 true + 1/2 loss-tilted, reweighted by
                     likelihood ratios.

Where the brief leaves a detail open, the choice made here is stated in the
function docstring so the comparison can be reproduced and challenged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc, t as student_t

from .model import (N_STATES, N_YEARS, SHAPE, TARGETS, Scenario,
                    annual_distribution, loss_grids, payoff_grids)

N_SCRAMBLES = 8


@dataclass
class Sampler:
    """Pre-computed tables for fast path simulation of one scenario.

    Annual states are sorted by annual portfolio loss (no-payout state first), so
    that the inverse CDF is monotone in loss. This matters for Sobol points.
    """
    scenario: Scenario
    payoff_table: dict | None = None      # optional {name: grid}; default = the five focus targets

    def __post_init__(self) -> None:
        sc = self.scenario
        table = self.payoff_table or {name: payoff_grids(sc)[name] for name in TARGETS}
        self.names = list(table)
        p = annual_distribution(sc).ravel()
        l1, l2 = loss_grids(sc)
        annual_loss = (l1 + l2).ravel()
        self.order = np.argsort(annual_loss, kind="stable")       # sorted position -> state index
        self.p = p[self.order]
        self.cdf = np.cumsum(self.p)
        self.cdf[-1] = 1.0
        self.coords = np.array(np.unravel_index(self.order, SHAPE)).T   # (100, 4)
        self.caps = np.array(SHAPE) - 1
        self.payoffs = np.stack([np.asarray(table[name], float).ravel() for name in self.names])
        self.p0 = self.p[0]                                       # annual no-payout probability
        assert annual_loss[self.order[0]] == 0.0
        # Conditional annual distribution given a payout that year.
        cond = self.p.copy()
        cond[0] = 0.0
        self.cdf_payout = np.cumsum(cond / cond.sum())
        self.cdf_payout[-1] = 1.0
        # Loss-tilted annual distribution: proportional to p(x) * annual loss(x).
        tilted = self.p * annual_loss[self.order]
        self.q = 0.5 * self.p + 0.5 * tilted / tilted.sum()
        self.cdf_q = np.cumsum(self.q)
        self.cdf_q[-1] = 1.0
        self.ratio = self.p / self.q                              # likelihood ratio per annual draw

    def draw(self, u: np.ndarray, cdf: np.ndarray) -> np.ndarray:
        """Inverse-CDF draw: uniforms -> sorted annual state positions."""
        return np.minimum(np.searchsorted(cdf, u, side="right"), N_STATES - 1)

    def payoff(self, positions: np.ndarray) -> np.ndarray:
        """Payoff of each target for paths given as (n, years) annual positions.

        Returns an array of shape (5, n).
        """
        term = np.minimum(self.coords[positions].sum(axis=1), self.caps)   # (n, 4)
        flat = np.ravel_multi_index(term.T, SHAPE)
        return self.payoffs[:, flat]


def monte_carlo(s: Sampler, n: int, rng: np.random.Generator) -> np.ndarray:
    """Plain Monte Carlo with n paths. Returns the five estimates."""
    positions = s.draw(rng.random((n, N_YEARS)), s.cdf)
    return s.payoff(positions).mean(axis=1)


def sobol(s: Sampler, n: int, rng: np.random.Generator, return_interval: bool = False):
    """Scrambled Sobol in three dimensions (one per year).

    The budget n is split over eight independently scrambled sequences of n / 8
    points. The estimate is the mean of the eight sequence means. With
    return_interval=True the 95% t-interval half-width on 7 degrees of freedom is
    also returned.
    """
    per = max(n // N_SCRAMBLES, 1)
    means = np.empty((N_SCRAMBLES, len(s.names)))
    for i in range(N_SCRAMBLES):
        engine = qmc.Sobol(d=N_YEARS, scramble=True, rng=rng)
        u = engine.random(per)
        means[i] = s.payoff(s.draw(u, s.cdf)).mean(axis=1)
    est = means.mean(axis=0)
    if not return_interval:
        return est
    half = student_t.ppf(0.975, N_SCRAMBLES - 1) * means.std(axis=0, ddof=1) / np.sqrt(N_SCRAMBLES)
    return est, half


def stratified(s: Sampler, n: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified sampling by year of first payout.

    Stratum k (k = 1..3) has weight p0^(k-1) (1 - p0): no payout before year k, a
    payout in year k (drawn from the conditional annual distribution), and
    unconditional draws afterwards. The no-payout stratum (weight p0^3) has zero
    payoff for all five targets and needs no samples. The budget is allocated
    proportionally to the stratum weights (a choice the brief leaves open).
    Not valid for payoffs that are non-zero when nothing is paid.
    """
    weights = np.array([s.p0 ** (k - 1) * (1.0 - s.p0) for k in range(1, N_YEARS + 1)])
    alloc = np.maximum(np.round(n * weights / weights.sum()).astype(int), 1)
    est = np.zeros(len(s.names))
    for k, (w, nk) in enumerate(zip(weights, alloc)):
        positions = np.zeros((nk, N_YEARS), dtype=int)
        positions[:, k] = s.draw(rng.random(nk), s.cdf_payout)
        if k + 1 < N_YEARS:
            positions[:, k + 1:] = s.draw(rng.random((nk, N_YEARS - k - 1)), s.cdf)
        est += w * s.payoff(positions).mean(axis=1)
    return est


def importance(s: Sampler, n: int, rng: np.random.Generator) -> np.ndarray:
    """Importance sampling with a half-true, half-loss-tilted annual proposal.

    Each annual draw comes from q = 1/2 p + 1/2 p_tilted, where
    p_tilted(x) is proportional to p(x) times the annual portfolio loss of x.
    Paths are reweighted by the product of annual likelihood ratios p / q.
    (The brief names the mixture; the form of the tilt is our reading of it.)
    """
    positions = s.draw(rng.random((n, N_YEARS)), s.cdf_q)
    weight = s.ratio[positions].prod(axis=1)
    return (s.payoff(positions) * weight).mean(axis=1)


BASELINES = {
    "Monte Carlo": monte_carlo,
    "Scrambled Sobol": sobol,
    "Stratified": stratified,
    "Importance sampling": importance,
}
