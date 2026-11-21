"""Exact variance formulas and the tuned proposal."""
import numpy as np
import pytest

from jointloss.model import SCENARIOS, TARGETS, annual_distribution, exact_targets, payoff_grids
from jointloss.tuned_is import (brief_proposal, optimise_proposal, path_payoffs, relative_variance,
                                sample_estimate)

SC = SCENARIOS["adb-base"]
P = annual_distribution(SC).ravel()


def test_monte_carlo_variance_of_a_probability():
    a = exact_targets(SC)[TARGETS[4]]
    f = path_payoffs(SC, payoff_grids(SC)[TARGETS[4]])
    assert relative_variance(P, P, f) == pytest.approx((1 - a) / a, rel=1e-9)


def test_tuned_proposal_beats_brief_and_is_unbiased():
    f = path_payoffs(SC, payoff_grids(SC)[TARGETS[4]])
    q_brief = brief_proposal(SC)
    q = optimise_proposal(P, f, q_brief)
    v_brief, v = relative_variance(P, q_brief, f), relative_variance(P, q, f)
    assert v < v_brief / 20
    rng = np.random.default_rng(5)
    est = np.array([sample_estimate(P, q, f, 4096, rng) for _ in range(200)])
    exact = exact_targets(SC)[TARGETS[4]]
    assert abs(est.mean() - exact) < 5 * est.std(ddof=1) / np.sqrt(len(est))
    # The simulated spread matches the exact variance formula.
    assert est.std(ddof=1) / exact == pytest.approx(np.sqrt(v / 4096), rel=0.25)
