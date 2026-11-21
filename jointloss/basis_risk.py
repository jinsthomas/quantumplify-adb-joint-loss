"""Basis-risk targets: does the bond payout match humanitarian need?

A parametric bond pays on a measured trigger; household need follows assessed
damage. The two can disagree. This module adds two illustrative targets of the
same form E_p[f] as the five focus targets, so the same estimators apply:

* unpaid need   : at least one country has high need while its bond pays nothing;
* paid, low need: at least one bond pays while that country's need is low.

ILLUSTRATIVE ASSUMPTION, NOT CALIBRATED. Whether need is "high" in a country over
the term is a random event whose probability depends only on that country's
payout fraction s = L_i / principal:

    P(high need | s = 0) = NEED_GIVEN_NO_PAYOUT          (damage below the trigger)
    P(high need | s > 0) = NEED_FLOOR + (1 - NEED_FLOOR) * s

and the two countries are independent given the state. Real values would come
from damage-assessment and affected-population data (Phase 2).
"""
from __future__ import annotations

import numpy as np

from .model import Scenario, loss_grids

NEED_GIVEN_NO_PAYOUT = 0.03
NEED_FLOOR = 0.60

BASIS_TARGETS = ("term_unpaid_need_probability", "term_paid_low_need_probability")


def need_probability(sc: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """P(high need) for each country on the state grid."""
    out = []
    for loss in loss_grids(sc):
        s = loss / sc.principal
        out.append(np.where(s > 0, NEED_FLOOR + (1.0 - NEED_FLOOR) * s, NEED_GIVEN_NO_PAYOUT))
    return out[0], out[1]


def basis_payoff_grids(sc: Scenario) -> dict[str, np.ndarray]:
    """Conditional probability of each basis-risk event given the state."""
    l1, l2 = loss_grids(sc)
    h1, h2 = need_probability(sc)
    miss1, miss2 = (l1 == 0) * h1, (l2 == 0) * h2              # need high, nothing paid
    over1, over2 = (l1 > 0) * (1 - h1), (l2 > 0) * (1 - h2)    # paid, need low
    return {
        BASIS_TARGETS[0]: 1.0 - (1.0 - miss1) * (1.0 - miss2),
        BASIS_TARGETS[1]: 1.0 - (1.0 - over1) * (1.0 - over2),
    }
