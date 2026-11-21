"""The exact model must reproduce Table 5 of the challenge brief."""
import numpy as np
import pytest

from jointloss.model import (LAMBDA_KYRGYZ, LAMBDA_TAJIK, SCENARIOS, TARGETS,
                             annual_distribution, eq_annual, exact_targets,
                             rain_annual, shared_eq_rate, term_distribution)

# Table 5: E[L] (USD m), both pay, L > 80m, E[loss above 80m] (USD), both exhausted.
TABLE5 = {
    "adb-base":         (10.06, 0.04414, 0.01205, 282_355, 0.000032),
    "adb-independent":  (10.06, 0.02716, 0.00719, 171_266, 0.000004),
    "adb-common-shock": (10.06, 0.05053, 0.01607, 385_043, 0.000044),
    "adb-rain-stress":  (11.17, 0.07046, 0.01322, 306_286, 0.000069),
    "adb-rain-cover":   (9.97,  0.04414, 0.00700, 170_927, 0.000032),
}


@pytest.mark.parametrize("name", list(TABLE5))
def test_table5(name):
    got = exact_targets(SCENARIOS[name])
    el, both, exceed, stop, exhausted = TABLE5[name]
    assert got[TARGETS[0]] / 1e6 == pytest.approx(el, abs=0.005)
    assert got[TARGETS[1]] == pytest.approx(both, abs=5e-6)
    assert got[TARGETS[2]] == pytest.approx(exceed, abs=5e-6)
    assert got[TARGETS[3]] == pytest.approx(stop, abs=1.0)
    assert got[TARGETS[4]] == pytest.approx(exhausted, abs=5e-7)


def test_table2_rates():
    assert LAMBDA_KYRGYZ == pytest.approx((0.016147, 0.005498, 0.006215, 0.011743), abs=5e-7)
    assert LAMBDA_TAJIK == pytest.approx((0.022711, 0.004020, 0.002915, 0.010395), abs=5e-7)


def test_intermediate_values_from_brief():
    base = SCENARIOS["adb-base"]
    assert shared_eq_rate(base) == pytest.approx(0.0029667, abs=5e-8)
    assert eq_annual(base)[0, 0] == pytest.approx(0.9262, abs=5e-5)
    rain = rain_annual(base)
    assert rain[0, 0] == pytest.approx(0.96526, abs=5e-6)
    assert rain[0, 1] == pytest.approx(0.01474, abs=5e-6)
    assert rain[1, 1] == pytest.approx(0.00526, abs=5e-6)
    assert annual_distribution(base)[0, 0, 0, 0] == pytest.approx(0.8940, abs=5e-5)


@pytest.mark.parametrize("name", list(TABLE5))
def test_distributions_sum_to_one(name):
    sc = SCENARIOS[name]
    assert np.isclose(annual_distribution(sc).sum(), 1.0)
    assert np.isclose(term_distribution(sc).sum(), 1.0)
