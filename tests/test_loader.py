"""The shared-shock loader must reproduce the exact distributions."""
import numpy as np
import pytest

from jointloss import loader as L
from jointloss.model import SCENARIOS, TARGETS, annual_distribution, exact_targets, term_distribution


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_loader_matches_two_country_model(name):
    sc = SCENARIOS[name]
    term = L.loaded_distribution(L.shared_shock_loader(sc, 2), 2)
    assert np.abs(term - term_distribution(sc)).max() < 2e-9
    annual = L.loaded_distribution(L.shared_shock_loader(sc, 2, years=1), 2)
    assert np.abs(annual - annual_distribution(sc)).max() < 1e-10


def test_brute_force_reference_matches_model():
    sc = SCENARIOS["adb-base"]
    assert np.abs(L.exact_distribution_brute_force(sc, 2) - term_distribution(sc)).max() < 1e-14


def test_loader_matches_three_country_brute_force():
    sc = SCENARIOS["adb-base"]
    got = L.loaded_distribution(L.shared_shock_loader(sc, 3), 3)
    assert np.abs(got - L.exact_distribution_brute_force(sc, 3)).max() < 1e-9


def test_mixture_formula_matches_focus_targets():
    sc = SCENARIOS["adb-base"]
    exact = exact_targets(sc)
    assert L.exact_product_event(sc, 2, L.EVENTS["all_bonds_pay"]) == pytest.approx(exact[TARGETS[1]], rel=1e-7)
    assert L.exact_product_event(sc, 2, L.EVENTS["all_bonds_exhausted"]) == pytest.approx(exact[TARGETS[4]], rel=1e-6)


@pytest.mark.parametrize("n,event", [(2, "all_bonds_pay"), (2, "all_bonds_exhausted"), (3, "all_bonds_exhausted")])
def test_grover_law_with_structured_loader(n, event):
    sc = SCENARIOS["adb-base"]
    a_circuit, objective = L.flag_state_preparation(sc, n, event)
    theta = np.arcsin(np.sqrt(L.exact_product_event(sc, n, L.EVENTS[event])))
    for power in (0, 1, 3):
        got = L.objective_probability(a_circuit, objective, power)
        assert got == pytest.approx(np.sin((2 * power + 1) * theta) ** 2, abs=1e-10)
