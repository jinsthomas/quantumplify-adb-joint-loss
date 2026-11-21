"""The circuits must encode the exact targets and follow the Grover rotation law."""
import numpy as np
import pytest

from jointloss import qae
from jointloss.model import SCENARIOS, TARGETS, exact_targets


@pytest.mark.parametrize("name", ["adb-base", "adb-rain-cover"])
@pytest.mark.parametrize("target", TARGETS)
def test_circuit_amplitudes(name, target):
    sc = SCENARIOS[name]
    a, f_max = qae.amplitude(sc, target)
    assert a * f_max == pytest.approx(exact_targets(sc)[target], rel=1e-12)
    theta = np.arcsin(np.sqrt(a))
    for power in (0, 1, 2, 5):
        got = qae.objective_probability(qae.qae_circuit(sc, target, power))
        assert got == pytest.approx(np.sin((2 * power + 1) * theta) ** 2, abs=1e-10)


def test_mlqae_recovers_rare_amplitude():
    a, _ = qae.amplitude(SCENARIOS["adb-base"], TARGETS[4])
    powers = qae.schedule(2048)
    est = qae.simulate_mlqae(a, powers, 32, np.random.default_rng(3), runs=40)
    assert np.sqrt(((est - a) ** 2).mean()) / a < 0.03


def test_basis_risk_targets_in_circuit():
    from jointloss.basis_risk import BASIS_TARGETS, basis_payoff_grids
    from jointloss.model import term_distribution
    sc = SCENARIOS["adb-base"]
    grids = basis_payoff_grids(sc)
    for t in BASIS_TARGETS:
        exact = float((term_distribution(sc) * grids[t]).sum())
        assert 0.0 < exact < 1.0
        a, f_max = qae.amplitude(sc, grids[t])
        assert a * f_max == pytest.approx(exact, rel=1e-12)
        got = qae.objective_probability(qae.qae_circuit(sc, grids[t], 1))
        assert got == pytest.approx(np.sin(3 * np.arcsin(np.sqrt(a))) ** 2, abs=1e-10)
