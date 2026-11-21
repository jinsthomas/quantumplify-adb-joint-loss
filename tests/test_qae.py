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
