"""Each baseline must be unbiased: its long-run average equals the exact value."""
import numpy as np
import pytest

from jointloss.baselines import BASELINES, Sampler
from jointloss.model import SCENARIOS, TARGETS, exact_targets


@pytest.mark.parametrize("method", list(BASELINES))
@pytest.mark.parametrize("name", ["adb-base", "adb-rain-cover"])
def test_unbiased(method, name):
    sc = SCENARIOS[name]
    exact = np.array([exact_targets(sc)[t] for t in TARGETS])
    s = Sampler(sc)
    rng = np.random.default_rng(1)
    runs = np.array([BASELINES[method](s, 8192, rng) for _ in range(300)])
    mean, sem = runs.mean(axis=0), runs.std(axis=0, ddof=1) / np.sqrt(len(runs))
    # Within 5 standard errors on every target (the rarest one included).
    assert np.all(np.abs(mean - exact) <= 5 * sem + 1e-12), (mean, exact, sem)
