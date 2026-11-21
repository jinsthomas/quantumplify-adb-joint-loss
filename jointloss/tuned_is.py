"""Stronger classical baseline: importance sampling tuned to each target.

The brief's importance sampler uses one loss-tilted proposal for all targets.
An expert would tune the proposal to the target. This module does two things:

1. `relative_variance` gives the EXACT relative variance of an importance-sampling
   estimator whose three annual draws are i.i.d. from any proposal q, by summing
   over all 100^3 paths. The number of paths for relative error e is then
   variance / e^2, with no simulation noise and no extrapolation.
2. `optimise_proposal` finds the annual proposal that minimises that variance
   for one target (gradient descent on the exact second moment).

Like the quantum loader, the tuned proposal is built from the exact annual
distribution, so the comparison with QAE is on equal information.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .model import N_STATES, SHAPE, Scenario, annual_distribution, loss_grids


def path_payoffs(sc: Scenario, payoff: np.ndarray) -> np.ndarray:
    """Payoff of every three-year path, shape (100, 100, 100)."""
    coords = np.array(np.unravel_index(np.arange(N_STATES), SHAPE)).T
    caps = np.array(SHAPE) - 1
    term = np.minimum(coords[:, None, None, :] + coords[None, :, None, :] + coords[None, None, :, :], caps)
    return np.asarray(payoff, float)[term[..., 0], term[..., 1], term[..., 2], term[..., 3]]


def brief_proposal(sc: Scenario) -> np.ndarray:
    """The brief's proposal: half true, half proportional to p(x) x annual loss."""
    p = annual_distribution(sc).ravel()
    l1, l2 = loss_grids(sc)
    tilted = p * (l1 + l2).ravel()
    return 0.5 * p + 0.5 * tilted / tilted.sum()


def relative_variance(p: np.ndarray, q: np.ndarray, f_paths: np.ndarray) -> float:
    """Exact relative variance (variance / mean^2) of one importance-sampled path."""
    mu = np.einsum("i,j,k,ijk->", p, p, p, f_paths)
    w = np.divide(p ** 2, q, out=np.zeros_like(p), where=q > 0)
    second = np.einsum("i,j,k,ijk->", w, w, w, f_paths ** 2)
    return float(second / mu ** 2 - 1.0)


def optimise_proposal(p: np.ndarray, f_paths: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Annual proposal minimising the exact second moment for one target."""
    g = f_paths ** 2                                   # symmetric in the three years

    def objective(z):
        q = np.exp(z - z.max())
        q /= q.sum()
        w = p ** 2 / q
        inner = np.einsum("j,k,ijk->i", w, w, g)       # d(second)/d(w_i) / 3
        second = float(w @ inner)
        d_q = 3.0 * inner * (-(p ** 2) / q ** 2)       # d(second)/d(q_i)
        d_z = q * (d_q - (d_q * q).sum())              # through the softmax
        return np.log(second), d_z / second

    z0 = np.log(np.maximum(start, 1e-300))
    res = minimize(objective, z0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    q = np.exp(res.x - res.x.max())
    return q / q.sum()


def sample_estimate(p: np.ndarray, q: np.ndarray, f_paths: np.ndarray, n: int,
                    rng: np.random.Generator) -> float:
    """One importance-sampling estimate from n paths (for checking the exact variance)."""
    idx = rng.choice(N_STATES, size=(n, 3), p=q)
    ratio = (p / q)[idx].prod(axis=1)
    return float((f_paths[idx[:, 0], idx[:, 1], idx[:, 2]] * ratio).mean())
