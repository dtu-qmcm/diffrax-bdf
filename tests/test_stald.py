"""Tests for stability limit detection.

The JAX port is checked against a line-by-line NumPy port of CVODE's `cvSLdet`, on
random histories and on histories recorded from a stalling solve. The solver tests
use a linear system with a damped oscillation 87 degrees from the negative real
axis, as in enzax's ZeLa experiment; without detection BDF stalls on it.
"""

from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffrax_bdf import BDF, BDFController
from diffrax_bdf._stald import stability_limit_flag

from stald_reference import sldet

HISTORIES = np.load(Path(__file__).parent / "data" / "stald_histories.npz")
N_RANDOM = 2000
STEADY_STATE = jnp.array([1e-2, 1e-4, 2e-6, 3e-4, 7e-3])
DAMPED_OSCILLATION = (-3.79e-3, 7.38e-2)
REAL_EIGENVALUES = (-88.6, -0.75, -5.2e-4)


def reference_flag(history, order):
    padded = np.zeros((6, 4))
    padded[1:, 1:] = history
    return sldet(padded, int(order))


def random_histories(n, seed=0):
    rng = np.random.default_rng(seed)
    histories, orders = [], []
    for _ in range(n):
        kind = rng.integers(4)
        if kind == 0:
            history = np.exp(3 * rng.normal(size=(5, 3)))
        else:
            ratio = rng.uniform(0.97, 1.0) if kind == 3 else rng.uniform(0.5, 1.2)
            noise = (1e-6, 1e-3, 1e-5)[kind - 1]
            scale = np.exp(2 * rng.normal(size=3))
            history = (
                scale[None, :]
                * ratio ** (-np.arange(5))[:, None]
                * (1 + noise * rng.normal(size=(5, 3)))
            )
            if rng.uniform() < 0.5:
                wobble = 1 + 0.3 * np.cos(rng.uniform(0, 3) * np.arange(5))
                history = history * wobble[:, None]
        histories.append(np.abs(history))
        orders.append(rng.integers(3, 6))
    return np.array(histories), np.array(orders)


def matrix(oscillates):
    blocks = np.zeros((5, 5))
    if oscillates:
        a, b = DAMPED_OSCILLATION
        blocks[:2, :2] = [[a, b], [-b, a]]
    else:
        blocks[0, 0], blocks[1, 1] = DAMPED_OSCILLATION[0], -DAMPED_OSCILLATION[1]
    blocks[2, 2], blocks[3, 3], blocks[4, 4] = REAL_EIGENVALUES
    rotation, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(5, 5)))
    return jnp.asarray(rotation @ blocks @ rotation.T)


def solve(solver, oscillates=True, t1=jnp.inf, steady_state=STEADY_STATE, **kwargs):
    a = matrix(oscillates)
    return diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: a @ (y - args)),
        solver,
        t0=0.0,
        t1=t1,
        dt0=1e-6,
        y0=1.3 * steady_state,
        args=steady_state,
        stepsize_controller=BDFController(rtol=1e-9, atol=1e-9, dtmax=1e6),
        max_steps=10_000,
        throw=False,
        **kwargs,
    )


def to_steady_state(solver, oscillates=True):
    return solve(
        solver,
        oscillates,
        event=diffrax.Event(diffrax.steady_state_event(rtol=1e-12, atol=1e-12)),
        saveat=diffrax.SaveAt(t1=True, solver_state=True),
    )


@pytest.mark.parametrize("source", ["recorded", "random"])
def test_the_flag_matches_cvode(source):
    if source == "recorded":
        histories, orders = HISTORIES["histories"], HISTORIES["orders"]
    else:
        histories, orders = random_histories(N_RANDOM)
    flags, roots = jax.jit(jax.vmap(stability_limit_flag))(
        jnp.asarray(histories), jnp.asarray(orders)
    )
    expected = [reference_flag(h, q) for h, q in zip(histories, orders)]
    assert np.asarray(flags).tolist() == [flag for flag, _ in expected]
    for root, (flag, expected_root) in zip(np.asarray(roots), expected):
        if flag > 0:
            assert root == pytest.approx(expected_root, rel=1e-9)


def test_the_recorded_histories_include_stability_limits():
    flags = {reference_flag(h, q)[0] for h, q in zip(*HISTORIES.values())}
    assert flags & {4, 5, 6}


def test_detection_lets_bdf_reach_steady_state_past_an_oscillatory_mode():
    detected = to_steady_state(BDF())
    undetected = to_steady_state(BDF(stability_limit_detection=False))
    assert detected.result == diffrax.RESULTS.event_occurred
    assert undetected.result == diffrax.RESULTS.max_steps_reached
    assert int(detected.solver_state.stable_order) < 3
    np.testing.assert_allclose(detected.ys[0], STEADY_STATE, rtol=1e-3)


def test_detection_leaves_real_eigenvalues_alone():
    detected = to_steady_state(BDF(), oscillates=False)
    undetected = to_steady_state(
        BDF(stability_limit_detection=False), oscillates=False
    )
    assert detected.result == diffrax.RESULTS.event_occurred
    assert int(detected.solver_state.stable_order) == 5
    assert int(detected.stats["num_steps"]) == int(undetected.stats["num_steps"])


def test_backpropagating_through_detection_gives_finite_gradients():
    def final_state(steady_state):
        solution = solve(
            BDF(),
            t1=600.0,
            steady_state=steady_state,
            adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=64),
            saveat=diffrax.SaveAt(t1=True, solver_state=True),
        )
        return solution.ys[0].sum(), solution.solver_state.stable_order

    (_, stable_order), gradient = jax.value_and_grad(final_state, has_aux=True)(
        STEADY_STATE
    )
    assert int(stable_order) < 5
    assert jnp.all(jnp.isfinite(gradient))
