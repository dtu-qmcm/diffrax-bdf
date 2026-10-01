import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from scipy.integrate import solve_ivp

from diffrax_bdf import BDF, BDFController, SemiExplicitDAETerm

from helpers import (
    FAST_EQUILIBRIUM_ARGS,
    FAST_EQUILIBRIUM_Y0,
    ROBERTSON_ARGS,
    ROBERTSON_Y0,
    fast_equilibrium_vf,
    observed_order,
    robertson_vf,
    seed_state,
)


TIGHT = diffrax.VeryChord(rtol=1e-13, atol=1e-13)
ROBERTSON_MASK = jnp.array([False, False, True])


def robertson_dae_vf(t, y, args):
    f = robertson_vf(t, y, args)
    return f.at[2].set(jnp.sum(y) - 1)


def linear_dae_vf(t, y, args):
    del args
    z, u = y
    return jnp.stack([u, u + z - jnp.cos(t)])


def linear_dae_exact(t):
    return jnp.stack([jnp.cos(t) + jnp.sin(t), jnp.cos(t) - jnp.sin(t)]) / 2


LINEAR_DAE_TERM = SemiExplicitDAETerm(linear_dae_vf, jnp.array([False, True]))


def solve_robertson(term, solver, t1=1e4):
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t1,
        dt0=1e-8,
        y0=ROBERTSON_Y0,
        args=ROBERTSON_ARGS,
        stepsize_controller=BDFController(rtol=1e-8, atol=1e-10),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=200_000,
    )


@pytest.mark.parametrize("suppress", [False, True])
@pytest.mark.parametrize("bdf_order", [None, 2, 5])
def test_robertson_dae_matches_scipy(bdf_order, suppress):
    t1 = 1e4
    reference = solve_ivp(
        lambda t, y: np.asarray(robertson_vf(t, jnp.asarray(y), ROBERTSON_ARGS)),
        (0.0, t1),
        np.asarray(ROBERTSON_Y0),
        method="BDF",
        rtol=1e-10,
        atol=1e-12,
    )
    term = SemiExplicitDAETerm(robertson_dae_vf, ROBERTSON_MASK)
    solver = BDF(bdf_order=bdf_order, suppress_algebraic_error=suppress)
    sol = solve_robertson(term, solver, t1)
    assert sol.result == diffrax.RESULTS.successful
    np.testing.assert_allclose(sol.ys[0], reference.y[:, -1], rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(float(jnp.sum(sol.ys[0])), 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_linear_dae_convergence_order(order):
    solver = BDF(bdf_order=order, root_finder=TIGHT)

    def error(h):
        sol = diffrax.diffeqsolve(
            LINEAR_DAE_TERM,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=h,
            y0=linear_dae_exact(0.0),
            stepsize_controller=diffrax.ConstantStepSize(),
            solver_state=seed_state(
                solver, LINEAR_DAE_TERM, linear_dae_exact, 0.0, h, order
            ),
            max_steps=1_000_000,
        )
        return np.abs(np.asarray(sol.ys[0] - linear_dae_exact(1.0)))

    finest = {1: 11, 2: 11, 3: 10, 4: 9, 5: 8}[order]
    hs = [1.0 / 2**m for m in range(finest - 4, finest)]
    errors = np.array([error(h) for h in hs])
    for component in range(2):
        assert observed_order(hs, errors[:, component]) == pytest.approx(
            order, abs=0.15
        )


@pytest.mark.parametrize("bdf_order", [None, 3])
def test_all_false_mask_is_bit_identical_to_ode(bdf_order):
    ode = solve_robertson(diffrax.ODETerm(robertson_vf), BDF(bdf_order=bdf_order))
    dae = solve_robertson(
        SemiExplicitDAETerm(robertson_vf, False), BDF(bdf_order=bdf_order)
    )
    np.testing.assert_array_equal(dae.ys, ode.ys)
    assert dae.stats["num_steps"] == ode.stats["num_steps"]
    assert dae.stats["num_accepted_steps"] == ode.stats["num_accepted_steps"]


def test_all_false_mask_is_bit_identical_at_steady_state():
    def solve(term):
        return diffrax.diffeqsolve(
            term,
            BDF(),
            t0=0.0,
            t1=jnp.inf,
            dt0=1e-6,
            y0=FAST_EQUILIBRIUM_Y0,
            args=FAST_EQUILIBRIUM_ARGS,
            stepsize_controller=BDFController(rtol=1e-8, atol=1e-10, dtmax=1e6),
            event=diffrax.Event(diffrax.steady_state_event(rtol=1e-10, atol=1e-10)),
            adjoint=diffrax.ImplicitAdjoint(),
            saveat=diffrax.SaveAt(t1=True),
        )

    ode = solve(diffrax.ODETerm(fast_equilibrium_vf))
    dae = solve(SemiExplicitDAETerm(fast_equilibrium_vf, False))
    assert dae.result == diffrax.RESULTS.event_occurred
    np.testing.assert_array_equal(dae.ys, ode.ys)
    assert dae.stats["num_steps"] == ode.stats["num_steps"]


@pytest.mark.parametrize("solver", [diffrax.Kvaerno5(), diffrax.Tsit5()])
def test_other_solvers_reject_the_dae_term(solver):
    term = SemiExplicitDAETerm(robertson_dae_vf, ROBERTSON_MASK)
    with pytest.raises(TypeError, match="diffrax_bdf.BDF"):
        diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=1e-3,
            y0=ROBERTSON_Y0,
            args=ROBERTSON_ARGS,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
        )


def test_suppressed_error_is_zero_on_algebraic_components():
    term = SemiExplicitDAETerm(robertson_dae_vf, ROBERTSON_MASK)
    root_finder = diffrax.VeryChord(rtol=1e-8, atol=1e-10)

    def first_step_error(suppress):
        solver = BDF(
            bdf_order=2, root_finder=root_finder, suppress_algebraic_error=suppress
        )
        state = solver.init(term, 0.0, 1e-4, ROBERTSON_Y0, ROBERTSON_ARGS)
        _, y_error, _, _, _ = solver.step(
            term, 0.0, 1e-4, ROBERTSON_Y0, ROBERTSON_ARGS, state, False
        )
        return np.asarray(y_error)

    included = first_step_error(False)
    suppressed = first_step_error(True)
    assert included[2] != 0
    assert suppressed[2] == 0
    np.testing.assert_array_equal(suppressed[:2], included[:2])


def test_suppression_does_not_change_an_ode():
    ode = solve_robertson(diffrax.ODETerm(robertson_vf), BDF())
    suppressed = solve_robertson(
        diffrax.ODETerm(robertson_vf), BDF(suppress_algebraic_error=True)
    )
    np.testing.assert_array_equal(suppressed.ys, ode.ys)
    assert suppressed.stats["num_steps"] == ode.stats["num_steps"]


def cubic_dae_vf(t, y, args):
    del t, args
    z, u = y
    return jnp.stack([u - z, u**3 + u - z])


def hopeless_dae_vf(t, y, args):
    del t, args
    z, u = y
    return jnp.stack([-z, u**2 + 1])


def solve_from(vector_field, y0, solver):
    return diffrax.diffeqsolve(
        SemiExplicitDAETerm(vector_field, jnp.array([False, True])),
        solver,
        t0=0.0,
        t1=1.0,
        dt0=1e-3,
        y0=y0,
        stepsize_controller=BDFController(rtol=1e-8, atol=1e-10),
        saveat=diffrax.SaveAt(t1=True),
        throw=False,
    )


def initial_history(vector_field, y0, solver):
    term = SemiExplicitDAETerm(vector_field, jnp.array([False, True]))
    tols = diffrax.VeryChord(rtol=1e-8, atol=1e-10)
    solver = eqx.tree_at(lambda s: s.root_finder, solver, tols)
    return solver.init(term, 0.0, 1e-3, y0, None)


def test_inconsistent_initial_value_is_corrected():
    state = initial_history(cubic_dae_vf, jnp.array([2.0, 5.0]), BDF())
    assert not state.initial_failed
    assert state.d_array[0, 0] == 2.0
    np.testing.assert_allclose(state.d_array[0, 1], 1.0, rtol=1e-10)
    corrected = solve_from(cubic_dae_vf, jnp.array([2.0, 5.0]), BDF())
    consistent = solve_from(cubic_dae_vf, jnp.array([2.0, 1.0]), BDF())
    assert corrected.result == diffrax.RESULTS.successful
    np.testing.assert_allclose(corrected.ys, consistent.ys, rtol=1e-7)


def test_hopeless_initial_value_fails_cleanly():
    state = initial_history(hopeless_dae_vf, jnp.array([1.0, 0.5]), BDF())
    assert state.initial_failed
    sol = solve_from(hopeless_dae_vf, jnp.array([1.0, 0.5]), BDF())
    assert sol.result == diffrax.RESULTS.nonlinear_divergence
    assert sol.stats["num_steps"] <= 1


def test_correction_can_be_skipped():
    solver = BDF(correct_initial_algebraic=False)
    state = initial_history(cubic_dae_vf, jnp.array([2.0, 5.0]), solver)
    assert not state.initial_failed
    np.testing.assert_array_equal(state.d_array[0], jnp.array([2.0, 5.0]))


RAPID_EQUILIBRIUM_ARGS = jnp.append(FAST_EQUILIBRIUM_ARGS, 0.5)


def rapid_equilibrium_dae_vf(t, y, args):
    del t
    inflow_1, inflow_2, out_1, out_2, out_3, log_k = args
    totals, log_conc = y
    y1, y2, y3 = jnp.exp(log_conc)
    rates = jnp.stack(
        [inflow_1 - out_1 * y1 - out_3 * y3, inflow_2 - out_2 * y2 - out_3 * y3]
    )
    constraints = jnp.stack(
        [
            jnp.log(y1 + y3) - jnp.log(totals[0]),
            jnp.log(y2 + y3) - jnp.log(totals[1]),
            log_conc[2] - log_conc[0] - log_conc[1] - log_k,
        ]
    )
    return rates, constraints


def rapid_equilibrium_reference(args):
    inflow_1, inflow_2, out_1, out_2, out_3, log_k = args

    def balance(conc, _):
        y1, y2 = conc
        y3 = jnp.exp(log_k) * y1 * y2
        return jnp.stack(
            [inflow_1 - out_1 * y1 - out_3 * y3, inflow_2 - out_2 * y2 - out_3 * y3]
        )

    conc = optx.root_find(
        balance, optx.Newton(rtol=1e-12, atol=1e-12), jnp.array([1.0, 1.0])
    ).value
    y3 = jnp.exp(log_k) * conc[0] * conc[1]
    return conc + y3, jnp.log(jnp.append(conc, y3))


def rapid_equilibrium_steady_state(args, suppress):
    sol = diffrax.diffeqsolve(
        SemiExplicitDAETerm(rapid_equilibrium_dae_vf, (False, True)),
        BDF(suppress_algebraic_error=suppress),
        t0=0.0,
        t1=jnp.inf,
        dt0=1e-6,
        y0=(jnp.array([1.0, 1.0]), jnp.zeros(3)),
        args=args,
        stepsize_controller=BDFController(rtol=1e-8, atol=1e-10, dtmax=1e6),
        event=diffrax.Event(diffrax.steady_state_event(rtol=1e-10, atol=1e-10)),
        adjoint=diffrax.ImplicitAdjoint(),
        saveat=diffrax.SaveAt(t1=True),
    )
    return sol


@pytest.mark.parametrize("suppress", [False, True])
def test_dae_steady_state_matches_reference(suppress):
    sol = rapid_equilibrium_steady_state(RAPID_EQUILIBRIUM_ARGS, suppress)
    totals, log_conc = rapid_equilibrium_reference(RAPID_EQUILIBRIUM_ARGS)
    assert sol.result == diffrax.RESULTS.event_occurred
    np.testing.assert_allclose(sol.ys[0][0], totals, rtol=1e-7)
    np.testing.assert_allclose(sol.ys[1][0], log_conc, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("suppress", [False, True])
def test_dae_steady_state_gradient_matches_reference(suppress):
    def steady_totals(args):
        return rapid_equilibrium_steady_state(args, suppress).ys[0][0]

    got = jax.jacrev(steady_totals)(RAPID_EQUILIBRIUM_ARGS)
    expected = jax.jacfwd(lambda a: rapid_equilibrium_reference(a)[0])(
        RAPID_EQUILIBRIUM_ARGS
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-10)
