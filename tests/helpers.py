"""Shared problem definitions and helpers for the test suite."""

import diffrax
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from diffrax_bdf._coeffs import D_ROWS


def seed_state(solver, term, exact, t0, h, order, args=None):
    """Build solver state whose history is the *exact* solution.

    Without this the opening order ramp contributes `O(h ** 2)` error and swamps
    the measured convergence slope of any higher-order formula.

    The cached Jacobian and factorisation are taken from the solver's own `init`, so
    this keeps working as the solver state grows.
    """
    samples = np.array([np.asarray(exact(t0 - i * h)) for i in range(order + 1)])
    d_array = np.zeros((D_ROWS,) + samples.shape[1:])
    column = samples.copy()
    for j in range(order + 1):
        d_array[j] = column[0]
        column = column[:-1] - column[1:]
    state = solver.init(term, t0, t0 + h, exact(t0), args)
    return eqx.tree_at(
        lambda s: (s.d_array, s.order, s.n_equal_steps, s.h_prev),
        state,
        (
            jnp.asarray(d_array),
            jnp.asarray(order, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(h),
        ),
    )


def observed_order(hs, errors):
    """Least-squares slope of log(error) against log(h)."""
    return float(np.polyfit(np.log(np.asarray(hs)), np.log(np.asarray(errors)), 1)[0])


# A stiff, linear, upper-triangular system with eigenvalues -1e6, -1e3, -1.
# Being linear it has a steady state and a parameter sensitivity we know exactly:
# y* = -inv(A) @ b and dy*/db = -inv(A).
STIFF_MATRIX = jnp.array(
    [
        [-1e6, 1e6, 0.0],
        [0.0, -1e3, 1e3],
        [0.0, 0.0, -1.0],
    ]
)


def stiff_linear_vf(t, y, args):
    del t
    return STIFF_MATRIX @ y + args


def stiff_linear_steady_state(b):
    return -jnp.linalg.solve(STIFF_MATRIX, b)


def fast_equilibrium_vf(t, y, args):
    """A stiff nonlinear network with a unique, non-degenerate steady state.

    A fast reversible binding `y1 + y2 <-> y3` sits on top of slow in- and out-flows.
    Unlike Robertson there is no conserved quantity, so the Jacobian at the steady
    state is non-singular and the steady state is genuinely isolated -- which is what
    both the terminating event and `ImplicitAdjoint` need.
    """
    del t
    inflow_1, inflow_2, out_1, out_2, out_3 = args
    y1, y2, y3 = y
    fast = 1e6 * (y1 * y2 - y3)
    return jnp.stack(
        [
            inflow_1 - out_1 * y1 - fast,
            inflow_2 - out_2 * y2 - fast,
            fast - out_3 * y3,
        ]
    )


FAST_EQUILIBRIUM_ARGS = jnp.array([1.0, 2.0, 0.1, 0.2, 0.5])
FAST_EQUILIBRIUM_Y0 = jnp.array([1.0, 1.0, 1.0])


def robertson_vf(t, y, args):
    """Robertson's chemical kinetics, the canonical stiff BDF test problem."""
    del t
    k1, k2, k3 = args
    y1, y2, y3 = y
    return jnp.stack(
        [
            -k1 * y1 + k3 * y2 * y3,
            k1 * y1 - k2 * y2**2 - k3 * y2 * y3,
            k2 * y2**2,
        ]
    )


ROBERTSON_ARGS = jnp.array([0.04, 3e7, 1e4])
ROBERTSON_Y0 = jnp.array([1.0, 0.0, 0.0])


def steady_state_solve(solver, vf, y0, args, rtol=1e-8, atol=1e-10, max_steps=100_000):
    """enzax's steady-state configuration: integrate to `inf` until `dy/dt` vanishes.

    `dtmax` is not optional here. As the residual collapses the controller's growth
    factor saturates at `factormax`, so the step grows by 10x every step without
    bound until it overflows the time variable and the solve livelocks.
    """
    return diffrax.diffeqsolve(
        diffrax.ODETerm(vf),
        solver,
        t0=0.0,
        t1=jnp.inf,
        dt0=1e-6,
        y0=y0,
        args=args,
        stepsize_controller=diffrax.PIDController(
            rtol=rtol, atol=atol, pcoeff=0.1, icoeff=0.3, dtmax=1e6
        ),
        event=diffrax.Event(diffrax.steady_state_event(rtol=1e-10, atol=1e-10)),
        adjoint=diffrax.ImplicitAdjoint(),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=max_steps,
    )
