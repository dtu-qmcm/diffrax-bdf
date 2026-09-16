"""Step-size control with CVODE's deadband.

A BDF only pays for itself if the factorisation of `I - c * J` can be reused across
steps, and that requires `c`, and hence the step size, to stay put. diffrax's stock
`PIDController` adjusts the step on every single step, so `c` changes every step and
the factorisation has to be redone every step -- at which point a BDF is doing the
same amount of linear algebra as an ESDIRK and has given up its advantage.

CVODE solves this with a deadband (`cvSetEta`): a successful step never changes the
step size unless the proposed growth factor is at least `eta_max_fx` (1.5 by
default), and never shrinks it at all. Rejected steps are unaffected. This gives up
a little step-size optimality and buys long runs at constant `c`.
"""

import diffrax
import jax.numpy as jnp
from jaxtyping import PyTree

from ._bdf import ErrorWithOrder


class BDFController(diffrax.PIDController):
    """`diffrax.PIDController` with CVODE's step-size deadband and variable-order support.

    All of `PIDController`'s arguments apply. The added `eta_max_fx` is the growth
    factor below which an accepted step keeps its step size unchanged.

    This controller is also required by a variable-order [`diffrax_bdf.BDF`][].
    `diffrax` asks a solver for its error order only once, so a solver whose order
    changes from step to step has to send the current one along with the error
    estimate; this controller knows how to read that.
    """

    eta_max_fx: float = 1.5

    def adapt_step_size(
        self,
        t0,
        t1,
        y0: PyTree,
        y1_candidate: PyTree,
        args: PyTree,
        y_error: PyTree,
        error_order,
        controller_state,
    ):
        if isinstance(y_error, ErrorWithOrder):
            # The step-size exponent is `1 / (p + 1)` for a method of order `p`, so at
            # variable order it has to follow the order that actually produced this
            # estimate rather than a value fixed before the solve began.
            error_order = y_error.order + 1
            y_error = y_error.error
        keep_step, next_t0, next_t1, made_jump, state, result = super().adapt_step_size(
            t0, t1, y0, y1_candidate, args, y_error, error_order, controller_state
        )
        # Work in signed step sizes throughout so that backwards integration, where
        # `diffrax` hands us a negative step, needs no special case.
        step = t1 - t0
        proposed = next_t1 - next_t0
        growth = proposed / jnp.where(step == 0, 1.0, step)
        frozen = keep_step & (growth < self.eta_max_fx)
        next_t1 = jnp.where(frozen, next_t0 + step, next_t1)
        return keep_step, next_t0, next_t1, made_jump, state, result


BDFController.__init__.__doc__ = """**Arguments:**

As [`diffrax.PIDController`][], plus:

- `eta_max_fx`: an accepted step keeps its step size unless the controller wants to
    grow it by at least this factor. Raising it reuses factorisations for longer at
    the cost of less optimal steps; setting it to 1 recovers `PIDController`.
"""
