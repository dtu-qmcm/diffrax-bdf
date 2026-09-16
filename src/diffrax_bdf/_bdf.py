"""A fixed-order BDF/NDF solver for diffrax.

The method is the one used by `scipy.integrate.BDF` and MATLAB's `ode15s`: a
backward differentiation formula applied to a quasi-constant-step history stored
as backward differences, with the Shampine & Reichelt NDF correction available.

Two properties of `diffrax`'s integration loop shape the implementation:

* `solver_state` is a `lax.while_loop` carry, so it must have a fixed pytree
  structure, shape and dtype on every step. The BDF order is therefore a *traced*
  integer used to mask fixed-size buffers, never a Python-level dimension.
* `solver_state` is rolled back when a step is rejected (`_integrate.py:423-425`),
  so nothing here may rely on remembering a rejected attempt. `n_equal_steps` is
  derived from `h == h_prev` at step entry rather than accumulated, which
  reproduces scipy's reset-on-rejection for free.

Note also that `t0` and `t1` are only ever passed to `terms.contr`/`terms.vf_prod`
and never subtracted. Under backwards integration `diffrax` wraps the terms so
that `contr` returns a *negative* step, and every formula below is consistent
under that sign provided the step is never recomputed as `t1 - t0`.
"""

from collections.abc import Callable
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
from diffrax import (
    AbstractAdaptiveSolver,
    AbstractImplicitSolver,
    AbstractTerm,
    LocalLinearInterpolation,
    RESULTS,
    VeryChord,
    with_stepsize_controller_tols,
)
from equinox.internal import ω
from jaxtyping import Array, Float, Int, PyTree

from ._coeffs import D_ROWS, MAX_ORDER, change_D, make_tables, weighted_row_sum


class _BDFState(eqx.Module):
    """The multistep history carried between steps."""

    d_array: PyTree[Float[Array, "8 ..."]]
    order: Int[Array, ""]
    n_equal_steps: Int[Array, ""]
    h_prev: Float[Array, ""]


def _bdf_residual(d, nonlinear_args):
    """Residual whose root is the total Newton correction from the predictor.

    Solving for the correction rather than for `y1` avoids forming `y1 - y_pred`,
    a cancellation between two `O(|y|)` quantities whose difference is
    `O(h ** (k + 1))`; that difference feeds straight into the error estimate.
    """
    vf_prod, t1, y_pred, args, control, inv_alpha, psi = nonlinear_args
    y = (y_pred**ω + d**ω).ω
    # `vf_prod(t1, y, args, control)` is `h * f(t1, y)`, so scaling by
    # `1 / alpha[order]` gives scipy's `c * f` with `c = h / alpha[order]`.
    c_f = (inv_alpha * vf_prod(t1, y, args, control) ** ω).ω
    return (c_f**ω - psi**ω - d**ω).ω


def _row_mask(mask, x):
    """Broadcast a length-`D_ROWS` row mask against a leaf of the difference array."""
    return jnp.reshape(mask, (-1,) + (1,) * (jnp.ndim(x) - 1))


def _update_difference_array(d_array, order, correction):
    """Fold an accepted Newton correction into the difference array.

    scipy writes this as

        D[order + 2] = d - D[order + 1]
        D[order + 1] = d
        for i in reversed(range(order + 1)): D[i] += D[i + 1]

    The loop is a reverse cumulative sum over rows `0..order + 1`, which is done
    here by masking rather than slicing so that the shapes stay static.
    """
    idx = jnp.arange(D_ROWS)

    def update(x, d):
        x = x.at[order + 2].set(d - x[order + 1])
        x = x.at[order + 1].set(d)
        summable = jnp.where(_row_mask(idx <= order + 1, x), x, 0)
        reverse_cumsum = jnp.cumsum(summable[::-1], axis=0)[::-1]
        return jnp.where(_row_mask(idx <= order, x), reverse_cumsum, x)

    return jtu.tree_map(update, d_array, correction)


class BDF(AbstractImplicitSolver, AbstractAdaptiveSolver):
    """Fixed-order backward differentiation formula.

    The order ramps from 1 up to `bdf_order` over the opening steps, since a
    `k`-step formula needs `k` points of history. Because a rejected step rolls the
    solver state back, the ramp is automatically correct: a step that did not
    happen does not advance the order.

    !!! warning

        The backward differences suffer heavy cancellation at high order. Orders
        above 2 are effectively noise in float32 — enable `jax_enable_x64`.
    """

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    bdf_order: int = eqx.field(static=True, default=2)
    use_ndf: bool = eqx.field(static=True, default=True)
    root_finder: optx.AbstractRootFinder = with_stepsize_controller_tols(VeryChord)()
    root_find_max_steps: int = 10

    def __check_init__(self):
        if not 1 <= self.bdf_order <= MAX_ORDER:
            raise ValueError(f"`bdf_order` must be between 1 and {MAX_ORDER}.")

    def order(self, terms):
        del terms
        return self.bdf_order

    def error_order(self, terms):
        del terms
        # The local error of a k-th order BDF step is O(h ** (k + 1)).
        return self.bdf_order + 1

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def init(self, terms, t0, t1, y0, args) -> _BDFState:
        control = terms.contr(t0, t1)
        if jnp.shape(control) != ():
            raise ValueError(
                "`BDF` requires a term whose control is a scalar (e.g. `ODETerm`); "
                f"got a control of shape {jnp.shape(control)}. The step-size ratio "
                "used to rescale the difference array is not defined otherwise."
            )
        # Start at order 1 with `D = [y0, h * f0]`, matching scipy. The remaining
        # rows are zero rather than uninitialised so that a high-order formula
        # applied before enough history exists degenerates gracefully.
        f0 = terms.vf_prod(t0, y0, args, control)
        d_array = jtu.tree_map(
            lambda y, f: jnp.zeros((D_ROWS,) + jnp.shape(y), jnp.result_type(y))
            .at[0]
            .set(y)
            .at[1]
            .set(f),
            y0,
            f0,
        )
        return _BDFState(
            d_array=d_array,
            order=jnp.asarray(1, dtype=jnp.int32),
            n_equal_steps=jnp.asarray(0, dtype=jnp.int32),
            h_prev=control,
        )

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        control = terms.contr(t0, t1)
        gamma_table, alpha_table, error_const = make_tables(self.use_ndf)

        d_array = solver_state.d_array
        order = solver_state.order
        # The difference array is scaled for the previous step size; rescale it
        # before use, or the history polynomial is evaluated at the wrong times.
        factor = control / solver_state.h_prev
        d_array = change_D(d_array, order, factor)
        n_equal_steps = jnp.where(factor == 1, solver_state.n_equal_steps + 1, 0)

        # A jump means the vector field was discontinuous at `t0`, so the history
        # polynomial is meaningless and we restart at order 1. `PIDController`
        # always reports `False` and `diffrax` keeps it a static bool where it can,
        # so in the common case this costs nothing at all.
        if not (isinstance(made_jump, bool) and not made_jump):
            d_array, order, n_equal_steps = self._restart(
                terms, t0, y0, args, control, d_array, order, n_equal_steps, made_jump
            )

        dtype = jnp.result_type(*jtu.tree_leaves(y0))
        indices = jnp.arange(MAX_ORDER + 1)
        active = indices <= order
        inv_alpha = 1 / jnp.asarray(alpha_table, dtype)[order]

        y_pred = weighted_row_sum(active.astype(dtype), d_array)
        psi_coeffs = jnp.where(active, jnp.asarray(gamma_table, dtype), 0.0)
        psi = (inv_alpha * weighted_row_sum(psi_coeffs, d_array) ** ω).ω

        correction, converged = self._solve(
            terms, t1, y_pred, args, control, inv_alpha, psi
        )
        y1 = (y_pred**ω + correction**ω).ω
        y_error = (jnp.asarray(error_const, dtype)[order] * correction**ω).ω
        # A failed Newton solve is reported as an infinite error estimate so that
        # the controller rejects the step, rather than as a non-successful
        # `RESULTS`, which `diffrax` would treat as a failure of the whole solve.
        y_error = jtu.tree_map(
            lambda e: jnp.where(converged, e, jnp.inf), y_error
        )

        d_array = _update_difference_array(d_array, order, correction)
        new_state = _BDFState(
            d_array=d_array,
            order=jnp.minimum(order + 1, self.bdf_order),
            n_equal_steps=n_equal_steps,
            h_prev=control,
        )
        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, new_state, RESULTS.successful

    def _restart(
        self, terms, t0, y0, args, control, d_array, order, n_equal_steps, made_jump
    ):
        """Rebuild the history from scratch after a discontinuity."""
        f0 = terms.vf_prod(t0, y0, args, control)
        idx = jnp.arange(D_ROWS)

        def reset(x, y, f):
            fresh = jnp.zeros_like(x).at[0].set(y).at[1].set(f)
            return jnp.where(made_jump, fresh, x)

        d_array = jtu.tree_map(reset, d_array, y0, f0)
        order = jnp.where(made_jump, 1, order)
        n_equal_steps = jnp.where(made_jump, 0, n_equal_steps)
        return d_array, order, n_equal_steps

    def _solve(self, terms, t1, y_pred, args, control, inv_alpha, psi):
        """One Newton solve for the correction to the predictor."""
        root_finder = self._retuned_root_finder(y_pred)
        guess = jtu.tree_map(jnp.zeros_like, y_pred)
        nonlinear_args = (
            terms.vf_prod,
            t1,
            y_pred,
            args,
            control,
            inv_alpha,
            psi,
        )
        solution = optx.root_find(
            _bdf_residual,
            root_finder,
            guess,
            nonlinear_args,
            throw=False,
            max_steps=self.root_find_max_steps,
        )
        converged = solution.result == optx.RESULTS.successful
        return solution.value, converged

    def _retuned_root_finder(self, y_pred):
        """Scale the root finder's `atol` to the size of the solution.

        Optimistix tests convergence against `atol + rtol * |variable|`, and our
        variable is the correction, which is tiny. Left alone that makes the test
        effectively absolute and far tighter than intended. scipy instead scales by
        the predictor, which is what we restore here. The controller's tolerances
        have already been injected into `self.root_finder` by `diffeqsolve`.
        """
        root_finder = self.root_finder
        atol = getattr(root_finder, "atol", None)
        rtol = getattr(root_finder, "rtol", None)
        norm = getattr(root_finder, "norm", None)
        if not (isinstance(atol, (int, float)) and isinstance(rtol, (int, float))):
            return root_finder
        scale = atol + rtol * norm(y_pred)
        return eqx.tree_at(lambda rf: rf.atol, root_finder, scale)


BDF.__init__.__doc__ = """**Arguments:**

- `bdf_order`: the order of the formula, between 1 and 5. The order ramps up to
    this value over the opening steps.
- `use_ndf`: whether to apply the Shampine & Reichelt NDF correction. `False`
    gives the classical BDF coefficients, which is the family SUNDIALS CVODE
    implements.
- `root_finder`: an [Optimistix](https://github.com/patrick-kidger/optimistix) root
    finder for the implicit problem at each step. Use a chord-style solver:
    `optimistix.Newton` relinearises on every iteration, costing one Jacobian per
    iteration rather than one per step.
- `root_find_max_steps`: the maximum number of root-finder steps before the step is
    rejected and retried with whatever smaller step the controller proposes.
"""
