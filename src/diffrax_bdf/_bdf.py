"""A fixed-order BDF/NDF solver for diffrax.

The method is the one used by `scipy.integrate.BDF` and MATLAB's `ode15s`: a
backward differentiation formula applied to a quasi-constant-step history stored
as backward differences, with the Shampine & Reichelt NDF correction available.

On top of that sits SUNDIALS CVODE's linear-algebra reuse policy, which is what
makes a BDF worth using over an ESDIRK. `Kvaerno5` already computes only one
Jacobian per step and reuses it across its five stages, so "one Newton solve per
step instead of five stage solves" is worth only a few linear solves. The real
saving is keeping `J` and the factorisation of `I - c * J` across *steps*, which
CVODE gates on two separate counters (see `_should_update`).

Three properties of `diffrax`'s integration loop shape the implementation:

* `solver_state` is a `lax.while_loop` carry, so it must have a fixed pytree
  structure, shape and dtype on every step. The BDF order is therefore a *traced*
  integer used to mask fixed-size buffers, never a Python-level dimension.
* `solver_state` is rolled back when a step is rejected (`_integrate.py:423-425`).
  This is why the reuse counters live here rather than in the controller: they must
  stay consistent with the stored factorisation, which is also rolled back. It also
  matches CVODE, whose `nst`-based counters likewise advance only on accepted steps.
* `t0` and `t1` are only ever passed to `terms.contr`/`terms.vf_prod` and never
  subtracted. Under backwards integration `diffrax` wraps the terms so that `contr`
  returns a *negative* step, and every formula below is consistent under that sign
  provided the step is never recomputed as `t1 - t0`.
"""

from collections.abc import Callable
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
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
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Int, PyTree

from ._coeffs import D_ROWS, MAX_ORDER, change_D, make_tables, weighted_row_sum
from ._newton import chord_solve, newton_tolerance, rms_norm


class ErrorWithOrder(eqx.Module):
    """An error estimate together with the BDF order that produced it.

    `diffrax` asks the solver for its error order once, outside the loop
    (`_integrate.py:388` calls `solver.error_order(terms)`, which sees neither the
    state nor the time), so a variable-order solver has no way to tell the
    controller which order the current error estimate belongs to. The step-size
    controller's only other input from the solver is `y_error`, and the integration
    loop treats that opaquely -- it only does a structure-agnostic `tree_map` over it
    (`_integrate.py:386`) -- so the order travels in here alongside the estimate.

    [`diffrax_bdf.BDFController`][] unpacks this. Stock `diffrax.PIDController`
    cannot: it tree-maps `y_error` against `y0` and will raise on the extra leaf.
    """

    error: PyTree
    order: Float[Array, ""]

    def __truediv__(self, other):
        # `PIDController` tree-maps its scaling function over `(y0, y1, y_error)`,
        # taking its structure from `y0`, so this whole object arrives where a leaf
        # was expected and the first thing done to it is a division. Intercepting
        # that turns an unreadable TypeError into an actionable one.
        del other
        raise TypeError(
            "A variable-order `BDF` reports its error estimate together with the "
            "order that produced it, and `diffrax.PIDController` cannot read that. "
            "Either pass `stepsize_controller=diffrax_bdf.BDFController(...)`, which "
            "can, or pin the order with `BDF(bdf_order=...)` to get a plain error "
            "estimate that any adaptive controller accepts."
        )


class _BDFState(eqx.Module):
    """The multistep history, plus the cached linear algebra and its age."""

    d_array: PyTree[Float[Array, "8 ..."]]
    order: Int[Array, ""]
    n_equal_steps: Int[Array, ""]
    h_prev: Float[Array, ""]
    jac: Float[Array, "n n"]
    lu: Float[Array, "n n"]
    piv: Int[Array, " n"]
    c_at_last_lu: Float[Array, ""]
    steps_since_lu: Int[Array, ""]
    steps_since_jac: Int[Array, ""]


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
    """Fixed-order backward differentiation formula with Jacobian reuse.

    The order ramps from 1 up to `bdf_order` over the opening steps, since a
    `k`-step formula needs `k` points of history. Because a rejected step rolls the
    solver state back, the ramp is automatically correct: a step that did not
    happen does not advance the order.

    Pair this with [`diffrax_bdf.BDFController`][]. With diffrax's stock
    `PIDController` the step size, and hence `c`, changes on every step, so the
    factorisation is rebuilt every step and the reuse below never engages.

    !!! warning

        The Jacobian is formed densely, so this is intended for small to medium
        systems. The backward differences also suffer heavy cancellation at high
        order: orders above 2 are effectively noise in float32, so enable
        `jax_enable_x64`.
    """

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    bdf_order: int | None = eqx.field(static=True, default=None)
    max_order: int = eqx.field(static=True, default=MAX_ORDER)
    use_ndf: bool = eqx.field(static=True, default=True)
    root_finder: optx.AbstractRootFinder = with_stepsize_controller_tols(VeryChord)()
    root_find_max_steps: int = 4
    # CVODE's `MSBP` and `MSBJ`, and its `DGMAX_LSETUP`.
    lu_reuse_steps: int = eqx.field(static=True, default=20)
    jac_reuse_steps: int = eqx.field(static=True, default=51)
    max_gamma_change: float = 0.3

    def __check_init__(self):
        if self.bdf_order is not None and not 1 <= self.bdf_order <= MAX_ORDER:
            raise ValueError(f"`bdf_order` must be between 1 and {MAX_ORDER}, or None.")
        if not 1 <= self.max_order <= MAX_ORDER:
            raise ValueError(f"`max_order` must be between 1 and {MAX_ORDER}.")

    @property
    def _variable_order(self) -> bool:
        return self.bdf_order is None

    @property
    def _order_cap(self) -> int:
        return self.max_order if self._variable_order else self.bdf_order

    def order(self, terms):
        del terms
        return self._order_cap

    def error_order(self, terms):
        del terms
        # The local error of a k-th order BDF step is O(h ** (k + 1)). At variable
        # order this is only the value used to pick the very first step size; the
        # per-step order reaches the controller through `ErrorWithOrder`.
        return self._order_cap + 1

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def _tolerances(self):
        rtol = getattr(self.root_finder, "rtol", None)
        atol = getattr(self.root_finder, "atol", None)
        if not isinstance(rtol, (int, float)) or not isinstance(atol, (int, float)):
            raise ValueError(
                "`BDF` needs concrete tolerances for its Newton iteration. Either use "
                "an adaptive step size controller, which supplies them, or construct "
                "the solver with e.g. "
                "`BDF(root_finder=diffrax.VeryChord(rtol=..., atol=...))`."
            )
        return rtol, atol

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
        flat, _ = ravel_pytree(y0)
        size = flat.size
        dtype = flat.dtype
        return _BDFState(
            d_array=d_array,
            order=jnp.asarray(1, dtype=jnp.int32),
            n_equal_steps=jnp.asarray(0, dtype=jnp.int32),
            h_prev=control,
            jac=jnp.zeros((size, size), dtype),
            lu=jnp.zeros((size, size), dtype),
            piv=jnp.zeros((size,), jnp.int32),
            c_at_last_lu=jnp.asarray(0.0, dtype),
            # Start both counters expired so the first step builds everything.
            steps_since_lu=jnp.asarray(self.lu_reuse_steps, jnp.int32),
            steps_since_jac=jnp.asarray(self.jac_reuse_steps, jnp.int32),
        )

    def _should_update(self, state, c):
        """CVODE's two-tier gate on the Jacobian and its factorisation.

        The asymmetry is the point. `J` costs a full automatic-differentiation pass
        through the vector field and survives ~51 steps; the factorisation costs an
        LU and is redone whenever `c` has moved appreciably or 20 steps have passed,
        reassembling `I - c * J` from the *stored* `J` without touching the vector
        field. So the expensive quantity is refreshed far less often than the cheap
        one. See `cvLsSetup` and `cvNls` in SUNDIALS.
        """
        need_jac = state.steps_since_jac >= self.jac_reuse_steps
        safe_previous = jnp.where(state.c_at_last_lu == 0, jnp.inf, state.c_at_last_lu)
        gamma_change = jnp.abs(c / safe_previous - 1)
        need_lu = (
            need_jac
            | (gamma_change > self.max_gamma_change)
            | (state.steps_since_lu >= self.lu_reuse_steps)
        )
        return need_jac, need_lu, gamma_change

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
        steps_since_jac = solver_state.steps_since_jac

        # A jump means the vector field was discontinuous at `t0`, so the history
        # polynomial is meaningless and we restart at order 1 with a fresh Jacobian.
        # `PIDController` always reports `False` and `diffrax` keeps it a static bool
        # where it can, so in the common case this costs nothing at all.
        if not (isinstance(made_jump, bool) and not made_jump):
            d_array, order, n_equal_steps, steps_since_jac = self._restart(
                terms,
                t0,
                y0,
                args,
                control,
                d_array,
                order,
                n_equal_steps,
                steps_since_jac,
                made_jump,
            )

        flat_y0, unravel = ravel_pytree(y0)
        dtype = flat_y0.dtype
        size = flat_y0.size
        indices = jnp.arange(MAX_ORDER + 1)
        active = indices <= order
        inv_alpha = 1 / jnp.asarray(alpha_table, dtype)[order]
        c = control * inv_alpha

        y_pred = weighted_row_sum(active.astype(dtype), d_array)
        psi_coeffs = jnp.where(active, jnp.asarray(gamma_table, dtype), 0.0)
        psi = (inv_alpha * weighted_row_sum(psi_coeffs, d_array) ** ω).ω
        flat_pred, _ = ravel_pytree(y_pred)
        flat_psi, _ = ravel_pytree(psi)

        def vector_field(flat):
            return ravel_pytree(terms.vf(t1, unravel(flat), args))[0]

        def residual(correction):
            y = unravel(flat_pred + correction)
            scaled = ravel_pytree(terms.vf_prod(t1, y, args, control))[0] * inv_alpha
            return scaled - flat_psi - correction

        state = eqx.tree_at(
            lambda s: s.steps_since_jac, solver_state, steps_since_jac
        )
        need_jac, need_lu, gamma_change = self._should_update(state, c)

        rtol, atol = self._tolerances()
        scale = atol + rtol * jnp.abs(flat_pred)
        tol = newton_tolerance(rtol, dtype)
        identity = jnp.eye(size, dtype=dtype)

        def factorise(jacobian):
            return jsl.lu_factor(identity - c * jacobian)

        jac = jax.lax.cond(
            need_jac, lambda: jax.jacfwd(vector_field)(flat_pred), lambda: state.jac
        )
        lu, piv = jax.lax.cond(
            need_lu, lambda: factorise(jac), lambda: (state.lu, state.piv)
        )
        guess = jnp.zeros_like(flat_pred)
        correction, converged, _ = chord_solve(
            residual, guess, lu, piv, scale, tol, self.root_find_max_steps
        )

        # A Newton failure against reused linear algebra is not evidence that the step
        # is too large, so rejecting would be the wrong response -- and it would not
        # even help, since a rejected step rolls the solver state back and the retry
        # would arrive with the same stale matrices. scipy and CVODE both refresh and
        # retry within the step instead.
        #
        # Which part to refresh matters. If the factorisation was stale with respect to
        # `c` -- because `c` moved less than `max_gamma_change`, or because the order
        # ramped and changed `alpha[order]` -- then rebuilding it from the *stored*
        # Jacobian is enough, and costs no vector-field evaluations. Only when the
        # factorisation was already current at this `c` is the Jacobian itself the
        # suspect. This is CVODE's `convfail` distinction.
        retry = jnp.invert(converged)
        # Is the factorisation we just used already built at this `c`? If so the
        # Jacobian inside it is the only thing left to blame. If not -- `c` drifted
        # within the deadband, or the order ramped and changed `alpha[order]` --
        # rebuilding the factorisation from the stored Jacobian is the cheap fix.
        lu_is_current = need_lu | (gamma_change < 1e-10)
        blame_jacobian = retry & lu_is_current

        def refresh():
            fresh_jac = jax.lax.cond(
                blame_jacobian,
                lambda: jax.jacfwd(vector_field)(flat_pred),
                lambda: jac,
            )
            fresh_lu, fresh_piv = factorise(fresh_jac)
            retried, retried_converged, _ = chord_solve(
                residual, guess, fresh_lu, fresh_piv, scale, tol, self.root_find_max_steps
            )
            return fresh_jac, fresh_lu, fresh_piv, retried, retried_converged

        jac, lu, piv, correction, converged = jax.lax.cond(
            retry,
            refresh,
            lambda: (jac, lu, piv, correction, converged),
        )
        did_jac = need_jac | blame_jacobian
        did_lu = need_lu | retry

        flat_y1 = flat_pred + correction
        y1 = unravel(flat_y1)
        y_error = unravel(jnp.asarray(error_const, dtype)[order] * correction)
        # A failed Newton solve is reported as an infinite error estimate so that
        # the controller rejects the step, rather than as a non-successful
        # `RESULTS`, which `diffrax` would treat as a failure of the whole solve.
        y_error = jtu.tree_map(lambda e: jnp.where(converged, e, jnp.inf), y_error)

        d_array = _update_difference_array(d_array, order, unravel(correction))

        if self._variable_order:
            next_order, n_equal_steps = self._select_order(
                d_array,
                order,
                correction,
                flat_y1,
                error_const,
                rtol,
                atol,
                n_equal_steps,
                dtype,
            )
            # The order that produced this error estimate is the one the controller
            # needs, not the one we have just chosen for the next step.
            y_error = ErrorWithOrder(
                error=y_error, order=order.astype(dtype)
            )
        else:
            # At fixed order the only movement is the opening ramp: a k-step formula
            # needs k points of history. A rejected step rolls the state back, so the
            # ramp is automatically correct -- a step that did not happen does not
            # advance the order.
            next_order = jnp.minimum(order + 1, self.bdf_order)

        new_state = _BDFState(
            d_array=d_array,
            order=next_order,
            n_equal_steps=n_equal_steps,
            h_prev=control,
            jac=jac,
            lu=lu,
            piv=piv,
            c_at_last_lu=jnp.where(did_lu, c, state.c_at_last_lu),
            steps_since_lu=jnp.where(did_lu, 0, state.steps_since_lu + 1).astype(
                jnp.int32
            ),
            steps_since_jac=jnp.where(did_jac, 0, state.steps_since_jac + 1).astype(
                jnp.int32
            ),
        )
        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, new_state, RESULTS.successful

    def _select_order(
        self,
        d_array,
        order,
        correction,
        flat_y1,
        error_const,
        rtol,
        atol,
        n_equal_steps,
        dtype,
    ):
        """scipy's order heuristic: compare the error at this order, one below, one above.

        The two rows of the difference array beyond `order` exist for exactly this.
        After `_update_difference_array`, row `order` carries what the error would
        have been one order lower and row `order + 2` what it would be one order
        higher, so all three candidates are available without re-solving anything.
        """
        table = jnp.asarray(error_const, dtype)
        scale = atol + rtol * jnp.abs(flat_y1)

        def row(index):
            return ravel_pytree(jtu.tree_map(lambda x: x[index], d_array))[0]

        error = rms_norm(table[order] * correction / scale)
        error_lower = rms_norm(table[order - 1] * row(order) / scale)
        error_higher = rms_norm(table[order + 1] * row(order + 2) / scale)
        # Candidates outside 1..cap are ruled out with an infinite error, which maps
        # to a zero factor and so can never win the argmax below.
        error_lower = jnp.where(order > 1, error_lower, jnp.inf)
        error_higher = jnp.where(order < self._order_cap, error_higher, jnp.inf)

        exponents = jnp.stack([order, order + 1, order + 2]).astype(dtype)
        candidates = jnp.stack([error_lower, error, error_higher])
        factors = candidates ** (-1 / exponents)
        delta = jnp.argmax(factors) - 1

        # scipy holds the order still until `order + 1` steps of equal size have been
        # taken. Both the error constants and the difference array assume an equally
        # spaced history, so changing order before that compares invalid estimates.
        allowed = n_equal_steps >= order + 1
        next_order = jnp.where(
            allowed, jnp.clip(order + delta, 1, self._order_cap), order
        )
        n_equal_steps = jnp.where(allowed & (delta != 0), 0, n_equal_steps)
        return next_order.astype(jnp.int32), n_equal_steps.astype(jnp.int32)

    def _restart(
        self,
        terms,
        t0,
        y0,
        args,
        control,
        d_array,
        order,
        n_equal_steps,
        steps_since_jac,
        made_jump,
    ):
        """Rebuild the history from scratch after a discontinuity."""
        f0 = terms.vf_prod(t0, y0, args, control)

        def reset(x, y, f):
            fresh = jnp.zeros_like(x).at[0].set(y).at[1].set(f)
            return jnp.where(made_jump, fresh, x)

        d_array = jtu.tree_map(reset, d_array, y0, f0)
        order = jnp.where(made_jump, 1, order)
        n_equal_steps = jnp.where(made_jump, 0, n_equal_steps)
        # The vector field changed, so the cached Jacobian is suspect too.
        steps_since_jac = jnp.where(
            made_jump, self.jac_reuse_steps, steps_since_jac
        ).astype(jnp.int32)
        return d_array, order, n_equal_steps, steps_since_jac


BDF.__init__.__doc__ = """**Arguments:**

- `bdf_order`: pin the formula to a fixed order between 1 and 5, or leave it as
    `None` (the default) to vary the order between 1 and `max_order` as scipy and
    CVODE do. Variable order needs [`diffrax_bdf.BDFController`][]; a fixed order
    works with any adaptive step size controller. Note that a fixed high order is
    less stable: BDF4 is A(73.35 degrees)-stable and BDF5 only
    A(51.84 degrees)-stable, so both can go unstable on strongly oscillatory systems.
- `max_order`: the highest order the variable-order heuristic may select.
- `use_ndf`: whether to apply the Shampine & Reichelt NDF correction. `False`
    gives the classical BDF coefficients, which is the family SUNDIALS CVODE
    implements.
- `root_finder`: only its `rtol` and `atol` are used, to set the Newton convergence
    tolerance. The iteration itself is a chord iteration against a stored
    factorisation, which is what allows that factorisation to be reused between
    steps.
- `root_find_max_steps`: maximum chord iterations per step. scipy uses 4, CVODE 3.
- `lu_reuse_steps`: refactorise `I - c * J` after this many accepted steps even if
    `c` has not moved. CVODE's `MSBP`.
- `jac_reuse_steps`: re-evaluate the Jacobian after this many accepted steps.
    CVODE's `MSBJ`. Much larger than `lu_reuse_steps` because a Jacobian costs an
    automatic-differentiation pass where a refactorisation does not.
- `max_gamma_change`: refactorise when `c` has moved by more than this relative
    amount since the last factorisation. CVODE's `DGMAX_LSETUP`.
"""
