"""Chord iteration against a stored LU factorisation.

This replaces the Optimistix root find used elsewhere in diffrax for one reason: we
need to keep the factorisation of `I - c * J` in the solver state and reuse it
across *steps*. Optimistix's root finders build their linearisation internally and
hold it as a `lineax` operator with a captured jaxpr, which cannot be carried
through a `lax.while_loop`. A pair of plain `(lu, piv)` arrays can.

The iteration and its convergence test follow `scipy.integrate.BDF`'s
`solve_bdf_system` closely, including the divergence test, so that behaviour is
comparable step by step.
"""

import equinox.internal as eqxi
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float, Int


def rms_norm(x: Float[Array, " n"]) -> Float[Array, ""]:
    """scipy's `norm`: the 2-norm scaled by the square root of the size."""
    return jnp.linalg.norm(x) / jnp.sqrt(x.size)


def newton_tolerance(rtol, dtype) -> Float[Array, ""]:
    """scipy's `newton_tol`.

    The `eps` must come from the *runtime* dtype. Hard-coding the float64 value
    would, in float32 at `rtol=1e-6`, give a tolerance above 1 and make the
    iteration "converge" immediately on its first step.
    """
    eps = jnp.finfo(dtype).eps
    return jnp.maximum(10 * eps / rtol, jnp.minimum(0.03, jnp.sqrt(rtol)))


def chord_solve(
    residual,
    guess: Float[Array, " n"],
    lu: Float[Array, "n n"],
    piv: Int[Array, " n"],
    scale: Float[Array, " n"],
    tol,
    max_iter: int,
):
    """Solve `residual(d) == 0` by chord iteration, holding `lu` fixed.

    Returns `(correction, converged, num_iterations)`.
    """
    zero = jnp.zeros((), dtype=guess.dtype)

    def cond(carry):
        _, _, _, converged, diverged = carry
        return jnp.invert(converged | diverged)

    def body(carry):
        step, correction, previous_norm, _, _ = carry
        residual_value = residual(correction)
        delta = jsl.lu_solve((lu, piv), residual_value)
        delta_norm = rms_norm(delta / scale)

        # On the first iteration there is no previous norm to compare against, so
        # neither the divergence nor the convergence rate test applies.
        has_history = step > 0
        safe_previous = jnp.where(previous_norm > 0, previous_norm, 1.0)
        rate = jnp.where(has_history, delta_norm / safe_previous, zero)
        # Guard the `1 - rate` denominators: they are only meaningful when the
        # iteration is contracting, and the `rate >= 1` test below catches the rest.
        safe_rate = jnp.minimum(rate, 0.99)
        remaining = jnp.maximum(max_iter - step, 1)

        diverging = has_history & (
            (rate >= 1)
            | (safe_rate**remaining / (1 - safe_rate) * delta_norm > tol)
        )
        diverging = diverging | jnp.invert(jnp.all(jnp.isfinite(residual_value)))
        # scipy breaks out *before* applying a diverging update.
        correction = jnp.where(diverging, correction, correction + delta)
        converged = jnp.invert(diverging) & (
            (delta_norm == 0)
            | (has_history & (safe_rate / (1 - safe_rate) * delta_norm < tol))
        )
        return step + 1, correction, delta_norm, converged, diverging

    init = (
        jnp.asarray(0, jnp.int32),
        guess,
        zero,
        jnp.asarray(False),
        jnp.asarray(False),
    )
    # `kind="bounded"` keeps this reverse-mode differentiable, which a plain
    # `lax.while_loop` would not be.
    steps, correction, _, converged, _ = eqxi.while_loop(
        cond, body, init, max_steps=max_iter, kind="bounded"
    )
    return correction, converged, steps
