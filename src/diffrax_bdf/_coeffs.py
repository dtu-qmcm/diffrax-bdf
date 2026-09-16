"""BDF/NDF coefficient tables and backward-difference array manipulation.

The representation follows `scipy.integrate.BDF` (in turn following the MATLAB
`ode15s` of Shampine & Reichelt 1997): the solution history is stored as an array
of backward differences `D`, with `D[j]` the `j`-th backward difference of `y`.

Everything here is written for static shapes: arrays are always sized for
`MAX_ORDER` and the active block is selected by masking against a *traced* order,
never by slicing with it. This is what makes the solver usable inside the
`lax.while_loop` that `diffrax` runs its integration in.
"""

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Float, PyTree


MAX_ORDER = 5

# Rows of the difference array. Rows `0..order` hold the differences themselves;
# row `order + 1` holds the accepted Newton correction `d`, and row `order + 2`
# holds `d - D_prev[order + 1]`. The last two feed the order-up error estimate.
D_ROWS = MAX_ORDER + 3

# Shampine & Reichelt's NDF correction. `kappa = 0` recovers the classical BDF
# formulae (which is what SUNDIALS CVODE uses); see `make_tables`.
KAPPA_NDF = np.array([0.0, -0.1850, -1 / 9, -0.0823, -0.0415, 0.0])
KAPPA_BDF = np.zeros(MAX_ORDER + 1)


def make_tables(use_ndf: bool = True):
    """Return `(gamma, alpha, error_const)` as float64 numpy arrays.

    With `use_ndf=False` this reduces to the classical BDF coefficients,
    `alpha[k] == sum_{j=1..k} 1/j` and `error_const[k] == 1/(k + 1)`, which is the
    formula family CVODE implements (its `gamma` is our `h / alpha[order]`).
    """
    kappa = KAPPA_NDF if use_ndf else KAPPA_BDF
    gamma = np.hstack((0.0, np.cumsum(1 / np.arange(1, MAX_ORDER + 1))))
    alpha = (1 - kappa) * gamma
    error_const = kappa * gamma + 1 / np.arange(1, MAX_ORDER + 2)
    return gamma, alpha, error_const


def compute_R(factor, order) -> Float[Array, "6 6"]:
    """The `(MAX_ORDER + 1, MAX_ORDER + 1)` difference-rescaling matrix.

    The top-left `(order + 1, order + 1)` block equals scipy's
    `compute_R(order, factor)`; outside that block the result is the identity, so
    that multiplying by it leaves the inactive rows of `D` untouched.

    Note the unmasked entries `M[i, j] = (i - 1 - factor * j) / i` do not depend on
    `order` at all, and `cumprod` along axis 0 only ever reads rows `<= i`. That is
    why building the full matrix and masking gives exactly the same active block as
    scipy's order-sized construction.
    """
    idx = jnp.arange(MAX_ORDER + 1)
    i = jnp.arange(1, MAX_ORDER + 1)[:, None]
    j = jnp.arange(1, MAX_ORDER + 1)[None, :]
    # `jnp.result_type(float)` respects the x64 config, so this is float64 when
    # x64 is enabled rather than being dragged down to float32 by a literal.
    dtype = jnp.result_type(factor, jnp.result_type(float))
    body = (i - 1 - factor * j) / i
    m = jnp.zeros((MAX_ORDER + 1, MAX_ORDER + 1), dtype=dtype)
    m = m.at[1:, 1:].set(body.astype(dtype))
    m = m.at[0].set(1.0)
    r = jnp.cumprod(m, axis=0)
    active = (idx[:, None] <= order) & (idx[None, :] <= order)
    return jnp.where(active, r, jnp.eye(MAX_ORDER + 1, dtype=dtype))


def change_D(
    d_array: PyTree[Float[Array, "8 ..."]], order, factor
) -> PyTree[Float[Array, "8 ..."]]:
    """Rescale the difference array for a change of step size by `factor`.

    This is scipy's `change_D`, and is exactly SUNDIALS' `cvRescale` expressed in
    the backward-difference basis rather than the Nordsieck one: writing `S` for the
    (fixed, order-only) change of basis, `(R @ U).T == inv(S) @ diag(factor**j) @ S`.
    """
    ru = compute_R(factor, order) @ compute_R(1.0, order)
    return jtu.tree_map(lambda x: _apply_rows(ru.T, x), d_array)


def _apply_rows(mat: Float[Array, "6 6"], x: Float[Array, "8 ..."]):
    """Left-multiply the first `MAX_ORDER + 1` rows of `x` by `mat`."""
    x = jnp.asarray(x)
    head = jnp.tensordot(mat.astype(x.dtype), x[: MAX_ORDER + 1], axes=1)
    return x.at[: MAX_ORDER + 1].set(head)


def weighted_row_sum(coeffs: Float[Array, " 6"], d_array: PyTree) -> PyTree:
    """Contract `coeffs` against rows `0..MAX_ORDER` of each leaf of `d_array`."""
    def contract(x):
        x = jnp.asarray(x)
        return jnp.tensordot(
            jnp.asarray(coeffs).astype(x.dtype), x[: MAX_ORDER + 1], axes=1
        )

    return jtu.tree_map(contract, d_array)
