from collections.abc import Callable

import jax.numpy as jnp
import jax.tree_util as jtu
from diffrax import AbstractTerm, ODETerm
from diffrax._term import WrapTerm
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Bool, PyTree


class SemiExplicitDAETerm(AbstractTerm):
    """A semi-explicit differential algebraic equation, for
    [`diffrax_bdf.BDF`][].

    Where `algebraic_mask` is false, the vector field gives derivatives,
    `dy/dt = f(t, y)`; where it is true, it gives constraints, `0 = g(t, y)`.
    The constraints must have index 1: their Jacobian with respect to the
    algebraic components must be nonsingular.

    Only `BDF` can solve this term. Other diffrax solvers would integrate the
    constraints as if they were derivatives, so they raise an error instead.
    """

    vector_field: Callable
    algebraic_mask: PyTree

    def _unsupported(self, *args, **kwargs):
        raise TypeError(
            "A `SemiExplicitDAETerm` can only be solved with `diffrax_bdf.BDF`, "
            "which treats its algebraic components as constraints. Other solvers "
            "would integrate them as if they were derivatives."
        )

    def vf(self, t, y, args):
        return self.vector_field(t, y, args)

    vf_prod = _unsupported
    prod = _unsupported

    def contr(self, t0, t1, **kwargs):
        return t1 - t0


SemiExplicitDAETerm.__init__.__doc__ = """**Arguments:**

- `vector_field`: a function `(t, y, args) -> f` returning a pytree with the
    structure of `y`.
- `algebraic_mask`: which components of `y` are algebraic. Each leaf is
    broadcast against the matching leaf of `y`, so `(False, True)` marks the
    second element of `y = (z, u)` as algebraic.
"""


def split_dae_term(terms) -> tuple[AbstractTerm, PyTree | None]:
    """Return an ODE view of `terms` and its algebraic mask, or `None` for an ODE
    term."""
    if isinstance(terms, SemiExplicitDAETerm):
        return ODETerm(terms.vector_field), terms.algebraic_mask
    if isinstance(terms, WrapTerm) and isinstance(terms.term, SemiExplicitDAETerm):
        ode_term = ODETerm(terms.term.vector_field)
        return WrapTerm(ode_term, terms.direction), terms.term.algebraic_mask
    return terms, None


def flat_algebraic_mask(mask: PyTree, y: PyTree) -> Bool[Array, " n"]:
    """Broadcast the mask against `y` and flatten it in `ravel_pytree`'s order."""
    broadcast = jtu.tree_map(
        lambda m, leaf: jnp.broadcast_to(jnp.asarray(m, bool), jnp.shape(leaf)),
        mask,
        y,
    )
    return ravel_pytree(broadcast)[0]
