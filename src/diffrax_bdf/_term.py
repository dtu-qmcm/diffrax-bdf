from collections.abc import Callable

import jax.numpy as jnp
import jax.tree_util as jtu
from diffrax import AbstractTerm, ODETerm
from diffrax._term import WrapTerm
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Bool, PyTree


class SemiExplicitDAETerm(AbstractTerm):
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


def split_dae_term(terms) -> tuple[AbstractTerm, PyTree | None]:
    if isinstance(terms, SemiExplicitDAETerm):
        return ODETerm(terms.vector_field), terms.algebraic_mask
    if isinstance(terms, WrapTerm) and isinstance(terms.term, SemiExplicitDAETerm):
        ode_term = ODETerm(terms.term.vector_field)
        return WrapTerm(ode_term, terms.direction), terms.term.algebraic_mask
    return terms, None


def flat_algebraic_mask(mask: PyTree, y: PyTree) -> Bool[Array, " n"]:
    broadcast = jtu.tree_map(
        lambda m, leaf: jnp.broadcast_to(jnp.asarray(m, bool), jnp.shape(leaf)),
        mask,
        y,
    )
    return ravel_pytree(broadcast)[0]
