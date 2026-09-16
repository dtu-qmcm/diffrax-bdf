# diffrax-bdf

A [backward differentiation formula](https://en.wikipedia.org/wiki/Backward_differentiation_formula)
solver for [diffrax](https://github.com/patrick-kidger/diffrax), aimed at stiff
steady-state problems — in particular the kinetic reaction networks in
[enzax](https://github.com/dtu-qmcm/enzax).

diffrax has no BDF ([#8](https://github.com/patrick-kidger/diffrax/issues/8)). This
package supplies one as an ordinary `AbstractSolver` subclass, with no patching of
diffrax itself.

```python
import diffrax
import jax.numpy as jnp
from diffrax_bdf import BDF

sol = diffrax.diffeqsolve(
    diffrax.ODETerm(vector_field),
    BDF(bdf_order=3),
    t0=0.0,
    t1=jnp.inf,
    dt0=1e-6,
    y0=guess,
    args=parameters,
    stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10, dtmax=1e6),
    event=diffrax.Event(diffrax.steady_state_event()),
    adjoint=diffrax.ImplicitAdjoint(),
    saveat=diffrax.SaveAt(t1=True),
)
```

## Status

Fixed order (the order ramps from 1 up to `bdf_order` over the opening steps), with
adaptive step size and reuse of the Jacobian and its factorisation across steps. Not
yet implemented: variable order.

## Method

The formulation follows `scipy.integrate.BDF` and MATLAB's `ode15s`: the history is
a backward-difference array rescaled on each change of step size, with the Shampine
& Reichelt NDF correction available via `use_ndf` (`False` gives the classical BDF
coefficients, the family SUNDIALS CVODE implements).

The linear-algebra reuse policy is CVODE's, because that is where CVODE's advantage
over scipy actually lives. `Kvaerno5` already computes one Jacobian per step and
reuses it across its five stages, so "one Newton solve per step instead of five
stage solves" buys only a few linear solves. The real saving is keeping `J` and the
factorisation of `I - c * J` across *steps*, gated on two separate counters:

- the factorisation is rebuilt when `c` has moved by more than `max_gamma_change`
  (0.3), or after `lu_reuse_steps` (20) accepted steps;
- the Jacobian itself is rebuilt only after `jac_reuse_steps` (51) accepted steps,
  so a refactorisation reassembles `I - c * J` from the stored `J` without touching
  the vector field.

`BDFController` adds CVODE's step-size deadband on top of `diffrax.PIDController`:
an accepted step keeps its step size unless the controller wants to grow it by at
least 1.5x, and never shrinks it. Holding `h` still holds `c` still, which is what
lets the factorisation survive.

Everything is written for static shapes -- the order is a *traced* integer used to
mask fixed-size buffers -- because diffrax carries the solver state through a
`lax.while_loop`.

## Caveats

- **Enable `jax_enable_x64`.** High-order backward differences suffer heavy
  cancellation; above order 2 they are noise in float32.
- **Set `dtmax` when integrating to `t1=inf`.** As the residual collapses the
  controller's growth factor saturates and the step grows without bound until it
  overflows the time variable.
- The Jacobian is formed densely, so this suits small to medium systems.
- Requires a term whose control is a scalar, i.e. `ODETerm`.

## Performance

Steady-state solves to `rtol=1e-8`, CPU, float64, medians of interleaved rounds.

**Whether BDF beats `Kvaerno5` depends entirely on the problem.** Where `Kvaerno5`'s
step control struggles, BDF wins by a wide margin; where it does not, `Kvaerno5`'s
higher order wins.

| problem | solver | steps | rejected | median |
|---|---|---|---|---|
| fast equilibrium, n=3 | `Kvaerno5` | 7784 | 4563 | 186.8 ms |
| fast equilibrium, n=3 | `BDF(3)` | 414 | 8 | 5.1 ms |
| fast equilibrium, n=3 | `BDF(5)` | 276 | 14 | **3.4 ms** |
| stiff random network, n=18 | `Kvaerno5` | 212 | 14 | **35.4 ms** |
| stiff random network, n=18 | `BDF(5)` | 593 | 11 | 45.0 ms |
| stiff random network, n=60 | `Kvaerno5` | 193 | 7 | **72.0 ms** |
| stiff random network, n=60 | `BDF(5)` | 586 | 6 | 137.6 ms |

The first problem is the interesting one: `Kvaerno5` rejects 59% of its steps, where
BDF rejects 2-5%. On the well-conditioned random networks `Kvaerno5` needs 3x fewer
steps and wins despite costing more per step.

Reuse is worth a consistent 16-18%:

| problem | `BDF(5)` | with reuse disabled |
|---|---|---|
| stiff random network, n=18 | 45.0 ms | 53.4 ms |
| stiff random network, n=60 | 137.6 ms | 168.5 ms |

Order matters more than reuse does -- `BDF(5)` is roughly 2x faster than `BDF(3)` at
n=60 -- which is the argument for implementing variable order next.

## Testing

`uv run pytest`. scipy's BDF is pure Python and is used as a white-box oracle: the
coefficient tables, the difference-array rescaling, and a single pinned step are all
asserted against it directly. The reuse policy and the deadband are tested directly
as well, since without them the solver has no advantage over `Kvaerno5` at all.
