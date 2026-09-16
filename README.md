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
from diffrax_bdf import BDF, BDFController

sol = diffrax.diffeqsolve(
    diffrax.ODETerm(vector_field),
    BDF(),                       # variable order 1-5, as scipy and CVODE do
    t0=0.0,
    t1=jnp.inf,
    dt0=1e-6,
    y0=guess,
    args=parameters,
    stepsize_controller=BDFController(rtol=1e-8, atol=1e-10, dtmax=1e6),
    event=diffrax.Event(diffrax.steady_state_event()),
    adjoint=diffrax.ImplicitAdjoint(),
    saveat=diffrax.SaveAt(t1=True),
)
```

Pin the order with `BDF(bdf_order=3)` to use any adaptive step size controller,
including diffrax's own `PIDController`.

## Status

Variable order 1-5, adaptive step size, and reuse of the Jacobian and its
factorisation across steps.

## Method

The formulation follows `scipy.integrate.BDF` and MATLAB's `ode15s`: the history is
a backward-difference array rescaled on each change of step size, with the Shampine
& Reichelt NDF correction available via `use_ndf` (`False` gives the classical BDF
coefficients, the family SUNDIALS CVODE implements). The order is chosen each step
by scipy's heuristic, comparing the error estimate at the current order against what
it would have been one order lower and one order higher.

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

It also carries the order. diffrax asks a solver for its error order once, before
the solve (`solver.error_order(terms)` sees neither state nor time), so a
variable-order solver cannot report a per-step order through that route. The only
other channel to the controller is `y_error`, which the integration loop treats
opaquely, so the order travels inside it.

Everything is written for static shapes -- the order is a *traced* integer used to
mask fixed-size buffers -- because diffrax carries the solver state through a
`lax.while_loop`.

## Order selection on a steady-state solve

Integrating the `n=3` problem below to steady state, the order climbs through the
transient and then falls back as the residual collapses and the step grows:

```
order:  1 2 2 3 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 4 4 4 4 4 4 2 1 1 1 1 1 1 1
step:   1.2e-07  ------------------------------------------------>  7.9e+05
```

The tail is the useful part: BDF1 with an enormous step *is* Newton's method on
`f(y) = 0`, so a correct variable-order BDF turns into a steady-state solver by
itself as it converges.

## Caveats

- **Enable `jax_enable_x64`.** High-order backward differences suffer heavy
  cancellation; above order 2 they are noise in float32.
- **Set `dtmax` when integrating to `t1=inf`.** As the residual collapses the
  controller's growth factor saturates and the step grows without bound until it
  overflows the time variable.
- Variable order requires `BDFController`; a fixed `bdf_order` works with any
  adaptive controller.
- The Jacobian is formed densely, so this suits small to medium systems.
- Requires a term whose control is a scalar, i.e. `ODETerm`.

## Performance

Steady-state solves to `rtol=1e-8`, CPU, float64, medians of interleaved rounds.
Compare within a block, not across them -- absolute timings drift between runs.

**Whether BDF beats `Kvaerno5` depends entirely on the problem.** Where `Kvaerno5`'s
step control struggles, BDF wins by a wide margin; where it does not, `Kvaerno5`
wins on step count.

| problem | solver | steps | rejected | median |
|---|---|---|---|---|
| fast equilibrium, n=3 | `Kvaerno5` | 7784 | 4563 | 192.4 ms |
| fast equilibrium, n=3 | `BDF(bdf_order=5)` | 276 | 14 | 3.5 ms |
| fast equilibrium, n=3 | `BDF()` variable | 231 | 6 | **3.3 ms** |
| stiff random network, n=18 | `Kvaerno5` | 212 | 14 | **23.6 ms** |
| stiff random network, n=18 | `BDF(bdf_order=5)` | 593 | 11 | 28.8 ms |
| stiff random network, n=18 | `BDF()` variable | 589 | 12 | 31.7 ms |
| stiff random network, n=60 | `Kvaerno5` | 193 | 7 | **73.6 ms** |
| stiff random network, n=60 | `BDF(bdf_order=5)` | 586 | 6 | **145.2 ms** |
| stiff random network, n=60 | `BDF()` variable | 580 | 7 | 153.7 ms |

The first problem is the interesting one: `Kvaerno5` rejects 59% of its steps, where
BDF rejects 2-5%.

Variable order helps where the order genuinely needs to move -- 231 steps against
276 for the best fixed order on the steady-state problem. On the random networks,
where order 5 is right throughout, it matches fixed order 5 on step count and pays
about 8% per step for the heuristic. Reuse is worth a further 16-18%.

## Testing

`uv run pytest`. scipy's BDF is pure Python and is used as a white-box oracle: the
coefficient tables, the difference-array rescaling, and a single pinned step are all
asserted against it directly. The reuse policy, the deadband and the order-selection
heuristic are tested directly as well, since without them the solver has no
advantage over `Kvaerno5` at all.
