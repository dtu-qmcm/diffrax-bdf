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

Fixed-order (the order ramps from 1 to `bdf_order` over the opening steps), with
adaptive step size via diffrax's stock `PIDController`. Not yet implemented:
Jacobian reuse across steps, and variable order.

## Method

The formulation follows `scipy.integrate.BDF` and MATLAB's `ode15s`: the history is
a backward-difference array rescaled on each change of step size, with the Shampine
& Reichelt NDF correction available via `use_ndf` (`False` gives the classical BDF
coefficients, the family SUNDIALS CVODE implements).

Everything is written for static shapes — the order is a *traced* integer used to
mask fixed-size buffers — because diffrax carries the solver state through a
`lax.while_loop`.

## Caveats

- **Enable `jax_enable_x64`.** High-order backward differences suffer heavy
  cancellation; above order 2 they are noise in float32.
- **Set `dtmax` when integrating to `t1=inf`.** As the residual collapses the
  controller's growth factor saturates and the step grows without bound until it
  overflows the time variable.
- Requires a term whose control is a scalar, i.e. `ODETerm`.

## Performance

Against `Kvaerno5` on a 3-state stiff steady-state solve to `rtol=1e-8`, CPU,
float64. `BDF` currently computes one Jacobian per step, the same as `Kvaerno5`; it
wins by taking fewer and more reliable steps, not yet by reusing Jacobians.

| problem | solver | steps | rejected | wall |
|---|---|---|---|---|
| stiff linear | `Kvaerno5` | 127 | 0 | 3.4 ms |
| stiff linear | `BDF(3)` | 596 | 10 | 4.9 ms |
| stiff linear | `BDF(4)` | 396 | 12 | 3.6 ms |
| stiff nonlinear | `Kvaerno5` | 4576 | 1570 | 114.8 ms |
| stiff nonlinear | `BDF(3)` | 451 | 9 | 3.7 ms |
| stiff nonlinear | `BDF(4)` | 310 | 10 | 2.6 ms |

The nonlinear case is the representative one: `Kvaerno5` rejects a third of its
steps, where BDF rejects about 2%. On the easy linear problem `Kvaerno5` needs far
fewer steps and the two are comparable in wall time.

## Testing

`uv run pytest`. scipy's BDF is pure Python and is used as a white-box oracle:
the coefficient tables, the difference-array rescaling, and a single pinned step are
all asserted against it directly.
