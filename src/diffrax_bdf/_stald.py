# Ported from cvSLdet in SUNDIALS CVODE (src/cvode/cvode.c, v7.1.1).
# Copyright (c) 2002-2024, Lawrence Livermore National Security
# and Southern Methodist University. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSES/SUNDIALS-LICENSE and LICENSES/SUNDIALS-NOTICE.

"""Stability limit detection for BDF, ported from SUNDIALS CVODE's `cvSLdet`.

At orders 3 and above, a BDF's step size can be held down by stability rather than
accuracy: a mode near the imaginary axis grows under the formula unless the step
stays small. `stability_limit_flag` looks for this in the last five steps' scaled
derivative norms. It estimates the dominant characteristic root of their growth,
then checks it against a relation that the three norms must satisfy.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

TINY = 1.0e-10
RR_CUT = 0.98
VRR_TOL = 1.0e-4
VRR_T2 = 5.0e-4
SQ_TOL = 1.0e-3
RR_TOL = 1.0e-2
N_NEWTON = 3
N_HISTORY = 5


def _divide(numerator, denominator):
    return numerator / jnp.where(denominator == 0, 1.0, denominator)


def _quartic(qc, rr):
    qc1, qc2, qc4, qc5 = qc
    return qc5 + rr * (qc4 + rr * rr * (qc2 + rr * qc1))


def stability_limit_flag(
    history: Float[Array, "5 3"], order: Int[Array, ""]
) -> tuple[Int[Array, ""], Float[Array, ""]]:
    """Return CVODE's flag and its estimate of the dominant characteristic root.

    `history[i, k]` holds the squared weighted norm of component `k` at the `i`-th
    most recent step, where components 0, 1 and 2 are the scaled derivatives of
    orders `order - 1`, `order` and `order + 1`. Flags 4, 5 and 6 mean the step is
    limited by stability; 1, 2 and 3 mean a stable root was found; negative flags
    mean no root could be estimated, for the reasons `cvSLdet` lists.
    """
    history = jax.lax.stop_gradient(history)
    s = [history[i] for i in range(N_HISTORY)]
    s_min = history.min(axis=0)
    s_max = history.max(axis=0)
    ss_max = s_max * s_max

    ratios = jnp.stack([_divide(s[i], s[i + 1]) for i in range(N_HISTORY - 1)])
    rav = ratios.mean(axis=0)
    vrat = jnp.abs((ratios * ratios).mean(axis=0) - rav * rav)

    qc5 = s[0] * s[2] - s[1] * s[1]
    qc4 = s[1] * s[2] - s[0] * s[3]
    qc2 = s[1] * s[4] - s[2] * s[3]
    qc1 = s[3] * s[3] - s[2] * s[4]
    qc = (qc1, qc2, qc4, qc5)

    flag = jnp.asarray(0, jnp.int32)
    done = jnp.asarray(False)

    def settle(flag, done, condition, value):
        hit = jnp.invert(done) & condition
        return jnp.where(hit, value, flag), done | hit

    flag, done = settle(flag, done, jnp.any(s_min < TINY * s_max), -1)

    normal = jnp.min(vrat) < VRR_TOL * VRR_TOL
    rr_normal = rav.mean()
    flag, done = settle(
        flag, done, normal & (jnp.max(vrat) > VRR_T2 * VRR_T2), -2
    )
    flag, done = settle(
        flag, done, normal & (jnp.max(jnp.abs(rav - rr_normal)) > VRR_T2), -3
    )
    flag, done = settle(flag, done, normal, 1)

    quartic = jnp.invert(normal)
    qco1, qco2, qco3, qco4, qco5 = (
        qc1,
        qc2,
        jnp.zeros_like(qc1),
        qc4,
        qc5,
    )
    flag, done = settle(
        flag, done, quartic & (jnp.abs(qco1[0]) < TINY * ss_max[0]), -4
    )
    rows = [qco2, qco3, qco4, qco5]
    for k in (1, 2):
        tem = _divide(qco1[k], qco1[0])
        rows = [row.at[k].add(-tem * row[0]) for row in rows]
    qco2, qco3, qco4, qco5 = rows
    flag, done = settle(
        flag, done, quartic & (jnp.abs(qco2[1]) < TINY * ss_max[1]), -4
    )
    tem = _divide(qco2[2], qco2[1])
    qco3, qco4, qco5 = (row.at[2].add(-tem * row[1]) for row in (qco3, qco4, qco5))
    flag, done = settle(
        flag, done, quartic & (jnp.abs(qco4[2]) < TINY * ss_max[2]), -4
    )
    rr_quartic = -_divide(qco5[2], qco4[2])
    flag, done = settle(
        flag, done, quartic & ((rr_quartic < TINY) | (rr_quartic > 100.0)), -5
    )
    qkr = _quartic(qc, rr_quartic)
    flag, done = settle(
        flag, done, quartic & (jnp.max(jnp.abs(qkr) / ss_max) < SQ_TOL), 2
    )

    rr_newton = rr_quartic
    newton_done = jnp.asarray(False)
    sq_min = jnp.asarray(jnp.inf, history.dtype)
    for _ in range(N_NEWTON):
        qp = qc4 + rr_newton * rr_newton * (3.0 * qc2 + rr_newton * 4.0 * qc1)
        drr = jnp.where(jnp.abs(qp) > TINY * ss_max, -_divide(qkr, qp), 0.0)
        candidates = rr_newton + drr
        values = jax.vmap(lambda r: _quartic(qc, r))(candidates)
        sq_max = jnp.max(jnp.abs(values) / ss_max, axis=1)
        best = jnp.argmin(sq_max)
        step_rr = candidates[best]
        step_sq = sq_max[best]
        rr_newton = jnp.where(newton_done, rr_newton, step_rr)
        sq_min = jnp.where(newton_done, sq_min, step_sq)
        qkr = jnp.where(newton_done, qkr, values[best])
        newton_done = newton_done | (step_sq < SQ_TOL)
    flag, done = settle(flag, done, quartic & newton_done, 3)
    flag, done = settle(flag, done, quartic, -6)

    rr = jnp.where(normal, rr_normal, jnp.where(flag == 2, rr_quartic, rr_newton))
    found = flag > 0

    rsa = s[0]
    rsb = s[1] * rr
    rsc = s[2] * rr * rr
    rsd = s[3] * rr * rr * rr
    rd1a = rsa - rsb
    rd1b = rsb - rsc
    rd1c = rsc - rsd
    rd2a = rd1a - rd1b
    rd2b = rd1b - rd1c
    rd3a = rd2a - rd2b
    cest1 = -_divide(rd3a, rd1b)
    invalid = (
        (jnp.abs(rd1b) < TINY * s_max) | (cest1 < TINY) | (cest1 > 4.0)
    )
    verified = found & jnp.invert(jnp.any(invalid))
    flag = jnp.where(found & jnp.any(invalid), -7, flag)
    corr1 = _divide(_divide(rd2b, cest1), rr * rr)
    sigsq = s[2] + corr1

    flag = jnp.where(verified & (sigsq[1] < TINY), -8, flag)
    verified = verified & (sigsq[1] >= TINY)
    ratp = _divide(sigsq[2], sigsq[1])
    ratm = _divide(sigsq[0], sigsq[1])
    q = order.astype(history.dtype)
    qfac1 = 0.25 * (q * q - 1.0)
    qfac2 = _divide(2.0, q - 1.0)
    bb = ratp * ratm - 1.0 - qfac1 * ratp
    tem = 1.0 - qfac2 * bb
    flag = jnp.where(verified & (jnp.abs(tem) < TINY), -8, flag)
    verified = verified & (jnp.abs(tem) >= TINY)
    rrb = _divide(1.0, tem)
    flag = jnp.where(verified & (jnp.abs(rrb - rr) > RR_TOL), -9, flag)
    verified = verified & (jnp.abs(rrb - rr) <= RR_TOL)

    flag = jnp.where(verified & (rr > RR_CUT), flag + 3, flag)
    return flag.astype(jnp.int32), rr
