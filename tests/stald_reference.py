# Ported from cvSLdet in SUNDIALS CVODE (src/cvode/cvode.c, v7.1.1).
# Copyright (c) 2002-2024, Lawrence Livermore National Security
# and Southern Methodist University. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSES/SUNDIALS-LICENSE and LICENSES/SUNDIALS-NOTICE.

"""A line-by-line NumPy port of SUNDIALS CVODE's `cvSLdet`, kept as the reference
for the JAX version. It keeps CVODE's one-based indices so that it can be read
against the C source.
"""

import numpy as np

TINY = 1.0e-10


def sldet(ssdat, q):
    """Return `(kflag, rr)`, as SUNDIALS CVODE's `cvSLdet`.

    `ssdat[i][k]` uses CVODE's one-based indices: `i` runs over the last five
    steps, newest first, and `k` over the three components.
    """
    ssdat = np.asarray(ssdat, dtype=float)
    rat = np.zeros((5, 4))
    rav = np.zeros(4)
    qkr = np.zeros(4)
    sigsq = np.zeros(4)
    smax = np.zeros(4)
    ssmax = np.zeros(4)
    drr = np.zeros(4)
    rrc = np.zeros(4)
    sqmx = np.zeros(4)
    qjk = np.zeros((4, 4))
    vrat = np.zeros(5)
    qc = np.zeros((6, 4))
    qco = np.zeros((6, 4))

    rrcut = 0.98
    vrrtol = 1.0e-4
    vrrt2 = 5.0e-4
    sqtol = 1.0e-3
    rrtol = 1.0e-2

    rr = 0.0
    kflag = 0
    kmin = 0

    for k in range(1, 4):
        smink = ssdat[1][k]
        smaxk = 0.0
        for i in range(1, 6):
            smink = min(smink, ssdat[i][k])
            smaxk = max(smaxk, ssdat[i][k])
        if smink < TINY * smaxk:
            return -1, rr
        smax[k] = smaxk
        ssmax[k] = smaxk * smaxk
        sumrat = 0.0
        sumrsq = 0.0
        for i in range(1, 5):
            rat[i][k] = ssdat[i][k] / ssdat[i + 1][k]
            sumrat = sumrat + rat[i][k]
            sumrsq = sumrsq + rat[i][k] * rat[i][k]
        rav[k] = 0.25 * sumrat
        vrat[k] = abs(0.25 * sumrsq - rav[k] * rav[k])
        qc[5][k] = ssdat[1][k] * ssdat[3][k] - ssdat[2][k] * ssdat[2][k]
        qc[4][k] = ssdat[2][k] * ssdat[3][k] - ssdat[1][k] * ssdat[4][k]
        qc[3][k] = 0.0
        qc[2][k] = ssdat[2][k] * ssdat[5][k] - ssdat[3][k] * ssdat[4][k]
        qc[1][k] = ssdat[4][k] * ssdat[4][k] - ssdat[3][k] * ssdat[5][k]
        for i in range(1, 6):
            qco[i][k] = qc[i][k]

    vmin = min(vrat[1], vrat[2], vrat[3])
    vmax = max(vrat[1], vrat[2], vrat[3])

    if vmin < vrrtol * vrrtol:
        if vmax > vrrt2 * vrrt2:
            return -2, rr
        rr = (rav[1] + rav[2] + rav[3]) / 3.0
        drrmax = 0.0
        for k in range(1, 4):
            drrmax = max(drrmax, abs(rav[k] - rr))
        if drrmax > vrrt2:
            return -3, rr
        kflag = 1
    else:
        if abs(qco[1][1]) < TINY * ssmax[1]:
            return -4, rr
        tem = qco[1][2] / qco[1][1]
        for i in range(2, 6):
            qco[i][2] = qco[i][2] - tem * qco[i][1]
        qco[1][2] = 0.0
        tem = qco[1][3] / qco[1][1]
        for i in range(2, 6):
            qco[i][3] = qco[i][3] - tem * qco[i][1]
        qco[1][3] = 0.0
        if abs(qco[2][2]) < TINY * ssmax[2]:
            return -4, rr
        tem = qco[2][3] / qco[2][2]
        for i in range(3, 6):
            qco[i][3] = qco[i][3] - tem * qco[i][2]
        if abs(qco[4][3]) < TINY * ssmax[3]:
            return -4, rr
        rr = -qco[5][3] / qco[4][3]
        if rr < TINY or rr > 100.0:
            return -5, rr
        for k in range(1, 4):
            qkr[k] = qc[5][k] + rr * (qc[4][k] + rr * rr * (qc[2][k] + rr * qc[1][k]))
        sqmax = 0.0
        for k in range(1, 4):
            saqk = abs(qkr[k]) / ssmax[k]
            if saqk > sqmax:
                sqmax = saqk
        if sqmax < sqtol:
            kflag = 2
        else:
            sqmin = 0.0
            for _ in range(1, 4):
                for k in range(1, 4):
                    qp = qc[4][k] + rr * rr * (3.0 * qc[2][k] + rr * 4.0 * qc[1][k])
                    drr[k] = 0.0
                    if abs(qp) > TINY * ssmax[k]:
                        drr[k] = -qkr[k] / qp
                    rrc[k] = rr + drr[k]
                for k in range(1, 4):
                    s = rrc[k]
                    sqmaxk = 0.0
                    for j in range(1, 4):
                        qjk[j][k] = qc[5][j] + s * (
                            qc[4][j] + s * s * (qc[2][j] + s * qc[1][j])
                        )
                        saqj = abs(qjk[j][k]) / ssmax[j]
                        if saqj > sqmaxk:
                            sqmaxk = saqj
                    sqmx[k] = sqmaxk
                sqmin = sqmx[1] + 1.0
                for k in range(1, 4):
                    if sqmx[k] < sqmin:
                        kmin = k
                        sqmin = sqmx[k]
                rr = rrc[kmin]
                if sqmin < sqtol:
                    kflag = 3
                    break
                for j in range(1, 4):
                    qkr[j] = qjk[j][kmin]
            if sqmin > sqtol:
                return -6, rr

    for k in range(1, 4):
        rsa = ssdat[1][k]
        rsb = ssdat[2][k] * rr
        rsc = ssdat[3][k] * rr * rr
        rsd = ssdat[4][k] * rr * rr * rr
        rd1a = rsa - rsb
        rd1b = rsb - rsc
        rd1c = rsc - rsd
        rd2a = rd1a - rd1b
        rd2b = rd1b - rd1c
        rd3a = rd2a - rd2b
        if abs(rd1b) < TINY * smax[k]:
            return -7, rr
        cest1 = -rd3a / rd1b
        if cest1 < TINY or cest1 > 4.0:
            return -7, rr
        corr1 = (rd2b / cest1) / (rr * rr)
        sigsq[k] = ssdat[3][k] + corr1

    if sigsq[2] < TINY:
        return -8, rr

    ratp = sigsq[3] / sigsq[2]
    ratm = sigsq[1] / sigsq[2]
    qfac1 = 0.25 * (q * q - 1.0)
    qfac2 = 2.0 / (q - 1.0)
    bb = ratp * ratm - 1.0 - qfac1 * ratp
    tem = 1.0 - qfac2 * bb
    if abs(tem) < TINY:
        return -8, rr
    rrb = 1.0 / tem
    if abs(rrb - rr) > rrtol:
        return -9, rr

    if rr > rrcut:
        kflag = kflag + 3
    return kflag, rr
