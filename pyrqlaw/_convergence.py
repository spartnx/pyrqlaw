"""
Check convergence
"""

import numpy as np
from numba import njit
from ._elements import mee_with_a2sv


@njit
def check_convergence_oe(oe, oeT, woe, wl, tol_oe, deltaL=0):
    """Stage 1 convergence: check convergence between oe and oeT"""
    check_array = np.zeros(6,)
    doe = np.abs(oeT - oe)  # FIXME is this ok? or should we be careful for angles?
    # check convergence for each slow element: a, f, g, h, k
    for idx in range(5):
        if woe[idx] > 0.0:
            if doe[idx] < tol_oe[idx]:
                check_array[idx] = 1
        # if we don't care about the idx^th element, we just say it has converged
        else:
            check_array[idx] = 1

    # check convergence for the fast variable: true longitude L
    if wl > 0:
        if abs(deltaL) < tol_oe[5]:
            check_array[5] = 1
    # if we don't care about the idx^th element, we just say it has converged
    else:
        check_array[5] = 1

    if np.sum(check_array)==6:
        return True
    else:
        return False

def check_convergence_sv(oe, oeT, tol_sv, mu):
    """Check convergence between the chaser's and target's 
    orbital state vectors (position and velocity)"""
    # convert elements to state vector [r,v]
    sv = mee_with_a2sv(oe, mu)
    svT = mee_with_a2sv(oeT, mu)
    # retrieve posiiton and velocity vectors
    r = sv[:3]
    rT = svT[:3]
    v = sv[3:6]
    vT = svT[3:6]
    # check convergence
    norms = np.array([np.linalg.norm(r-rT), np.linalg.norm(v-vT)])
    if (norms < tol_sv).all():
        return True
    else:
        return False

def check_convergence_q(tol_q, eval_q, eval_lmax, eval_fdot, eval_gdot, eval_dfdoe, eval_dgdoe, 
                        mu, accel_thrust, oe_iter, oeT_iter, 
                        rpmin, m_petro1, n_petro1, r_petro1,
                        k_petro1, wp1, woe1, wl1, wscl1, l_mesh, deltaL=0):
    """Check convergence of the Lyapunov function to below some tolerance."""
    lmax_f, lmax_g = eval_lmax(mu, accel_thrust, oe_iter, eval_fdot, eval_gdot, l_mesh=l_mesh)
    fdot_xx = eval_fdot(mu, accel_thrust, np.concatenate((oe_iter[:5],[lmax_f])))
    gdot_xx = eval_gdot(mu, accel_thrust, np.concatenate((oe_iter[:5],[lmax_g])))
    dfdoe_max = eval_dfdoe(mu, accel_thrust, np.concatenate((oe_iter[:5],[lmax_f])))
    dgdoe_max = eval_dgdoe(mu, accel_thrust, np.concatenate((oe_iter[:5],[lmax_g])))

    q_val = accel_thrust**2 * eval_q(mu, accel_thrust, oe_iter, oeT_iter,
                                        rpmin, m_petro1, n_petro1, r_petro1,
                                        k_petro1, wp1, woe1, fdot_xx, gdot_xx, 
                                        dfdoe_max, dgdoe_max, wl1, wscl1, deltaL)
    return q_val < tol_q

@njit
def elements_safety(oe, oe_min, oe_max):
    """Ensure osculating elements stays within bounds
    
    Args:
        oe (np.array): current osculating elements
        oe_min (np.array): minimum values for each osculating element
        oe_max (np.array): maximum values for each osculating element

    Returns:
        (np.array): "cleaned" osculating elements
    """
    oe_clean = oe
    for idx in range(6):
        oe_clean[idx] = min(max(oe_min[idx], oe[idx]), oe_max[idx])
    return oe_clean