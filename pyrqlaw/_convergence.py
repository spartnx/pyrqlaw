"""
Check convergence
"""

import numpy as np
from numba import njit


@njit
def check_convergence(oe, oeT, woe, wl, tol_oe, deltaL=0):
    """Stage 1 convergence: check convergence between oe[:5] and oeT[:5]"""
    check_array = np.zeros(6,)
    doe = np.abs(oeT - oe)  # FIXME is this ok? or should we be careful for angles?
    # print(doe)
    # print(doe/tol_oe)
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
        if deltaL < tol_oe[5]:
            check_array[5] = 1
    # if we don't care about the idx^th element, we just say it has converged
    else:
        check_array[5] = 1

    if np.sum(check_array)==6:
        return True
    else:
        return False

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