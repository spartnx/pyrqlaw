"""
Lyapunov feedback control
"""

import numpy as np
import numpy.linalg as la



def lyapunov_control_angles(
        fun_lyapunov_control,
        mu, 
        f, 
        oe, 
        oeT, 
        rpmin, m_petro, n_petro, 
        r_petro, k_petro, 
        wp, 
        woe,
    ):
    """Compute thrust angles from Lyapunov feedback control law
    
    Args:
        fun_lyapunov_control (callable): symbolic-generated Lyapunov control law
        mu (float): gravitational parameter
        f (float): thrust-acceleration
        oe (np.array): current osculating elements
        oeT (np.array): target osculating elements
        rpmin (float): minimum periapsis
        k_petro (float): scalar factor k on minimum periapsis
        m_petro (float): scalar factor m to prevent non-convergence, default is 3.0
        n_petro (float): scalar factor n to prevent non-convergence, default is 4.0
        r_petro (float): scalar factor r to prevent non-convergence, default is 2.0
        wp (float): penalty scalar on minimum periapsis, default is 1.0
        woe (np.array): weight on each osculating element
    
    Returns:
        (tuple): alpha, beta, vector u, list of columns of psi
    """
    # compute u direction
    d_raw, psi = fun_lyapunov_control(
        mu, 
        f, 
        oe, 
        oeT, 
        rpmin, m_petro, n_petro, 
        r_petro, k_petro, 
        wp, 
        woe
    )

    # compute thrust angles
    d_float = np.array([float(el) for el in d_raw]) # D2, D1, D3 
    d_float_norm = np.sqrt( d_float[0]**2 + d_float[1]**2 + d_float[2]**2 )
    d = d_float/d_float_norm

    alpha = np.arctan2(-d[0],-d[1]) 
    beta = np.arctan(-d[2]/np.sqrt(d[0]**2 + d[1]**2))
    return alpha, beta, d_float, psi
