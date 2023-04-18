"""
Lyapunov feedback control
"""

import numpy as np
import numpy.linalg as la



def lyapunov_control_angles(
        fun_lyapunov_control,
        fun_eval_fdot, 
        fun_eval_gdot, 
        fun_eval_dfdoe, 
        fun_eval_dgdoe,
        mu, 
        accel, 
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
        fun_eval_fdot (callable): symbolic-generated f_dot_x
        fun_eval_gdot (callable): symbolic-generated g_dot_x
        fun_eval_dfdoe (callable): symbolic-generated partial derivatives of f_dot_x wrt oe
        fun_eval_dgdoe (callable): symbolic-generated partial derivatives of f_dot_x wrt oe 
        mu (float): gravitational parameter
        accel (float): thrust-acceleration
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
    l_range = np.linspace(0,2*np.pi,num=100,endpoint=False)
    oe_5 = oe[:5] # a, f, g, h, k

    # evaluate the max and maximizer of fdot_x for L in [0,2pi)
    fdot_x = np.array([fun_eval_fdot(mu, accel, np.concatenate((oe_5,[l]))) for l in l_range])
    fdot_xx = np.max(fdot_x)
    l_max_f = l_range[np.argmax(fdot_x)]

    # evaluate the max and maximizer of gdot_x for L in [0,2pi)
    gdot_x = np.array([fun_eval_gdot(mu, accel, np.concatenate((oe_5,[l]))) for l in l_range])
    gdot_xx = np.max(gdot_x)
    l_max_g = l_range[np.argmax(gdot_x)]

    # compute dfdoe_max, dgdoe_max
    dfdoe_max = fun_eval_dfdoe(mu, accel, np.concatenate((oe_5,[l_max_f])))
    dgdoe_max = fun_eval_dgdoe(mu, accel, np.concatenate((oe_5,[l_max_g])))

    # compute D1, D2, D3
    d_raw, psi = fun_lyapunov_control(
        mu, 
        accel, 
        oe, 
        oeT, 
        rpmin, m_petro, n_petro, 
        r_petro, k_petro,  
        wp, 
        woe,
        fdot_xx, 
        gdot_xx, 
        dfdoe_max, 
        dgdoe_max
    )

    # compute thrust angles
    d_float = np.array([float(el) for el in d_raw]) # D2, D1, D3 
    d_float_norm = np.sqrt( d_float[0]**2 + d_float[1]**2 + d_float[2]**2 )
    d = d_float/d_float_norm

    alpha = np.arctan2(-d[0],-d[1]) 
    beta = np.arctan(-d[2]/np.sqrt(d[0]**2 + d[1]**2))
    return alpha, beta, d_float, psi
