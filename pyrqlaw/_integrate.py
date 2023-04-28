"""
integration
"""

import numpy as np
from numba import njit

@njit
def eom_mee_with_a_gauss(t, state, param):
    """Equations of motion for gauss with MEE"""
    # unpack parameters
    mu, u, psi_c0, psi_c1, psi_c2 = param

    # unpack elements
    sma,f,g,h,k,l = state

    # compute additional parameters
    ecc = np.sqrt(f**2 + g**2)
    p = sma*(1 - ecc**2)

    # Gauss perturbation
    sinL = np.sin(l)
    cosL = np.cos(l)
    w = 1 + f*cosL + g*sinL
    psi = np.array([   
        # [2*sma**2*ecc*np.sin(ta) / ang_mom, (2*sma**2*p) / (r*ang_mom), 0.0],
        # [ sqrt_pmu*sinL, sqrt_pmu/w*((w+1)*cosL + f), -g/w*sqrt_pmu*(h*sinL - k*cosL)],
        # [-sqrt_pmu*cosL, sqrt_pmu/w*((w+1)*sinL + g),  f/w*sqrt_pmu*(h*sinL - k*cosL)],
        # [0.0, 0.0, sqrt_pmu/w * 0.5*(1 + h**2 + k**2)*cosL],
        # [0.0, 0.0, sqrt_pmu/w * 0.5*(1 + h**2 + k**2)*sinL],
        # [0.0, 0.0., sqrt_pmu/w* (h*sinL - k*cosL)]
        [psi_c0[0], psi_c1[0], psi_c2[0]],
        [psi_c0[1], psi_c1[1], psi_c2[1]],
        [psi_c0[2], psi_c1[2], psi_c2[2]],
        [psi_c0[3], psi_c1[3], psi_c2[3]],
        [psi_c0[4], psi_c1[4], psi_c2[4]],
        [psi_c0[5], psi_c1[5], psi_c2[5]]
    ])

    # combine
    doe = np.dot(psi,u) + np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(mu*p)*(w/p)**2])
    return doe

def integrate_next_step(integrator, eom, 
                        t, t_step, 
                        oe, oeT, 
                        ode_params, ode_paramsT, 
                        abs_tol, rel_tol, 
                    ):
    """Propagate chaser's and target's eoms over the next step"""
    # Propagate chaser's state
    oe_next, h_next = integrator(
        eom, 
        t,
        t_step,
        oe,
        ode_params,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    # Propagate target's state
    oeT_next, _ = integrator(
        eom, 
        t,
        t_step,
        oeT,
        ode_paramsT,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    return h_next, oe_next, oeT_next

@njit
def rk4(rhs, t, h, y, p, abs_tol=None, rel_tol=None):
    """Perform single-step Runge-Kutta 4th order
    
    Args:
        rhs (callable): ODE right-hand side expressions
        t (float): current time
        h (float): time-step
        y (np.array): current state-vector
        p (Real or tuple): additional parameters passed to `rhs`

    Returns:
        (np.array); state-vector at time t+h
    """
    k1 = h * rhs(t, y, p)
    k2 = h * rhs(t + 0.5 * h, y + 0.5 * k1, p)
    k3 = h * rhs(t + 0.5 * h, y + 0.5 * k2, p)
    k4 = h * rhs(t + h, y + k3, p)
    y_next = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
    h_next = h
    return y_next, h_next


@njit
def rkf45(rhs, t, h, y, p, abs_tol=1e-7, rel_tol=1e-9):
    """Perform single-step Runge-Kutta 4th order with 5th order step correction
    
    Args:
        rhs (callable): ODE right-hand side expressions
        t (float): current time
        h (float): time-step
        y (np.array): current state-vector
        p (Real or tuple): additional parameters passed to `rhs`
        abs_tol (float): absolute tolerance
        rel_tol (float): relative tolerance

    Returns:
        (np.array); state-vector at time t+h
    """
    k1 = h * rhs(t, y, p)
    k2 = h * rhs(t + 1/4 * h,   y + 1/4*h * k1, p)
    k3 = h * rhs(t + 3/8 * h,   y + 3/32*h*k1 + 9/32*h*k2, p)
    k4 = h * rhs(t + 12/13 * h, y + 1932/2197*h*k1 - 7200/2197*h*k2 + 7296/2197*h*k3, p)
    k5 = h * rhs(t + h,         y + 439/216*h*k1 - 8*h*k2 + 3680/513*h*k3 - 845/4104*h*k4, p)
    k6 = h * rhs(t + 1/2*h,     y - 8/27*h*k1 + 2*h*k2 - 3544/2565*h*k3 + 1859/4104*h*k4 - 11/40*h*k5, p)
    y_next = y + 25/216*k1 + 1408/2565*k3 + 2197/4101*k4 - 1/5*k5
    z_next = y + 16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6
    sc = abs_tol + rel_tol*np.maximum(np.abs(y), np.abs(y_next))
    error = np.linalg.norm((z_next - y_next)/sc)
    if error != 0:
        h_next = h*(0.38*np.sqrt(6)/error)**(1/5)
    else:
        h_next = h
    return y_next, h_next

@njit
def dopri5(rhs, t, h, y, p, abs_tol=1e-7, rel_tol=1e-9):
    """Perform single-step Runge-Kutta Dormand-Prince 5th order integration with 4th order step correction
    
    Args:
        rhs (callable): ODE right-hand side expressions
        t (float): current time
        h (float): time-step
        y (np.array): current state-vector
        p (Real or tuple): additional parameters passed to `rhs`
        abs_tol (float): absolute tolerance
        rel_tol (float): relative tolerance

    Returns:
        (np.array); state-vector at time t+h
    """
    k1 = h * rhs(t + 0    * h, y, p)
    k2 = h * rhs(t + 1/5  * h, y + 1/5        * h*k1, p)
    k3 = h * rhs(t + 3/10 * h, y + 3/40       * h*k1 + 9/40       * h*k2, p)
    k4 = h * rhs(t + 4/5  * h, y + 44/45      * h*k1 - 56/15      * h*k2 + 32/9       * h*k3, p)
    k5 = h * rhs(t + 8/9  * h, y + 19372/6561 * h*k1 - 25360/2187 * h*k2 + 64448/6561 * h*k3 - 212/729 * h*k4, p)
    k6 = h * rhs(t + 1    * h, y + 9017/3168  * h*k1 - 355/33     * h*k2 + 46732/5247 * h*k3 + 49/176  * h*k4 - 5103/18656 * h*k5, p)
    k7 = h * rhs(t + 1    * h, y + 35/384     * h*k1 + 0          * h*k2 + 500/1113   * h*k3 + 125/192 * h*k4 - 2187/6784  * h*k5 + 11/84 * h*k6, p)
    y_next = y + 35/384*k1     + 0*k2 + 500/1113*k3   + 125/192*k4 - 2187/6784*k5    + 11/84*k6    + 0*k7
    z_next = y + 5179/57600*k1 + 0*k2 + 7571/16695*k3 + 393/640*k4 - 92097/339200*k5 + 187/2100*k6 + 1/40*k7
    sc = abs_tol + rel_tol*np.maximum(np.abs(y), np.abs(y_next))
    error = np.linalg.norm((z_next - y_next)/sc)
    if error != 0:
        h_next = h*(0.38*np.sqrt(6)/error)**(1/5)
    else:
        h_next = h
    return y_next, h_next