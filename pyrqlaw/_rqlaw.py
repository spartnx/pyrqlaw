"""
main functions for Q-Law transfers
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from ._symbolic import symbolic_rqlaw_mee_with_a
from ._lyapunov import lyapunov_control_angles, eval_lmax
from ._integrate import eom_mee_with_a_gauss, integrate_next_step, rk4, rkf45, dopri5
from ._convergence import check_convergence, elements_safety
from ._elements import (
    mee_with_a2sv, get_orbit_coordinates, mee_with_a2kep
)
from ._plot_helper import plot_sphere_wireframe, set_equal_axis


class RQLaw:
    """Object for Q-law based transfer problem.
    The overall procedure for using this class is:
    1. Create object via `prob = RQLaw()`
    2. Set problem parameters via `prob.set_problem()`
    3. solve problem via `prob.solve()`

    Exitcodes:
    0 : initial value (problem not yet attempted)
    1 : solved within tolerance
    2 : solved within relaxed tolerance
    -1 : mass is below threshold
    -2 : target elements could not be reached within allocated time
    -3 : thrust angles from feedback control law is nan

    Args:
        mu (float): gravitational parameter, default is 1.0
        rpmin (float): minimum periapsis
        k_petro (float): scalar factor k on minimum periapsis
        m_petro (float): scalar factor m to prevent non-convergence, default is 3.0
        n_petro (float): scalar factor n to prevent non-convergence, default is 4.0
        r_petro (float): scalar factor r to prevent non-convergence, default is 2.0
        wp (float): penalty scalar on minimum periapsis, default is 1.0
        integrator (str): "rk4", "rkf45", or "dopri5"
        verbosity (int): verbosity level for Q-law
        t_mesh (int): number of evaluation points along orbit for Q-law effectivity 
        tol_oe (np.array or None): tolerance on 5 elements targeted
        oe_min (np.array): minimum values of elements for safe-guarding
        oe_max (np.array): minimum values of elements for safe-guarding
        nan_angles_threshold (int): number of times to ignore `nan` thrust angles

    Attributes:
        print_frequency (int): if verbosity >= 2, prints at this frequency
    """
    def __init__(
        self, 
        mu=1.0,
        rpmin=0.5, 
        k_petro1=100, k_petro2=100,
        m_petro1=3, m_petro2=3,
        n_petro1=4, n_petro2=4,
        r_petro1=2, r_petro2=2,
        wp1=1, wp2=1,
        integrator="dopri5",
        t_mesh=20, l_mesh=100,
        oe_min=None, oe_max=None,
        tol_oe=None,
        nan_angles_threshold=10,
        verbosity=1,
        print_frequency=200,
        abs_tol=1e-7,
        rel_tol=1e-9
    ):
        """Construct RQLaw object"""
        # dynamics
        self.mu = mu

        # Q-law parameters
        self.rpmin = rpmin
        self.k_petro1 = k_petro1
        self.k_petro2 = k_petro2
        self.m_petro1 = m_petro1 
        self.m_petro2 = m_petro2
        self.n_petro1 = n_petro1 
        self.n_petro2 = n_petro2
        self.r_petro1 = r_petro1 
        self.r_petro2 = r_petro2
        self.wp1 = wp1
        self.wp2 = wp2
        self.t_mesh = t_mesh
        self.l_mesh = l_mesh

        # settings
        self.verbosity = verbosity

        # tolerance for convergence
        if tol_oe is None:
            self.tol_oe = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 3e-3])
        else:
            assert len(tol_oe)==6, "tol_oe must have 6 components"
            self.tol_oe = np.array(tol_oe)
        self.tol_oe_relaxed = 10*self.tol_oe  # relaxed tolerance -> why?
        self.exit_at_relaxed = 25

        # minimum bounds on elements
        if oe_min is None:
            self.oe_min = np.array([0.05, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        else:
            assert len(oe_min)==6, "oe_max must have 6 components" 
            self.oe_min = np.array(oe_min)

        # bounds on elements
        if oe_max is None:
            self.oe_max = np.array([1e2, 10.0, 10.0, np.inf, np.inf, np.inf])
        else:
            assert len(oe_max)==6, "oe_max must have 6 components" 
            self.oe_max = np.array(oe_max)

        # number of times to accept nan angles
        self.nan_angles_threshold = nan_angles_threshold

        # construct element names
        self.element_names = ["a", "f", "g", "h", "k", "L"]
        self.eom = eom_mee_with_a_gauss
        fun_lyapunov_control, fun_eval_psi, _, fun_eval_fdot, fun_eval_gdot, fun_eval_dfdoe, fun_eval_dgdoe, _, fun_eval_qdot = symbolic_rqlaw_mee_with_a()
        self.lyap_fun = fun_lyapunov_control
        self.psi_fun = fun_eval_psi
        self.eval_fdot = fun_eval_fdot
        self.eval_gdot = fun_eval_gdot
        self.eval_dfdoe = fun_eval_dfdoe
        self.eval_dgdoe = fun_eval_dgdoe
        self.eval_qdot = fun_eval_qdot

        # Integrator parameters
        self.step_min = 1e-4
        self.step_max = 2.0
        self.abs_tol = abs_tol # absolute tolerance
        self.rel_tol = rel_tol # relative tolerance
        if integrator == "rk4":
            self.integrator = rk4
        elif integrator == "rkf45":
            self.integrator = rkf45
        elif integrator == "dopri5":
            self.integrator = dopri5
        else:
            raise ValueError("integrator name invalid!")

        # print frequency
        self.print_frequency = print_frequency  

        # checks -> may need to modify/add to these attributes for RQ-Law
        self.ready = False
        self.converge = False
        self.exitcode = 0
        return
    

    def set_problem(
        self, 
        oe0, 
        oeT, 
        mass0, 
        tmax, 
        mdot, 
        tf_max, 
        t_step=0.1,
        mass_min=0.1,
        woe1=None,
        woe2=None,
        wl=0.06609, 
        wscl=3.3697
    ):
        """Set transfer problem
        
        Args:
            oe0 (np.array): chaser's initial state, in Keplerian elements (6 components)
            oeT (np.array): target's initial state, in Keplerian elements (6 components)
            mass0 (float): initial mass
            tmax (float): max thrust
            mdot (float): mass-flow rate
            tf_max (float): max time allocated to transfer
            t_step (float): initial time-step size to be used for integration
            mass_min (float): minimum mass
            woe (np.array): weight on each osculating element
        """
        assert len(oe0)==6, "oe0 must have 6 components"
        assert mass_min >= 1e-2, "mass should be above 0.01 to avoid numerical difficulties"
        assert len(oeT)==6, "oeT must have 6 components"
        # Stage 1 weight parameters
        if woe1 is None:
            self.woe1 = np.array([2, 50, 50, 1, 1])
        else:
            assert len(woe1)==5, "woe1 must have 5 components"
            self.woe1 = np.array(woe1)
        self.wl1 = 0
        self.wscl1 = 0

        # Stage 2 weight parameters
        if woe2 is None:
            self.woe2 = np.array([10, 1, 1, 1, 1])
        else:
            assert len(woe2)==5, "woe2 must have 5 components"
            self.woe2 = np.array(woe2)
        self.wl2 = wl
        self.wscl2 = wscl

        # time parameters
        self.tf_max = tf_max
        self.t_step = t_step
        # spacecraft parameters
        self.oe0 = oe0
        self.oeT = oeT 
        self.mass0 = mass0
        self.tmax  = tmax
        self.mdot  = mdot
        self.mass_min = mass_min
        self.ready = True  # toggle
        return


    def solve_stage1(self, eta_r=0.0):
        """Propagate and solve control problem for stage 1 (orbital transfer)
        ws = wscl = 0
        eta_r >= 0
        
        Args:
            eta_r (float): relative effectivity, `0.0 <= eta_r <= 1.0`
        """
        assert self.ready == True, "Please first call `set_problem()`"

        # efficiency thresholds
        self.eta_r = eta_r

        # initialize values for propagation
        t_iter = 0.0
        oe_iter = self.oe0
        oeT_iter = self.oeT
        mass_iter = self.mass0

        # initialize storage
        self.times1 = [t_iter,]
        self.states1 = [oe_iter,]
        self.statesT1 = [oeT_iter,]
        self.masses = [mass_iter,]
        self.controls1 = []
        n_nan_angles = 0

        if self.verbosity >= 2:
            header = " iter   |  time      |  del1       |  del2       |  del3       |  del4       |  del5       |  el6        |"
    
        # iterate until tf_max
        idx = 0
        while t_iter < self.tf_max:
            print("Stage 1 : " + str(round(t_iter/self.tf_max*100,2)))

            # ensure numerical stabilty
            oe_iter = elements_safety(oe_iter, self.oe_min, self.oe_max)

            # compute instantaneous acceleration magnitude due to thrust
            accel_thrust = self.tmax/mass_iter

            # evaluate Lyapunov function
            alpha, beta, _, psi = lyapunov_control_angles(
                fun_lyapunov_control=self.lyap_fun,
                fun_eval_fdot=self.eval_fdot, 
                fun_eval_gdot=self.eval_gdot, 
                fun_eval_dfdoe=self.eval_dfdoe, 
                fun_eval_dgdoe=self.eval_dgdoe,
                mu=self.mu, 
                accel=accel_thrust, 
                oe=oe_iter, 
                oeT=oeT_iter, 
                rpmin=self.rpmin, 
                m_petro=self.m_petro1, 
                n_petro=self.n_petro1, 
                r_petro=self.r_petro1, 
                k_petro=self.k_petro1, 
                wp=self.wp1, 
                woe=self.woe1,
                l_mesh=self.l_mesh,
                wl=self.wl1,
                wscl=self.wscl1
            )

            # ensure angles are not nan and otherwise compute thrust vector
            # nan values occur at singularities, i.e., when ecc = 0, ecc = 1, or inc = pi
            throttle = 1   # initialize
            if np.isnan(alpha) == True or np.isnan(beta) == True:
                alpha, beta = 0.0, 0.0
                throttle = 0  # turn off
                u = np.array([0.0,0.0,0.0])
                n_nan_angles += 1
                if n_nan_angles > self.nan_angles_threshold:
                    if self.verbosity > 0:
                        print("Breaking as angles are nan")
                    self.exitcode = -3
                    break
            else:
                u = accel_thrust*np.array([
                    np.cos(beta)*np.sin(alpha),
                    np.cos(beta)*np.cos(alpha),
                    np.sin(beta),
                ])

                # check effectivity to decide whether to thrust or coast
                if self.eta_r > 0: 
                    # Compute current qdot
                    lmax_f, lmax_g = eval_lmax(self.mu, accel_thrust, oe_iter, self.eval_fdot, self.eval_gdot, l_mesh=self.l_mesh)
                    fdot_xx = self.eval_fdot(self.mu, accel_thrust, np.concatenate((oe_iter[:5],[lmax_f])))
                    gdot_xx = self.eval_gdot(self.mu, accel_thrust, np.concatenate((oe_iter[:5],[lmax_g])))
                    dfdoe_max = self.eval_dfdoe(self.mu, accel_thrust, np.concatenate((oe_iter[:5],[lmax_f])))
                    dgdoe_max = self.eval_dgdoe(self.mu, accel_thrust, np.concatenate((oe_iter[:5],[lmax_g])))
                    qdot_current = self.eval_qdot(self.mu, accel_thrust, oe_iter, oeT_iter,
                                                    self.rpmin, self.m_petro1, self.n_petro1, self.r_petro1,
                                                    self.k_petro1, self.wp1, self.woe1, fdot_xx, gdot_xx, 
                                                    dfdoe_max, dgdoe_max, self.wl1, self.wscl1, u/accel_thrust, 0)
                    # compute max and min qdot along osculating orbit
                    qdot_min, qdot_max = self.evaluate_osculating_qdot( 
                        t_iter, oe_iter, oeT_iter, accel_thrust
                    )
                    val_eta_r = (qdot_current - qdot_max)/(qdot_min - qdot_max)
                    print("val_eta_r = " + str(val_eta_r))
                    print("qdot_current = " + str(qdot_current))
                    print("qdot_min = " + str(qdot_max))
                    print("qdot_max = " + str(qdot_min))
                    # turn thrust off if below threshold
                    if val_eta_r < self.eta_r:
                        throttle = 0  # turn off
                        u = np.zeros((3,))

            # ODE parameters
            ode_params = (self.mu, u, psi[0], psi[1], psi[2])
            psiT = self.psi_fun(self.mu, oeT_iter)
            ode_paramsT = (self.mu, np.zeros((3,)), psiT[0], psiT[1], psiT[2])
            h_next, oe_iter, oeT_iter = integrate_next_step(self.integrator, self.eom, 
                                                                t_iter, self.t_step, 
                                                                oe_iter, oeT_iter, 
                                                                ode_params, ode_paramsT, 
                                                                self.abs_tol, self.rel_tol, 
                                                            )
            t_iter += self.t_step  # update time
            mass_iter -= self.mdot*self.t_step*throttle
            self.t_step = max(self.step_min, min(self.step_max,h_next))
            print("l = " + str(oe_iter[5]))
            print("lT = " + str(oeT_iter[5]))
            print("deltaL = " + str(self.eval_deltaL(oe_iter[5], oeT_iter[5])))
                
            # check convergence
            if check_convergence(oe_iter, oeT_iter, self.woe1, self.wl1, self.tol_oe) == True:
                self.exitcode = 1
                self.converge = True
                break
            
            # print progress
            if self.verbosity >= 2 and np.mod(idx,self.print_frequency)==0:
                if np.mod(idx, 20*self.print_frequency) == 0:
                    print("\n" + header)
                t_fraction = t_iter/self.tf_max
                print(f"Stage 1: {idx:6.0f} | {t_fraction: 1.3e} | {oe_iter[0]-oeT_iter[0]: 1.4e} | {oe_iter[1]-oeT_iter[1]: 1.4e} | {oe_iter[2]-oeT_iter[2]: 1.4e} | {oe_iter[3]-oeT_iter[3]: 1.4e} | {oe_iter[4]-oeT_iter[4]: 1.4e} | {oe_iter[5]-oeT_iter[5]: 1.4e} |")

            # check if mass is below threshold
            if mass_iter <= self.mass_min:
                if self.verbosity > 0:
                    print("Breaking as mass is now under mass_min")
                self.exitcode = -1
                break

            # store
            self.times1.append(t_iter)
            self.states1.append(oe_iter) 
            self.statesT1.append(oeT_iter)
            self.masses.append(mass_iter)
            self.controls1.append([alpha, beta, throttle])

            # index update
            idx += 1

        # Storage over full timeline (stage 1 and stage 2)
        self.states = copy.deepcopy(self.states1)
        self.statesT = copy.deepcopy(self.statesT1)
        self.times = copy.deepcopy(self.times1)
        self.controls = copy.deepcopy(self.controls1)

        if self.converge == False:
            if self.verbosity > 0:
                print("Could not arrive to target elements within time")
            self.exitcode = -2
        else:
            if self.verbosity > 0:
                print("Target elements successfully reached!")
        return

    
    def evaluate_osculating_qdot(self, t0, oe, oeT, accel_thrust):
        """Evaluate Qdot over the entire orbit. For stage 1 only (since eta_r is set to 0 in stage 2).
        
        Args:
            t0 (float): current time
            oe (np.array): chaser's current osculating elements
            oeT (np.array): target's current osculating elements
            accel_thrust (float): magnitude of thrust acceleration at t0, tmax/mass

        Returns:
            (tuple): min and max Qdot
        """
        # Period of osculating orbit
        T = 2*np.pi*np.sqrt(oe[0]**3/self.mu)
        # points in time where to evaluate qdot
        eval_pts = np.linspace(t0, t0+T, self.t_mesh+1)
        qdot_list = []
        oe_iter = oe
        oeT_iter = oeT
        for time in eval_pts:
            t = t0
            while t < time:
                # propagate chaser's and target's elements without thrust force
                psi = self.psi_fun(self.mu, oe_iter)
                ode_params = (self.mu, np.zeros((3,)), psi[0], psi[1], psi[2])
                psiT = self.psi_fun(self.mu, oeT_iter)
                ode_paramsT = (self.mu, np.zeros((3,)), psiT[0], psiT[1], psiT[2])

                h_next, oe_iter, oeT_iter = integrate_next_step(self.integrator, self.eom, 
                                                                t, self.t_step, 
                                                                oe_iter, oeT_iter, 
                                                                ode_params, ode_paramsT, 
                                                                self.abs_tol, self.rel_tol, 
                                                            )
                t += self.t_step  # update time
                self.t_step = max(self.step_min, min(self.step_max,h_next))
            oe_test = oe_iter
            oeT_test = oeT_iter   

            # evaluate Lyapunov function
            alpha, beta, _, _ = lyapunov_control_angles(
                fun_lyapunov_control=self.lyap_fun,
                fun_eval_fdot=self.eval_fdot, 
                fun_eval_gdot=self.eval_gdot, 
                fun_eval_dfdoe=self.eval_dfdoe, 
                fun_eval_dgdoe=self.eval_dgdoe,
                mu=self.mu, 
                accel=accel_thrust, 
                oe=oe_test, 
                oeT=oeT_test,
                rpmin=self.rpmin, 
                m_petro=self.m_petro1, 
                n_petro=self.n_petro1, 
                r_petro=self.r_petro1, 
                k_petro=self.k_petro1, 
                wp=self.wp1, 
                woe=self.woe1,
                l_mesh=self.l_mesh,
                wl=self.wl1,
                wscl=self.wscl1,
            )
            u_test = accel_thrust*np.array([
                np.cos(beta)*np.sin(alpha), 
                np.cos(beta)*np.cos(alpha),
                np.sin(beta),
            ])

            # evaluate qdot
            lmax_f, lmax_g = eval_lmax(self.mu, accel_thrust, oe_test, self.eval_fdot, self.eval_gdot, l_mesh=self.l_mesh)
            fdot_xx = self.eval_fdot(self.mu, accel_thrust, np.concatenate((oe_test[:5],[lmax_f])))
            gdot_xx = self.eval_gdot(self.mu, accel_thrust, np.concatenate((oe_test[:5],[lmax_g])))
            dfdoe_max = self.eval_dfdoe(self.mu, accel_thrust, np.concatenate((oe_test[:5],[lmax_f])))
            dgdoe_max = self.eval_dgdoe(self.mu, accel_thrust, np.concatenate((oe_test[:5],[lmax_g])))
            qdot_test = self.eval_qdot(self.mu, accel_thrust, oe_test, oeT_test,
                                        self.rpmin, self.m_petro1, self.n_petro1, self.r_petro1,
                                        self.k_petro1, self.wp1, self.woe1, fdot_xx, gdot_xx, 
                                        dfdoe_max, dgdoe_max, self.wl1, self.wscl1, u_test/accel_thrust, 0)
            qdot_list.append(qdot_test)
        return min(qdot_list), max(qdot_list)

    def check_doe_stage2(self):
        return check_convergence(self.oe0, self.oeT, self.woe2, self.wl2, self.tol_oe)

    def solve_stage2(self):
        """Propagate and solve control problem for stage 2 (phasing)
        ws, wscl > 0
        eta_r = 0
        
        Args:
            eta_r (float): relative effectivity, `0.0 <= eta_r <= 1.0`
        """
        assert self.ready == True, "Please first call `set_problem()`"

        self.converge = False

        # initialize values for propagation
        t_iter = self.times1[-1]
        oe_iter = self.states1[-1]
        oeT_iter = self.statesT1[-1]
        mass_iter = self.masses[-1]

        # initialize storage
        n_nan_angles = 0
        self.states2 = [oe_iter,]
        self.statesT2 = [oeT_iter,]
        self.times2 = [t_iter,]
        self.controls2 = [self.controls[-1],]

        if self.verbosity >= 2:
            header = " iter   |  time      |  del1       |  del2       |  del3       |  del4       |  del5       |  el6        |"
    
        # iterate until tf_max
        idx = 0
        while t_iter < self.tf_max:
            print("Stage 2 : " + str(round(t_iter/self.tf_max*100,2)))

            # ensure numerical stabilty
            oe_iter = elements_safety(oe_iter, self.oe_min, self.oe_max)

            # compute instantaneous acceleration magnitude due to thrust
            accel_thrust = self.tmax/mass_iter

            # Compute deltaL in [-pi, pi)
            deltaL = self.eval_deltaL(oe_iter[5], oeT_iter[5]) 

            # evaluate Lyapunov function
            alpha, beta, _, psi = lyapunov_control_angles(
                fun_lyapunov_control=self.lyap_fun,
                fun_eval_fdot=self.eval_fdot, 
                fun_eval_gdot=self.eval_gdot, 
                fun_eval_dfdoe=self.eval_dfdoe, 
                fun_eval_dgdoe=self.eval_dgdoe,
                mu=self.mu, 
                accel=accel_thrust, 
                oe=oe_iter, 
                oeT=oeT_iter, 
                rpmin=self.rpmin, 
                m_petro=self.m_petro2, 
                n_petro=self.n_petro2, 
                r_petro=self.r_petro2,
                k_petro=self.k_petro2,
                wp=self.wp2, 
                woe=self.woe2, 
                l_mesh=self.l_mesh,
                wl=self.wl2,
                wscl=self.wscl2,
                deltaL=deltaL
            )

            # ensure angles are not nan and otherwise compute thrust vector
            # nan values occur at singularities, i.e., when ecc = 0, ecc = 1, or inc = pi
            throttle = 1   # initialize
            if np.isnan(alpha) == True or np.isnan(beta) == True:
                alpha, beta = 0.0, 0.0
                throttle = 0  # turn off
                u = np.array([0.0,0.0,0.0])
                n_nan_angles += 1
                if n_nan_angles > self.nan_angles_threshold:
                    if self.verbosity > 0:
                        print("Breaking as angles are nan")
                    self.exitcode = -3
                    break
            else:
                u = accel_thrust*np.array([
                    np.cos(beta)*np.sin(alpha),
                    np.cos(beta)*np.cos(alpha),
                    np.sin(beta),
                ])

            # ODE parameters
            ode_params = (self.mu, u, psi[0], psi[1], psi[2])
            psiT = self.psi_fun(self.mu, oeT_iter)
            ode_paramsT = (self.mu, np.zeros((3,)), psiT[0], psiT[1], psiT[2])
            h_next, oe_iter, oeT_iter = integrate_next_step(self.integrator, self.eom, 
                                                                t_iter, self.t_step, 
                                                                oe_iter, oeT_iter, 
                                                                ode_params, ode_paramsT, 
                                                                self.abs_tol, self.rel_tol, 
                                                            )
            t_iter += self.t_step  # update time
            mass_iter -= self.mdot*self.t_step*throttle
            self.t_step = max(self.step_min, min(self.step_max,h_next))
                
            # check convergence
            if check_convergence(oe_iter, oeT_iter, self.woe2, self.wl2, self.tol_oe, deltaL=deltaL) == True:
                self.exitcode = 1
                self.converge = True
                break
            
            # print progress
            if self.verbosity >= 2 and np.mod(idx,self.print_frequency)==0:
                if np.mod(idx, 20*self.print_frequency) == 0:
                    print("\n" + header)
                t_fraction = t_iter/self.tf_max
                print(f"Stage 2: {idx:6.0f} | {t_fraction: 1.3e} | {oe_iter[0]-oeT_iter[0]: 1.4e} | {oe_iter[1]-oeT_iter[1]: 1.4e} | {oe_iter[2]-oeT_iter[2]: 1.4e} | {oe_iter[3]-oeT_iter[3]: 1.4e} | {oe_iter[4]-oeT_iter[4]: 1.4e} | {oe_iter[5]-oeT_iter[5]: 1.4e} |")

            # check if mass is below threshold
            if mass_iter <= self.mass_min:
                if self.verbosity > 0:
                    print("Breaking as mass is now under mass_min")
                self.exitcode = -1
                break

            # store
            self.times2.append(t_iter)
            self.states2.append(oe_iter) 
            self.statesT2.append(oeT_iter)
            self.masses.append(mass_iter)
            self.controls2.append([alpha, beta, throttle])

            # index update
            idx += 1

        # Update storage over full timeline (stage 1 and stage 2)
        self.states += self.states2
        self.statesT += self.statesT2
        self.times += self.times2
        self.controls += self.controls2

        if self.converge == False:
            if self.verbosity > 0:
                print("Could not arrive to target elements within time")
            self.exitcode = -2
        else:
            if self.verbosity > 0:
                print("Target elements successfully reached!")
        return

    def eval_deltaL(self, l, lT):
        """Bring l - lT from [-2pi, 2pi) to [-pi,pi)"""
        # Bring l and lT back to [0,2pi] -> l-lT in [-2pi, 2pi)
        l = l%(2*np.pi)
        lT = lT%(2*np.pi)
        # Bring l-lT in [-pi,pi)
        return (l - lT)/2

    def plot_elements_history(self, figsize=(6,4), to_keplerian=False, 
                                    time_scale=1, distance_scale=1, 
                                    time_unit=None, distance_unit=None,
                                    degrees=True, to_plot=0):
        """Plot elements time history"""
        assert to_plot in [0,1,2], "to_plot must be 0, 1, or 2"
        if to_plot == 0:
            states = self.states
            statesT = self.statesT
            times = np.array(self.times) * time_scale
        elif to_plot == 1:
            states = self.states1
            statesT = self.statesT1
            times = np.array(self.times1) * time_scale
        else: 
            states = self.states2
            statesT = self.statesT2
            times = np.array(self.times2) * time_scale
        oes = np.zeros((6,len(times)))
        oesT = np.zeros((6,len(times)))
        if to_keplerian:
            labels = ["a", "e", "i", "raan", "om", "ta"]
            if distance_unit != None:
                assert type(distance_unit) == str, "The time unit must be a string"
                labels[0] += " ["+distance_unit+"]"
            for lab_idx in range(2,6):
                if degrees:
                    labels[lab_idx] += " [deg]"
                else:
                    labels[lab_idx] += " [rad]"
        else:
            labels = self.element_names
        time_label = "Time"
        if time_unit != None:
            time_label += " ["+time_unit+"]"
        
        for idx in range(len(times)):
            if to_keplerian:
                oes[:,idx] = mee_with_a2kep(states[idx])
                oes[0,idx] *= distance_scale
                oesT[:,idx] = mee_with_a2kep(statesT[idx])
                oesT[0,idx] *= distance_scale
                if degrees:
                    oes[2:6,idx] = np.degrees(oes[2:6,idx]) % 360
                    oesT[2:6,idx] = np.degrees(oesT[2:6,idx]) % 360
                else:
                    oes[2:6,idx] = oes[2:6,idx] % (2*np.pi)
                    oesT[2:6,idx] = oesT[2:6,idx] % (2*np.pi)
            else:
                oes[:,idx] = states[idx]
                oesT[:,idx] = statesT[idx]
                
        fig, axs = plt.subplots(3,2,figsize=figsize)
        i = 0
        j = 0
        for idx in range(6):
            axs[i, j].plot(times, oes[idx,:])
            axs[i, j].plot(times, oesT[idx,:], c="crimson")
            axs[i, j].set(xlabel=time_label, ylabel=labels[idx])
            i = (i + 1) % 3
            j = (j + 1) % 2
        axs[i, 1].set(xlabel=time_label, ylabel=labels[5])
        return fig, axs


    def plot_controls(self, time_scale, time_unit=None, figsize=(9,6), to_plot=0):
        """Plot control time history"""
        assert to_plot in [0,1,2], "to_plot must be 0, 1, or 2"
        if to_plot == 0:
            controls = self.controls
            times = np.array(self.times)[0:-1] * time_scale
        elif to_plot == 1:
            controls = self.controls1
            times = np.array(self.times1)[0:-1] * time_scale
        else: 
            controls = self.controls2
            times = np.array(self.times2) * time_scale
        alphas, betas, throttles = [], [], []
        for ctl in controls:
            alphas.append(ctl[0])
            betas.append(ctl[1])
            throttles.append(ctl[2])
        time_label = "Time"
        if time_unit != None:
            time_label += " ["+time_unit+"]"
        fig, axs = plt.subplots(3,1,figsize=figsize)
        axs[0].plot(times, np.array(alphas)*180/np.pi, marker='o', markersize=2)
        axs[1].plot(times, np.array(betas)*180/np.pi,  marker='o', markersize=2)
        axs[2].plot(times, np.array(throttles)*100,  marker='o', markersize=2)
        axs[0].set(xlabel=time_label, ylabel="Alpha control [deg]")
        axs[1].set(xlabel=time_label, ylabel="Beta control [deg]")
        axs[2].set(xlabel=time_label, ylabel="Throttle [%]")
        return fig, axs


    def interpolate_states(self, states, times):
        """Create interpolation states"""
        # prepare states matrix
        state_matrix = np.zeros((6,len(states)))
        for idx,state in enumerate(states):
            state_matrix[:,idx] = state
        f_a = interpolate.interp1d(times, state_matrix[0,:])
        f_f = interpolate.interp1d(times, state_matrix[1,:])
        f_g = interpolate.interp1d(times, state_matrix[2,:])
        f_h = interpolate.interp1d(times, state_matrix[3,:])
        f_k = interpolate.interp1d(times, state_matrix[4,:])
        f_l = interpolate.interp1d(times, state_matrix[5,:])
        return (f_a, f_f, f_g, f_h, f_k, f_l)


    def get_cartesian_history(self, states, times, interpolate=True, steps=None):
        """Get Cartesian history of states"""
        if interpolate:
            # interpolate orbital elements
            f_a, f_f, f_g, f_h, f_k, f_l = self.interpolate_states(states, times)
            if steps is None:
                steps = min(8000, int(round(times[-1]/0.1))) # -> why divide by 0.1
                print(f"Using {steps} steps for evaluation")
            t_evals = np.linspace(times[0], times[-1], steps)
            cart = np.zeros((6,steps))
            for idx, t in enumerate(t_evals):
                cart[:,idx] = mee_with_a2sv([f_a(t), f_f(t), f_g(t), f_h(t), f_k(t), f_l(t)], self.mu) 
        else:
            cart = np.zeros((6,len(times)))
            for idx in range(len(times)):
                cart[:,idx] = mee_with_a2sv(states[idx], self.mu)
        return cart


    def plot_trajectory_2d(
        self, 
        figsize=(6,6),
        interpolate=True, 
        steps=None,
        to_plot=0
    ):
        """Plot trajectory in xy-plane"""
        assert to_plot in [0,1,2], "to_plot must be 0, 1, or 2"
        if to_plot == 0:
            states = self.states
            times = self.times
        elif to_plot == 1:
            states = self.states1
            times = self.times1
        else: 
            states = self.states2
            times = self.times2
        # get cartesian history
        cart = self.get_cartesian_history(states, times, interpolate, steps)

        fig, ax = plt.subplots(1,1,figsize=figsize)
        # plot initial and final orbit
        coord_orb0 = get_orbit_coordinates(mee_with_a2kep(self.oe0), self.mu)
        coord_orbT = get_orbit_coordinates(mee_with_a2kep(self.oeT), self.mu)
        ax.plot(coord_orb0[0,:], coord_orb0[1,:], label="Initial", c="darkblue")
        ax.plot(coord_orbT[0,:], coord_orbT[1,:], label="Final", c="forestgreen")

        # plot transfer
        ax.plot(cart[0,:], cart[1,:], label="transfer", c="crimson", lw=0.4)
        ax.scatter(cart[0,0], cart[1,0], label=None, c="crimson", marker="x")
        ax.scatter(cart[0,-1], cart[1,-1], label=None, c="crimson", marker="o")
        ax.set_aspect('equal')
        return fig, ax


    def plot_trajectory_3d(
        self, 
        figsize=(6,6), 
        interpolate=True, 
        steps=None, 
        plot_sphere=True,
        sphere_radius=0.35,
        scale=1.02,
        to_plot=0
    ):
        """Plot trajectory in xyz"""
        assert to_plot in [0,1,2], "to_plot must be 0, 1, or 2"
        if to_plot == 0:
            states = self.states
            statesT = self.statesT
            times = self.times
        elif to_plot == 1:
            states = self.states1
            statesT = self.statesT1
            times = self.times1
        else: 
            states = self.states2
            statesT = self.statesT2
            times = self.times2
        # get cartesian history
        cart = self.get_cartesian_history(states, times, interpolate, steps)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')

        # plot center body
        if plot_sphere:
            plot_sphere_wireframe(ax, sphere_radius)
        xlims = [min(cart[0,:]), max(cart[0,:])]
        ylims = [min(cart[1,:]), max(cart[1,:])]
        zlims = [min(cart[2,:]), max(cart[2,:])]
        set_equal_axis(ax, xlims, ylims, zlims, scale=scale)

        # plot transfer
        ax.plot(cart[0,:], cart[1,:], cart[2,:], label="Transfer", c="crimson", lw=0.15)
        
        # plot initial and final orbits
        coord_orb0 = get_orbit_coordinates(mee_with_a2kep(self.oe0), self.mu) 
        coord_orbT = get_orbit_coordinates(mee_with_a2kep(self.oeT), self.mu)
        ax.plot(coord_orb0[0,:], coord_orb0[1,:], coord_orb0[2,:], label="Initial orbit", c="darkblue", lw=1.5)
        ax.plot(coord_orbT[0,:], coord_orbT[1,:], coord_orbT[2,:], label="Final orbit", c="forestgreen", lw=1.5)

        # plot chaser's initial and final positions
        ax.scatter(cart[0,0], cart[1,0], cart[2,0], label="Chaser's initial position", c="black", marker="x")
        ax.scatter(cart[0,-1], cart[1,-1], cart[2,-1], label="Chaser's final position", c="black", marker="o")

        # plot target's initial and final positions
        sv0 = mee_with_a2sv(statesT[0], self.mu)
        svf = mee_with_a2sv(statesT[-1], self.mu)
        ax.scatter(sv0[0], sv0[1], sv0[2], label="Target's initial position", c="yellow", marker="x")
        ax.scatter(svf[0], svf[1], svf[2], label="Target's final position", c="yellow", marker="o")
        
        # final plot settings
        ax.set_aspect('equal')
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        leg = ax.legend()
        leg.legendHandles[0].set_linewidth(1.5) # set linewidth for Transfer's legend 
        return fig, ax


    def pretty(self):
        """Pretty print"""
        print(f"\nStage 1 orbital transfer:")
        print(f"  {self.element_names[0]}  : {self.oe0[0]:1.4e} -> {self.oeT[0]:1.4e} (weight: {self.woe1[0]:2.2f})")
        print(f"  {self.element_names[1]}  : {self.oe0[1]:1.4e} -> {self.oeT[1]:1.4e} (weight: {self.woe1[1]:2.2f})")
        print(f"  {self.element_names[2]}  : {self.oe0[2]:1.4e} -> {self.oeT[2]:1.4e} (weight: {self.woe1[2]:2.2f})")
        print(f"  {self.element_names[3]}  : {self.oe0[3]:1.4e} -> {self.oeT[3]:1.4e} (weight: {self.woe1[3]:2.2f})")
        print(f"  {self.element_names[4]}  : {self.oe0[4]:1.4e} -> {self.oeT[4]:1.4e} (weight: {self.woe1[4]:2.2f})")
        return


    def pretty_results(self):
        """Pretty print results"""
        print(f"\nExit code : {self.exitcode}")
        print(f"Converge  : {self.converge}")
        print(f"Final state:")
        print(f"  {self.element_names[0]}  : {self.states[-1][0]:1.4e} (error: {abs(self.states[-1][0]-self.statesT[-1][0]):1.4e})")
        print(f"  {self.element_names[1]}  : {self.states[-1][1]:1.4e} (error: {abs(self.states[-1][1]-self.statesT[-1][1]):1.4e})")
        print(f"  {self.element_names[2]}  : {self.states[-1][2]:1.4e} (error: {abs(self.states[-1][2]-self.statesT[-1][2]):1.4e})")
        print(f"  {self.element_names[3]}  : {self.states[-1][3]:1.4e} (error: {abs(self.states[-1][3]-self.statesT[-1][3]):1.4e})")
        print(f"  {self.element_names[4]}  : {self.states[-1][4]:1.4e} (error: {abs(self.states[-1][4]-self.statesT[-1][4]):1.4e})")
        print(f"  {self.element_names[5]}  : {self.states[-1][5]:1.4e} (error: {abs(self.states[-1][5]-self.statesT[-1][5]):1.4e})")
        print(f"Transfer time : {self.times[-1]}")
        print(f"Final mass    : {self.masses[-1]}")
        return


    def pretty_settings(self):
        """Pretty print settings"""
        print(f"\nElements type  : MEE with sma")
        print(f"Elements names : {self.element_names}")
        print(f"Integrator    : {self.integrator}")
        print(f"Tolerance     : {self.tol_oe}")
        print(f"Relaxed tolerance : {self.tol_oe_relaxed}")
        print(f"Exit at relaxed   : {self.exit_at_relaxed}")
        return
