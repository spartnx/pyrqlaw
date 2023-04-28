"""
This script describes how to set up the code to compute dynamic 
low-thrust orbital rendezvous using the RQ-Law algo implemented in pyrqlaw.
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import time

sys.path.append("../")
import pyrqlaw


def scenario_full(fig_display=True, fig_save=True):
    """Show how to define problem for a full maneuver, i.e., stage 1 and stage 2"""
    # start measuring time
    start = time.time()

    #########################
    ######  CONSTANTS  ######
    #########################
    # Dynamics constants
    mu = 3.9860e14 # Earth's gravitational parameter, m^3/s^2
    g0 = 9.81 # Earth's sea-level gravitational acceleration, m/s^2
    Re = 6378000.0 # Earth's radius, m

    # Standardizing constants
    DU = Re
    TU = np.sqrt(DU**3/mu) # period of an orbit of radius DU divided by 2pi, s
    #########################

    ######################
    ######  INPUTS  ######
    ######################
    # Chaser's initial state (Keplerian elements)
    sma_C = DU + 2e6 # semi-major axis, m
    ecc_C = 0.2 # eccentricity - can't be 0 nor 1 due to singularities in the RQ-Law formulation
    inc_C = 0 # inclination, rad - can't be pi due to singularities in the MEE (tan(inc/2) in h and k)
    raan_C = 0 # RAAN, rad
    aop_C = 0 # argument of periapse, rad
    ta_C = np.radians(30) # true anomaly, rad

    # Target's initial state (Keplerian elements)
    sma_T = DU + 3e6 # semi-major axis, m
    ecc_T = 1e-3 # eccentricity - can't be 0 nor 1 due to singularities in the RQ-Law formulation
    inc_T = np.pi/2 # inclination, rad - can't be pi due to singularities in the MEE (tan(inc/2) in h and k)
    raan_T = np.pi/2 # RAAN, rad
    aop_T = np.pi/2 # argument of periapse, rad
    ta_T = np.pi/2 # true anomaly, rad

    # spacecraft parameters
    mass0 = MU = 450 # chaser's initial mass, kg
    isp = 3300 # engine specific impulse, s
    eta = 0.65 # engine efficiency
    power = 5000 # engine power, W
    thrust = 2*eta*power/(g0*isp) # thrust, N
    mdot = thrust/(g0*isp) # mass flow rate, kg/s

    # Parameters common to both stages
    tf_max = 300 * (24*3600) # max time of flight, s
    t_step = 1 # integration step, s
    rpmin = DU # min periapse radius, m
    l_mesh = 100 # number of points where to evaluate fdot_x and gdot_x as needed in RQ-Law algo - the larger this number, the slower but more accurate the algo

    # Stage 1 RQ-Law parameters: orbital transfer -> matching chaser's and target's orbits
    k_petro1 = 100 # Penalty function parameter
    wp1 = 1 # Penalty function weight
    woe1 = [2, 50, 50, 1, 1] # Lyapunov function weights, in terms of MEE with sma ([sma, f, g, h, k])
    m_petro1, n_petro1, r_petro1 = 3, 4, 2 # For weight function Sa associated with sma
    eta_r1 = 0 # Relative effectivity threshold - automatically set to 0 in stage 2
    t_mesh = 20 # number of points where to evaluate qdot as needed in RQ-Law algo - the larger this number, the slower but more accurate the algo (used by algo only when eta_r > 0)

    # Stage 2 RQ-Law parameters: phasing -> matching chaser's and target's positions
    standalone_stage2 = False # run stage 2 alone, without stage 1? Set to True when Stage 2 only is computed
    k_petro2 = 100 # Penalty function parameter
    wp2 = 1 # Penalty function weight
    woe2 = [10, 1, 1, 1, 1] # Lyapunov function weights, in terms of MEE with sma ([sma, f, g, h, k])
    m_petro2, n_petro2, r_petro2 = 3, 4, 2 # For weight function Sa associated with sma
    wl2 = 0.06609 # amplitude weight in augmented target sma - automatically set to 0 for stage 1 (no tracking of true longitude in stage 1)
    wscl2 = 3.3697 # frequency weight in augmented target sma - automatically set to 0 for stage 1 (no tracking of true longitude in stage 1)
    #########################

    ######################################
    ######  INPUTS STANDARDIZATION  ######
    ######################################
    sma_C /= DU
    sma_T /= DU
    rpmin /= DU
    tf_max /= TU
    t_step /= TU
    thrust /= (mass0*DU/TU**2)
    mdot /= (mass0/TU)
    mass0 /= MU
    mu = 1
    #################################

    #############################
    ######  SOLVE PROBLEM  ######
    #############################
    # Chaser's initial orbital elements
    oeC = [sma_C, ecc_C, inc_C, raan_C, aop_C, ta_C] 
    # Target's initial orbital elements
    oeT = [sma_T, ecc_T, inc_T, raan_T, aop_T, ta_T]
    # Instantiate RQLaw object
    prob = pyrqlaw.RQLaw(
                    mu=mu, rpmin=rpmin, 
                    k_petro1=k_petro1, k_petro2=k_petro2,
                    m_petro1=m_petro1, m_petro2=m_petro2,
                    n_petro1=n_petro1, n_petro2=n_petro2,
                    r_petro1=r_petro1, r_petro2=r_petro2,
                    wp1=wp1, wp2=wp2,
                    l_mesh=l_mesh, t_mesh=t_mesh,
                    verbosity=2
                    )
    # Define problem
    prob.set_problem(
                    oeC, oeT, 
                    mass0, thrust, mdot, tf_max, t_step, 
                    woe1=woe1, woe2=woe2, wl=wl2, wscl=wscl2,
                    eta_r=eta_r1,
                    standalone_stage2=standalone_stage2
                )
    prob.pretty_settings()
    prob.pretty()

    # Solve the problem
    prob.solve_stage1()
    prob.solve_stage2()
    prob.pretty_results() 

    run_time = time.time() - start
    print("\nRuntime: " + str(round(run_time,2)) + " sec")
    
    #############################
    ######  PLOT RESULTS  #######
    #############################
    # Using states over Stage 1 and Stage 2
    fig11, _ = prob.plot_elements_history(to_keplerian=True, 
                                            time_scale=TU/(24*3600), distance_scale=DU/1000, 
                                            time_unit="days", distance_unit="km", to_plot=0)
    fig21, _ = prob.plot_trajectory_3d(sphere_radius=Re/DU, to_plot=0)
    fig31, _ = prob.plot_controls(time_scale=TU/(24*3600), time_unit="days", to_plot=0)
    # Using states over Stage 1 only
    fig12, _ = prob.plot_elements_history(to_keplerian=True, 
                                            time_scale=TU/(24*3600), distance_scale=DU/1000, 
                                            time_unit="days", distance_unit="km", to_plot=1)
    fig22, _ = prob.plot_trajectory_3d(sphere_radius=Re/DU, to_plot=1)
    fig32, _ = prob.plot_controls(time_scale=TU/(24*3600), time_unit="days", to_plot=1)
    # Using states over Stage 2 only
    fig13, _ = prob.plot_elements_history(to_keplerian=True, 
                                            time_scale=TU/(24*3600), distance_scale=DU/1000, 
                                            time_unit="days", distance_unit="km", to_plot=2)
    fig23, _ = prob.plot_trajectory_3d(sphere_radius=Re/DU, to_plot=2)
    fig33, _ = prob.plot_controls(time_scale=TU/(24*3600), time_unit="days", to_plot=2)

    if fig_save:
        fig11.savefig(f"../plots/test_pyrqlaw/Elements history (stages 1, 2) - full")
        fig21.savefig(f"../plots/test_pyrqlaw/Trajectory (stages 1, 2) - full")
        fig31.savefig(f"../plots/test_pyrqlaw/Controls history (stages 1, 2) - full")
        fig12.savefig(f"../plots/test_pyrqlaw/Elements history (stage 1) - full")
        fig22.savefig(f"../plots/test_pyrqlaw/Trajectory (stage 1) - full")
        fig32.savefig(f"../plots/test_pyrqlaw/Controls history (stage 1) - full")
        fig13.savefig(f"../plots/test_pyrqlaw/Elements history (stage 2) - full")
        fig23.savefig(f"../plots/test_pyrqlaw/Trajectory (stage 2) - full")
        fig33.savefig(f"../plots/test_pyrqlaw/Controls history (stage 2) - full")
    if fig_display:
        plt.show()
    #############################

    #############################
    #########  OUTPUTS  #########
    #############################
    end_mass = prob.masses[-1] * MU # kg
    mass_delta = MU - end_mass # kg
    end_time = prob.times[-1] * TU/(24*3600) # days
    exitcode = prob.exitcode 
    return end_mass, mass_delta, end_time, exitcode


def scenario_stage1(fig_display=True, fig_save=True):
    """Show how to define problem for stage 1 only"""
    # start measuring time
    start = time.time()

    #########################
    ######  CONSTANTS  ######
    #########################
    # Dynamics constants
    mu = 3.9860e14 # Earth's gravitational parameter, m^3/s^2
    g0 = 9.81 # Earth's sea-level gravitational acceleration, m/s^2
    Re = 6378000.0 # Earth's radius, m

    # Standardizing constants
    DU = Re
    TU = np.sqrt(DU**3/mu) # period of an orbit of radius DU divided by 2pi, s
    #########################

    ######################
    ######  INPUTS  ######
    ######################
    # Chaser's initial state (Keplerian elements)
    sma_C = DU + 2e6 # semi-major axis, m
    ecc_C = 0.2 # eccentricity - can't be 0 nor 1 due to singularities in the RQ-Law formulation
    inc_C = 0 # inclination, rad - can't be pi due to singularities in the MEE (tan(inc/2) in h and k)
    raan_C = 0 # RAAN, rad
    aop_C = 0 # argument of periapse, rad
    ta_C = np.radians(30) # true anomaly, rad

    # Target's initial state (Keplerian elements)
    sma_T = DU + 3e6 # semi-major axis, m
    ecc_T = 1e-3 # eccentricity - can't be 0 nor 1 due to singularities in the RQ-Law formulation
    inc_T = np.pi/2 # inclination, rad - can't be pi due to singularities in the MEE (tan(inc/2) in h and k)
    raan_T = np.pi/2 # RAAN, rad
    aop_T = np.pi/2 # argument of periapse, rad
    ta_T = np.pi/2 # true anomaly, rad

    # spacecraft parameters
    mass0 = MU = 450 # chaser's initial mass, kg
    isp = 3300 # engine specific impulse, s
    eta = 0.65 # engine efficiency
    power = 5000 # engine power, W
    thrust = 2*eta*power/(g0*isp) # thrust, N
    mdot = thrust/(g0*isp) # mass flow rate, kg/s

    # Parameters common to both stages
    tf_max = 300 * (24*3600) # max time of flight, s
    t_step = 1 # integration step, s
    rpmin = DU # min periapse radius, m
    l_mesh = 100 # number of points where to evaluate fdot_x and gdot_x as needed in RQ-Law algo - the larger this number, the slower but more accurate the algo

    # Stage 1 RQ-Law parameters: orbital transfer -> matching chaser's and target's orbits
    k_petro1 = 100 # Penalty function parameter
    wp1 = 1 # Penalty function weight
    woe1 = [2, 50, 50, 1, 1] # Lyapunov function weights, in terms of MEE with sma ([sma, f, g, h, k])
    m_petro1, n_petro1, r_petro1 = 3, 4, 2 # For weight function Sa associated with sma
    eta_r1 = 0 # Relative effectivity threshold - automatically set to 0 in stage 2
    t_mesh = 20 # number of points where to evaluate qdot as needed in RQ-Law algo - the larger this number, the slower but more accurate the algo (used by algo only when eta_r > 0)
    #########################

    ######################################
    ######  INPUTS STANDARDIZATION  ######
    ######################################
    sma_C /= DU
    sma_T /= DU
    rpmin /= DU
    tf_max /= TU
    t_step /= TU
    thrust /= (mass0*DU/TU**2)
    mdot /= (mass0/TU)
    mass0 /= MU
    mu = 1
    #################################

    #############################
    ######  SOLVE PROBLEM  ######
    #############################
    # Chaser's initial orbital elements
    oeC = [sma_C, ecc_C, inc_C, raan_C, aop_C, ta_C] 
    # Target's initial orbital elements
    oeT = [sma_T, ecc_T, inc_T, raan_T, aop_T, ta_T]
    # Instantiate RQLaw object
    prob = pyrqlaw.RQLaw(
                    mu=mu, rpmin=rpmin, 
                    k_petro1=k_petro1,
                    m_petro1=m_petro1,
                    n_petro1=n_petro1,
                    r_petro1=r_petro1,
                    wp1=wp1, 
                    l_mesh=l_mesh, t_mesh=t_mesh,
                    verbosity=2
                    )
    # Define problem
    prob.set_problem(
                    oeC, oeT, 
                    mass0, thrust, mdot, tf_max, t_step, 
                    woe1=woe1,
                    eta_r=eta_r1
                )
    prob.pretty_settings()
    prob.pretty()

    # Solve the problem
    prob.solve_stage1()
    prob.pretty_results() 

    run_time = time.time() - start
    print("\nRuntime: " + str(round(run_time,2)) + " sec")
    
    #############################
    ######  PLOT RESULTS  #######
    #############################
    # Using states over Stage 1 only
    fig12, _ = prob.plot_elements_history(to_keplerian=True, 
                                            time_scale=TU/(24*3600), distance_scale=DU/1000, 
                                            time_unit="days", distance_unit="km", to_plot=1)
    fig22, _ = prob.plot_trajectory_3d(sphere_radius=Re/DU, to_plot=1)
    fig32, _ = prob.plot_controls(time_scale=TU/(24*3600), time_unit="days", to_plot=1)

    if fig_save:
        fig12.savefig(f"../plots/test_pyrqlaw/Elements history (stage 1) - stage 1 only")
        fig22.savefig(f"../plots/test_pyrqlaw/Trajectory (stage 1) - stage 1 only")
        fig32.savefig(f"../plots/test_pyrqlaw/Controls history (stage 1) - stage 1 only")
    if fig_display:
        plt.show()
    #############################

    #############################
    #########  OUTPUTS  #########
    #############################
    end_mass = prob.masses[-1] * MU # kg
    mass_delta = MU - end_mass # kg
    end_time = prob.times[-1] * TU/(24*3600) # days
    exitcode = prob.exitcode 
    return end_mass, mass_delta, end_time, exitcode


def scenario_stage2(fig_display=True, fig_save=True):
    """Show how to define problem for stage 2 only"""
    # start measuring time
    start = time.time()

    #########################
    ######  CONSTANTS  ######
    #########################
    # Dynamics constants
    mu = 3.9860e14 # Earth's gravitational parameter, m^3/s^2
    g0 = 9.81 # Earth's sea-level gravitational acceleration, m/s^2
    Re = 6378000.0 # Earth's radius, m

    # Standardizing constants
    DU = Re
    TU = np.sqrt(DU**3/mu) # period of an orbit of radius DU divided by 2pi, s
    #########################

    ######################
    ######  INPUTS  ######
    ######################
    # Chaser's initial state (Keplerian elements)
    sma_C = DU + 3e6 # semi-major axis, m
    ecc_C = 1e-3 # eccentricity - can't be 0 nor 1 due to singularities in the RQ-Law formulation
    inc_C = np.pi/2 # inclination, rad - can't be pi due to singularities in the MEE (tan(inc/2) in h and k)
    raan_C = np.pi/2 # RAAN, rad
    aop_C = np.pi/2 # argument of periapse, rad
    ta_C = np.radians(30) # true anomaly, rad

    # Target's initial state (Keplerian elements)
    sma_T = DU + 3e6 # semi-major axis, m
    ecc_T = 1e-3 # eccentricity - can't be 0 nor 1 due to singularities in the RQ-Law formulation
    inc_T = np.pi/2 # inclination, rad - can't be pi due to singularities in the MEE (tan(inc/2) in h and k)
    raan_T = np.pi/2 # RAAN, rad
    aop_T = np.pi/2 # argument of periapse, rad
    ta_T = np.pi/2 # true anomaly, rad

    # spacecraft parameters
    mass0 = MU = 450 # chaser's initial mass, kg
    isp = 3300 # engine specific impulse, s
    eta = 0.65 # engine efficiency
    power = 5000 # engine power, W
    thrust = 2*eta*power/(g0*isp) # thrust, N
    mdot = thrust/(g0*isp) # mass flow rate, kg/s

    # Parameters common to both stages
    tf_max = 300 * (24*3600) # max time of flight, s
    t_step = 1 # integration step, s
    rpmin = DU # min periapse radius, m
    l_mesh = 100 # number of points where to evaluate fdot_x and gdot_x as needed in RQ-Law algo - the larger this number, the slower but more accurate the algo

    # Stage 2 RQ-Law parameters: phasing -> matching chaser's and target's positions
    standalone_stage2 = True # run stage 2 alone, without stage 1? Set to True when Stage 2 only is computed
    k_petro2 = 100 # Penalty function parameter
    wp2 = 1 # Penalty function weight
    woe2 = [10, 1, 1, 1, 1] # Lyapunov function weights, in terms of MEE with sma ([sma, f, g, h, k])
    m_petro2, n_petro2, r_petro2 = 3, 4, 2 # For weight function Sa associated with sma
    wl2 = 0.06609 # amplitude weight in augmented target sma - automatically set to 0 for stage 1 (no tracking of true longitude in stage 1)
    wscl2 = 3.3697 # frequency weight in augmented target sma - automatically set to 0 for stage 1 (no tracking of true longitude in stage 1)
    #########################

    ######################################
    ######  INPUTS STANDARDIZATION  ######
    ######################################
    sma_C /= DU
    sma_T /= DU
    rpmin /= DU
    tf_max /= TU
    t_step /= TU
    thrust /= (mass0*DU/TU**2)
    mdot /= (mass0/TU)
    mass0 /= MU
    mu = 1
    #################################

    #############################
    ######  SOLVE PROBLEM  ######
    #############################
    # Chaser's initial orbital elements
    oeC = [sma_C, ecc_C, inc_C, raan_C, aop_C, ta_C] 
    # Target's initial orbital elements
    oeT = [sma_T, ecc_T, inc_T, raan_T, aop_T, ta_T]
    # Instantiate RQLaw object
    prob = pyrqlaw.RQLaw(
                    mu=mu, rpmin=rpmin, 
                    k_petro2=k_petro2,
                    m_petro2=m_petro2,
                    n_petro2=n_petro2,
                    r_petro2=r_petro2,
                    wp2=wp2,
                    l_mesh=l_mesh,
                    verbosity=2
                    )
    # Define problem
    prob.set_problem(
                    oeC, oeT, 
                    mass0, thrust, mdot, tf_max, t_step, 
                    woe2=woe2, wl=wl2, wscl=wscl2,
                    standalone_stage2=standalone_stage2
                )
    prob.pretty_settings()
    prob.pretty()

    # Solve the problem
    prob.solve_stage2()
    prob.pretty_results() 

    run_time = time.time() - start
    print("\nRuntime: " + str(round(run_time,2)) + " sec")
    
    #############################
    ######  PLOT RESULTS  #######
    #############################
    # Using states over Stage 2 only
    fig13, _ = prob.plot_elements_history(to_keplerian=True, 
                                            time_scale=TU/(24*3600), distance_scale=DU/1000, 
                                            time_unit="days", distance_unit="km", to_plot=2)
    fig23, _ = prob.plot_trajectory_3d(sphere_radius=Re/DU, to_plot=2)
    fig33, _ = prob.plot_controls(time_scale=TU/(24*3600), time_unit="days", to_plot=2)

    if fig_save:
        fig13.savefig(f"../plots/test_pyrqlaw/Elements history (stage 2) - stage 2 only")
        fig23.savefig(f"../plots/test_pyrqlaw/Trajectory (stage 2) - stage 2 only")
        fig33.savefig(f"../plots/test_pyrqlaw/Controls history (stage 2) - stage 2 only")
    if fig_display:
        plt.show()
    #############################

    #############################
    #########  OUTPUTS  #########
    #############################
    end_mass = prob.masses[-1] * MU # kg
    mass_delta = MU - end_mass # kg
    end_time = prob.times[-1] * TU/(24*3600) # days
    exitcode = prob.exitcode 
    return end_mass, mass_delta, end_time, exitcode



if __name__ == "__main__":
    # run = 0, 1, or 2
    # 0 - run stage 1 + stage 2
    # 1 - run stage 1 only
    # 2 - run stage 2 only
    run = 1
    if run == 0:
        end_mass, mass_delta, end_time, exitcode = scenario_full(fig_display=True, fig_save=True)
    elif run == 1:
        end_mass, mass_delta, end_time, exitcode = scenario_stage1(fig_display=True, fig_save=True)
    elif run == 2:
        end_mass, mass_delta, end_time, exitcode = scenario_stage2(fig_display=True, fig_save=True)
    print(f"\nFinal spacecraft mass    : {round(end_mass,3)} kg")
    print(f"Propellant mass consumed : {round(mass_delta,3)} kg")
    print(f"Time of flight           : {round(end_time,3)} days")
    print(f"Exit code                : {exitcode}")
    
    


    