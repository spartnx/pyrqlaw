import numpy as np
import sys
import matplotlib.pyplot as plt
import time

sys.path.append("../")
import pyrqlaw


def scenario(ta0, fig_display=True, fig_save=True):
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
    ta_C = ta0 # true anomaly, rad

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

    # Integration parameters
    tf_max = 300 * (24*3600) # max time of flight, s
    t_step = 0.001 # integration step, non-dimensional

    # RQ-Law parameters common to both stages
    rpmin = DU
    l_mesh = 100
    t_mesh = 20

    # RQ-Law parameters: Stage 1 (orbital transfer - matching chaser's and target's orbits)
    k_petro1 = 100 # Penalty function parameter
    wp1 = 1 # Penalty function weight
    woe1 = [2, 50, 50, 1, 1] # Lyapunov function weights, in terms of MEE with sma ([sma, f, g, h, k])
    m_petro1 = 3 # For weight function Sa associated with sma
    n_petro1 = 4 # For weight function Sa associated with sma
    r_petro1 = 2 # For weight function Sa associated with sma
    eta_r1 = 0 # Relative effectivity threshold - automatically set to 0 in stage 2

    # RQ-Law parameters: Stage 2 (phasing - matching chaser's and target's positions)
    k_petro2 = 100 # Penalty function parameter
    wp2 = 1 # Penalty function weight
    woe2 = [10, 1, 1, 1, 1] # Lyapunov function weights, in terms of MEE with sma ([sma, f, g, h, k])
    m_petro2 = 3 # For weight function Sa associated with sma
    n_petro2 = 4 # For weight function Sa associated with sma
    r_petro2 = 2 # For weight function Sa associated with sma
    wl2 = 0.06609 # amplitude weight in augmented target sma - automatically set to 0 in stage 1
    wscl2 = 3.3697 # frequency weight in augmented target sma - automatically set to 0 in stage 1
    #########################

    #################################
    ######  INPUTS CONVERSION  ######
    #################################
    # Standardization
    sma_C /= DU
    sma_T /= DU
    rpmin /= DU
    tf_max /= TU
    thrust /= (mass0*DU/TU**2)
    mdot /= (mass0/TU)
    mass0 /= MU
    mu = 1

    # Convert keplerian elements into Modified Equinoctial Elements with sma
    oe0 = pyrqlaw.kep2mee_with_a(np.array([sma_C, ecc_C, inc_C, raan_C, aop_C, ta_C]))
    oeT = pyrqlaw.kep2mee_with_a(np.array([sma_T, ecc_T, inc_T, raan_T, aop_T, ta_T])) 
    #################################

    #############################
    ######  SOLVE PROBLEM  ######
    #############################
    # Construct the problem object -> this only solves Stage 1 for now...
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
    prob.set_problem(
                    oe0, oeT, 
                    mass0, thrust, mdot, tf_max, t_step, 
                    woe1=woe1, woe2=woe2, wl=wl2, wscl=wscl2
                )
    prob.pretty_settings()
    prob.pretty()

    # Solve the problem
    prob.solve_stage1(eta_r=eta_r1)
    prob.solve_stage2()
    prob.pretty_results() 

    run_time = time.time() - start
    print("\nRuntime: " + str(round(run_time,2)) + " sec")
    
    #############################
    ######  PLOT RESULTS  #######
    #############################
    # Using states over Stage 1 and Stage 2
    fig11, ax11 = prob.plot_elements_history(to_keplerian=True, 
                                            time_scale=TU/(24*3600), distance_scale=DU/1000, 
                                            time_unit="days", distance_unit="km", to_plot=0)
    fig21, ax21 = prob.plot_trajectory_3d(sphere_radius=Re/DU, to_plot=0)
    fig31, ax31 = prob.plot_controls(time_scale=TU/(24*3600), time_unit="days", to_plot=0)
    # Using states over Stage 1 only
    fig12, ax12 = prob.plot_elements_history(to_keplerian=True, 
                                            time_scale=TU/(24*3600), distance_scale=DU/1000, 
                                            time_unit="days", distance_unit="km", to_plot=1)
    fig22, ax22 = prob.plot_trajectory_3d(sphere_radius=Re/DU, to_plot=1)
    fig32, ax32 = prob.plot_controls(time_scale=TU/(24*3600), time_unit="days", to_plot=1)
    # Using states over Stage 2 only
    fig13, ax13 = prob.plot_elements_history(to_keplerian=True, 
                                            time_scale=TU/(24*3600), distance_scale=DU/1000, 
                                            time_unit="days", distance_unit="km", to_plot=2)
    fig23, ax23 = prob.plot_trajectory_3d(sphere_radius=Re/DU, to_plot=2)
    fig33, ax33 = prob.plot_controls(time_scale=TU/(24*3600), time_unit="days", to_plot=2)

    if fig_save:
        fig11.savefig(f"../plots/chaser_departure_point/Elements history (stages 1, 2) - ta0 {int(np.degrees(ta0))}")
        fig21.savefig(f"../plots/chaser_departure_point/Trajectory (stages 1, 2) - ta0 {int(np.degrees(ta0))}")
        fig31.savefig(f"../plots/chaser_departure_point/Controls history (stages 1, 2) - ta0 {int(np.degrees(ta0))}")
        fig12.savefig(f"../plots/chaser_departure_point/Elements history (stage 1) - ta0 {int(np.degrees(ta0))}")
        fig22.savefig(f"../plots/chaser_departure_point/Trajectory (stage 1) - ta0 {int(np.degrees(ta0))}")
        fig32.savefig(f"../plots/chaser_departure_point/Controls history (stage 1) - ta0 {int(np.degrees(ta0))}")
        fig13.savefig(f"../plots/chaser_departure_point/Elements history (stage 2) - ta0 {int(np.degrees(ta0))}")
        fig23.savefig(f"../plots/chaser_departure_point/Trajectory (stage 2) - ta0 {int(np.degrees(ta0))}")
        fig33.savefig(f"../plots/chaser_departure_point/Controls history (stage 2) - ta0 {int(np.degrees(ta0))}")
    if fig_display:
        plt.show()
        #############################

    #############################
    #########  OUTPUTS  #########
    #############################
    end_mass = prob.masses[-1] * MU
    mass_delta = MU - end_mass
    end_time = prob.times[-1] * TU/(24*3600)
    exitcode = prob.exitcode
    return end_mass, mass_delta, end_time, exitcode
    

if __name__ == "__main__":
    # Compare the figures with Figure 4 in Narayanaswamy and Damaren 
    # (Equinoctial Lyapunov Control Law for Low-Thrust Rendezvous)
    ta0 = 0
    scenario(ta0, fig_display=True, fig_save=False)

    # Compute final mass and tofs in all 8 scenarios
    end_masses = []
    mass_deltas = []
    tofs = []
    exitcodes = []
    true_anomalies = np.linspace(0,2*np.pi, num=8, endpoint=False)
    for ta0 in true_anomalies:
        end_mass, mass_delta, tof, exitcode = scenario(ta0, fig_display=False, fig_save=True)
        end_masses.append(end_mass)
        mass_deltas.append(mass_delta)
        tofs.append(tof)
        exitcodes.append(exitcode)

    # Outputs to be compared with Table 5 in Narayanaswamy and Damaren 
    # (Equinoctial Lyapunov Control Law for Low-Thrust Rendezvous)
    print(list(np.degrees(true_anomalies)))
    print(end_masses)
    print(mass_deltas)
    print(tofs)
    print(exitcodes)

    


    