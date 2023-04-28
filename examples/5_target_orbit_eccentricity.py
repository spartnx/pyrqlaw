import numpy as np
import sys
import matplotlib.pyplot as plt
import time

sys.path.append("../")
import pyrqlaw


def scenario(eT, wl, wscl, woe=[10, 1, 1, 1, 1], fig_display=True, fig_save=True):
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
    sma_C = DU + 2e7 # semi-major axis, m
    ecc_C = eT # eccentricity - can't be 0 nor 1 due to singularities in the RQ-Law formulation
    inc_C = np.pi/2 # inclination, rad - can't be pi due to singularities in the MEE (tan(inc/2) in h and k)
    raan_C = np.pi/2 # RAAN, rad
    aop_C = np.pi/2 # argument of periapse, rad
    ta_C = np.pi # true anomaly, rad

    # Target's initial state (Keplerian elements)
    sma_T = DU + 2e7 # semi-major axis, m
    ecc_T = eT # eccentricity - can't be 0 nor 1 due to singularities in the RQ-Law formulation
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
    tf_max = 100 * (24*3600) # max time of flight, s
    t_step = 0.001 # integration step, non-dimensional

    # RQ-Law parameters common to both stages
    rpmin = DU
    l_mesh = 100
    t_mesh = 20

    # RQ-Law parameters: Stage 2 (phasing - matching chaser's and target's positions)
    k_petro2 = 100 # Penalty function parameter
    wp2 = 1 # Penalty function weight
    woe2 = woe # Lyapunov function weights, in terms of MEE with sma ([sma, f, g, h, k])
    m_petro2 = 3 # For weight function Sa associated with sma
    n_petro2 = 4 # For weight function Sa associated with sma
    r_petro2 = 2 # For weight function Sa associated with sma
    wl2 = wl # amplitude weight in augmented target sma - automatically set to 0 in stage 1
    wscl2 = wscl # frequency weight in augmented target sma - automatically set to 0 in stage 1
    standalone_stage2 = True
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
                    l_mesh=l_mesh, t_mesh=t_mesh,
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

    # Solve the problem
    prob.solve_stage2()
    prob.pretty_results()

    run_time = time.time() - start
    print("\nRuntime: " + str(round(run_time,2)) + " sec")
    
    #############################
    ######  PLOT RESULTS  #######
    #############################
    # Using states over Stage 2 only
    fig1, _ = prob.plot_elements_history(to_keplerian=True, 
                                            time_scale=TU/(24*3600), distance_scale=DU/1000, 
                                            time_unit="days", distance_unit="km", to_plot=2)
    fig2, _ = prob.plot_trajectory_3d(sphere_radius=Re/DU, to_plot=2)
    fig3, _ = prob.plot_controls(time_scale=TU/(24*3600), time_unit="days", to_plot=2)

    if fig_save:
        fig1.savefig(f"../plots/target_orbit_eccentricity/Elements history (stage 2) - eT {int(eT*1000)}")
        fig2.savefig(f"../plots/target_orbit_eccentricity/Trajectory (stage 2) - eT {int(eT*1000)}")
        fig3.savefig(f"../plots/target_orbit_eccentricity/Controls history (stage 2) - eT {int(eT*1000)}")
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
    # Compare the figures with Figure 10 in Narayanaswamy and Damaren 
    # (Equinoctial Lyapunov Control Law for Low-Thrust Rendezvous)
    eT = 0.7
    wl = 0.209375
    wscl = 3.22762143
    woe = [15.11160714,
           5.03348214,
           8.64955357,
           1.00669643,
           1.28794643]
    scenario(eT, wl, wscl, woe=woe, fig_display=True, fig_save=False)

    # Outputs to be compared with Figure 11 in Narayanaswamy and Damaren 
    # (Equinoctial Lyapunov Control Law for Low-Thrust Rendezvous)
    eT = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    Wl = [0.0594, 0.0621, 0.0621, 0.0661, 0.0548, 0.0702, 0.1123, 0.1, 0.2]
    Wscl = [3.623, 3.6952, 3.6952, 3.3714, 5.4849, 3.6663, 1.9891, 3.5557, 3.2053]
    Woe = [[10, 5, 5, 1, 1],
           [10, 5, 5, 1, 1],
           [10, 1, 1, 1, 1],
           [10, 1, 1, 1, 1],
           [10, 1, 1, 1, 1],
           [10, 1, 1, 1, 1],
           [10, 1, 1, 1, 1],
           [15, 5, 5, 1, 1],
           [15, 5, 5, 1, 1]]
    end_masses = []
    mass_deltas = []
    tofs = []
    exitcodes = []
    for j in range(len(eT)):
        end_mass, mass_delta, tof, exitcode = scenario(eT[j], Wl[j], Wscl[j], woe=Woe[j], fig_display=False, fig_save=True)
        end_masses.append(end_mass)
        mass_deltas.append(mass_delta)
        tofs.append(tof)
        exitcodes.append(exitcode)
    print(eT)
    print(Wl)
    print(Wscl)
    print(end_masses)
    print(mass_deltas)
    print(tofs)
    print(exitcodes)

    


    