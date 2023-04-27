import numpy as np
import sys

sys.path.append("../")
import pyrqlaw


def scenario():
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
    sma_C = DU + 36e6 # semi-major axis, m
    ecc_C = 1e-3 # eccentricity - can't be 0 nor 1 due to singularities in the RQ-Law formulation
    inc_C = 0 # inclination, rad - can't be pi due to singularities in the MEE (tan(inc/2) in h and k)
    raan_C = 0 # RAAN, rad
    aop_C = 0 # argument of periapse, rad
    ta_C = np.pi # true anomaly, rad

    # Target's initial state (Keplerian elements)
    sma_T = DU + 37e6 # semi-major axis, m
    ecc_T = np.radians(5) # eccentricity - can't be 0 nor 1 due to singularities in the RQ-Law formulation
    inc_T = 0 # inclination, rad - can't be pi due to singularities in the MEE (tan(inc/2) in h and k)
    raan_T = 0 # RAAN, rad
    aop_T = 0 # argument of periapse, rad
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
    t_step = 0.01 # integration step, non-dimensional

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
    woe2 = [23.54823751, 40.23722676, 5.69083197, 31.76804207, 43.3191163] # Lyapunov function weights, in terms of MEE with sma ([sma, f, g, h, k])
    m_petro2 = 3 # For weight function Sa associated with sma
    n_petro2 = 4 # For weight function Sa associated with sma
    r_petro2 = 2 # For weight function Sa associated with sma
    wl2 = 0.17370297 # amplitude weight in augmented target sma - automatically set to 0 in stage 1
    wscl2 = 2.26701138 # frequency weight in augmented target sma - automatically set to 0 in stage 1
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
                    verbosity=0
                    )
    prob.set_problem(
                    oe0, oeT, 
                    mass0, thrust, mdot, tf_max, t_step, 
                    woe1=woe1, woe2=woe2, wl=wl2, wscl=wscl2, 
                    eta_r=eta_r1
                )
    prob.pretty_settings()
    prob.pretty()

    return prob
    

if __name__ == "__main__":
    # Define the problem over which to fine tune the RQ-Law algorithm
    prob = scenario()

    # Tune RQ-Law over stage 1
    tuner = pyrqlaw.RQLawTuner(prob, stage="Stage 1", tune_bounds=[50]*5)
    tuner.tune(pop_size=5, solver="de")
    tuner.pretty()

    # # Tune RQ-Law over stage 2
    # tuner = pyrqlaw.RQLawTuner(prob, stage="Stage 2", tune_bounds=[50]*5+[1,10])
    # tuner.tune(pop_size=5, solver="de")
    # tuner.pretty()


    


    