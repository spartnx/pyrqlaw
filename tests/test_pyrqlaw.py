import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("../")
import pyrqlaw

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
VU = DU/TU # speed along an orbit of radius DU, m/s
#########################


######################
######  INPUTS  ######
######################
# Chaser's initial state (Keplerian elements)
sma_C = DU + 2e6 # semi-major axis, m
ecc_C = 0.2 # eccentricity - can't be 0 nor 1 due to singularities in RQ-Law
inc_C = 0 # inclination, rad - can't be pi due to singularities in RQ-Law
raan_C = 0 # RAAN, rad
aop_C = 0 # argument of periapse, rad
ta_C = 0 # true anomaly, rad

# Target's initial state (Keplerian elements)
sma_T = DU + 3e6 # semi-major axis, m
ecc_T = 1e-3 # eccentricity - can't be 0 nor 1 due to singularities in RQ-Law
inc_T = np.pi/2 # inclination, rad - can't be pi due to singularities in RQ-Law
raan_T = np.pi/2 # RAAN, rad
aop_T = np.pi/2 # argument of periapse, rad
ta_T = np.pi/2 # true anomaly, rad

# spacecraft parameters
mass0 = 450.0 # chaser's initial mass, kg
isp = 3300 # engine specific impulse, s
eta = 0.65 # engine efficiency
power = 5000 # engine power, W
thrust = 2*eta*power/(g0*isp) # thrust, N
mdot = thrust/(g0*isp) # mass flow rate, kg/s

# Integration parameters
tf_max = 300 * (24*3600) # max time of flight, s
t_step = 1 # integration step, s

# RQ-Law parameters: Stage 1 (orbital transfer)
k_petro_1 = 100
rpmin_1 = DU
wp_1 = 1
woe_1 = [2, 50, 50, 1, 1] # Lyapunov function weights, in terms of MEE with sma ([sma, f, g, h, k])
m_petro_1 = 3
n_petro_1 = 4
r_petro_1 = 2

# RQ-Law parameters: Stage 2 (phasing)
k_petro_2 = 100
rpmin_2 = DU
wp_2 = 1
woe_2 = [10, 1, 1, 1, 1] # Lyapunov function weights, in terms of MEE with sma ([sma, f, g, h, k])
m_petro_2 = 3
n_petro_2 = 4
r_petro_2 = 2
l_tol = 3e-3 # error tolerance on true longitude, rad
wl = 0.06609 # set 0 in stage 1 (automatically done)
wscl = 3.3697 # set 0 in stage 1 (automatically done)
#########################


#################################
######  INPUTS CONVERSION  ######
#################################
# Standardization
sma_C /= DU
sma_T /= DU
rpmin_1 /= DU
rpmin_2 /= DU
tf_max /= TU
t_step /= TU
thrust /= (mass0*DU/TU**2)
mdot /= (mass0/TU)
mass0 = 1
mu = 1

# Convert keplerian elements into Modified Equinoctial Elements with sma
oe0 = pyrqlaw.kep2mee_with_a(np.array([sma_C, ecc_C, inc_C, raan_C, aop_C, ta_C]))
oeT = pyrqlaw.kep2mee_with_a(np.array([sma_T, ecc_T, inc_T, raan_T, aop_T, ta_T]))[:5] # -> remove [:5] once stage 2 is implemented
#################################


#############################
######  SOLVE PROBLEM  ######
#############################
# Construct the problem object -> this only solves Stage 1 for now...
prob = pyrqlaw.RQLaw(mu=mu,
                     rpmin=rpmin_1, 
                     k_petro=k_petro_1, # -> add inputs for Stage 2
                     m_petro=m_petro_1, # -> add Qtol for Stage 1 convergence check
                     n_petro=n_petro_1,
                     r_petro=r_petro_1, 
                     wp=wp_1,
                    )
prob.set_problem(oe0, oeT, mass0, thrust, mdot, tf_max, t_step, woe=woe_1)
prob.pretty_settings()
prob.pretty()

# Solve the problem
prob.solve()
prob.pretty_results() 

# Plots
fig1, ax1 = prob.plot_elements_history(to_keplerian=True, 
                                       time_scale=TU/(24*3600), distance_scale=DU/1000, 
                                       time_unit="days", distance_unit="km")
fig2, ax2 = prob.plot_trajectory_3d(sphere_radius=Re/DU)
fig3, ax3 = prob.plot_controls()
plt.show()
#############################

# -> get output data such as final mass and convert back into input units