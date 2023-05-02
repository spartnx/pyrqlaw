# pyrqlaw
Python implementation of the RQ-law feedback control for low-thrust orbital transfer and rendezvous developed by Narayanaswamy & Damaren [4]. Adapted from [pyqlaw](https://github.com/Yuricst/pyqlaw), a version of the Q-Law algorithm implemented in Python by [Yuricst](https://github.com/Yuricst). Utility functions are available to tune the RQ-Law algorithm in cases when it cannot converge. 

### Dependencies

- `sympy`, `numpy`, `matplotilb`, `numba`, `pygmo`

### Important notes
- Dynamics & spacecraft parameters are given in non-dimensional quantities, scaling `GM = 1.0` (which may be modified, but it is numerically desirable to use this scaling). 
- All angles are defined in radians.
- The RQ-Law formulation presents singularities when the eccentricity of the chaser's orbit is equal to 0 or 1 or its inclination is 180 degrees. Thus, initial circular orbits for either the target or chaser must be defined with small eccentricities on the order of 1e-3 and not 0. Similarly, retrograde orbits must be defined with inclinations slightly less than 180 degrees.
- A rendezvous maneuver is performed in two stages. Stage 1 is the orbital transfer aiming to match the chaser and target's slow elements (i.e., semi-major axis, eccentricity, inclination, RAAN, argument of periapse). Stage 2 is the phasing matching the position of the chaser and target after the chaser reaches the target's orbit at the end of stage 1. 
- The RQ-Law algorithm implemented in `pyrqlaw` natively converts the classical elements into the Modified Equinoctial Elements with the semi-latus rectum replaced with the semi-major axis.

### Basic usage

Examples are included in `./examples/`. Example `1_` gives a template on how to use `pyrqlaw` on the full trajectory (stage 1 + stage 2), the orbital transfer (stage 1) only, and the phasing (stage 2) only. The example is, hopefully, self explanatory and can be used as is with user-defined inputs such as the initial orbital elements of the chaser and targets. Examples `2_`, `3_`, and `5_` are examples inspired from Narayanaswamy & Damaren (2023) [4]. Examples `7_` and `8_` are examples involving rendezvous in geosynchronous orbits. Finally, examples `4_` and `6_` demonstrate the use of utility functions tuning the algorithm in highly eccentric and inclined orbits and in the GEO regime, respectively.

Here, we present a basic example whose complete code can be found in `./examples/3_rendezvous_with_coasting.py`. 

We first start by importing the module

```python
#import sys
#sys.path.append("../")  # make sure the pyrqlaw folder is exposed
import pyrqlaw
```

Define the initial keplerian elements of the chaser and target, along with weighting. Stage 2 needs two additional weights compared to Stage 1. 

```python
# Chaser's and target's initial orbital elements (always in order: [SMA, ECC, INC, RAAN, AOP, TA])
oeC = [1.03, 0.2, 0, 0, 0, np.pi]
oeT = [1.05, 1e-3, , np.pi/2, np.pi/2, np.pi/2, np.pi/2]
woe1 = [2, 50, 50, 1, 1] # Stage 1 weights
woe2 = [10, 1, 1, 1, 1] # Stage 2 weights (set 1)
wl, wscl = 0.06609, 3.3697 # Stage 2 weights (set 2)
```

Provide spacecraft parameters (max thrust and mass-flow rate), max transfer time, the relative effectivity tolerance for Stage 1, and initial time step for integration. (By default, the algorithm uses a Dormand-Prince 5(4) integration scheme.)

```python
# spacecraft parameters
mass0 = 1.0
thrust = 4.6e-5
mdot = 1.1e-5
tf_max = 400.0
eta_r = 0.1
t_step = 0.001
```

Construct the problem object, then set the problem parameters

```python
prob = pyrqlaw.RQLaw()
prob.set_problem(oe0, oeT, mass0, thrust, mdot, tf_max, t_step, woe1=woe1, woe2=woe2, wl=wl, wscl=wscl, eta_r=eta_r)
prob.pretty()  # print info
```

```
Stage 1 orbital transfer:
  a  : 1.3136e+00 -> 1.4704e+00 (weight: 2.00)
  f  : 2.0000e-01 -> -1.0000e-03 (weight: 50.00)
  g  : 0.0000e+00 -> 1.2246e-19 (weight: 50.00)
  h  : 0.0000e+00 -> 6.1232e-17 (weight: 1.00)
  k  : 0.0000e+00 -> 1.0000e+00 (weight: 1.00)
```

Solve the problem

```python
prob.solve_stage1()
prob.solve_stage2()
prob.pretty_results()   # print info
```

```
Target elements successfully reached!
Converge  : True
Final state:
  a  : 1.4698e+00 (error: 5.9319e-04)
  f  : -6.4898e-04 (error: 3.5102e-04)
  g  : -2.6536e-04 (error: 2.6536e-04)
  h  : -2.6799e-06 (error: 2.6799e-06)
  k  : 1.0000e+00 (error: 9.1120e-07)
  L  : 7.2732e+03 (error: 3.6059e-03)
```

Some conveninence methods for plotting:

```python
fig1, ax1 = prob.plot_elements_history()
fig2, ax2 = prob.plot_trajectory_3d()
```

Stage 1 Keplerian elements (blue = chaser's; red = target's)
<p align="center">
  <img src="./plots//rendezvous_with_coasting//Elements history (stage 1).png" width="400" title="transfer">
</p>

Stage 1 trajectory
<p align="center">
  <img src="./plots//rendezvous_with_coasting//Trajectory (stage 1).png" width="400" title="transfer">
</p>

Stage 2 Keplerian elements (blue = chaser's; red = target's)
<p align="center">
  <img src="./plots//rendezvous_with_coasting//Elements history (stage 2).png" width="400" title="transfer">
</p>

Stage 2 trajectory (note that the yellow and black dots are at the same location)
<p align="center">
  <img src="./plots//rendezvous_with_coasting//Trajectory (stage 2).png" width="400" title="transfer">
</p>



### Some things to be careful!

- The `pyrqlaw` implementation was compared to the case studies presented in [4] which proposed the RQ-Law. For some of the examples, the outputs of pyrqlaw don't quite match the results from [4], likely because of differences between the `pyrqlaw` implementation and the authors' implementation. More work will be done towards addressing this mismatch in the future.


### References

[1] Petropoulos, A. E. (2003). Simple Control Laws for Low-Thrust Orbit Transfers. AAS Astrodynamics Specialists Conference.

[2] Petropoulos, A. E. (2004). Low-thrust orbit transfers using candidate Lyapunov functions with a mechanism for coasting. AIAA/AAS Astrodynamics Specialist Conference, August. https://doi.org/10.2514/6.2004-5089

[3] Petropoulos, A. E. (2005). Refinements to the Q-law for low-thrust orbit transfers. Advances in the Astronautical Sciences, 120(I), 963–982.

[4] Narayanaswamy, S., & Damaren, C. J. (2023). Equinoctial Lyapunov Control Law for Low-Thrust Rendezvous. Journal of Guidance, Control, and Dynamics, Vol. 46, No. 4. https://doi.org/10.2514/1.G006662

[5] [Modified Equinoctial Elements](https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/EquinoctalElements-modified.pdf)
