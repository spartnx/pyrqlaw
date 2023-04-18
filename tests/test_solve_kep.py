"""
Test for constructing QLaw object
"""


import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
import pyrqlaw


def test_object():
    # construct problem
    prob = pyrqlaw.RQLaw(
        elements_type="keplerian",
        integrator="rk4"
    )#, verbosity=2)
    # initial and final elements: [a,e,i,RAAN,omega,ta]
    oe0 = np.array([1.5, 0.2, 0.3, 1e-2, 1e-2, 1e-3])
    oeT = np.array([2.2, 0.3, 1.1, 0.3, 0.0])
    woe = [1.0, 1.0, 1.0, 1.0, 1.0]
    # spacecraft parameters
    mass0 = 1.0
    tmax = 1e-3
    mdot = 1e-4
    tf_max = 3000.0
    t_step = 1.0
    # set problem
    prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
    prob.pretty()

    # solve
    prob.solve()
    prob.pretty_results()

    # plot
    fig1, ax1 = prob.plot_elements_history()
    fig2, ax2 = prob.plot_trajectory_3d(sphere_radius=0.1)
    return fig1, fig2


if __name__=="__main__":
    figs = test_object()
    plt.show()
    print("Done!")