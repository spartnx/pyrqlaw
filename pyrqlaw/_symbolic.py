"""
Symbolic derivation for feedback control law
"""

"""
sympy dev
"""

import numpy as np
import sympy as sym
from sympy import lambdify


def symbolic_rqlaw_mee_with_a():
    """Generate symbolic function for Q-Law using MEE with sma"""
    # define parameters
    rpmin, k_petro, m_petro, n_petro, r_petro = sym.symbols("rpmin k_petro m_petro n_petro r_petro") 
    wp = sym.symbols("Wp")
    mu = sym.symbols("mu")
    accel = sym.symbols("accel") # thrust/mass

    # orbital elements
    a, f, g, h, k, l = sym.symbols("a f g h k l")
    oe = [a,f,g,h,k,l]

    # targeted orbital elements
    aT, fT, gT, hT, kT, lT = sym.symbols("a_T f_T g_T h_T k_T l_T")
    oeT = [aT, fT, gT, hT, kT] # -> add lt in the list???

    # weights on orbital elements
    wa, wf, wg, wh, wk = sym.symbols("w_a w_f w_g w_h w_k")
    woe = [wa, wf, wg, wh, wk]


    def quotient(mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, k_petro, wp, woe):
        # unpack elements
        a, f, g, h, k, l = oe
        aT, fT, gT, hT, kT = oeT # -> add true longitude of the target lt

        # compute semi-parameter p, periapsis rp
        e = sym.sqrt(f**2 + g**2)
        p = a * (1 - e**2)
        ang_mom = sym.sqrt(a*mu*(1-e**2))
        rp = a*(1 - e)
        ta = l - sym.atan(g/f)
        r = ang_mom**2/(mu*(1+e*sym.cos(ta)))
        s_squared = 1 + h**2 + k**2

        # -------- compute quotient Q -------- #
        doe = [ # -> replace oeT elements with their augmented counterparts
            a-aT, 
            f-fT, 
            g-gT, 
            h-hT, 
            k-kT,
        ]

        # compute max oedot over controls and true longitude
        sqrt_pmu = sym.sqrt(p/mu)
        adot_xx = 2*accel*a*sym.sqrt(a/mu) * sym.sqrt((1 + sym.sqrt(f**2 + g**2)) / (1 - sym.sqrt(f**2 + g**2)))
        fdot_xx = 2*accel*sqrt_pmu # -> replace with max of the fdot_max_alpha_beta function by Yuan et al. (see 2023 paper)
        gdot_xx = 2*accel*sqrt_pmu # -> replace with max of the gdot_max_alpha_beta function by Yuan et al. (see 2023 paper)
        hdot_xx = 0.5*accel*sqrt_pmu * s_squared/(sym.sqrt(1 - g**2) + f)
        kdot_xx = 0.5*accel*sqrt_pmu * s_squared/(sym.sqrt(1 - f**2) + g)
        oedot = [
            adot_xx,
            fdot_xx,
            gdot_xx,
            hdot_xx,
            kdot_xx,
        ]
        
        # compute periapsis radius constraint P
        penalty_rp = sym.exp(k_petro*(1.0 - rp/rpmin))
        # compute scaling for each element Soe
        soe = [
            (1 + (sym.sqrt((a-aT)**2)/(m_petro*aT))**n_petro)**(1/r_petro), # -> replace aT with augmented aT
            1.0, 1.0, 1.0, 1.0
        ]
        # compute quotient Q
        q = (1 + wp*penalty_rp) * (
            woe[0]*soe[0]*(doe[0]/oedot[0])**2 +
            woe[1]*soe[1]*(doe[1]/oedot[1])**2 + 
            woe[2]*soe[2]*(doe[2]/oedot[2])**2 +
            woe[3]*soe[3]*(doe[3]/oedot[3])**2 +
            woe[4]*soe[4]*(doe[4]/oedot[4])**2 
        )

        # -------- compute Gauss differential equation terms -------- #
        cosL = sym.cos(l)
        sinL = sym.sin(l)
        w = 1 + f*cosL + g*sinL
        # let psi be column major!
        psi = [ # -> add line for true longitude L 
            # multiplies f_r
            [
                2*a**2/ang_mom*e*sym.sin(ta), # this is for a
                sqrt_pmu* sinL, # this is for f
                -sqrt_pmu* cosL, # this is for g
                0.0, # this is for h
                0.0, # this is for k
            ],
            # multiplies f_theta
            [
                2*a**2/ang_mom * p/r,  # this is for a
                sqrt_pmu/w * ((w+1)*cosL + f),
                sqrt_pmu/w * ((w+1)*sinL + g),
                0.0,
                0.0,
            ],
            # multiplies f_h
            [
                0.0,  # this is for a
                sqrt_pmu/w* (-g*(h*sinL - k*cosL)),
                sqrt_pmu/w* ( f*(h*sinL - k*cosL)),
                sqrt_pmu/w* 0.5*(1 + h**2 + k**2)*cosL,
                sqrt_pmu/w* 0.5*(1 + h**2 + k**2)*sinL,
            ]
        ]

        # -------- Apply Lyapunov descent direction -------- #
        dqdoe0 = sym.diff(q, oe[0])   # a
        dqdoe1 = sym.diff(q, oe[1])   # f
        dqdoe2 = sym.diff(q, oe[2])   # g 
        dqdoe3 = sym.diff(q, oe[3])   # h
        dqdoe4 = sym.diff(q, oe[4])   # k

        # compute thrust vector components
        # -> need to modify with extra terms related to true longitudes, as shown in 2023 paper
        d2 = (psi[0][0]*dqdoe0 + psi[0][1]*dqdoe1 + psi[0][2]*dqdoe2 + psi[0][3]*dqdoe3 + psi[0][4]*dqdoe4) # D2 in 2023 paper
        d1 = (psi[1][0]*dqdoe0 + psi[1][1]*dqdoe1 + psi[1][2]*dqdoe2 + psi[1][3]*dqdoe3 + psi[1][4]*dqdoe4) # D1 in 2023 paper
        d3 = (psi[2][0]*dqdoe0 + psi[2][1]*dqdoe1 + psi[2][2]*dqdoe2 + psi[2][3]*dqdoe3 + psi[2][4]*dqdoe4) # D3 in 2023 paper

        fun_lyapunov_control = lambdify(
            [mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, k_petro, wp, woe], 
            [[d2, d1, d3], psi], 
            "numpy",
        )

        fun_eval_psi = lambdify(
            [mu, oe], 
            psi, 
            "numpy",
        )

        fun_eval_dqdoe = lambdify(
            [
                mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, 
                k_petro, wp, woe
            ], 
            [dqdoe0,dqdoe1,dqdoe2,dqdoe3,dqdoe4], 
            "numpy",
        )
        return fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe

    # create function
    print("Generating MEE-SMA lyapunov control funcion with sympy")
    fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe = quotient(mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, k_petro, wp, woe)
    return fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe 





def symbolic_rqlaw_mee_with_a_improved():
    """Generate symbolic function for Q-Law using MEE with sma"""
    # define parameters
    rpmin, k_petro, m_petro, n_petro, r_petro = sym.symbols("rpmin k_petro m_petro n_petro r_petro") 
    wp = sym.symbols("Wp")
    mu = sym.symbols("mu")
    accel = sym.symbols("accel") # thrust/mass

    # orbital elements
    a, f, g, h, k, l = sym.symbols("a f g h k l")
    oe = [a,f,g,h,k,l]

    # targeted orbital elements
    aT, fT, gT, hT, kT, lT = sym.symbols("a_T f_T g_T h_T k_T l_T")
    oeT = [aT, fT, gT, hT, kT] # -> add lt in the list???

    # weights on orbital elements
    wa, wf, wg, wh, wk = sym.symbols("w_a w_f w_g w_h w_k")
    woe = [wa, wf, wg, wh, wk]


    def quotient(mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, k_petro, wp, woe):
        # unpack elements
        a, f, g, h, k, l = oe
        aT, fT, gT, hT, kT = oeT # -> add true longitude of the target lt

        # compute semi-parameter p, periapsis rp
        e = sym.sqrt(f**2 + g**2)
        p = a * (1 - e**2)
        ang_mom = sym.sqrt(a*mu*(1-e**2))
        rp = a*(1 - e)
        ta = l - sym.atan(g/f)
        r = ang_mom**2/(mu*(1+e*sym.cos(ta)))
        s_squared = 1 + h**2 + k**2

        # -------- compute quotient Q -------- #
        doe = [ # -> replace oeT elements with their augmented counterparts
            a-aT, 
            f-fT, 
            g-gT, 
            h-hT, 
            k-kT,
        ]

        # compute max oedot over controls and true longitude
        sqrt_pmu = sym.sqrt(p/mu)
        adot_xx = 2*accel*a*sym.sqrt(a/mu) * sym.sqrt((1 + sym.sqrt(f**2 + g**2)) / (1 - sym.sqrt(f**2 + g**2)))
        fdot_xx = sym.symbols("fdot_xx")
        gdot_xx = sym.symbols("gdot_xx")
        hdot_xx = 0.5*accel*sqrt_pmu * s_squared/(sym.sqrt(1 - g**2) + f)
        kdot_xx = 0.5*accel*sqrt_pmu * s_squared/(sym.sqrt(1 - f**2) + g)
        oedot = [
            adot_xx,
            fdot_xx,
            gdot_xx,
            hdot_xx,
            kdot_xx,
        ]
        
        # compute periapsis radius constraint P
        penalty_rp = sym.exp(k_petro*(1.0 - rp/rpmin))
        # compute scaling for each element Soe
        soe = [
            (1 + (sym.sqrt((a-aT)**2)/(m_petro*aT))**n_petro)**(1/r_petro), # -> replace aT with augmented aT
            1.0, 1.0, 1.0, 1.0
        ]
        # compute quotient Q
        q = (1 + wp*penalty_rp) * (
            woe[0]*soe[0]*(doe[0]/oedot[0])**2 +
            woe[1]*soe[1]*(doe[1]/oedot[1])**2 + 
            woe[2]*soe[2]*(doe[2]/oedot[2])**2 +
            woe[3]*soe[3]*(doe[3]/oedot[3])**2 +
            woe[4]*soe[4]*(doe[4]/oedot[4])**2 
        )

        # -------- compute Gauss differential equation terms -------- #
        cosL = sym.cos(l)
        sinL = sym.sin(l)
        w = 1 + f*cosL + g*sinL
        # let psi be column major!
        psi = [ # -> add line for true longitude L 
            # multiplies f_r
            [
                2*a**2/ang_mom*e*sym.sin(ta), # this is for a
                sqrt_pmu* sinL, # this is for f
                -sqrt_pmu* cosL, # this is for g
                0.0, # this is for h
                0.0, # this is for k
            ],
            # multiplies f_theta
            [
                2*a**2/ang_mom * p/r,  # this is for a
                sqrt_pmu/w * ((w+1)*cosL + f),
                sqrt_pmu/w * ((w+1)*sinL + g),
                0.0,
                0.0,
            ],
            # multiplies f_h
            [
                0.0,  # this is for a
                sqrt_pmu/w* (-g*(h*sinL - k*cosL)),
                sqrt_pmu/w* ( f*(h*sinL - k*cosL)),
                sqrt_pmu/w* 0.5*(1 + h**2 + k**2)*cosL,
                sqrt_pmu/w* 0.5*(1 + h**2 + k**2)*sinL,
            ]
        ]

        # -------- Apply Lyapunov descent direction -------- #
        dqdoe0 = sym.diff(q, oe[0])   # a
        dqdoe1 = sym.diff(q, oe[1])   # f
        dqdoe2 = sym.diff(q, oe[2])   # g 
        dqdoe3 = sym.diff(q, oe[3])   # h
        dqdoe4 = sym.diff(q, oe[4])   # k
        dqdf = sym.diff(q, fdot_xx)   
        dqdg = sym.diff(q, gdot_xx)

        # define the partial derivatives of fdot_x and gdot_x wrt oe at the maximizers Lmax_f and Lmax_f
        dfda_max, dgda_max = sym.symbols("dfda_max dgda_max")
        dfdf_max, dgdf_max = sym.symbols("dfdf_max dgdf_max")
        dfdg_max, dgdg_max = sym.symbols("dfdg_max dgdg_max")
        dfdh_max, dgdh_max = sym.symbols("dfdh_max dgdh_max")
        dfdk_max, dgdk_max = sym.symbols("dfdk_max dgdk_max")
        dfdoe_max = [dfda_max, dfdf_max, dfdg_max, dfdh_max, dfdk_max]
        dgdoe_max = [dgda_max, dgdf_max, dgdg_max, dgdh_max, dgdk_max]

        # compute D terms
        # D2 in 2023 paper
        d2 = psi[0][0]*(dqdoe0 + dqdf*dfdoe_max[0] + dqdg*dgdoe_max[0])   \
             + psi[0][1]*(dqdoe1 + dqdf*dfdoe_max[1] + dqdg*dgdoe_max[1]) \
             + psi[0][2]*(dqdoe2 + dqdf*dfdoe_max[2] + dqdg*dgdoe_max[2]) \
             + psi[0][3]*(dqdoe3 + dqdf*dfdoe_max[3] + dqdg*dgdoe_max[3]) \
             + psi[0][4]*(dqdoe4 + dqdf*dfdoe_max[4] + dqdg*dgdoe_max[4])
        # D1 in 2023 paper
        d1 = psi[1][0]*(dqdoe0 + dqdf*dfdoe_max[0] + dqdg*dgdoe_max[0])   \
             + psi[1][1]*(dqdoe1 + dqdf*dfdoe_max[1] + dqdg*dgdoe_max[1]) \
             + psi[1][2]*(dqdoe2 + dqdf*dfdoe_max[2] + dqdg*dgdoe_max[2]) \
             + psi[1][3]*(dqdoe3 + dqdf*dfdoe_max[3] + dqdg*dgdoe_max[3]) \
             + psi[1][4]*(dqdoe4 + dqdf*dfdoe_max[4] + dqdg*dgdoe_max[4])
        # D3 in 2023 paper
        d3 = psi[2][0]*(dqdoe0 + dqdf*dfdoe_max[0] + dqdg*dgdoe_max[0])   \
             + psi[2][1]*(dqdoe1 + dqdf*dfdoe_max[1] + dqdg*dgdoe_max[1]) \
             + psi[2][2]*(dqdoe2 + dqdf*dfdoe_max[2] + dqdg*dgdoe_max[2]) \
             + psi[2][3]*(dqdoe3 + dqdf*dfdoe_max[3] + dqdg*dgdoe_max[3]) \
             + psi[2][4]*(dqdoe4 + dqdf*dfdoe_max[4] + dqdg*dgdoe_max[4])

        # -------- fdot_x and gdot_x -------- #
        fdot_x = accel/w*sqrt_pmu*sym.sqrt((f+sym.sin(l)*(w+1))**2 + (w*sym.sin(l))**2 + g**2*(k*sym.cos(l) - h*sym.sin(l))**2)
        gdot_x = accel/w*sqrt_pmu*sym.sqrt((g+sym.sin(l)*(w+1))**2 + (w*sym.cos(l))**2 + f**2*(k*sym.cos(l) - h*sym.sin(l))**2)

        # -------- Derivatives of fdot_x wrt oe  -------- #
        dfdoe0 = sym.diff(fdot_x, oe[0])   # a
        dfdoe1 = sym.diff(fdot_x, oe[1])   # f
        dfdoe2 = sym.diff(fdot_x, oe[2])   # g 
        dfdoe3 = sym.diff(fdot_x, oe[3])   # h
        dfdoe4 = sym.diff(fdot_x, oe[4])   # k
        dfdoe = [dfdoe0, dfdoe1, dfdoe2, dfdoe3, dfdoe4]

        # -------- Derivatives of gdot_x wrt oe  -------- #
        dgdoe0 = sym.diff(gdot_x, oe[0])   # a
        dgdoe1 = sym.diff(gdot_x, oe[1])   # f
        dgdoe2 = sym.diff(gdot_x, oe[2])   # g 
        dgdoe3 = sym.diff(gdot_x, oe[3])   # h
        dgdoe4 = sym.diff(gdot_x, oe[4])   # k
        dgdoe = [dgdoe0, dgdoe1, dgdoe2, dgdoe3, dgdoe4]

        fun_lyapunov_control = lambdify(
            [
                mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, k_petro,  
                wp, woe, fdot_xx, gdot_xx, dfdoe_max, dgdoe_max
            ], 
            [[d2, d1, d3], psi], 
            "numpy",
        )

        fun_eval_psi = lambdify(
            [
                mu, oe
            ], 
            psi, 
            "numpy",
        )

        fun_eval_dqdoe = lambdify(
            [
                mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, k_petro,
                wp, woe, fdot_xx, gdot_xx, dfdoe_max, dgdoe_max
            ], 
            [
                dqdoe0 + dqdf*dfdoe_max[0] + dqdg*dgdoe_max[0],
                dqdoe1 + dqdf*dfdoe_max[1] + dqdg*dgdoe_max[1],
                dqdoe2 + dqdf*dfdoe_max[2] + dqdg*dgdoe_max[2],
                dqdoe3 + dqdf*dfdoe_max[3] + dqdg*dgdoe_max[3],
                dqdoe4 + dqdf*dfdoe_max[4] + dqdg*dgdoe_max[4]
            ], 
            "numpy",
        )

        fun_eval_fdot = lambdify(
            [
                mu, accel, oe
            ], 
            fdot_x, 
            "numpy",
        )

        fun_eval_gdot = lambdify(
            [
                mu, accel, oe
            ], 
            gdot_x, 
            "numpy",
        )

        fun_eval_dfdoe = lambdify(
            [
                mu, accel, oe
            ], 
            dfdoe, 
            "numpy",
        )

        fun_eval_dgdoe = lambdify(
            [
                mu, accel, oe
            ], 
            dgdoe, 
            "numpy",
        )
        return fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe, fun_eval_fdot, fun_eval_gdot, fun_eval_dfdoe, fun_eval_dgdoe

    # create functions
    print("Generating MEE-SMA lyapunov control funcion with sympy")
    return quotient(mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, k_petro, wp, woe)