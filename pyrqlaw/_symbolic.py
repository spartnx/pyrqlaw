"""
Symbolic derivation for feedback control law
"""

"""
sympy dev
"""

import sympy as sym

def symbolic_rqlaw_mee_with_a():
    """Generate symbolic function for Q-Law using MEE with sma"""
    # define parameters
    rpmin, k_petro, m_petro, n_petro, r_petro = sym.symbols("rpmin k_petro m_petro n_petro r_petro") 
    wp, wl, wscl = sym.symbols("Wp WL Wscl")
    mu = sym.symbols("mu")
    accel = sym.symbols("accel") # thrust/mass

    # weights on orbital elements
    wa, wf, wg, wh, wk = sym.symbols("w_a w_f w_g w_h w_k")
    woe = [wa, wf, wg, wh, wk]

    # orbital elements
    a, f, g, h, k, l = sym.symbols("a f g h k l")
    oe = [a,f,g,h,k,l]

    # targeted orbital elements
    aT, fT, gT, hT, kT, lT = sym.symbols("a_T f_T g_T h_T k_T l_T")
    oeT = [aT, fT, gT, hT, kT, lT] 

    # deltaL = LC - LT
    deltaL = sym.symbols("deltaL")

    # Thrust direction 
    u1, u2, u3 = sym.symbols("u1 u2 u3")
    u = [u1, u2, u3]

    # compute chaser parameters
    e = sym.sqrt(f**2 + g**2)
    p = a * (1 - e**2)
    rp = a*(1 - e)
    s_squared = 1 + h**2 + k**2
    cosL = sym.cos(l)
    sinL = sym.sin(l)
    w = 1 + f*cosL + g*sinL

    # compute target parameters
    eT = sym.sqrt(fT**2 + gT**2)
    pT = aT * (1 - eT**2)
    cosLT = sym.cos(lT)
    sinLT = sym.sin(lT)
    wT = 1 + fT*cosLT + gT*sinLT

    # -------- Augmented target elements -------- #
    # -> deltaL only
    aT_aug = 2*wl/sym.pi * (aT - rpmin/(1 - e)) * sym.atan(wscl*deltaL) + aT # -> deltaL only
    fT_aug = fT
    gT_aug = gT
    hT_aug = hT
    kT_aug = kT

    # -------- quotient Q -------- #
    doe = [
        a-aT_aug, #-> deltaL only
        f-fT_aug, 
        g-gT_aug, 
        h-hT_aug, 
        k-kT_aug,
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
        (1 + (sym.sqrt((a-aT_aug)**2)/(m_petro*aT_aug))**n_petro)**(1/r_petro), # -> deltaL only
        1.0, 1.0, 1.0, 1.0
    ]
    # compute quotient Q -> deltaL only
    q = (1 + wp*penalty_rp) * (
        woe[0]*soe[0]*(doe[0]/oedot[0])**2 +
        woe[1]*soe[1]*(doe[1]/oedot[1])**2 + 
        woe[2]*soe[2]*(doe[2]/oedot[2])**2 +
        woe[3]*soe[3]*(doe[3]/oedot[3])**2 +
        woe[4]*soe[4]*(doe[4]/oedot[4])**2 
    )

    # -------- compute Gauss differential equation terms -------- #
    # -> l only
    # let psi be column major!
    psi = [ 
        # multiplies f_r
        [
           sqrt_pmu* 2*a*(f*sinL - g*cosL)/(1-e**2), # this is for a
            sqrt_pmu* sinL, # this is for f
            -sqrt_pmu* cosL, # this is for g
            0.0, # this is for h
            0.0, # this is for k
            0.0, # this is for l
        ],
        # multiplies f_theta
        [
            sqrt_pmu* 2*a*w/(1-e**2),  # this is for a
            sqrt_pmu/w * ((w+1)*cosL + f),
            sqrt_pmu/w * ((w+1)*sinL + g),
            0.0,
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
            sqrt_pmu/w* (h*sinL - k*cosL)
        ]
    ]

    # The constant term of the differential equation - associated with true longitude
    b = sym.sqrt(mu*p)*(w/p)**2 # chaser -> l only (in w)
    bT = sym.sqrt(mu*pT)*(wT/pT)**2 # target -> lT only (in wT)

    # -------- Apply Lyapunov descent direction -------- #
    # -> any derivative of q w.r.t any variable involves deltaL only (not l nor lT)
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

    # Define the partial derivatives of Q wrt the true longitudes
    dqdl = (+1)*sym.diff(q, deltaL) # wrt l -> replace this with (+1)*sym.diff(q, deltaL) - before : sym.diff(q, oeT[5])
    dqdlT = (-1)*sym.diff(q, deltaL) # wrt lT -> replace this with (-1)*sym.diff(q, deltaL) - before : sym.diff(q, oeT[5])

    # compute D terms -> deltaL in every term involving Q; l in every term involving psi
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
            + psi[2][4]*(dqdoe4 + dqdf*dfdoe_max[4] + dqdg*dgdoe_max[4]) \
            + dqdl*psi[2][5]

    # -------- qdot ---------- #
    # NB: in the 2023 paper, the authors change the order of F_theta and F_r for some reason...
    qdot = accel*(d1*u2 + d2*u1 + d3*u3) + dqdl*b + dqdlT*bT 
    # -> deltaL in every term involving Q (d1, d2, d3, dqdl); l in b; lT in bT

    # -------- fdot_x and gdot_x -------- #
    fdot_x = accel/w*sqrt_pmu*sym.sqrt((f+sym.sin(l)*(w+1))**2 + (w*sym.sin(l))**2 + g**2*(k*sym.cos(l) - h*sym.sin(l))**2)
    gdot_x = accel/w*sqrt_pmu*sym.sqrt((g+sym.sin(l)*(w+1))**2 + (w*sym.cos(l))**2 + f**2*(k*sym.cos(l) - h*sym.sin(l))**2)
    # -> l only

    # -------- Derivatives of fdot_x wrt oe  -------- #
    dfdoe0 = sym.diff(fdot_x, oe[0])   # a
    dfdoe1 = sym.diff(fdot_x, oe[1])   # f
    dfdoe2 = sym.diff(fdot_x, oe[2])   # g 
    dfdoe3 = sym.diff(fdot_x, oe[3])   # h
    dfdoe4 = sym.diff(fdot_x, oe[4])   # k
    dfdoe = [dfdoe0, dfdoe1, dfdoe2, dfdoe3, dfdoe4]
    # -> l only

    # -------- Derivatives of gdot_x wrt oe  -------- #
    dgdoe0 = sym.diff(gdot_x, oe[0])   # a
    dgdoe1 = sym.diff(gdot_x, oe[1])   # f
    dgdoe2 = sym.diff(gdot_x, oe[2])   # g 
    dgdoe3 = sym.diff(gdot_x, oe[3])   # h
    dgdoe4 = sym.diff(gdot_x, oe[4])   # k
    dgdoe = [dgdoe0, dgdoe1, dgdoe2, dgdoe3, dgdoe4]
    # -> l only

    # -------- The functions to be used in RQ-Law algorithm -------- #
    fun_lyapunov_control = sym.lambdify(
        [
            mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, k_petro,  
            wp, woe, fdot_xx, gdot_xx, dfdoe_max, dgdoe_max, wl, wscl, deltaL
        ], 
        [[d2, d1, d3], psi], 
        "numpy",
    )
    # -> deltaL, l, lT in d1, d2, d3
    # -> l in psi

    fun_eval_psi = sym.lambdify(
        [
            mu, oe
        ], 
        psi, 
        "numpy",
    )
    # -> l

    fun_eval_dqdoe = sym.lambdify(
        [
            mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, k_petro,
            wp, woe, fdot_xx, gdot_xx, dfdoe_max, dgdoe_max, wl, wscl, deltaL
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
    # -> deltaL only

    fun_eval_fdot = sym.lambdify(
        [
            mu, accel, oe
        ], 
        fdot_x, 
        "numpy",
    )
    # -> l only

    fun_eval_gdot = sym.lambdify(
        [
            mu, accel, oe
        ], 
        gdot_x, 
        "numpy",
    )
    # -> l only

    fun_eval_dfdoe = sym.lambdify(
        [
            mu, accel, oe
        ], 
        dfdoe, 
        "numpy",
    )
    # -> l only

    fun_eval_dgdoe = sym.lambdify(
        [
            mu, accel, oe
        ], 
        dgdoe, 
        "numpy",
    )
    # -> l only

    fun_eval_q = sym.lambdify(
        [
            mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, k_petro,  
            wp, woe, fdot_xx, gdot_xx, dfdoe_max, dgdoe_max, wl, wscl, deltaL
        ], 
        q, 
        "numpy",
    )
    # -> deltaL only (no need for l nor lT)

    fun_eval_qdot = sym.lambdify(
        [
            mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, k_petro,  
            wp, woe, fdot_xx, gdot_xx, dfdoe_max, dgdoe_max, wl, wscl, u, deltaL
        ], 
        qdot, 
        "numpy",
    )
    # -> need deltaL, l, and lT
    print("Generating MEE-SMA lyapunov control funcion with sympy")
    return fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe, fun_eval_fdot, fun_eval_gdot, fun_eval_dfdoe, fun_eval_dgdoe, fun_eval_q, fun_eval_qdot