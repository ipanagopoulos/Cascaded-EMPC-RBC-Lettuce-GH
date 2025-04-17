import casadi as ca
import numpy as np

def vanHenten_LettuceModel():
    p1 = 0.544; p2 = 2.65e-07; p3 = 53; p4 = 3.55e-09;
    p5 = 5.11e-06; p6 = 0.00023; p7 = 0.000629;
    p8 = 5.2e-05; p9 = 4.1; p10 = 4.87e-07;
    p11 = 7.5e-06; p12 = 8.31; p13 = 273.15;
    p14 = 101325; p15 = 0.044; p16 = 30000;
    p17 = 1290; p18 = 6.1; p19 = 0.2; p20 = 4.1;
    p21 = 0.0036; p22 = 9348; p23 = 8314;
    p24 = 273.15; p25 = 17.4; p26 = 239;
    p27 = 17.269; p28 = 238.3;
    
    w1 = 0; w2 = 0; w3 = 0; w4 = 0; # No disturbances
    
    # Dimensions
    nx = 4
    nu = 3
    nd = 4
    ny = 4
    
    # Declare variables
    x  = ca.SX.sym('x', nx)  # state
    u  = ca.SX.sym('u', nu)  # control
    d  = ca.SX.sym('d', nd)  # exogenous inputs
    y  = ca.SX.sym('y', ny)  # outputs
    
    y[0] = 10**(3)*x[0];                                                                 # Weight in g m^{-2}
    y[1] = p12*(x[2]+p13)/(p14*p15)*x[1]*10**(3);                                 # Indoor CO2 in ppm 10^{3}
    y[2] = x[2];                                                                       # Air Temp in  ^oC
    y[3] = p12*(x[2] + p13)/(11*ca.exp(p27*x[2]/(x[2]+p28)))*x[3]*10**(2);             # RH C_{H2O} in %
    
    phi       = p4*d[0] + (-p5*x[2]**2 + p6*x[2] - p7)*(x[1] - p8);
    PhiPhot_c = (1-ca.exp(-p3*x[0]))*(p4*d[0]*(-p5*x[2]**2 + p6*x[2] - p7)*(x[1]-p8))/phi;         # gross canopy phootsynthesis rate
    PhiVent_c = (u[1]*10**(-3) + p11)*(x[1]-d[1]);                                           # mass exhcnage of CO2 thorought the vents
    PhiVent_h = (u[1]*10**(-3) + p11)*(x[3] - d[3]);                                         # canopy transpiration
    PhiTransp_h = p21*(1 - ca.exp(-p3*x[0]))*(p22/(p23*(x[2]+p24))*ca.exp(p25*x[2]/(x[2]+p26))-x[3]); # mass exchange of H2) through the vents
    
    dx1dt = (p1*PhiPhot_c - p2*x[0]*2**(x[2]/10 - 5/2))*(1+w1);
    dx2dt = 1/p9*(-PhiPhot_c + p10*x[0]*2**(x[2]/10 - 5/2) + u[0]*10**(-6) - PhiVent_c)*(1+w2);
    dx3dt = 1/p16*(u[2] - (p17*u[1]*10**(-3) + p18)*(x[2] - d[2]) + p19*d[0])*(1+w3);
    dx4dt = 1/p20*(PhiTransp_h - PhiVent_h)*(1+w4);
    
    ode = ca.vertcat(dx1dt, dx2dt, dx3dt, dx4dt)
    
    dae = {'x':x, 'p':ca.vertcat(u,d), 'ode':ode}
    
    # Continuous time climate-crop dynamics
    f = ca.Function('f', [x, u, d], [ode])
    
    h_meas = ca.Function('h_meas', [x], [y])
    return f, h_meas
