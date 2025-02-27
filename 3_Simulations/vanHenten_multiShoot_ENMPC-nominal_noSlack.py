import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
# Decide if you want to store the results
storage = False
unique_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#%%%%%%%%%%%%%%% CLIMATE-CROP MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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



#%%%%%%%%%%%%%%%% COLLOCATION %%%%%%%%%%%%%%%%%%%%%%%%%%%
h = 300*3
integration_method = "RK4"

if integration_method == "collocation":
    # Number of finite elements
    n = 1

    # Degree of interpolating polynomial
    d = 2

    # Get collocation points
    tau_root = np.append(0, ca.collocation_points(d, 'legendre'))

    # Coefficients of the collocation equation
    C = np.zeros((d+1,d+1))

    # Coefficients of the continuity equation
    D = np.zeros(d+1)

    # Coefficients of the quadrature function
    B = np.zeros(d+1) 

    # Construct polynomial basis
    for j in range(d+1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d+1):
            C[j,r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)
        
    # Total number of variables for one finite element
    X0 =ca.MX.sym('X0',nx)
    U  = ca.MX.sym('U',nu)
    W  = ca.MX.sym('W', nd)
    V = ca.MX.sym('V',d*nx)

    # Get the state at each collocation point
    X = [X0] + ca.vertsplit(V,[r*nx for r in range(d+1)])

    # Get the collocation equations (that define V)
    V_eq = []
    for j in range(1,d+1):
      # Expression for the state derivative at the collocation point
      xp_j = 0
      for r in range (d+1):
        xp_j += C[r,j]*X[r]

      # Append collocation equations
      f_j = f(X[j],U,W)
      V_eq.append(h*f_j - xp_j)

    # Concatenate constraints
    V_eq = ca.vertcat(*V_eq)

    # Root-finding function, implicitly defines V as a function of X0 and P
    vfcn = ca.Function('vfcn', [V, X0, U, W], [V_eq])

    # Convert to SX to decrease overhead
    vfcn_sx = vfcn#.expand()

    # Create a implicit function instance to solve the system of equations
    ifcn = ca.rootfinder('ifcn', 'newton', vfcn_sx)
    V = ifcn(ca.MX(),X0,U,W)
    X = [X0 if r==0 else V[(r-1)*nx:r*nx] for r in range(d+1)]

    # Get an expression for the state at the end of the finite element
    XF = 0
    for r in range(d+1):
      XF += D[r]*X[r]

    # Get the discrete time dynamics
    # F = ca.Function('F', [X0,U,W],[XF])
    F = ca.Function('F', [X0, U, W], [XF],['x0','u', 'd'],['xf'])

    # # Do this iteratively for all finite elements
    # X = X0
    # for i in range(n):
    #   X = F(X,U,W)

    
    # # Fixed-step integrator
    # irk_integrator = ca.Function('irk_integrator', {'x0':X0, 'p':ca.vertcat(U, W), 'xf':X}, 
    #                           ca.integrator_in(), ca.integrator_out())


    # # Create a convensional integrator for reference
    # ref_integrator = ca.integrator('ref_integrator', 'cvodes', dae, 0, h)
elif integration_method == "RK4":
    n_rk4 = 1
    delta_rk4 = h / n_rk4
    x_rk4 = x
    for i in range(n_rk4):
        k_1 = f(x, u, d)
        k_2 = f(x + 0.5 * delta_rk4 * k_1, u, d)
        k_3 = f(x + 0.5 * delta_rk4 * k_2, u, d)
        k_4 = f(x + delta_rk4 * k_3, u, d)
        x_rk4 = x_rk4 + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * delta_rk4
    # Get the discrete time dynamics
    F = ca.Function('F', [x,u,d],[x_rk4],['x0','u', 'd'],['xf'])
        



# Load Initial Conditions and Reference Values
x0_val_irk  = np.array([0.0035, 0.001, 15, 0.008])
x0_val_ref  = x0_val_irk
u_val = np.loadtxt('../1_InputData/vanHentenENMPCinputs.csv',delimiter=',',skiprows=0)
d_val = np.loadtxt('../1_InputData/vanHentenENMPCweather.csv',delimiter=',',skiprows=0)
x_val = np.loadtxt('../1_InputData/vanHentenENMPCstates.csv',delimiter=',',skiprows=0)
y_val = np.loadtxt('../1_InputData/vanHentenENMPCoutputs.csv',delimiter=',',skiprows=0)

#%%%%%%%%%%%%%%%%%%% MPC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###############################################################################

Delta = h
N = 4*6                      # Prediction horizon = 6 hours for h = 15 min
x0 = x_val[:,0]    # Initial Conditions
# Define cost parameters
c_co2 = 0.42*Delta;         # per kg/s of CO2
c_q = 6.35E-9*Delta;        # per W of heat
c_dw = -16;                 # price per kg of dry weight

Nsim = 96*28
t0 = 0
# Initialize closed-loop state and input variables
x_cl = np.zeros((nx,Nsim+1))
y_cl = np.zeros((nx,Nsim+1))
nsp = 4
x_sp_cl = np.zeros((nsp,Nsim))
u_cl = np.zeros((3,Nsim))
slack_cl = np.zeros((1,Nsim))

# Set initial condition
x_cl[:,0] = x0
y_cl[:, 0] = h_meas(x0).full().ravel()
# Set Initial Guess for Control Inputs and Setpoints
init_guess = [0, 0, 0, 0]
init_guess = np.tile(init_guess, (N, 1)).T
# Set weather profile 
d_cl = d_val
# Xsp_ventline_last = init_guess[4]
# Xsp_heatline_last = init_guess[5]
# Xsp_rhmax_last = init_guess[7]
print("Simulation step:")

exec_time = np.zeros((1,Nsim))

for j in range(Nsim):
    print(j)
    # Create empty NLP problem 
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []
    
    d_p = ca.MX.sym('d_p', nd, N)
    
    # "Lift" initial conditions
    Xk = ca.MX.sym('X0', nx)
    w += [Xk]
    lbw += [x0[0], x0[1], x0[2], x0[3]]
    ubw += [x0[0], x0[1], x0[2], x0[3]]
    w0 += [x0[0], x0[1], x0[2], x0[3]]
    
    # Formulate the NLP
    for k in range(N):
        # Control Input: CO2 ext
        Uco2k = ca.MX.sym('Uco2_' + str(k))
        w   += [Uco2k]
        lbw += [0]
        ubw += [1.2]
        w0  += [init_guess[0,k]]
        
        # Control Input: Ventilation flow 
        Uventk = ca.MX.sym('Uvent_' + str(k))
        w   += [Uventk]
        lbw += [0]
        ubw += [7.5]
        w0  += [init_guess[1,k]]
    
        # Control Input: Heating
        Uheatk = ca.MX.sym('Uheat_' + str(k))
        w   += [Uheatk]
        lbw += [0]
        ubw += [150]
        w0  += [init_guess[2,k]]
        
        # # Slack Variable 
        # slackVar_k = ca.MX.sym('slackVar_k_' + str(k))
        # w   += [slackVar_k]
        # lbw += [0]
        # ubw += [20]
        # w0  += [init_guess[3]]
            
        # Integrate till the end of the interval
        Fk = F(x0=Xk, u=ca.vertcat(Uco2k, Uventk, Uheatk), d=d_p[:,k])
        Xk_end = Fk['xf']
        
        # Cost Function        
        J=J + c_dw*(Xk_end[0] - Xk[0]) + c_q*Uheatk  + c_co2*Uco2k*10**(-6) #+ ca.mtimes(slackVar_k.T, slackVar_k)*10**(0) 
        # J = J + Uheatk  + Uventk + 10*Uco2k  + 10*ca.mtimes(slackVar_k.T, slackVar_k)
                                    
        # Add output inequality constraints
        g   += [h_meas(Xk_end)]#-ca.vertcat(0,0,0,slackVar_k)]    
        lbg += [0, 0, 10, 0]
        ubg += [600, 1.6, 25, 80]
    
        # New NLP variable for state at end of interval
        Xk = ca.MX.sym('X_' + str(k+1), nx)
        w   += [Xk]
        lbw += [0, 0, 0, 0]
        ubw += [  600,  0.009, 50, 1] 
        w0  += [x0[0], x0[1], x0[2], x0[3]]
    
        # Add equality constraint
        g   += [Xk_end-Xk]    
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]
        
    # J=J - 10**(3)*h_meas(Xk_end)[0]
                            
    # Create an NLP solver
    # Solver options
    opts = {
        'ipopt': {
            'print_level': 0,      # IPOPT verbosity (0-12, where 5 is moderate)
            'max_iter': 3000,      # Maximum number of iterations
            'warm_start_init_point': 'yes'
            # 'tol': 1e-6,           # Convergence tolerance
            
        },
        'print_time': False,        # Print timing information
        'verbose': False            # CasADi-level verbosity
    }
    prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g), 'p': d_p}
    solver = ca.nlpsol('solver', 'ipopt', prob, opts);
    
    # Start timer
    start_time = time.time()
    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=d_val[:,t0+j:(t0+j+N)])
    # End timer
    end_time = time.time()  
    # Compute elapsed time
    elapsed_time = end_time - start_time
    exec_time[0,j] = elapsed_time
    print(f"Execution time: {elapsed_time:.6f} seconds")
    print("EXIT status:", solver.stats()['return_status'])
    if solver.stats()['return_status'] != 'Solve_Succeeded':
        breakpoint()
        

    w_opt = sol['x'].full().flatten()
    
    # Plot the solution
    x0_opt = w_opt[0::7]
    x1_opt = w_opt[1::7]
    x2_opt = w_opt[2::7]
    x3_opt = w_opt[3::7]
    uco2_opt = w_opt[4::7]
    uvent_opt = w_opt[5::7]
    uheat_opt = w_opt[6::7]    
    #slack_opt = w_opt[7::8]
    
    y_opt = np.zeros((nx, N))
    for i in range(N):
        y_opt[:,i] = h_meas(ca.vertcat(x0_opt[i], x1_opt[i], x2_opt[i], x3_opt[i])).full().ravel()
    
    # Take only the *first* input & Simulate system
    u_cl[0,j] = uco2_opt[0]
    u_cl[1,j] = uvent_opt[0]
    u_cl[2,j] = uheat_opt[0]
    # slack_cl[0,j] = slack_opt[0]
    x_cl[:,j+1] = F(x0=x_cl[:,j], u=ca.vertcat(u_cl[0,j],u_cl[1,j],u_cl[2,j]), d=d_cl[:,t0+j])['xf'].full().ravel()
    y_cl[:, j+1] = h_meas(x_cl[:, j+1]).full().ravel()
    # Update the next initial condition
    x0 = x_cl[:,j+1]
    init_guess[0,:] = np.hstack((uco2_opt[1:], uco2_opt[23]))#u_cl[0,j]
    init_guess[1,:] = np.hstack((uvent_opt[1:], uvent_opt[23]))#u_cl[1,j]
    init_guess[2,:] = np.hstack((uheat_opt[1:], uheat_opt[23]))#u_cl[2,j]
    
    print('next iter \n')

#%%%%%%%%%%%%%%%%%%%%%%%% Plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import matplotlib.image as mpimg  # To read images

# Load an image
img = mpimg.imread('../1_InputData/lettuce_image.png')

# Create a 3x4 grid of subplots
fig, axes = plt.subplots(3, 4, figsize=(15, 10))  # 3 rows, 4 columns

# Generate some example data
x = np.linspace(0, Nsim-1, Nsim)*h/(60*60*24)
# x = np.linspace(0, Nsim-1, Nsim)
# Plot weather conditions
ax = axes[0, 0]
ax.plot(x, d_cl[0,0:Nsim], color='g', label='$[W m^{-2}]$')  # Example plot
ax.set_title('Incoming radiation')  # Title for each subplot
ax.set_ylabel('$[W m^{-2}]$')  # Display legend
ax.grid(True)  # Add grid

ax = axes[0, 1]
ax.plot(x, d_cl[1,0:Nsim], color='g', label='$[m^{-3}]$')  # Example plot
ax.set_title('Outside CO2')  # Title for each subplot
ax.set_ylabel('$[m^{-3}]$')  # Display ylabel
ax.grid(True)  # Add grid

ax = axes[0, 2]
ax.plot(x, d_cl[2,0:Nsim], color='g', label='$[^oC]$')  # Example plot
ax.set_title('Outdoor tempetarure')  # Title for each subplot
ax.set_ylabel('$[^oC]$')  # Display ylabel
ax.grid(True)  # Add grid

ax = axes[0, 3]
ax.plot(x, d_cl[3,0:Nsim], color='g', label='$[kg m^{-3}]$')  # Example plot
ax.set_title('Outdoor humidity content $C_{H2O}$')  # Title for each subplot
ax.set_ylabel('$[kg m^{-3}]$')  # Display ylabel
ax.grid(True)  # Add grid

# Plot control inputs
ax = axes[1, 0]
ax.plot(x, u_val[0,0:Nsim], color='#FFA500', label='EMPC') 
ax.plot(x, u_cl[0,0:Nsim], color='b', label='EMPC') 
ax.set_title('Supply rate of CO2 ')  # Title for each subplot
ax.set_ylabel('$[mg*m^{-2}*s^{-1}]$')  # Display ylabel
ax.legend(loc='best') # Display legend
ax.grid(True)  # Add grid

ax = axes[1, 1]
ax.plot(x, u_val[1,0:Nsim], color='#FFA500', label='EMPC') 
ax.plot(x, u_cl[1,0:Nsim], color='b', label='EMPC') 
ax.set_title('Ventilation rate through the vents')  # Title for each subplot
ax.set_ylabel('$[mm s^{-1}]$')  # Display ylabel
ax.legend(loc='best') # Display legend
ax.grid(True)  # Add grid

ax = axes[1, 2]
ax.plot(x, u_val[2,0:Nsim], color='#FFA500', label='EMPC') 
ax.plot(x, u_cl[2,0:Nsim], color='b', label='EMPC') 
ax.set_title('Energy supply by the heating system')  # Title for each subplot
ax.set_ylabel('$[W*m^{-2}]$')  # Display ylabel
ax.legend(loc='best') # Display legend
ax.grid(True)  # Add grid

ax = axes[1, 3]
ax.imshow(img)  # Example plot
ax.axis('off')

# Plot states and setpoints
ax = axes[2, 0]
ax.plot(x, y_val[0,0:Nsim], color='#FFA500', label='EMPC') 
ax.plot(x, y_cl[0,0:Nsim], label='EMPC')  
ax.set_title('weight')  # Title for each subplot
ax.set_ylabel('$[g m^{-2}]$')  # Display ylabel
ax.legend(loc='best') # Display legend
ax.grid(True)  # Add grid

ax = axes[2, 1]
ax.plot(x, x_val[1,0:Nsim], color='#FFA500', label='EMPC') 
ax.plot(x, x_cl[1,0:Nsim], label='EMPC')  
# ax.plot(x, x_sp_cl[0,0:Nsim+1], color='r', linestyle='-', dashes=[5, 2], label='Co2 SP')
ax.set_title('indoor CO2')  # Title for each subplot
ax.set_ylabel('$[kg m^{-3}]$')  
ax.legend(loc='best') # Display legend
ax.grid(True)  # Add grid

ax = axes[2, 2]
ax.plot(x, y_val[2,0:Nsim], color='#FFA500', label='EMPC') 
ax.plot(x, y_cl[2,0:Nsim], label='EMPC')  
# ax.plot(x, x_sp_cl[1,0:Nsim+1], color='b', linestyle='-', dashes=[5, 1], label='Ventline')
# ax.plot(x, x_sp_cl[2,0:Nsim+1], color='r', linestyle='-', dashes=[5, 1], label='Heatline')
ax.set_title('air temperature')  # Title for each subplot
ax.set_ylabel('$[^oC]$')  # Display ylabel
ax.legend(loc='best') # Display legend
ax.grid(True)  # Add grid

ax = axes[2, 3]
ax.plot(x, y_val[3,0:Nsim], color='#FFA500', label='EMPC') 
ax.plot(x, y_cl[3,0:Nsim], label='EMPC')  
# ax.plot(x, x_sp_cl[3,0:Nsim+1], color='r', linestyle='-', dashes=[5, 1], label='RHmax')
ax.set_title('humidity')  # Title for each subplot
ax.set_ylabel('$[\%]$')  # Display ylabel
ax.legend(loc='best') # Display legend
ax.grid(True)  # Add grid

# Adjust layout to prevent overlap
plt.tight_layout()

if storage:
    # Save the figure with the unique identifier in the filename
    filename = f'../5_PostSimAnalysis/ENMPC-nominal/{unique_id}_vanHenten_ENMPC-nominal.png'
    plt.savefig(filename, dpi=300)  # Save as a PNG with high resolution

# Show the plot
plt.show()

#%%%%%%%%%%%%%%%% Store Data and Figure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save as CSV
if storage:
    np.savetxt(f'../4_OutputData/ENMPC-nominal/{unique_id}_vanHenten_ENMPC-nominal_states.csv', x_cl, delimiter=",", fmt='%f')  
    np.savetxt(f'../4_OutputData/ENMPC-nominal/{unique_id}_vanHenten_ENMPC-nominal_outputs.csv', y_cl, delimiter=",", fmt='%f')  
    np.savetxt(f'../4_OutputData/ENMPC-nominal/{unique_id}_vanHenten_ENMPC-nominal_inputs.csv', u_cl, delimiter=",", fmt='%f')  