import casadi as ca 
import numpy as np 

def collocation(h, nx, nu, nd, f):
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
    return F
    # # Do this iteratively for all finite elements
    # X = X0
    # for i in range(n):
    #   X = F(X,U,W)
    
    
    # # Fixed-step integrator
    # irk_integrator = ca.Function('irk_integrator', {'x0':X0, 'p':ca.vertcat(U, W), 'xf':X}, 
    #                           ca.integrator_in(), ca.integrator_out())
    
    
    # # Create a convensional integrator for reference
    # ref_integrator = ca.integrator('ref_integrator', 'cvodes', dae, 0, h)