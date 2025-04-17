import casadi as ca 

def RK4(h, x, u, d, f):
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
     return F