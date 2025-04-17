## `RK4_integration.py`
Runge-Kutta 4 integration method receiving as inputs the time interval h, the symbolic state vector x, the symbolic input vector u, the symbolic disturbance vector d and the symbolic function f (in this case the lettuce model). 

## `collocation.py`
Python code implementing the collocation integration method receiving as inputs the time interval h, the dimension of the state vector (nx), the dimension of the input vector (nu), the dimension of the disturbance vector (nd) and the symbolic function f (in this case the lettuce model). 

## `lettuce_model.py`
The climate-crop lettuce model as originally presented in [1] written in a symbolic form using casAdi [2].


## References
[1] van Henten, E.J. (1994). Greenhouse climate management: an optimal control approach. Ph.D. thesis, Wageningen
University & Research, Wageningen, Netherlands. PhD
Dissertation

[2] Andersson, J., Gillis, J., Horn, G., Rawlings, J., and Diehl,
M. (2018). *Casadi: a software framework for nonlinear optimization and optimal control.* Mathematical Pro-
gramming Computation, 11.
