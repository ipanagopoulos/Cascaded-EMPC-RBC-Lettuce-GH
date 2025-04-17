
# Input Files for Modelling and Simulation

This folder contains files that are used as inputs for modelling and simulation purposes.

## `weather_5min.csv`
Contains weather data that can be used as disturbances/exogenous inputs for the *van Henten* lettuce model. 

The weather data were taken from:
- [https://github.com/davkat1/GreenLight](https://github.com/davkat1/GreenLight)
- [https://doi.org/10.1016/j.biosystemseng.2020.03.010](https://doi.org/10.1016/j.biosystemseng.2020.03.010)

These represent the weather in Amsterdam, the Netherlands, for two winter months (January and February).

`Weather Variables:`
- `d1`: Incoming radiation in $[\mathrm{W \ m}^{-2}$]
- `d2`: Outside CO2 in  $[\mathrm{kg \ m}^{-3}]$
- `d3`: Outdoor temperature in $[^o \mathrm{C}]$
- `d4`: Outdoor humidity content $\mathrm{C}_{\mathrm{H2O}}$ in $[\mathrm{kg \ m}^{-3}]$

---

## `vanHentenENMPCxxxxx.csv`
Contains input, output, weather and state data from a simulated experiment run on the *van Henten* lettuce model. The numerical experiment represents the results from an **Economic Nonlinear Model Predictive Controller (ENMPC)** run for the nominal case. The ENMPC problem was solved using **Opti** in **CasADi**. 

The weather data used are stored in `vanHentenENMPCweather.csv`.

### `ENMPC Formulation:`
$$
\begin{aligned}
\min_{x, u} & \quad \sum_{k=k_{0}}^{k_{0}+N}  -c_{\mathrm{dw}} \Delta x_{\mathrm{dw}}(k)  + c^{T}_{u} u(k)
 \\
\textrm{s.t.} & \quad x(k+1) = f(x(k), u(k), d(k)), \\
& \quad y(k) = h(x(k)), \\
& \quad \underline{u} \leq u(k) \leq \overline{u}, \\
& \quad \underline{y} \leq y(k) \leq \overline{y}, \quad k = k_0, \dots, k_0 + N,
\end{aligned}
$$

where $c_{\mathrm{dw}} = 16$, $c_{u} = \left(378 \cdot 10^{-6} , \ 0, \ 5715 \cdot 10^{-9} \right)^{T}$  are weights for the dry weight change ($\Delta x_{\mathrm{dw}}$), and actuation effort ($u$),  the input constraints are $\underline{u}= \left( 0, \ 0, \ 0 \right)^{T}$ and $\overline{u}= \left(1.2, \ 7.5, \ 150 \right)^{T}$, the output constraints $\underline{y} = \left(0, \ 0.4, \ 10, \ 40\right)^{T}$,  $\quad \overline{y} = \left(\infty, \ 1.6, \ 25, \ 80\right)^{T}$


---

### `vanHentenENMPCstates.csv`
- `x1`: Dry weight in $[\mathrm{kg \ m}^{-2}]$
- `x2`: Indoor CO2 in $[\mathrm{kg \ m}^{-3}]$
- `x3`: Air temperature in $[^o \mathrm{C}]$
- `x4`: Humidity in $[\mathrm{kg \ m}^{-3}]$

### `vanHentenENMPCoutputs.csv`
- `y1`: Weight in $[\mathrm{g \ m}^{-2}]$
- `y2`: Indoor CO2 in $[\mathrm{ppm} \ 10^{3}]$
- `y3`: Air temperature in $[^o \mathrm{C}]$
- `y4`: Relative humidity  $\mathrm{C}_{\mathrm{H2O}}$ in [%]

### `vanHentenENMPCinputs.csv`
- `u1`: Supply rate of  $\mathrm{C}_{2}$ in $[\mathrm{mg \ m}^{-2} \mathrm{s}^{-1}]$
- `u2`: Ventilation rate through the vents in $[\mathrm{mm \ s}^{-1}]$
- `u3`: Energy supply by the heating system in $[\mathrm{W \ m}^{-2}]$

### `vanHentenENMPCweather.csv`
- `d1`: Incoming radiation in $[\mathrm{W \ m}^{-2}$]
- `d2`: Outside CO2 in  $[\mathrm{kg \ m}^{-3}]$
- `d3`: Outdoor temperature in $[^o \mathrm{C}]$
- `d4`: Outdoor humidity content $\mathrm{C}_{\mathrm{H2O}}$ in $[\mathrm{kg \ m}^{-3}]$
