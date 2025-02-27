vanHentenENMPCweather: csv file containing weather data coming from Bleisjiwk NL 
			(Source of Weather Data: https://github.com/davkat1/GreenLight || https://doi.org/10.1016/j.biosystemseng.2020.03.010)
			d1: Incoming radiation in W m^{-2}
			d2: Outside CO2 in kg m^{-3}
			d3: Outdoor tempetarure in ^oC
			d4: Outdoor humidity content C_{H2O} in kg m^{-3}



vanHentenENMPCinputs.csv vanHentenENMPCoutput.csv vanHentenENMPCstates.csv refer to input, output and state 
data coming from a simulated experiment run on the van Henten letture model. The numerical experiment
represents the results from an Economic Nonlinear Model Predictive Controller run for the nominal 
case. The weather data used live in vanHentenENMPCweather.csv. 

The ENMPC formulation can be found below:

\min_{x, u} \sum_{k=0}^{N-1} c_{dw}(x_{dw}(k+1) - x_{dw}(k)) + c_q u_{heat}(k) + c_co2 u_{co2}(k)
	s.t.
		0 <= x(k)
		[0 0 10 0] <= y(k) <= [inf, 1.6, 30, 80]
		[0 0 0] <= u(k) <= [1.2 7.5 150]