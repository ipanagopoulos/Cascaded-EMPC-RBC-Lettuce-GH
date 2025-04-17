import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os


storage = True

# u_val = np.loadtxt('../1_InputData/vanHentenENMPCinputs.csv',delimiter=',',skiprows=0)
# d_val = np.loadtxt('../1_InputData/vanHentenENMPCweather.csv',delimiter=',',skiprows=0)
# x_val = np.loadtxt('../1_InputData/vanHentenENMPCstates.csv',delimiter=',',skiprows=0)
# y_val = np.loadtxt('../1_InputData/vanHentenENMPCoutputs.csv',delimiter=',',skiprows=0)

u_val = np.loadtxt('../4_OutputData/ENMPC-nominal/2025-02-12_19-41-15_vanHenten_ENMPC-nominal_inputs.csv',delimiter=',',skiprows=0)
d_val = np.loadtxt('../1_InputData/vanHentenENMPCweather.csv',delimiter=',',skiprows=0)
x_val = np.loadtxt('../4_OutputData/ENMPC-nominal/2025-02-12_19-41-15_vanHenten_ENMPC-nominal_states.csv',delimiter=',',skiprows=0)
y_val = np.loadtxt('../4_OutputData/ENMPC-nominal/2025-02-12_19-41-15_vanHenten_ENMPC-nominal_outputs.csv',delimiter=',',skiprows=0)


u_cl = np.loadtxt('../4_OutputData/ENMPC-RBC-nominal/2025-02-12_16-17-37_vanHenten_ENMPC-RBC-nominal_inputs.csv',delimiter=',',skiprows=0)
x_sp_cl = np.loadtxt('../4_OutputData/ENMPC-RBC-nominal/2025-02-12_16-17-37_vanHenten_ENMPC-RBC-nominal_setpoints.csv',delimiter=',',skiprows=0)
x_cl = np.loadtxt('../4_OutputData/ENMPC-RBC-nominal/2025-02-12_16-17-37_vanHenten_ENMPC-RBC-nominal_states.csv',delimiter=',',skiprows=0)
y_cl = np.loadtxt('../4_OutputData/ENMPC-RBC-nominal/2025-02-12_16-17-37_vanHenten_ENMPC-RBC-nominal_outputs.csv',delimiter=',',skiprows=0)
d_cl = d_val



#%%%%%%%%%%%%%%%%%%%%%%%% Post Sim Analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nsim_full = 96*28
h = 300*3

# CO2 Injection
time = np.linspace(0, Nsim_full-1, Nsim_full)*h
total_co2_ENMPC = np.trapz(u_val[0,0:Nsim_full], time)*1e-3 # in mg*m^{-2}
total_co2_ENMPC_RBC = np.trapz(u_cl[0,:], time)*1e-3
print('ENMPC CO2 Injection in g*m^{-2}: ', total_co2_ENMPC)
print('ENMPC-RBC CO2 Injection in g*m^{-2}: ', total_co2_ENMPC_RBC)
print('Abs Diff: CO2 Injection in g*m^{-2}: ', total_co2_ENMPC - total_co2_ENMPC_RBC)
print('Rel Diff: CO2 Injection in g*m^{-2}: ', (total_co2_ENMPC_RBC - total_co2_ENMPC)/total_co2_ENMPC*100)
print('  ')

# ventilation
time = np.linspace(0, Nsim_full-1, Nsim_full)*h
total_vent_ENMPC = np.trapz(u_val[1,0:Nsim_full], time) # in mg*m^{-2}
total_vent_ENMPC_RBC = np.trapz(u_cl[1,:], time)
print('ENMPC vent in mm: ', total_vent_ENMPC)
print('ENMPC-RBC vent in mm: ', total_vent_ENMPC_RBC)
print('Abs Diff: vent in mm: ', total_vent_ENMPC - total_vent_ENMPC_RBC)
print('Rel Diff: vent in mm: ', (total_vent_ENMPC_RBC - total_vent_ENMPC)/total_vent_ENMPC*100)
print('  ')

# Heating
total_heat_ENMPC =  np.sum(u_val[2,0:Nsim_full])*h/3.6e6# in kWh m^{-2}
total_heat_ENMPC_RBC = np.sum(u_cl[2,:])*h/3.6e6
print('ENMPC heat supply in kWh*m^{-2}: ', total_heat_ENMPC)
print('ENMPC-RBC heat supply in kWh*m^{-2}: ', total_heat_ENMPC_RBC)
print('Abs Diff: heat supply in kWh*m^{-2}: ', total_heat_ENMPC - total_heat_ENMPC_RBC)
print('Rel Diff: heat supply in kWh*m^{-2}: ', (total_heat_ENMPC_RBC - total_heat_ENMPC)/total_heat_ENMPC*100)
print('  ')

# Heating Efficiency= yield/heating
heatEf_ENMPC = x_val[0,Nsim_full]/total_heat_ENMPC
heatEf_ENMPC_RBC = x_cl[0,Nsim_full]/total_heat_ENMPC_RBC
print('ENMPC heat effigicieny: ', heatEf_ENMPC)
print('ENMPC-RBC heat effigicieny: ', heatEf_ENMPC_RBC)
print('heat efficiency Rel.Diff: ', (heatEf_ENMPC_RBC - heatEf_ENMPC)/heatEf_ENMPC*100)
print('  ')

# CO2 Efficiency= yield/co2
co2Ef_ENMPC = x_val[0,Nsim_full]/total_co2_ENMPC
co2Ef_ENMPC_RBC = x_cl[0,Nsim_full]/total_co2_ENMPC_RBC
print('ENMPC co2 effigicieny: ', co2Ef_ENMPC)
print('ENMPC-RBC co2 effigicieny: ', co2Ef_ENMPC_RBC)
print('co2 efficiency Rel.Diff: ', (co2Ef_ENMPC_RBC - co2Ef_ENMPC)/co2Ef_ENMPC*100)
print('  ')

# Dry yield difference
print('ENMPC Dry weight x_{dw}(N) in kg*m^{-2}: ', x_val[0,Nsim_full-1])
print('ENMPC-RBC Dry weight x_{dw}(N) in kg*m^{-2}: ', x_cl[0,Nsim_full-1])
print('Abs Diff: Dry Weight Difference in kg*m^{-2}: ', x_val[0,Nsim_full-1] - x_cl[0,Nsim_full-1])
print('Rel Diff: Dry Weight Difference in kg*m^{-2}: ', (x_cl[0,Nsim_full-1] - x_val[0,Nsim_full-1])/x_val[0,Nsim_full-1]*100)
print('  ')


#%%%%%%%%%%%%%%%%%%%%%% Data Preproccesing for Plotting %%%%%%%%%%%%%%%%%%%%%%%
Nsim = 96*28

# Convert weather units 
p1 = 0.544; p2 = 2.65e-07; p3 = 53; p4 = 3.55e-09;
p5 = 5.11e-06; p6 = 0.00023; p7 = 0.000629;
p8 = 5.2e-05; p9 = 4.1; p10 = 4.87e-07;
p11 = 7.5e-06; p12 = 8.31; p13 = 273.15;
p14 = 101325; p15 = 0.044; p16 = 30000;
p17 = 1290; p18 = 6.1; p19 = 0.2; p20 = 4.1;
p21 = 0.0036; p22 = 9348; p23 = 8314;
p24 = 273.15; p25 = 17.4; p26 = 239;
p27 = 17.269; p28 = 238.3;

# Co2 mass exchange rate through the vents
mEx_ENMPC = np.zeros((1,Nsim));
mEx_ENMPC = (u_val[1,0:Nsim]*10**(-3) + p11)*(x_val[1,0:Nsim]-d_val[1,0:Nsim]) # kg*s^{-1}

mEx_ENMPC_RBC = np.zeros((1,Nsim));
mEx_ENMPC_RBC = (u_cl[1,0:Nsim]*10**(-3) + p11)*(x_cl[1,0:Nsim]-d_cl[1,0:Nsim])

# Co2 mass exchange 
# print(np.trapz(mEx_ENMPC[0:2400],time[0:2400]))
# print(np.trapz(mEx_ENMPC_RBC[0:2400],time[0:2400]))
print('CO2 mass exchange')
print(np.trapz(mEx_ENMPC[0:Nsim],time[0:Nsim]))
print(np.trapz(mEx_ENMPC_RBC[0:Nsim],time[0:Nsim]))
print((np.trapz(mEx_ENMPC_RBC[0:Nsim],time[0:Nsim])-np.trapz(mEx_ENMPC[0:Nsim],time[0:Nsim]))/np.trapz(mEx_ENMPC[0:Nsim],time[0:Nsim])*100)
print(' ')
# print(np.trapz(mEx_ENMPC[2400:2400+96*2],time[2400:2400+96*2]))
# print(np.trapz(mEx_ENMPC_RBC[2400:2400+96*2],time[2400:2400+96*2]))
# print(' ')

# heat Exchange with the environment
hEx_ENMPC = np.zeros((1,Nsim));
hEx_ENMPC = (p17*u_val[1,0:Nsim]*10**(-3) + p18)*(x_val[2,0:Nsim] - d_val[2,0:Nsim])

hEx_ENMPC_RBC = np.zeros((1,Nsim));
hEx_ENMPC_RBC = (p17*u_cl[1,0:Nsim]*10**(-3) + p18)*(x_cl[2,0:Nsim] - d_cl[2,0:Nsim])
print('Heat Exchange')
print(np.trapz(hEx_ENMPC,time)*h/3.6e6) # kWh m^{-2}
print(np.trapz(hEx_ENMPC_RBC,time)*h/3.6e6) # kWh m^{-2}
print((np.trapz(hEx_ENMPC_RBC,time)*h/3.6e6-np.trapz(hEx_ENMPC,time)*h/3.6e6)/np.trapz(hEx_ENMPC,time)*h/3.6e6*100)

# Total photosynthesis
print(np.trapz((1-np.exp(-p3*x_val[0,0:Nsim]))*(p4*d_val[0,0:Nsim]*(-p5*x_val[2,0:Nsim]**2 \
    + p6*x_val[2,0:Nsim] - p7)*(x_val[1,0:Nsim]-p8))/(p4*d_val[0,0:Nsim] + (-p5*x_val[2,0:Nsim]**2 + p6*x_val[2,0:Nsim] - p7)*(x_val[1,0:Nsim] - p8)),time))
    
print(np.trapz((1-np.exp(-p3*x_cl[0,0:Nsim]))*(p4*d_cl[0,0:Nsim]*(-p5*x_cl[2,0:Nsim]**2 \
    + p6*x_cl[2,0:Nsim] - p7)*(x_cl[1,0:Nsim]-p8))/(p4*d_cl[0,0:Nsim] + (-p5*x_cl[2,0:Nsim]**2 + p6*x_cl[2,0:Nsim] - p7)*(x_cl[1,0:Nsim] - p8)),time))

##################################################################################################################################
# start = 0#2400 
# end = 2496#2688#2400 + 96
start = 2400 + 96
end = 2400 + 96 * 2
Nsim = end-start
u_val = u_val[:,start:end]
d_val = d_val[:,start:end]
x_val = x_val[:,start:end]
y_val = y_val[:,start:end]

u_cl = u_cl[:,start:end]
d_cl = d_cl[:,start:end]
x_cl = x_cl[:,start:end]
x_sp_cl = x_sp_cl[:,start:end]
y_cl = y_cl[:,start:end]
 
co2Ppm      = p12*(d_val[2,:]+p13)/(p14*p15)*d_val[1,:]*10**(3);                                 # Indoor CO2 in ppm 10^{3}              
d_val[1,:] = co2Ppm       
x_co2sp_ppm = p12*(x_cl[2,:Nsim]+p13)/(p14*p15)*x_sp_cl[0,:]*10**(3);                                                 
RHperCent = p12*(d_val[2,:] + p13)/(11*np.exp(p27*d_val[2,:]/(d_val[2,:]+p28)))*d_val[3,:]*10**(2);             # RH C_{H2O} in %
d_val[3,:] = RHperCent

#%%%%%%%%%%%%%%%%%%%%%%%% Plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import matplotlib.image as mpimg  # To read images
# start = 0; end = 28; 

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "CMU Serif"
})

# Create a 3x4 grid of subplots
fig, axes = plt.subplots(3, 4, figsize=(15, 10))  # 3 rows, 4 columns

# Generate some example data
# x = np.linspace(0, Nsim-1, Nsim)*h/(60*60*24)
x = np.linspace(start, end-1, Nsim)*h/(60*60*24)
# x = np.linspace(0, Nsim-1, Nsim)
# Plot weather conditions
ax = axes[0, 0]
ax.plot(x, d_cl[0,0:Nsim], color='black', linewidth=2, label='$[\mathrm{W}\!\cdot\!\mathrm{m}^{-2}]$')  # Example plot
ax.set_title('Incoming radiation', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$d_{\mathrm{I}} \ \ [\mathrm{W}\!\cdot\!\mathrm{m}^{-2}]$', fontsize = 14.0)  # Display legend
ax.grid(True)  # Add grid

ax = axes[0, 1]
ax.plot(x, d_cl[1,0:Nsim], color='black', linewidth=2, label='$[\mathrm{ppm} \ 10^{3}]$')  # Example plot
ax.set_title('Outside CO$_{2}$', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$d_{\mathrm{CO}_{2}} \ \ [\mathrm{ppm} \ 10^{3}]$', fontsize = 14.0)  # Display ylabel
ax.grid(True)  # Add grid
ax.set_ylim(0.2,0.6)

ax = axes[0, 2]
ax.plot(x, d_cl[2,0:Nsim], color='black', linewidth=2, label='$[^o\mathrm{C}]$')  # Example plot
ax.set_title('Outdoor tempetarure', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$d_{\mathrm{T}} \ \ [^o\mathrm{C}]$', fontsize = 14.0)  # Display ylabel
ax.grid(True)  # Add grid

ax = axes[0, 3]
ax.plot(x, d_cl[3,0:Nsim], color='black', linewidth=2, label='$[\%]$')  # Example plot
ax.set_title('Outdoor relative humidity', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$d_{\mathrm{H}} \ \ [\%]$', fontsize = 14.0)  # Display ylabel
ax.grid(True)  # Add grid

# Plot control inputs
ax = axes[1, 0]
ax.plot(x, u_val[0,0:Nsim], color='darkorange', linewidth=2, label='EMPC') 
ax.plot(x, u_cl[0,0:Nsim], color='darkviolet', linewidth=2, linestyle='-', dashes=[4, 1], label='EMPC-RBC') 
ax.set_title('CO$_{2}$ supply rate ', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$u_{\mathrm{CO}_{2}} \ \ [\mathrm{mg}\!\cdot\!\mathrm{m}^{-2}\!\cdot\!\mathrm{s}^{-1}]$', fontsize = 14.0)  # Display ylabel
ax.grid(True)  # Add grid

ax = axes[1, 1]
ax.plot(x, u_val[1,0:Nsim], color='darkorange', linewidth=2, label='EMPC') 
ax.plot(x, u_cl[1,0:Nsim], color='darkviolet', linewidth=2, linestyle='-', dashes=[4, 1], label='EMPC-RBC') 
ax.set_title('Ventilation rate', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$u_{\mathrm{vent}} \ \ [\mathrm{mm}\!\cdot\!\mathrm{s}^{-1}]$', fontsize = 14.0)  # Display ylabel
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)

ax = axes[1, 2]
ax.plot(x, u_val[2,0:Nsim], color='darkorange', linewidth=2, label='EMPC') 
ax.plot(x, u_cl[2,0:Nsim], color='darkviolet', linewidth=2, linestyle='-', dashes=[4, 1], label='EMPC-RBC') 
ax.set_title('Heating energy supply', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$u_{\mathrm{heat}} \ \ [\mathrm{W}\!\cdot\!\mathrm{m}^{-2}]$', fontsize = 14.0)  # Display ylabel
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)

# Add a global legend specifically in axes[1, 3]
handles = [
    plt.Line2D([], [], color='darkorange', linewidth=2, label='EMPC (Benchmark)'),
    plt.Line2D([], [], color='darkviolet', linewidth=2, linestyle='-', dashes=[4, 1], label='EMPC with RBC integration')
]

# Turn off the axes[1, 3] subplot and use it for the legend
axes[1, 3].axis('off')  # Disable the plot area of axes[1, 3]
axes[1, 3].legend(handles=handles, loc='center', fontsize=12, frameon=False)  # Add legend to the center

# ax = axes[1, 3]
# # ax.imshow(img)  # Example plot
# ax.axis('off')

# Plot states and setpoints
ax = axes[2, 0]
ax.plot(x, y_val[0,0:Nsim], color='darkorange', linewidth=2, label='EMPC') 
ax.plot(x, y_cl[0,0:Nsim], color='darkviolet', linewidth=2, linestyle='-', dashes=[4, 1], label='EMPC-RBC')  
ax.set_title('Dry weight', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$y_{\mathrm{dw}} \ \ [\mathrm{g}\!\cdot\!\mathrm{m}^{-2}]$', fontsize = 14.0)  # Display ylabel
# ax.legend(loc='best') # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
ax.set_xlabel('Time [Days]', fontsize = 14.0)


ax = axes[2, 1]
ax.plot(x, y_val[1,0:Nsim], color='darkorange', linewidth=2)#, label='EMPC') 
ax.plot(x, y_cl[1,0:Nsim], color='darkviolet', linewidth=2, linestyle='-', dashes=[4, 1])#, label='EMPC-RBC')  
ax.plot(x, x_co2sp_ppm[0:Nsim], color='red', linewidth=1.5, linestyle='-', dashes=[5, 0], label='$s_{\mathrm{CO}_{2}}$')
ax.set_title('Indoor CO$_{2}$', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$y_{\mathrm{CO}_{2}} \ \ [\mathrm{ppm} \ 10^{3}]$', fontsize = 14.0)  
ax.legend(loc='best', fontsize=12.0) # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
ax.set_xlabel('Time [Days]', fontsize = 14.0)

ax = axes[2, 2]
ax.plot(x, y_val[2,0:Nsim], color='darkorange', linewidth=2)#, label='EMPC') 
ax.plot(x, y_cl[2,0:Nsim], color='darkviolet', linewidth=2, linestyle='-', dashes=[4, 1])#, label='EMPC-RBC')  
ax.plot(x, x_sp_cl[1,0:Nsim], color='darkcyan',  linewidth=1.5, linestyle='-', dashes=[5, 0], label='$s_{\mathrm{vent}}$')
ax.plot(x, x_sp_cl[2,0:Nsim], color='red',  linewidth=1.5, linestyle='-', dashes=[5, 0], label='$s_{\mathrm{heat}}$')
ax.set_title('Air temperature', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$y_{\mathrm{T}} \ \ [^o\mathrm{C}]$', fontsize = 14.0)  # Display ylabel
ax.legend(loc='best', fontsize = 12.0) # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
ax.set_xlabel('Time [Days]', fontsize = 14.0)

ax = axes[2, 3]
ax.plot(x, y_val[3,0:Nsim], color='darkorange', linewidth=2)#, label='EMPC') 
ax.plot(x, y_cl[3,0:Nsim], color='darkviolet', linewidth=2, linestyle='-', dashes=[4, 1])#, label='EMPC-RBC')  
ax.plot(x, x_sp_cl[3,0:Nsim], color='red', linewidth=1.5, label='$s_{\mathrm{RHmax}}$')
ax.set_title('Relative Humidity', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$y_{\mathrm{RH}} \ \ [\%]$', fontsize = 14.0)  # Display ylabel
ax.legend(loc='best', fontsize = 12.0) # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
ax.set_xlabel('Time [Days]', fontsize = 14.0)

# Adjust layout to prevent overlap
plt.tight_layout()

if storage:
    # Save the figure with the unique identifier in the filename
    filename = f'../5_PostSimAnalysis/ENMPC_vs_ENMPC-RBC_Comparison/Simulation_Results.png'
    plt.savefig(filename, dpi=300)  # Save as a PNG with high resolution

# Show the plot
plt.show()
