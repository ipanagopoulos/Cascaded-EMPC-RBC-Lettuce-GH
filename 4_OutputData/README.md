This directory contains the outputs for the two simulated experiments (Integrated ENMPC, Cascaded EMPC with RBC) presented in our paper. The data stored in the files are as follows:

### **states**:
- `x1`: Dry weight in $[\mathrm{kg \ m}^{-2}]$
- `x2`: Indoor CO2 in $[\mathrm{kg \ m}^{-3}]$
- `x3`: Air temperature in $[^o \mathrm{C}]$
- `x4`: Humidity in $[\mathrm{kg \ m}^{-3}]$

### **outputs**:
- `y1`: Weight in $[\mathrm{g \ m}^{-2}]$
- `y2`: Indoor CO2 in $[\mathrm{ppm} \ 10^{3}]$
- `y3`: Air temperature in $[^o \mathrm{C}]$
- `y4`: Relative humidity  $\mathrm{C}_{\mathrm{H2O}}$ in $[\%]$

### **inputs**:
- `u1`: Supply rate of  $\mathrm{CO}_{2}$ in $[\mathrm{mg \ m}^{-2} \mathrm{s}^{-1}]$
- `u2`: Ventilation rate through the vents in $[\mathrm{mm \ s}^{-1}]$
- `u3`: Energy supply by the heating system in $[\mathrm{W \ m}^{-2}]$

### **setpoints**:
- `sp1`: $\mathrm{CO}_{2}$ bound in $[\mathrm{kg \ m}^{-3}]$
- `sp2`: Ventilation line in $[^o \mathrm{C}]$
- `sp3`: Heating line in $[^o \mathrm{C}]$
- `sp4`: RH max in $[\%]$

