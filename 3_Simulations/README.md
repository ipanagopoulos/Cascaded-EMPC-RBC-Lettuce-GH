
# Simulation Scripts for EMPC and MPC-RBC  

This directory contains simulation scripts used to generate the results presented in [1].  
To store the simulated results, the boolen variable `storage` should be set to `true`. 

## Scripts  

### `vanHenten_multiShoot_ENMPC-nominal_noSlack.py`  
This Python script simulates the nominal Economic Model Predictive Controller (EMPC) as described in Equation (12) of [1].  

### `vanHenten_multiShoot_RBC-MPC-nominal_warmStart_slackVar.py`  
This Python script simulates a cascaded control approach, where the nominal EMPC computes the optimal decision based on the receding horizon principle. This decision is then passed to a Rule-Based Controller (RBC), which determines the greenhouse control inputs. The mathematical formulation of this control problem is presented in Equation (8) of [1].  

## Results  

The numerical experiment results are stored in `../4_Outputdata` in CSV format. Figures depicting the full simulation length are saved in `../5_PostSimAnalysis`.  

## Reference  
[1] Panagopoulos, I., McAllister, K., Keviczky, T., and van Mourik, S. (2025). *A Cascaded Economic Model Predictive Control Approach to Greenhouse Climate Control.*
