import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import os

### Initialization

# some configuration parameters
global Input_Filename    #!!! name of the input data file
Input_Filename = 'MainProjFailprobVSNoise251016.pkl'
Output_Filename = 'Proj_PfailVSNoise_251016.eps'

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
pkl_file_path = os.path.join(script_dir, Input_Filename)
plot1_file_path = os.path.join(script_dir, Output_Filename)

### Load data from the .pkl file; load all the variables
with open(pkl_file_path, 'rb') as file:
    loaded_data = pickle.load(file)

# Update global namespace with loaded variables
globals().update(loaded_data)


Tr_Dist_m2 = Result_m2_List[:,0]
Fail_Prob_m2 = Result_m2_List[:,1]

Tr_Dist_m3 = Result_m3_List[:,0]
Fail_Prob_m3 = Result_m3_List[:,1]


### Plot the results

# Define fitting functions
def linear_func(x, a):
    return a * np.asarray(x)

def quadratic_func(x, c):
    return c * np.asarray(x)**2

# Define color schemes
colors_failure = {
    'former': '#2ca02c',       # Medium green
    'former_light': '#98df8a', # Light green
    'current': '#d62728',      # Red
    'current_light': '#ff9896',# Light pink
    'hw_line': '#555555'       # Gray for hardware line
}

colors_trace = {
    'former': '#4C72B0',       # Soft blue
    'former_light': '#7FA6E1', # Lighter blue
    'current': '#C44E52',      # Soft red
    'current_light': '#E7969C',# Lighter pink
    'hw_line': '#555555'       # Gray for hardware line
}

## Plot the failure probability with green color scheme
NumFit = 10

# Perform the linear fit
popt_fail_disp, _ = curve_fit(linear_func, p_dep_list[:NumFit], Fail_Prob_m3[:NumFit])
b = popt_fail_disp[0]

# Plot failure probabilities
# plt.figure(figsize=(10, 6))
plt.plot(p_dep_list, Fail_Prob_m2, 
        label='m=2', 
        color=colors_failure['former'], 
        marker='o',
        markersize=6,
        linewidth=2)
plt.plot(p_dep_list, Fail_Prob_m3, 
        label='m=3', 
        color=colors_failure['current'], 
        marker='s',
        markersize=6,
        linewidth=2)

plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Physical Error Rate (p_dep)', fontsize=12)
plt.ylabel('Failure Probability', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.6)
plt.axvline(x=1e-3, color=colors_failure['hw_line'], linestyle=':', label='current hardware')
plt.legend(fontsize=10, framealpha=0.8)

# Save the plot
plt.savefig(plot1_file_path, format='eps', dpi=1000, bbox_inches='tight')
plt.show(block=False)
plt.pause(3)  # Show for 3 seconds
plt.close()   # Then close

print("Plot shown briefly and saved! Program completed.")

