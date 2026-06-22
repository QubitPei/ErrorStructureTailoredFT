import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import os

### Initialization

# some configuration parameters
global Input_Filename    #!!! name of the input data file
Input_Filename = 'MainProjectionNoise260109_Errorbar.pkl'
Output1_Filename = 'Projection_Pfail_260109.eps'
Output2_Filename = 'Projection_ErrorRate_260109_Errorbar.eps'

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
pkl_file_path = os.path.join(script_dir, Input_Filename)
plot1_file_path = os.path.join(script_dir, Output1_Filename)
plot2_file_path = os.path.join(script_dir, Output2_Filename)

### Load data from the .pkl file; load all the variables
with open(pkl_file_path, 'rb') as file:
    loaded_data = pickle.load(file)

# Update global namespace with loaded variables
globals().update(loaded_data)


Tr_Dist_Dep = Result_Dep_List[:,0]
Fail_Prob_Dep = Result_Dep_List[:,1]
Std_Tr_Dist_Dep = Result_Dep_List[:,2]

Tr_Dist_Disp = Result_Disp_List[:,0]
Fail_Prob_Disp = Result_Disp_List[:,1]
Std_Tr_Dist_Disp = Result_Disp_List[:,2]

### Plot the results

'''
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
popt_fail_disp, _ = curve_fit(linear_func, p_dep_list[:NumFit], Fail_Prob_Disp[:NumFit])
b = popt_fail_disp[0]

# Plot failure probabilities
# plt.figure(figsize=(10, 6))
plt.plot(p_dep_list, Fail_Prob_Dep, 
        label='former scheme', 
        color=colors_failure['former'], 
        marker='o',
        markersize=6,
        linewidth=2)
plt.plot(p_dep_list, Fail_Prob_Disp, 
        label='current scheme', 
        color=colors_failure['current'], 
        marker='s',
        markersize=6,
        linewidth=2)

# Add fitting formula if needed
# formula_label_disp = f"Fit: ${b:.4f} \\cdot p$"
# plt.plot(p_dep_list, linear_func(p_dep_list, *popt_fail_disp), 
#          '--', 
#          color=colors_failure['current_light'], 
#          label=formula_label_disp,
#          linewidth=1.5)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Physical error rate (p)', fontsize=12)
plt.ylabel('Failure probability', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.6)
plt.axvline(x=1e-3, color=colors_failure['hw_line'], linestyle=':', label='current hardware')
plt.legend(fontsize=10, framealpha=0.8)

# Save the plot
plt.savefig(plot1_file_path, format='eps', dpi=1000, bbox_inches='tight')
plt.show(block=False)
plt.pause(3)  # Show for 3 seconds
plt.close()   # Then close

## Plot trace distances with original color scheme
plt.figure()
popt_error_I, _ = curve_fit(quadratic_func, p_dep_list[:NumFit], Tr_Dist_Disp[:NumFit])
b = popt_error_I[0]/phi

# Create the fitting formula string
formula_label_disp = f"Fit: ${b:.4f} \\cdot \\varphi p^2$"

# plt.figure(figsize=(10, 6))
plt.plot(p_dep_list, Tr_Dist_Dep, 
        label='former scheme', 
        color=colors_trace['former'], 
        marker='o',
        markersize=6,
        linewidth=2)
plt.plot(p_dep_list, Tr_Dist_Disp, 
        label='current scheme', 
        color=colors_trace['current'], 
        marker='s',
        markersize=6,
        linewidth=2)

# Plot the quadratic fit
plt.plot(p_dep_list, quadratic_func(p_dep_list, *popt_error_I), 
        '--', 
        color=colors_trace['current_light'], 
        label=formula_label_disp,
        linewidth=1.5)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Physical error rate (p)', fontsize=12)
plt.ylabel('Trace distance', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.6)
plt.axvline(x=1e-3, color=colors_trace['hw_line'], linestyle=':', label='current hardware')
plt.legend(fontsize=10, framealpha=0.8)

# Save the plot
plt.savefig(plot2_file_path, format='eps', dpi=1000, bbox_inches='tight')
plt.show(block=False)
plt.pause(3)  # Show for 3 seconds
plt.close()   # Then close

print("Plot shown briefly and saved! Program completed.")
'''


import numpy as np
import matplotlib.pyplot as plt

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
popt_fail_disp, _ = curve_fit(linear_func, p_dep_list[:NumFit], Fail_Prob_Disp[:NumFit])
b = popt_fail_disp[0]

# Plot failure probabilities with improved styling
plt.figure(figsize=(8, 6))
plt.plot(p_dep_list, Fail_Prob_Dep, 
        label='former scheme', 
        color=colors_failure['former'], 
        marker='o',
        markersize=8,
        markerfacecolor='white',
        markeredgecolor=colors_failure['former'],
        markeredgewidth=2,
        linewidth=2.5,
        linestyle='-')

plt.plot(p_dep_list, Fail_Prob_Disp, 
        label='current scheme', 
        color=colors_failure['current'], 
        marker='s',
        markersize=8,
        markerfacecolor='white',
        markeredgecolor=colors_failure['current'],
        markeredgewidth=2,
        linewidth=2.5,
        linestyle='-')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Physical error rate (p)', fontsize=16)
plt.ylabel('Failure probability', fontsize=16)

# Improved grid (no alpha)
plt.grid(True, which="both", color='#f0f0f0', linestyle='-', linewidth=0.5)

plt.axvline(x=1e-3, color=colors_failure['hw_line'], linestyle=':', 
           linewidth=2, label='current hardware')

# Legend with solid background
plt.legend(fontsize=14, frameon=True, facecolor='white', 
          edgecolor='lightgray', loc='best')

# Increase tick label sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()

# Save the plot
plt.savefig(plot1_file_path, format='eps', dpi=1000, bbox_inches='tight', facecolor='white')
plt.show(block=False)
plt.pause(3)  # Show for 3 seconds
plt.close()   # Then close

## Plot trace distances with error bars
plt.figure(figsize=(8, 6))

# Perform quadratic fit
popt_error_I, _ = curve_fit(quadratic_func, p_dep_list[:NumFit], Tr_Dist_Disp[:NumFit])
b = popt_error_I[0]/phi

# Create the fitting formula string
formula_label_disp = f"Fit: ${b:.4f} \\cdot \\varphi p^2$"

# Plot with error bars (using Std_Tr_Dist_Dep and Std_Tr_Dist_Disp)
plt.errorbar(p_dep_list, Tr_Dist_Dep, 
            yerr=2*Std_Tr_Dist_Dep,  # 2 standard errors
            label='former scheme', 
            color=colors_trace['former'], 
            marker='o',
            markersize=8,
            markerfacecolor='white',
            markeredgecolor=colors_trace['former'],
            markeredgewidth=2,
            linewidth=2.5,
            linestyle='-',
            ecolor=colors_trace['former_light'],  # Light blue error bars
            capsize=5,
            capthick=1.5,
            elinewidth=1.5)

plt.errorbar(p_dep_list, Tr_Dist_Disp, 
            yerr=2*Std_Tr_Dist_Disp,  # 2 standard errors
            label='current scheme', 
            color=colors_trace['current'], 
            marker='s',
            markersize=8,
            markerfacecolor='white',
            markeredgecolor=colors_trace['current'],
            markeredgewidth=2,
            linewidth=2.5,
            linestyle='-',
            ecolor=colors_trace['current_light'],  # Light pink error bars
            capsize=5,
            capthick=1.5,
            elinewidth=1.5)

# Plot the quadratic fit (solid line for better log-scale visibility)
plt.plot(p_dep_list, quadratic_func(p_dep_list, *popt_error_I), 
        '--', 
        color=colors_trace['current_light'], 
        label=formula_label_disp,
        linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Physical error rate (p)', fontsize=16)
plt.ylabel('Trace distance', fontsize=16)

# Improved grid (no alpha)
plt.grid(True, which="both", color='#f0f0f0', linestyle='-', linewidth=0.5)

plt.axvline(x=1e-3, color=colors_trace['hw_line'], linestyle=':', 
           linewidth=2, label='current hardware')

# Legend with solid background
plt.legend(fontsize=14, frameon=True, facecolor='white', 
          edgecolor='lightgray', loc='best')

# Increase tick label sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add annotation for error bars
plt.text(0.02, 0.98, 'Error bars: ±2σ', 
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgray'))

plt.tight_layout()

# Save the plot
plt.savefig(plot2_file_path, format='eps', dpi=1000, bbox_inches='tight', facecolor='white')
plt.show(block=False)
plt.pause(3)  # Show for 3 seconds
plt.close()   # Then close

print("Plot shown briefly and saved! Program completed.")
