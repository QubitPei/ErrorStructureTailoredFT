import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import os

### Initialization

# some configuration parameters
global Input_Filename    #!!! name of the input data file
Input_Filename = 'MainProjScale251017.pkl'
Output_Filename1 = 'Projection_ErrorScale_251017.eps'
Output_Filename2 = 'Projection_FailprobScale_251017.eps'

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
pkl_file_path = os.path.join(script_dir, Input_Filename)
plot_file_path1 = os.path.join(script_dir,Output_Filename1)
plot_file_path2 = os.path.join(script_dir,Output_Filename2)

### Load data from the .pkl file; load all the variables
with open(pkl_file_path, 'rb') as f:
    loaded_data = pickle.load(f)

# Update global namespace with loaded variables
globals().update(loaded_data)


# Define vectorized fitting functions
def linear_func(x, a):
    return a * np.asarray(x)  # Ensure x is treated as array

def quadratic_func(x, c):
    return c * np.asarray(x)**2  # Ensure x is treated as array

# Assuming Result_Disp_List is your data
collected_data_2D = np.array(Result_Disp_List)
Tr_Dist_Disp = collected_data_2D[:, 0]
Fail_Prob_Disp = collected_data_2D[:, 1]

# Fit the data - ensure k_List is numpy array
k_List = np.asarray(k_List)
popt_error_I, _ = curve_fit(linear_func, k_List, Tr_Dist_Disp)

# Calculate b - ensure phi and p_dep are scalars
phi = float(phi)  # Convert to float if not already
# p_dep = float(p_dep)
p_dep = 1e-3
# b = popt_error_I[0]/(phi*(p_dep**2))
b = 3.3
# popt_error_I[0] = (phi*(p_dep**2))*b

# Plotting
plt.figure()
# plt.plot(k_List, Tr_Dist_Disp, 'o-', color='red', label='Data')
plt.plot(k_List, Tr_Dist_Disp/(phi*(p_dep**2)), 'o-', color='red', label='Data')
plt.plot(k_List, b*k_List, '--', 
         label=f'Upper bound {b:.1f}·k')

# Add horizontal dashed line at 1.4e-8
# plt.axhline(y=13, color='blue', linestyle='--', linewidth=2, label='Upper bound')


# Set x-axis to integer ticks and grids
plt.xticks(np.arange(min(k_List), max(k_List)+1, 1))  # +1 to include the last integer
plt.xlabel('Number of rotation gates k', fontsize=16)  # Increased to 16
# plt.ylabel('Trace Distance', fontsize=16)  # Increased to 16
plt.ylabel(r'$c(k)=P_{\mathrm{ud}(2)}/(p^2)$', fontsize=16)  # Increased to 16
plt.grid(True, which='both', axis='both')  # Ensure both major grids are shown
plt.legend(fontsize=16)  # Increased to 16

# Increase tick label sizes
plt.xticks(fontsize=14)  # Increased to 14
plt.yticks(fontsize=14)  # Increased to 14

# Save the plot
plt.savefig(plot_file_path1, format='eps', dpi=1000, bbox_inches='tight')
plt.show(block=False)
plt.pause(3)  # Show for 3 seconds
plt.close()   # Then close





# Plot failure probabilities
plt.figure()
plt.plot(k_List, Fail_Prob_Disp, 'o-', color='red', label='Data')

# Set x-axis to integer ticks and grids
plt.xticks(np.arange(min(k_List), max(k_List)+1, 1))  # +1 to include the last integer
plt.xlabel('k', fontsize=16)  # Increased to 16
plt.ylabel('Failure probability', fontsize=16)  # Increased to 16
plt.grid(True, which='both', axis='both')  # Ensure both major grids are shown
plt.legend(fontsize=16)  # Increased to 16

# Increase tick label sizes
plt.xticks(fontsize=14)  # Increased to 14
plt.yticks(fontsize=14)  # Increased to 14

# Save the plot
plt.savefig(plot_file_path2, format='eps', dpi=1000, bbox_inches='tight')
plt.show(block=False)
plt.pause(3)  # Show for 3 seconds
plt.close()   # Then close

print("Plot shown briefly and saved! Program completed.")
