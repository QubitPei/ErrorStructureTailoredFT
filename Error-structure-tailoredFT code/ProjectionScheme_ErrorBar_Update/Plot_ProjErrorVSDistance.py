import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import os

### Initialization

# some configuration parameters
global Input_Filename    #!!! name of the input data file
Input_Filename = 'MainProjScale251219m3_Errorbar.pkl'
Output_Filename1 = 'Projection_ErrorScale_Errorbar.eps'
Output_Filename2 = 'Projection_FailprobScale.eps'

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

import numpy as np
import matplotlib.pyplot as plt

# Assuming Result_Disp_List is your data
collected_data_2D = np.array(Result_Disp_List)
Tr_Dist_Disp = collected_data_2D[:, 0]
Fail_Prob_Disp = collected_data_2D[:, 1]
Std_Tr_Dist_Disp = collected_data_2D[:, 2]

# Fit the data - ensure k_List is numpy array
k_List = np.asarray(k_List)
popt_error_I, _ = curve_fit(linear_func, k_List, Tr_Dist_Disp)

# Calculate b - ensure phi and p_dep are scalars
phi = float(phi)  # Convert to float if not already
p_dep = 1e-3
Std_ck_Disp = Std_Tr_Dist_Disp/(phi*(p_dep**2))
b = 0.6

# Calculate the scaled values for plotting
Tr_Dist_Disp_scaled = Tr_Dist_Disp/(phi*(p_dep**2))
# Error bars in the scaled units (2 standard errors)
error_bars_scaled = 2 * Std_Tr_Dist_Disp/(phi*(p_dep**2))

# Plotting with improved styling
plt.figure(figsize=(8, 6))  # Better figure size

# Plot data points with enhanced error bars
plt.errorbar(k_List, Tr_Dist_Disp_scaled, 
             yerr=error_bars_scaled, 
             fmt='o-', 
             color='#2E86AB',  # Better blue color
             linewidth=2.5,    # Thicker line
             markersize=8,     # Larger markers
             markerfacecolor='white',  # White fill for better visibility
             markeredgecolor='#2E86AB',  # Blue edge
             markeredgewidth=2,  # Thicker marker edge
             ecolor='#A23B72',   # Different color for error bars (purple)
             capsize=8,          # Larger caps
             capthick=2,         # Thicker caps
             elinewidth=2,       # Thicker error bar lines
             alpha=0.9,          # Slight transparency
             label='Error bars: ±2σ'
             )

plt.fill_between(k_List, 
                 Tr_Dist_Disp_scaled - error_bars_scaled/2,
                 Tr_Dist_Disp_scaled + error_bars_scaled/2,
                 color='lightblue',  # No alpha, just light color
                 edgecolor='none',   # No edge
                 zorder=1)           # Behind everything

# Upper bound line with improved styling
# plt.plot(k_List, b*k_List, '--', 
#          color='#F18F01',  # Orange color for contrast
#          linewidth=2.5,
#          label=f'Upper bound {b:.1f}·k')

# Set x-axis to integer ticks and grids
plt.xticks(np.arange(min(k_List), max(k_List)+1, 1))
plt.xlabel('Number of rotation gates $k$', fontsize=16)
plt.ylabel(r'$c(k)=D_{Tr}/(p^2\cdot |\varphi|)$', fontsize=16)

# Improved grid styling
plt.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)

# Set axis limits with padding for error bars
y_max = np.max(Tr_Dist_Disp_scaled + error_bars_scaled)
plt.ylim(bottom=0, top=y_max * 1.15)  # 15% padding at top

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# Customize legend
plt.legend(fontsize=14, framealpha=0.9, loc='upper left')

# Increase tick label sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add a title if desired
# plt.title(r'$c(k)$ with $2\sigma$ Error Bars', fontsize=16, pad=15)

# Tight layout
plt.tight_layout()

# Add annotation about error bars
# plt.text(0.02, 0.98, 'Error bars: ±2σ', 
#          transform=plt.gca().transAxes,
#          fontsize=12,
#          verticalalignment='top',
#          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Optional: Add value labels on top of points
for i, (x, y, err) in enumerate(zip(k_List, Tr_Dist_Disp_scaled, error_bars_scaled)):
    plt.text(x, y + err + 0.02, f'{y+err:.2f}', 
             ha='center', va='bottom', fontsize=12, alpha=0.7)

# Save the plot
plt.savefig(plot_file_path1, format='eps', dpi=1000, bbox_inches='tight')
plt.savefig(plot_file_path1.replace('.eps', '.png'), dpi=300, bbox_inches='tight')  # Also save as PNG for quick view

plt.show(block=False)
plt.pause(3)  # Show for 3 seconds
plt.close()   # Then close





# Plot failure probabilities with comfortable red colors
plt.figure(figsize=(8, 6))

# Define comfortable red colors
dark_red = '#C62828'      # Rich, deep red
medium_red = '#EF5350'    # Medium red  
light_red = '#FFCDD2'     # Light, comfortable red
grid_color = '#FFF5F5'    # Very light red for grid

# Plot data points with comfortable red styling
plt.plot(k_List, Fail_Prob_Disp, 
         'o-', 
         color=dark_red,        # Line color
         linewidth=2.5,
         markersize=8,
         markerfacecolor=light_red,    # Light red fill
         markeredgecolor=dark_red,     # Dark red edge
         markeredgewidth=2,
         label='Failure probability')

# Set x-axis to integer ticks
plt.xticks(np.arange(min(k_List), max(k_List)+1, 1))
plt.xlabel('Number of rotation gates k', fontsize=16)
plt.ylabel('Failure probability', fontsize=16)

# Light red grid
plt.grid(True, color=grid_color, linestyle='-', linewidth=1)

# Set y-axis limits
y_min = np.min(Fail_Prob_Disp)
y_max = np.max(Fail_Prob_Disp)
plt.ylim(bottom=y_min * 0.95, top=y_max * 1.05)

# Add a reference line at starting value (around 0.6)
if y_min > 0.5:  # If data starts around 0.6
    plt.axhline(y=y_min, color='#FFAB91', linestyle='--', linewidth=1.5, 
                label=f'Starting value: {y_min:.3f}')

# Customize legend
plt.legend(fontsize=14, frameon=True, facecolor='white', 
           edgecolor='lightgray', loc='best')

# Increase tick label sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add value labels with white background
for i, (x, y) in enumerate(zip(k_List, Fail_Prob_Disp)):
    offset = 0.02 * (plt.ylim()[1] - plt.ylim()[0])
    plt.text(x, y + offset, 
             f'{y:.3f}', 
             ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', 
                      edgecolor=light_red, linewidth=0.5))

# Tight layout
plt.tight_layout()

# Save
plt.savefig(plot_file_path2, format='eps', dpi=1000, 
            bbox_inches='tight', facecolor='white')
plt.savefig(plot_file_path2.replace('.eps', '.png'), 
            dpi=300, bbox_inches='tight')

plt.show(block=False)
plt.pause(3)
plt.close()

print("Plot shown briefly and saved! Program completed.")
