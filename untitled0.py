# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 2025

@modified_by: ChatGPT (based on Dana's script)
Purpose: Full-day 3D ADMD comparison plot (00:00–23:59) with same baseline-difference style.
"""

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import maximum_filter

start_time = time.time()

############################################################################
#                           Enter inputs here                              #
############################################################################

WORKING_DIR = r"C:\Users\danap\OCHRE_Working"
  
ninety_fifth_output_file  = "D1_180111_ADMD_RC_95.csv"
mean_output_file          = "D1_180111_ADMD_RC_mean.csv"
fifth_output_file         = "D1_180111_ADMD_RC_5.csv"
alt                       = "D1_180111_ADMD_NoRamp_mean.csv"

ninety_fifth_file_name = os.path.join(WORKING_DIR, ninety_fifth_output_file)
mean_file_name         = os.path.join(WORKING_DIR, mean_output_file)
fifth_file_name        = os.path.join(WORKING_DIR, fifth_output_file)
alt                    = os.path.join(WORKING_DIR, alt)

write_percent_error = False

############################################################################
#                             Program Start                                #
############################################################################

# Read data 
ninety_fifth_df = pd.read_csv(ninety_fifth_file_name)
mean_df         = pd.read_csv(mean_file_name)
fifth_df        = pd.read_csv(fifth_file_name)
alt_df          = pd.read_csv(alt)

# Remove unnecessary columns
for df in [ninety_fifth_df, mean_df, fifth_df, alt_df]:
    if 'Unnamed: 0' in df.columns:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)

############################################################################
#                        Full-day range (00:00–23:59)                      #
############################################################################

intervals_per_hour = 4  # 15-min intervals → 96 columns/day
start_col = 0
end_col   = 96

# Use full-day data
ninety_fifth_df = ninety_fifth_df.iloc[:, start_col:end_col]
mean_df         = mean_df.iloc[:, start_col:end_col]
fifth_df        = fifth_df.iloc[:, start_col:end_col]
alt_df          = alt_df.iloc[:, start_col:end_col]

x_labels = ninety_fifth_df.columns.tolist()
x = np.arange(len(x_labels))

############################################################################
#                            Prepare Data                                  #
############################################################################

y = ninety_fifth_df.index
X, Y = np.meshgrid(x, y)

Z_95th = ninety_fifth_df.values
Z_Mean = mean_df.values
Z_5th  = fifth_df.values
Z_Alt_Mean = alt_df.values

# Define y-range
y_min, y_max = 10, 500
X = X[y_min:y_max, :]
Y = Y[y_min:y_max, :]
Z_95th     = Z_95th[y_min:y_max, :]
Z_Mean     = Z_Mean[y_min:y_max, :]
Z_5th      = Z_5th[y_min:y_max, :]
Z_Alt_Mean = Z_Alt_Mean[y_min:y_max, :]

# Apply smoothing
size = 1
Z_95th     = maximum_filter(Z_95th, size=size)
Z_Mean     = maximum_filter(Z_Mean, size=size)
Z_5th      = maximum_filter(Z_5th, size=size)
Z_Alt_Mean = maximum_filter(Z_Alt_Mean, size=size)

############################################################################
#                               Plotting                                   #
############################################################################

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

# Plot differences from alt mean
ax.plot_surface(X, Y, Z_95th, color='xkcd:goldenrod',
                edgecolor='black', rcount=12, ccount=32, shade=False, label='97.5%')
ax.plot_surface(X, Y, Z_Mean, color='lightgreen',
                edgecolor='black', rcount=12, ccount=32, shade=False, label='Mean')
ax.plot_surface(X, Y, Z_5th, color='plum',
                edgecolor='black', rcount=12, ccount=32, shade=False, label='2.5%')

############################################################################
#                               Axis setup                                 #
############################################################################

A = 16  # label font
B = 14  # tick font

# Y-axis
y_tick_frequency = 100
y_ticks = np.arange(y_min, y_max, y_tick_frequency)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks, fontsize=B)

# Z limits
# ax.set_zlim(-0.28, 0.25)

# Flip X (time) direction if desired
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(0.5 * y_min, y_max)

############################################################################
#                    Flexible X-axis Time Labels (Full Day)                #
############################################################################

intervals_per_hour = 4           # 15-min data → 4 intervals/hour
tick_every_hours   = 4           # <-- change to 1, 2, or 3 as desired

# compute tick positions in column indices
tick_positions = np.arange(0, len(x_labels), intervals_per_hour * tick_every_hours)

# extract first two characters (hour only, e.g. "06" from "06:15")
tick_labels = [str(label)[:2] for label in np.array(x_labels)[tick_positions]]

# ensure 23:59 (end of day) is included if not already
if tick_positions[-1] != len(x_labels) - 1:
    tick_positions = np.append(tick_positions, len(x_labels) - 1)
    tick_labels.append("23")

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=0, ha='left')
ax.set_xlabel('Time [H]', fontsize=A, labelpad=4)



############################################################################
#                           Other Labels & Style                           #
############################################################################

ax.set_ylabel('Units', fontsize=A, labelpad=7)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('Power [p.u.]', rotation=-90, fontsize=A, labelpad=16)

# View and style
ax.view_init(elev=20, azim=45)
ax.grid(True, linestyle='--', linewidth=0.3, color='white')
ax.tick_params(axis='x', labelsize=B, pad=-3)
ax.tick_params(axis='y', labelsize=B, pad=-1)
ax.tick_params(axis='z', labelsize=B, pad=8)
ax.legend(bbox_to_anchor=(0.88, 0.05), ncol=3, fontsize=14, frameon=False)

plt.tight_layout()

RESULTS_DIR = r"C:\Users\danap\OCHRE_Working\Figs\ADMD_3D_FullDay_BaselineDiff.pdf"
# plt.savefig(RESULTS_DIR, format='pdf', bbox_inches='tight', pad_inches=1)
plt.show()

############################################################################
#                            Timing Output                                 #
############################################################################

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")
print(f"Execution time: {execution_time/60:.2f} minutes")
