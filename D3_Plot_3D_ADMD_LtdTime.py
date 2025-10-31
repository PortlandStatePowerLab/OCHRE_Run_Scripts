# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 09:55:05 2025

@author: danap
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
# alt                       = "D1_180111_1_15_Baseline_Mean.csv"
alt                       = "D1_180111_1_15_NR_Mean.csv"

ninety_fifth_file_name = os.path.join(WORKING_DIR, ninety_fifth_output_file)
mean_file_name         = os.path.join(WORKING_DIR, mean_output_file)
fifth_file_name        = os.path.join(WORKING_DIR, fifth_output_file)
alt                    = os.path.join(WORKING_DIR, alt)

write_percent_error = False

############################################################################
#                             Program Start                                #
############################################################################

# read data 
ninety_fifth_df = pd.read_csv(ninety_fifth_file_name)
mean_df         = pd.read_csv(mean_file_name)
fifth_df        = pd.read_csv(fifth_file_name)
alt_df          = pd.read_csv(alt)

# remove the columns we don't need
for df in [ninety_fifth_df, mean_df, fifth_df, alt_df]:
    if 'Unnamed: 0' in df.columns:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)

############################################################################
#                      Limit data to 6 AM – 10 AM                          #
############################################################################

# 96 columns per day (15-min intervals)
intervals_per_hour = 4
start_col = 3 * intervals_per_hour   # 6 AM → 24
end_col   = 12 * intervals_per_hour  # 10 AM → 40

# Slice all DataFrames to that window
ninety_fifth_df = ninety_fifth_df.iloc[:, start_col:end_col]
mean_df         = mean_df.iloc[:, start_col:end_col]
fifth_df        = fifth_df.iloc[:, start_col:end_col]
alt_df          = alt_df.iloc[:, start_col:end_col]

# Update x_labels
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

# Slice the data before plotting
X = X[y_min:y_max, :]
Y = Y[y_min:y_max, :]
Z_95th = Z_95th[y_min:y_max, :]
Z_Mean = Z_Mean[y_min:y_max, :]
Z_5th  = Z_5th[y_min:y_max, :]
Z_Alt_Mean = Z_Alt_Mean[y_min:y_max, :]

# Apply smoothing (temporary)
size = 1
Z_95th     = maximum_filter(Z_95th, size=size)
Z_Mean     = maximum_filter(Z_Mean, size=size)
Z_5th      = maximum_filter(Z_5th, size=size)
Z_Alt_Mean = maximum_filter(Z_Alt_Mean, size=size)

############################################################################
#                               Plotting                                   #
############################################################################

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')

# Plot the differences from the alt mean
# ax.plot_surface(X, Y, Z_95th - Z_Alt_Mean, color='xkcd:goldenrod',
#                 edgecolor='black', rcount=10, ccount=16, shade=False, label='97.5%')
ax.plot_surface(X, Y, Z_Mean - Z_Alt_Mean, color='lightgreen',
                edgecolor='black', rcount=10, ccount=16, shade=False, label='Mean')
# ax.plot_surface(X, Y, Z_5th - Z_Alt_Mean, color='plum',
#                 edgecolor='black', rcount=10, ccount=16, shade=False, label='2.5%')
# ax.plot_surface(X, Y, Z_Alt_Mean, color='plum',
#                 edgecolor='black', rcount=10, ccount=16, shade=False, label='test')

# Y-axis ticks
A = 16
B = 14
y_tick_frequency = 100
y_ticks = np.arange(y_min, y_max, y_tick_frequency)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks, fontsize=B)

# Axis limits
# ax.set_ylim(0, 500)
# ax.set_yticks([50, 150, 250, 350, 450])
ax.set_ylim(0.5 * y_min, y_max)
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_zlim(-0.8, 0.1)

############################################################################
#                       Fix X-axis Time Labels (6–10 AM)                   #
############################################################################

start_hour = 0  # our slice starts at 6 AM
D = intervals_per_hour  # 1 tick per hour

tick_positions = np.arange(0, len(x_labels), D)
tick_hours = [start_hour + i for i in range(len(tick_positions))]
tick_labels = [f"{h:02d}" for h in tick_hours]

# ensure counts match
assert len(tick_positions) == len(tick_labels), "Tick/label mismatch!"

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
ax.view_init(elev=20, azim=110)
ax.set_title('')
ax.grid(True, linestyle='--', linewidth=0.3, color='white')
ax.tick_params(axis='x', labelsize=B, pad=-3)
ax.tick_params(axis='y', labelsize=B, pad=-1)
ax.tick_params(axis='z', labelsize=B, pad=8)
ax.legend(bbox_to_anchor=(0.88, 0.05), ncol=3, fontsize=14, frameon=False)

plt.tight_layout()

RESULTS_DIR = r"C:\Users\danap\OCHRE_Working\Figs\ADMD_3D_baselineDiff.pdf"
# plt.savefig(RESULTS_DIR, format='pdf', bbox_inches='tight', pad_inches=1)
plt.show()

############################################################################
#                            Timing Output                                 #
############################################################################

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")
print(f"Execution time: {execution_time/60:.2f} minutes")
