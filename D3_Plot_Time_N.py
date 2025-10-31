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
from matplotlib.patches import Patch
from scipy.ndimage import maximum_filter

start_time = time.time()

############################################################################
#                           Enter inputs here                              #
############################################################################

WORKING_DIR = r"C:\Users\danap\OCHRE_Working"
  
ninety_fifth_output_file  = "D1_180117_2_15_RCJ_95.csv"
mean_output_file  = "D1_180117_2_15_RCJ_Mean.csv"
fifth_output_file  = "D1_180117_2_15_RCJ_5.csv"

alt                       = "D1_180117_2_15_RCJ_Baseline_Mean.csv"

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
#                        Limited Day Range               #
############################################################################

# intervals_per_hour = 20  # 15-min intervals → 4 per hour

# # Define the start/end hours for display
# display_start_hour = 5
# display_end_hour   = 11
# # Tick every N hours
# tick_every_hours = 5 # can change to 1, 2, etc.
# tick_every_hours = int(tick_every_hours * intervals_per_hour / 4)

# Define your time resolution
interval_minutes = 15
intervals_per_hour = int(60 / interval_minutes)

# Display window in hours
display_start_hour = 5
display_end_hour   = 11

# Tick spacing (in hours)
tick_every_hours = 1 # e.g. tick every 5 hours
tick_every = int(tick_every_hours * intervals_per_hour)


# Convert to column indices
start_col = display_start_hour * intervals_per_hour
end_col   = display_end_hour * intervals_per_hour + 1

# Slice data columns to the selected time range
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

Z_95th     = ninety_fifth_df.values
Z_Mean     = mean_df.values
Z_5th      = fifth_df.values
Z_Alt_Mean = alt_df.values

# Define y-range (rows)
y_min, y_max = 10, 500
X = X[y_min:y_max, :]
Y = Y[y_min:y_max, :]
Z_95th     = Z_95th[y_min:y_max, :]
Z_Mean     = Z_Mean[y_min:y_max, :]
Z_5th      = Z_5th[y_min:y_max, :]
Z_Alt_Mean = Z_Alt_Mean[y_min:y_max, :]

# # Apply smoothing
# size = 3
# Z_95th     = maximum_filter(Z_95th, size=size)
# Z_Mean     = maximum_filter(Z_Mean, size=size)
# Z_5th      = maximum_filter(Z_5th, size=size)
# Z_Alt_Mean = maximum_filter(Z_Alt_Mean, size=size)



############################################################################
#                               Plotting                                   #
############################################################################

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')



############################################################################
#                Highlight windows (6 AM – 10 AM) setup                    #
############################################################################

highlight_windows = [
    (6, 10),  # 6 AM → 10 AM
    (17, 20),
]

def make_facecolors(base_color, X, hilite_color):
    base_rgba = np.array(plt.matplotlib.colors.to_rgba(base_color))
    highlight_rgba = np.array(plt.matplotlib.colors.to_rgba(hilite_color))
    fc = np.tile(base_rgba, (X.shape[0], X.shape[1], 1))
    
    for start_hr, end_hr in highlight_windows:
        # Compute relative column indices based on the sliced data
        start_idx = int(start_hr * intervals_per_hour) - start_col
        end_idx   = int(end_hr * intervals_per_hour) - start_col
        fc[:, start_idx:end_idx, :] = highlight_rgba
    
    return fc


c = min(48, X.shape[1])  # horizontal resolution
r = min(10, X.shape[0])  # vertical resolution
# c = min(48, X.shape[1])  # horizontal resolution
# r = min(10, X.shape[0])  # vertical resolution
# --- Plot 95th percentile ---
ax.plot_surface(
    X, Y, Z_95th - Z_Alt_Mean,
    facecolors=make_facecolors('xkcd:goldenrod', X, 'darkorange'),
    edgecolor='black', rcount=r, ccount=c, #96 for 24 hrs 48 for 12 hrs
    linewidth=1, shade=False, label='97.5%'
)

# --- Plot mean ---
ax.plot_surface(
    X, Y, Z_Mean - Z_Alt_Mean,
    facecolors=make_facecolors('lightgreen', X, 'xkcd:leaf green'),
    edgecolor='black', rcount=r, ccount=c,
    linewidth=1, shade=False, label='Mean'
)

# --- Plot 5th percentile ---
ax.plot_surface(
    X, Y, Z_5th - Z_Alt_Mean,
    facecolors=make_facecolors('plum', X, 'orchid'),
    edgecolor='black', rcount=r, ccount=c,
    linewidth=1, shade=False, label='2.5%'
)


############################################################################
#                               Axis setup                                 #
############################################################################

A = 16  # label font
B = 14  # tick font

# # Y-axis
# y_tick_frequency = 150
# y_ticks = np.arange(y_min, y_max, y_tick_frequency)
# ax.set_yticks(y_ticks)
# ax.set_yticklabels(y_ticks, fontsize=B)

# Desired Y-axis ticks
y_min, y_max = 10, 500
ax.set_ylim(y_min, y_max)
y_ticks = [10, 125, 250, 375, 500]

# Set the ticks
ax.set_yticks(y_ticks)

# Optional: set labels with specific font size
ax.set_yticklabels(y_ticks, fontsize=B,  ha='left', va='bottom')  # ha='left', va='center' // ha='right', va='bottom'


# Z limits
# ax.set_zlim(-0.28, 0.25)

# Flip X (time) direction if desired
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(0, y_max)
# ax.yaxis._axinfo['label']['space_factor'] = 5  # moves label away from ticks/grid


############################################################################
#                    Flexible X-axis Time Labels (Full Day)                #
############################################################################


# Compute tick positions based on selected interval spacing
tick_positions = np.arange(0, len(x_labels), tick_every)

# Generate hour labels corresponding to tick positions
tick_labels = [str(display_start_hour + i * tick_every_hours).zfill(2) for i in range(len(tick_positions))]

# Ensure last tick (end of day) is included
if tick_positions[-1] != len(x_labels) - 1:
    tick_positions = np.append(tick_positions, len(x_labels) - 1)
    tick_labels.append(str(display_end_hour).zfill(2))

# Apply ticks to axis
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=0, ha='center')
ax.set_xlabel('Time [H]', fontsize=A, labelpad=8)


############################################################################
#                           Other Labels & Style                           #
############################################################################

# ax.set_ylabel('Units', fontsize=A, labelpad=18)
ax.set_ylabel('')  # remove the default one

# Add a custom label in data or axis coordinates
ax.text2D(0.96, 0.25, "Units", transform=ax.transAxes,
          rotation=69, fontsize=A, va='center', ha='center')

ax.zaxis.set_rotate_label(False)
z_min, z_max = -0.25, 0.25
ax.set_zlim(z_min, z_max)
z_ticks = [-0.2, -0.1, 0, 0.1, 0.2]
ax.set_zticks(z_ticks)
ax.set_zlabel('Power [p.u.]', rotation=-90, fontsize=A, labelpad=25)


# View and style
ax.view_init(elev=13, azim=102) # (15, 102) (17, 67)
ax.grid(True, linestyle='--', linewidth=0.3, color='white')
ax.tick_params(axis='x', labelsize=B, pad=0)
ax.tick_params(axis='y', labelsize=B, pad=2)
ax.tick_params(axis='z', labelsize=B, pad=13)




# ax.legend(bbox_to_anchor=(0.88, 0.05), ncol=3, fontsize=14, frameon=False)
legend_elements = [
    Patch(facecolor='xkcd:goldenrod', edgecolor='black', label='97.5%'),
    Patch(facecolor='lightgreen', edgecolor='black', label='Mean'),
    Patch(facecolor='plum', edgecolor='black', label='2.5%')
]

ax.legend(handles=legend_elements, bbox_to_anchor=(0.88, 0.08), ncol=3, fontsize=14, frameon=False)


# plt.tight_layout()

RESULTS_DIR = r"C:\Users\danap\OCHRE_Working\Figs\ADMD_3D_diffMean2.pdf"
plt.savefig(RESULTS_DIR, format='pdf', bbox_inches='tight', pad_inches=1)
# plt.show()

############################################################################
#                    Create Summary Table per 15-min Step                  #
############################################################################

# Compute differences from the alternative mean (baseline)
diff_95th = Z_95th - Z_Alt_Mean
diff_5th  = Z_5th - Z_Alt_Mean
diff_mean = diff_95th - diff_5th



# # calculate the difference between regions
# diff_95th = Z_95th
# diff_5th = Z_5th
# # diff_mean = Z_95th - Z_5th
# diff_mean = Z_Mean

# Select the last row (n = 500 or last row in sliced data)
last_row_index = -1  # last row
diff_95th_last = diff_95th[last_row_index, :]
diff_mean_last = diff_mean[last_row_index, :]
diff_5th_last  = diff_5th[last_row_index, :]



# Initialize table
summary_table = pd.DataFrame(
    columns=['Time', '5th_diff', 'Mean',  '95th_diff']
)

# Fill table with values from the last row
for i, col_label in enumerate(x_labels):
    summary_table.loc[i] = [
        col_label,
        diff_5th_last[i],
        diff_mean_last[i],
        diff_95th_last[i],

    ]

# Display
print(summary_table)

# Save to CSV
# summary_table.to_csv(os.path.join(WORKING_DIR, "ADMD_summary.csv"), index=False)




############################################################################
#                            Timing Output                                 #
############################################################################

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")
print(f"Execution time: {execution_time/60:.2f} minutes")
