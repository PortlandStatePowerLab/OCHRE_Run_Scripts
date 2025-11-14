# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 10:51:13 2025

@author: danap
"""

import pandas as pd
import os
import re
# import os
import time
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# from scipy.ndimage import maximum_filter


plot_start_hour = 5 # 24 hr
plot_end_hour = 11 # 24 hr
tick_spacing = 1  # hours

Prated = 0.5   # kW
Nhouses = 409

WORKING_DIR = r"C:\Users\danap\OCHRE_Working"

files = [
    "180110_1_3_Shed125_Control.parquet",
    "180110_1_3_Shed120_Control.parquet",
    "180110_1_3_Shed115_Control.parquet",
    "180110_1_3_Shed110_Control.parquet"
]

setpoints = [110, 115, 120, 125]  # both setpoints

def process_parquet(filename, working_dir, db_values=[5, 10, 15]):
    """
    Process a Parquet file, aggregate power by deadband (°F), and
    return a dict of Series like {'DB_15_120': <Series>, ...}.
    """
    file_path = os.path.join(working_dir, filename)
    df = pd.read_parquet(file_path)

    # Ensure datetime is rounded
    df['Time'] = df['Time'].dt.round('min')

    # Convert Celsius → Fahrenheit
    df['DB_F'] = df['Deadband_C'] * 9 / 5
    df = df.drop(columns=['Deadband_C'])

    results = {}

    # Extract the numeric part from "Shed120" for suffix
    match = re.search(r"Shed(\d+)", filename)
    suffix = match.group(1) if match else "Unknown"

    for db in db_values:
        df_db = df[df['DB_F'] == db]
        if df_db.empty:
            print(f"⚠️ No data found for {db}°F in {filename}")
            continue

        power_sum = df_db.groupby('Time')['Water Heating Electric Power (kW)'].sum()
        key = f"DB_{int(db)}_{suffix}"
        results[key] = power_sum

    return results



all_results = {}

for sp in setpoints:
    for f in files:
        res = process_parquet(f, WORKING_DIR)
        all_results.update(res)



start_time = time.time()

############################################################################
#                           User Inputs                                    #
############################################################################

WORKING_DIR = r"C:\Users\danap\OCHRE_Working"
RESULTS_DIR = os.path.join(WORKING_DIR, "Figs")
os.makedirs(RESULTS_DIR, exist_ok=True)

# setpoint = 120  # °F setpoint temperature to plot
deadbands = [5, 10, 15]  # °F deadband levels

# Colors per layer
color_map = {
    5: ('plum', 'orchid'),
    10: ('xkcd:light green', 'xkcd:leaf green'),
    15: ('xkcd:goldenrod', 'xkcd:orange'),
}

# Expected global variable: all_results dict with DB_xx_xxx = Series
# Example: all_results["DB_15_120"]

############################################################################
#                       Data Preparation                                   #
############################################################################

series_list = []
valid_db = []
for db in deadbands:
    key = f"DB_{db}_{setpoint}"
    if key in all_results:
        series_list.append(all_results[key])
        valid_db.append(db)
    else:
        print(f"⚠️ Missing data for {key}")

if not series_list:
    raise ValueError("No matching data found in all_results.")

# Align data on common time index
df = pd.concat(series_list, axis=1)
df.columns = [f"DB_{db}" for db in valid_db]
df = df.dropna()

# Convert datetime to numeric hours
time_hours = (df.index - df.index[0]).total_seconds() / 3600
X, Y = np.meshgrid(time_hours, valid_db)
Z = df.T.values  # shape: (len(DB), len(Time))

# Optional smoothing
# Z = maximum_filter(Z, size=2)

############################################################################
#                              Plotting                                    #
############################################################################

# =============================================================================
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(projection='3d')
# 
# # Optional: highlight time ranges (e.g., morning/evening windows)
# highlight_windows = [
#     (6, 10),   # 6 AM to 10 AM
#     (17, 20),  # 5 PM to 8 PM
# ]
# 
# def make_facecolors(base_color, X, hilite_color):
#     """Generate a matrix of facecolors with highlighted windows."""
#     base_rgba = np.array(plt.matplotlib.colors.to_rgba(base_color))
#     highlight_rgba = np.array(plt.matplotlib.colors.to_rgba(hilite_color))
#     fc = np.tile(base_rgba, (X.shape[0], X.shape[1], 1))
#     for start_hr, end_hr in highlight_windows:
#         start_idx = np.searchsorted(time_hours, start_hr)
#         end_idx = np.searchsorted(time_hours, end_hr)
#         fc[:, start_idx:end_idx, :] = highlight_rgba
#     return fc
# 
# # --- Plot each deadband layer ---
# r = min(5, X.shape[0])
# c = min(96, X.shape[1])
# 
# Z_pu = Z / (Prated * Nhouses)
# 
# for i, db in enumerate(valid_db):
#     base_color, hi_color = color_map.get(db, ('lightgray', 'gray'))
#     ax.plot_surface(
#         X, Y, Z_pu,  # use normalized Z
#         facecolors=make_facecolors(base_color, X, hi_color),
#         edgecolor='black',
#         linewidth=0.6,
#         rcount=2,
#         ccount=min(96, Z.shape[1]),
#         shade=False
#     )
# 
#     break  # plot all at once (Z already holds all DBs)
# 
# =============================================================================

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')

highlight_windows = [
    (6, 10),   # 6 AM – 10 AM
    (17, 20),  # 5 PM – 8 PM
]

# Same make_facecolors function as your ADMD plot
def make_facecolors(base_color, X, hilite_color):
    base_rgba = np.array(plt.matplotlib.colors.to_rgba(base_color))
    highlight_rgba = np.array(plt.matplotlib.colors.to_rgba(hilite_color))
    fc = np.tile(base_rgba, (X.shape[0], X.shape[1], 1))
    for start_hr, end_hr in highlight_windows:
        start_idx = np.searchsorted(time_hours, start_hr)
        end_idx   = np.searchsorted(time_hours, end_hr)
        fc[:, start_idx:end_idx, :] = highlight_rgba
    return fc

# Colors per Tset from your ADMD plot
tset_colors = {
    120: ('lightgreen', 'xkcd:leaf green'),      # 120°F
    125: ('xkcd:goldenrod', 'darkorange'),       # 125°F
    115: ('plum', 'orchid'),                      # 115 F
    110: ('lightpink', 'darkpink')
}



for setpoint in [115, 120, 125]:
    # Collect all deadbands for this setpoint
    series_list = []
    valid_db = []
    for db in deadbands:
        key = f"DB_{db}_{setpoint}"
        if key in all_results:
            series_list.append(all_results[key])
            valid_db.append(db)
    if not series_list:
        continue

    df_sp = pd.concat(series_list, axis=1).dropna()
    time_hours = (df_sp.index - df_sp.index[0]).total_seconds() / 3600
    X = np.tile(time_hours, (len(valid_db), 1))
    Y = np.array(valid_db)[:, None] * np.ones_like(X)
    Z = df_sp.T.values
    Z_pu = Z / (Prated * Nhouses)
    
    
    # Filter df_sp to only the times within the window
    time_hours = (df_sp.index - df_sp.index[0]).total_seconds() / 3600
    mask = (time_hours >= plot_start_hour) & (time_hours <= plot_end_hour)
    df_sp = df_sp.iloc[mask, :]
    time_hours = time_hours[mask]
    
    # Now recreate X, Y, Z for plotting
    X = np.tile(time_hours, (len(valid_db), 1))
    Y = np.array(valid_db)[:, None] * np.ones_like(X)
    Z = df_sp.T.values
    Z_pu = Z / (Prated * Nhouses)


    base_color, hi_color = tset_colors[setpoint]

    ax.plot_surface(
        X, Y, Z_pu,
        facecolors=make_facecolors(base_color, X, hi_color),
        edgecolor='black',
        linewidth=0.6,
        rcount=len(valid_db),          # connect all deadbands
        ccount=min(96, Z_pu.shape[1]),
        shade=False,
        alpha=0.85
    )


legend_elements = [
    Patch(facecolor='lightgreen', edgecolor='black', label='Tset 120°F'),
    Patch(facecolor='xkcd:goldenrod', edgecolor='black', label='Tset 125°F')
]
ax.legend(handles=legend_elements, fontsize=12, frameon=False)



# deadband_step = 1 / len(valid_db) * (max(valid_db) - min(valid_db) + 1)  # fraction of total range
# for i, db in enumerate(valid_db):
#     base_color, hi_color = color_map.get(db, ('lightgray', 'gray'))

#     # Y-layer extends ± half-step for width
#     Y_layer = np.array([
#         [db - 0.5*deadband_step]*len(time_hours),
#         [db + 0.5*deadband_step]*len(time_hours)
#     ])
    
#     X_layer = np.tile(time_hours, (2, 1))  # same X for both rows
#     Z_layer = np.vstack([Z[i, :], Z[i, :]])  # duplicate Z for the two rows

#     ax.plot_surface(
#         X_layer, Y_layer, Z_layer,
#         facecolors=make_facecolors(base_color, X_layer, hi_color),
#         edgecolor='black',
#         linewidth=0.6,
#         rcount=2,
#         ccount=min(96, Z.shape[1]),
#         shade=False
#     )




############################################################################
#                           Axis Formatting                                #
############################################################################

A = 16  # axis label font
B = 13  # tick font

ax.set_xlabel("Time [H]", fontsize=A, labelpad=10)
ax.set_ylabel("TDB [F]", fontsize=A, labelpad=8)
ax.set_zlabel("Power [p.u.]", fontsize=A, rotation=90, labelpad=20)

# Set X-axis ticks and labels
ax.set_xlim(plot_start_hour, plot_end_hour)

# Flip X-axis so time decreases left → right
ax.set_xlim(ax.get_xlim()[::-1])

# Set X-axis ticks and labels
xticks = np.arange(plot_start_hour, plot_end_hour + 0.1, tick_spacing)
ax.set_xticks(xticks)
ax.set_xticklabels([f"{int(h):02d}" for h in xticks], fontsize=B)
ax.set_xlabel('Time [H]', fontsize=A, labelpad=8)
ax.set_yticks(valid_db)
ax.set_yticklabels(valid_db, fontsize=B)
ax.tick_params(axis='z', labelsize=B)

ax.view_init(elev=15, azim=110)
ax.grid(True, linestyle='--', linewidth=0.4, color='white')

# # Legend
# legend_elements = [
#     Patch(facecolor=color_map[db][0], edgecolor='black', label=f"DB {db}°F")
#     for db in valid_db
# ]
# ax.legend(handles=legend_elements, bbox_to_anchor=(0.85, 0.1),
#           ncol=3, fontsize=12, frameon=False)

ax.set_title(f"3D Power Surface — Setpoint {setpoint}°F", fontsize=A + 2, pad=15)

############################################################################
#                            Save and Finish                               #
############################################################################

output_file = os.path.join(RESULTS_DIR, f"PowerSurface_{setpoint}.pdf")
# plt.savefig(output_file, format='pdf', bbox_inches='tight', pad_inches=0.5)


############################################################################
#                           Timing Output                                  #
############################################################################

elapsed = time.time() - start_time
print(f"Execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} min)")
# print(f"Figure saved to: {output_file}")