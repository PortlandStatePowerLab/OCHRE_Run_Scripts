# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:46:36 2025

@author: danap
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

############################################################################
#                           User Inputs                                    #
############################################################################

WORKING_DIR = r"C:\Users\danap\OCHRE_Working"
input_file_name = "D0_180111_1_15_Baseline.csv"

# MCS parameters
MCS_runs = 1000
n_values = [500, 1000, 5000]  # number of units to test

############################################################################
#                           Helper Functions                               #
############################################################################

def sample_data(input_df, units):
    """Randomly sample N rows with replacement and drop 'Home' column."""
    df_sampled = input_df.sample(n=units, replace=True)
    df_sampled = df_sampled.drop(['Home'], axis=1)
    return df_sampled

def get_MCS_run(N, input_df, MCS_runs):
    """Generate MCS table for given N."""
    times = input_df.drop(['Home'], axis=1).columns
    MCS_table = pd.DataFrame(np.nan, index=range(MCS_runs), columns=times)
    
    for j in range(MCS_runs):
        df_sampled = sample_data(input_df, N)
        agg_sample = df_sampled.sum()
        MCS_table.loc[j] = agg_sample
    
    # normalize if needed (as in your original script)
    MCS_table = MCS_table.div(0.5 * N)
    return MCS_table

############################################################################
#                               Main Script                                #
############################################################################

# Load the data
input_file = os.path.join(WORKING_DIR, input_file_name)
df = pd.read_csv(input_file)

# Convert time labels to something easier to control (assuming "HH:MM" format)
times = df.drop(['Home'], axis=1).columns

# Choose tick interval (every 2, 3, or 4 hours)
tick_interval = 3 # every tick_interval hours
tick_interval = tick_interval * 4
x_tick_positions = np.arange(0, len(times), tick_interval)
x_tick_labels = [times[i] for i in x_tick_positions]

A = 16
# Set up plot
plt.figure(figsize=(8, 5))

colors = ['red', 'blue', 'green']

for n, c in zip(n_values, colors):
    print(f"Running MCS for n = {n} ...")
    MCS_table = get_MCS_run(n, df, MCS_runs)
    
    # Plot all 1000 runs in faint color (no legend)
    for i in range(MCS_runs):
        plt.plot(times, MCS_table.iloc[i].values, color=c, alpha=0.05, linewidth=0.5)
    
    # Plot mean curve in yellow
    mean_curve = MCS_table.mean()
    plt.plot(times, mean_curve, color='lightblue', linewidth=1, label=f'Mean n={n}')


plt.xlabel('Time', fontsize=A)
plt.ylabel('Power [p.u.]', fontsize=A)
# Create custom legend lines to match the sample colors (red, blue, green)
from matplotlib.lines import Line2D

legend_lines = [
    Line2D([0], [0], color=c, linewidth=2, label=f'MCS for n={n}')
    for n, c in zip(n_values, colors)
]
plt.legend(handles=legend_lines)

plt.grid(True, alpha=0.3)
plt.xticks(x_tick_positions, x_tick_labels, rotation=45)
plt.tick_params(axis='x', labelsize=A)
plt.tick_params(axis='y', labelsize=A)
plt.tight_layout()

# Save plot
output_path = os.path.join(WORKING_DIR, "MCS_runs_plot.png")
# plt.savefig(output_path, dpi=300)
plt.show()

print(f"\nPlot saved to: {output_path}")
