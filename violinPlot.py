# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 10:36:01 2025

@author: danap
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Processing...")

# ----------------- USER SETTINGS -----------------
WORKING_DIR = r"C:\Users\danap\OCHRE_Working\Input Files"
power_col = 'Water Heating Electric Power (kW)'


# Shed periods (start time, duration in hours)
shed_periods = {
    "Morning Shed": ("06:00", 4),
    "Evening Shed": ("17:00", 3),
    "Full Day": ("00:00", 24)
}

# ----------------- HELPER FUNCTIONS -----------------
def percent_diff(base, control):
    """Percent difference, safe for zero denominators"""
    if base == 0:
        return float('nan')
    return (control - base) / base * 100

def energy_from_power_series(series):
    if len(series) < 2:
        return 0.0
    return series.sum() 

def get_shed_mask(index, day, start_time_str, duration_hours):
    """Returns boolean mask for a shed period for a given day."""
    start = pd.to_datetime(f"{day} {start_time_str}")
    end = start + pd.Timedelta(hours=duration_hours)
    return (index >= start) & (index < end)

# ----------------- COLLECT DATA -----------------
homes = [os.path.join(WORKING_DIR, d) for d in os.listdir(WORKING_DIR)
         if os.path.isdir(os.path.join(WORKING_DIR, d))]

records = []

for home in homes:
    results_dir = os.path.join(home, "Results")
    baseline_file = os.path.join(results_dir, "hpwh_baseline.csv")
    control_file = os.path.join(results_dir, "hpwh_controlled.csv")
    
    if not (os.path.exists(baseline_file) and os.path.exists(control_file)):
        continue
    
    # Load data
    df_base = pd.read_csv(baseline_file, index_col=0, parse_dates=True)
    df_ctrl = pd.read_csv(control_file, index_col=0, parse_dates=True)
    
    # Keep only power column
    df_base = df_base[[power_col]]
    df_ctrl = df_ctrl[[power_col]]
    
    # Clean bogus "off" values (0.001 -> 0)
    df_base[power_col] = df_base[power_col].replace(0.001, 0)
    df_ctrl[power_col] = df_ctrl[power_col].replace(0.001, 0)
    
    # Loop over unique days
    unique_days = pd.to_datetime(df_ctrl.index.date).unique()
    
    for day in unique_days:
        for period_name, (start_time, duration) in shed_periods.items():
            mask_base = get_shed_mask(df_base.index, day, start_time, duration)
            mask_ctrl = get_shed_mask(df_ctrl.index, day, start_time, duration)
            
            energy_base = energy_from_power_series(df_base[power_col][mask_base])
            energy_ctrl = energy_from_power_series(df_ctrl[power_col][mask_ctrl])
            
            records.append({
                'House': os.path.basename(home),
                'Day': day,
                'Period': period_name,
                'Energy_Base': energy_base,
                'Energy_Control': energy_ctrl,
                '%Diff': percent_diff(energy_base, energy_ctrl)
            })

# ----------------- CREATE DATAFRAME -----------------
df = pd.DataFrame(records)


plt.figure(figsize=(10, 6))

loadColor = 'xkcd:light grass green'
shedColor = 'xkcd:orangey yellow'
controlColor = 'xkcd:robin egg blue'

morning = 'xkcd:spruce'
evening = 'red'
full = 'xkcd:rich blue'

colors = [loadColor, shedColor, controlColor]
dot = [morning, evening, full]

# --- 1) Violin plot with pastel colors ---
sns.violinplot(
    x='Period',
    y='%Diff',
    hue='Period',        # color by period
    data=df,
    inner=None,          # remove quartiles, we'll add our own
    palette=colors,
    dodge=False,         # make sure violins don't split by hue
    density_norm='count',
    legend=False
)

# --- 2) Raw data points ---
sns.stripplot(
    x='Period',
    y='%Diff',
    data=df,
    hue='Period',
    palette=dot,
    size=2,
    alpha=0.5,
    zorder=1,
    legend=False
)


# --- 3) Median + 95% percentile interval ---
summary = df.groupby("Period")["%Diff"].agg(
    lower=lambda x: x.quantile(0.025),
    median="median",
    upper=lambda x: x.quantile(0.975)
).reset_index()

plt.errorbar(
    x=summary["Period"],
    y=summary["median"],
    yerr=[summary["median"] - summary["lower"],
          summary["upper"] - summary["median"]],
    fmt="o",
    color="black",
    markerfacecolor="black",
    markeredgecolor="black",
    capsize=20,
    elinewidth=1.5,
    markersize=8,
    zorder=100
)


# Set ylim dynamically: 15% above the highest upper 95% bound
ylim_upper = summary["upper"].max() * 1.5

# --- Styling ---
A = 18
plt.ylim(-100, ylim_upper)
plt.ylabel("%Diff (Baseline vs Control)", fontsize=A)
plt.xlabel('')
plt.xticks(fontsize=A)
plt.yticks(fontsize=A)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




print("Done!")