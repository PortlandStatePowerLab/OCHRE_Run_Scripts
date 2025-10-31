# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 15:03:33 2025

@author: danap
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
from datetime import timedelta

print("Processing...")

# User settings
WORKING_DIR = r"C:\Users\danap\OCHRE_Working\Input Files"
A = 24
baseColor = 'blue'
pwrColor = 'red'
controlColor = 'darkorchid'
COPconstant = 4.16748

# Define the specific columns we care about
Ttrue_col = 'Water Heating Control Temperature (C)'
power_col = 'Water Heating Electric Power (kW)'

def ET(TF, COP, Tset):
    """Energy take calculation."""
    c = 0.291
    p = 8.3454
    V = 50
    return c * p * V * (Tset - TF) / COP

def collapse_to_composite(series, how="sum"):
    """ Collapse series into a single-day profile by summing/averaging by time of day. """
    df = series.copy().to_frame("val")
    df["time_of_day"] = df.index.time
    if how == "sum":
        return df.groupby("time_of_day")["val"].sum()
    elif how == "mean":
        return df.groupby("time_of_day")["val"].mean()
    else:
        raise ValueError("how must be 'sum' or 'mean'")

# Initialize cumulative series
cumsum_ET_base = None
cumsum_ET_control = None
cumsum_power_base = None
cumsum_power_control = None





# Load all home results
homes = [os.path.join(WORKING_DIR, d) for d in os.listdir(WORKING_DIR)
         if os.path.isdir(os.path.join(WORKING_DIR, d))]

valid_homes = []
for home in homes:
    results_dir = os.path.join(home, "Results")
    baseline_file = os.path.join(results_dir, "hpwh_baseline.csv")
    control_file = os.path.join(results_dir, "hpwh_controlled.csv")
    
    if not (os.path.exists(baseline_file) and os.path.exists(control_file)):
        continue
    
    try:
        df_base = pd.read_csv(baseline_file, index_col=0, parse_dates=True)
        df_ctrl = pd.read_csv(control_file, index_col=0, parse_dates=True)
    except:
        continue

    # Only keep the temperature + power columns
    df_base = df_base[[Ttrue_col, power_col]]
    df_ctrl = df_ctrl[[Ttrue_col, power_col]]

    # Deduplicate / average if duplicate timestamps exist
    df_base = df_base.groupby(df_base.index).mean(numeric_only=True)
    df_ctrl = df_ctrl.groupby(df_ctrl.index).mean(numeric_only=True)

    valid_homes.append(home)
    
    # Convert temps to F
    TbaselineF = 9/5 * df_base[Ttrue_col] + 32
    TcontrolF = 9/5 * df_ctrl[Ttrue_col] + 32
    
    # Extract power
    power_base = df_base[power_col]
    power_control = df_ctrl[power_col]
    
    # Compute EnergyTake
    ET_base = ET(TbaselineF, COPconstant, 130)
    ET_ctrl = ET(TcontrolF, COPconstant, 130)
    
    # Initialize cumulative series if None
    if cumsum_ET_base is None:
        cumsum_ET_base = ET_base.copy()
        cumsum_ET_control = ET_ctrl.copy()
        cumsum_power_base = power_base.copy()
        cumsum_power_control = power_control.copy()
    else:
        # Align on timestamps before summing (important!)
        cumsum_ET_base = cumsum_ET_base.add(ET_base, fill_value=0)
        cumsum_ET_control = cumsum_ET_control.add(ET_ctrl, fill_value=0)
        cumsum_power_base = cumsum_power_base.add(power_base, fill_value=0)
        cumsum_power_control = cumsum_power_control.add(power_control, fill_value=0)

if not valid_homes:
    raise RuntimeError("No valid homes found with results!")

# Collapse to composite day
comp_ET_base = collapse_to_composite(cumsum_ET_base/1000, how="sum")   # kWh
comp_ET_control = collapse_to_composite(cumsum_ET_control/1000, how="sum")
comp_power_base = collapse_to_composite(cumsum_power_base, how="sum")
comp_power_control = collapse_to_composite(cumsum_power_control, how="sum")

# Convert back to datetime index for plotting (midnight → midnight reference day)
ref_date = pd.to_datetime("2000-01-01")
comp_ET_base.index = pd.to_datetime(comp_ET_base.index.astype(str), format="%H:%M:%S").map(
    lambda t: ref_date.replace(hour=t.hour, minute=t.minute, second=0)
)
comp_ET_control.index = comp_ET_base.index
comp_power_base.index = comp_ET_base.index
comp_power_control.index = comp_ET_base.index

# Plot composite daily profiles
fig, (ax_base, ax_control) = plt.subplots(2,1,figsize=(12,8), sharex=True, gridspec_kw={'height_ratios':[1,1]})
ax_base_twin = ax_base.twinx()
ax_control_twin = ax_control.twinx()

# Baseline
ax_base.plot(comp_ET_base.index, comp_ET_base, color=baseColor)
ax_base_twin.plot(comp_ET_base.index, comp_power_base, color=pwrColor)

# Control
ax_control.plot(comp_ET_control.index, comp_ET_control, color=controlColor)
ax_control_twin.plot(comp_ET_control.index, comp_power_control, color=pwrColor)

# Formatting
for ax, ax_twin, ylabel, color in zip([ax_base, ax_control],
                                      [ax_base_twin, ax_control_twin],
                                      ["(Normal)\nEnergyTake (kWh)","(LoadUp/Shed)\nEnergyTake (kWh)"],
                                      [baseColor, controlColor]):
    ax.set_ylabel(ylabel, color=color, fontsize=A)
    ax.tick_params(axis='y', labelcolor=color, labelsize=A)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

for ax_twin in [ax_base_twin, ax_control_twin]:
    ax_twin.set_ylabel("Power (kW)", color=pwrColor, fontsize=A)
    ax_twin.tick_params(axis='y', labelcolor=pwrColor, labelsize=A)

# X-axis formatting (midnight to midnight)
time_format = mdates.DateFormatter('%H:%M')
for ax in [ax_base, ax_control]:
    ax.xaxis.set_major_formatter(time_format)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    # ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    ax.tick_params(axis='x', labelrotation=65, labelsize=A)
    ax.grid(True)
    ax.set_xlim(ref_date, ref_date + timedelta(days=1))

# Define schedule (same every day)
my_schedule = {
    'M_LU_time': '03:00', 'M_LU_duration': 3,
    'M_S_time': '06:00', 'M_S_duration': 4,
    'E_ALU_time': '16:00', 'E_ALU_duration': 1,
    'E_S_time': '17:00', 'E_S_duration': 3
}





periods = {
    "M_LU": {"color": "xkcd:light grass green"},
    "M_S": {"color": "xkcd:orangey yellow"},
    "E_ALU": {"color": "xkcd:light grass green"},
    "E_S": {"color": "xkcd:orangey yellow"}
}

def get_time_range(base_date, key_prefix):
    start = pd.to_datetime(f"{base_date.date()} {my_schedule[f'{key_prefix}_time']}")
    end = start + pd.Timedelta(hours=my_schedule[f'{key_prefix}_duration'])
    return start, end

# Shade regions on both plots
for key, info in periods.items():
    start, end = get_time_range(ref_date, key)
    for ax in [ax_base, ax_control]:
        ax.axvspan(start, end, color=info['color'], alpha=0.2)
        
        

# EnergyTake (kWh)
max_ET = max(comp_ET_base.max(), comp_ET_control.max())
ax_base.set_ylim(0, max_ET * 1.05)
ax_control.set_ylim(0, max_ET * 1.05)

# Power (kW)
max_power = max(comp_power_base.max(), comp_power_control.max())
ax_base_twin.set_ylim(0, max_power * 1.05/2)
ax_control_twin.set_ylim(0, max_power * 1.05)


plt.tight_layout()
points = plt.ginput(4)
print(points)
plt.show()
