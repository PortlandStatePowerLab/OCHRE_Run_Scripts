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

filename_baseline = 'hpwh_baseline_3min_Master_180111.csv'
filename_controlled = '180111_1_3_RC_2whAgg_contr.csv'
filename_controlled2 = 'hpwh_controlled_3min_Master_180111.csv'
WORKING_DIR = r"C:\Users\danap\OCHRE_Working"


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

# -------------------------------
# Load aggregated CSV
# -------------------------------
agg_controlled = os.path.join(WORKING_DIR, filename_controlled)
agg_controlled2 = os.path.join(WORKING_DIR, filename_controlled2)
agg_baseline = os.path.join(WORKING_DIR, filename_baseline)
# df_controlled = pd.read_csv(agg_controlled, index_col=0, parse_dates=True)
# df_baseline = pd.read_csv(agg_baseline, index_col=0, parse_dates=True)

# # Aggregate by timestamp across all homes
# df_controlled = df_controlled.groupby(df_controlled.index).agg({
#     'Water Heating Electric Power (kW)': 'sum',        # total power across homes
#     'Water Heating Control Temperature (C)': 'mean'    # average temp across homes
# })
# df_baseline = df_baseline.groupby(df_baseline.index).agg({
#     'Water Heating Electric Power (kW)': 'sum',        # total power across homes
#     'Water Heating Control Temperature (C)': 'mean'    # average temp across homes
# })

# -------------------------------
# Aggregate and compute ET at each timestep
# -------------------------------

def calc_ET_instant(tempC, COP, Tset_F):
    """Compute instantaneous ET in kWh at each timestep."""
    tempF = 9/5 * tempC + 32
    return ET(tempF, COP, Tset_F) / 1000.0  # kWh

def process_dataset(path):
    # Load raw CSV
    df_raw = pd.read_csv(path, index_col=0, parse_dates=True)

    # Compute instantaneous ET
    df_raw["ET (kWh)"] = calc_ET_instant(df_raw[Ttrue_col], COPconstant, 130)

    # Sum power and ET across all buildings at each timestep
    df_out = df_raw.groupby(df_raw.index).agg({
        'Water Heating Electric Power (kW)': 'sum',
        'ET (kWh)': 'sum'
    })

    # Add helper columns for time-of-day and date
    df_out["time_of_day"] = df_out.index.time
    df_out["date"] = df_out.index.date

    return df_out

# Process controlled and baseline datasets
df_controlled = process_dataset(agg_controlled)
df_controlled2 = process_dataset(agg_controlled2)
df_baseline   = process_dataset(agg_baseline)

# Extract instantaneous ET and power
power_control = df_controlled[power_col]
power_control2 = df_controlled2[power_col]
power_base    = df_baseline[power_col]

# Collapse to composite by time of day (sum/average across days)
ET_control = df_controlled.groupby("time_of_day")["ET (kWh)"].mean()
ET_control2 = df_controlled2.groupby("time_of_day")["ET (kWh)"].mean()
ET_base    = df_baseline.groupby("time_of_day")["ET (kWh)"].mean()


# -------------------------------
# Plot composite daily profiles
# -------------------------------

# Convert time_of_day index into datetime (midnight reference day)
ref_date = pd.to_datetime("2000-01-01")
ET_base.index = pd.to_datetime(ET_base.index.astype(str), format="%H:%M:%S").map(
    lambda t: ref_date.replace(hour=t.hour, minute=t.minute, second=0)
)
ET_control.index = ET_base.index  # ensure same reference

# Collapse power by time of day (to match ET)
comp_power_base = df_baseline.groupby("time_of_day")["Water Heating Electric Power (kW)"].mean()
comp_power_control = df_controlled.groupby("time_of_day")["Water Heating Electric Power (kW)"].mean()
comp_power_control2 = df_controlled2.groupby("time_of_day")["Water Heating Electric Power (kW)"].mean()
comp_power_base.index = ET_base.index
comp_power_control.index = ET_base.index
comp_power_control2.index = ET_base.index

# Create subplots
fig, (ax_base, ax_control) = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[1,1]}
)
ax_base_twin = ax_base.twinx()
ax_control_twin = ax_control.twinx()

# Baseline
ax_base.plot(ET_base.index, ET_base, color=baseColor, label="ET (kWh)")
ax_base_twin.plot(comp_power_base.index, comp_power_base, color=pwrColor, label="Power (kW)")

# Control
ax_control.plot(ET_control.index, ET_control, color=controlColor, label="ET (kWh)")
ax_control_twin.plot(comp_power_control.index, comp_power_control, color=pwrColor, label="Power (kW)")
ax_control_twin.plot(comp_power_control2.index, comp_power_control2, color='green', label='Power (kW')

# Formatting
for ax, ax_twin, ylabel, color in zip(
    [ax_base, ax_control],
    [ax_base_twin, ax_control_twin],
    ["(Normal)\nEnergyTake (kWh)", "(LoadUp/Shed)\nEnergyTake (kWh)"],
    [baseColor, controlColor]
):
    ax.set_ylabel(ylabel, color=color, fontsize=A)
    ax.tick_params(axis='y', labelcolor=color, labelsize=A)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

for ax_twin in [ax_base_twin, ax_control_twin]:
    ax_twin.set_ylabel("Power (kW)", color=pwrColor, fontsize=A)
    ax_twin.tick_params(axis='y', labelcolor=pwrColor, labelsize=A)

# X-axis formatting (midnight → midnight)
time_format = mdates.DateFormatter('%H:%M')
for ax in [ax_base, ax_control]:
    ax.xaxis.set_major_formatter(time_format)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.tick_params(axis='x', labelrotation=65, labelsize=A)
    ax.grid(True)
    ax.set_xlim(ref_date, ref_date + timedelta(days=1))

# -------------------------------
# Shade schedule periods
# -------------------------------
my_schedule = {
    'M_LU_time': '03:00', 'M_LU_duration': 3,
    'M_S_time': '06:00', 'M_S_duration': 4,
    'E_ALU_time': '14:00', 'E_ALU_duration': 3,
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

for key, info in periods.items():
    start, end = get_time_range(ref_date, key)
    for ax in [ax_base, ax_control]:
        ax.axvspan(start, end, color=info['color'], alpha=0.2)

# Y-limits
max_ET = max(ET_base.max(), ET_control.max())
ax_base.set_ylim(0, max_ET * 1.05)
ax_control.set_ylim(0, max_ET * 1.05)

max_power = max(comp_power_base.max(), comp_power_control.max())
ax_base_twin.set_ylim(0, max_power * 1.05)
ax_control_twin.set_ylim(0, max_power * 1.05)

plt.tight_layout()
plt.show()
