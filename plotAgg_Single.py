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

filename_baseline = '180407_1_15_xx_Baseline.parquet'
filename_controlled = '180407_1_15_xx_Control.parquet'
filename_controlled2 = '180407_1_15_xx_Control.parquet'
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

# -------------------------------
# Aggregate and compute ET at each timestep
# -------------------------------

def calc_ET_instant(tempC, COP, Tset_F):
    """Compute instantaneous ET in kWh at each timestep."""
    tempF = 9/5 * tempC + 32
    return ET(tempF, COP, Tset_F) / 1000.0  # kWh

# def process_dataset(path):
#     # Load raw CSV
#     df_raw = pd.read_parquet(path)

#     # Compute instantaneous ET
#     df_raw["ET (kWh)"] = calc_ET_instant(df_raw[Ttrue_col], COPconstant, 130)

#     # Sum power and ET across all buildings at each timestep
#     df_out = df_raw.groupby(df_raw.index).agg({
#         'Water Heating Electric Power (kW)': 'sum',
#         'ET (kWh)': 'sum'
#     })

#     # Add helper columns for time-of-day and date
#     df_out["time_of_day"] = df_out.index.time
#     df_out["date"] = df_out.index.date

#     return df_out

def process_dataset(path):
    """Load Parquet, ensure datetime index, compute ET, and aggregate."""
    df_raw = pd.read_parquet(path)

    # If the index is not datetime, try to find a timestamp column
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        possible_time_cols = [c for c in df_raw.columns if "time" in c.lower() or "date" in c.lower()]
        if len(possible_time_cols) == 0:
            raise ValueError(
                f"No datetime index or time column found in {os.path.basename(path)}. "
                f"Columns available: {list(df_raw.columns)}"
            )
        time_col = possible_time_cols[0]
        df_raw[time_col] = pd.to_datetime(df_raw[time_col], errors="coerce")
        df_raw = df_raw.set_index(time_col)
        if df_raw.index.isnull().any():
            raise ValueError(f"Datetime conversion failed for column {time_col} in {os.path.basename(path)}")

    # Compute instantaneous ET
    df_raw["ET (kWh)"] = calc_ET_instant(df_raw[Ttrue_col], COPconstant, 130)

    # Sum power and ET across all buildings at each timestep
    df_out = df_raw.groupby(df_raw.index).agg({
        power_col: 'sum',
        'ET (kWh)': 'sum'
    })

    # Add helper columns
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
comp_power_base = df_baseline.groupby("time_of_day")["Water Heating Electric Power (kW)"].sum()
comp_power_control = df_controlled.groupby("time_of_day")["Water Heating Electric Power (kW)"].sum()
comp_power_control2 = df_controlled2.groupby("time_of_day")["Water Heating Electric Power (kW)"].sum()
comp_power_base.index = ET_base.index
comp_power_control.index = ET_base.index
comp_power_control2.index = ET_base.index

# Create subplots
# fig, (ax_base, ax_control) = plt.subplots(
#     2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[1,1]}
# )
# ax_base_twin = ax_base.twinx()
# ax_control_twin = ax_control.twinx()

# =============================================================================
# # -------------------------------
# # Create stacked subplots (baseline on top, controlled on bottom)
# # -------------------------------
# fig, (ax_base, ax_control) = plt.subplots(
#     2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[1,1]}
# )
# 
# # -------------------------------
# # Baseline subplot
# # -------------------------------
# ax_base.plot(comp_power_base.index, comp_power_base, color=pwrColor, label="Baseline Power (kW)")
# ax_base.set_ylabel("Baseline\nPower (kW)", fontsize=A, color=pwrColor)
# ax_base.tick_params(axis='y', labelcolor=pwrColor, labelsize=A)
# ax_base.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# ax_base.grid(True)
# 
# # -------------------------------
# # Controlled subplot
# # -------------------------------
# ax_control.plot(comp_power_control.index, comp_power_control, color=pwrColor, label="Controlled Power (kW)")
# ax_control.plot(comp_power_control2.index, comp_power_control2, color="green", label="Controlled2 Power (kW)")
# ax_control.set_ylabel("Controlled\nPower (kW)", fontsize=A, color=pwrColor)
# ax_control.tick_params(axis='y', labelcolor=pwrColor, labelsize=A)
# ax_control.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# ax_control.grid(True)
# 
# # -------------------------------
# # X-axis formatting (shared)
# # -------------------------------
# time_format = mdates.DateFormatter('%H:%M')
# ax_control.xaxis.set_major_formatter(time_format)
# ax_control.xaxis.set_major_locator(mdates.HourLocator(interval=2))
# ax_control.tick_params(axis='x', labelrotation=65, labelsize=A)
# 
# # X-axis range: midnight → midnight
# for ax in [ax_base, ax_control]:
#     ax.set_xlim(ref_date, ref_date + timedelta(days=1))
# =============================================================================

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

# # Apply shading to both subplots
# for key, info in periods.items():
#     start, end = get_time_range(ref_date, key)
#     for ax in [ax_base, ax_control]:
#         ax.axvspan(start, end, color=info['color'], alpha=0.2)

    # Apply shading only to the controlled subplot
    for key, info in periods.items():
        start, end = get_time_range(ref_date, key)
        ax_control.axvspan(start, end, color=info['color'], alpha=0.2)


# -------------------------------
# Y-limits (synchronized)
# -------------------------------
max_power = max(comp_power_base.max(), comp_power_control.max(), comp_power_control2.max())
# ax_base.set_ylim(0, max_power * 1.05)


# plt.tight_layout()
# plt.show()

# -------------------------------
# Create only one subplot (controlled)
# -------------------------------
fig, ax_control = plt.subplots(figsize=(9, 5))

ax_control.set_ylim(0, max_power * 1.05)

# Controlled power curves
ax_control.plot(comp_power_control.index, comp_power_base, color=pwrColor, linewidth=3)
ax_control.plot(comp_power_control2.index, comp_power_control2, color="darkgreen", linewidth=3)
ax_control.fill_between(comp_power_control.index, comp_power_control, comp_power_control2,
                        where=(comp_power_control < comp_power_control2), alpha=1, linewidth=0, color='lightgreen')
ax_control.fill_between(comp_power_control.index, comp_power_control, comp_power_control2,
                        where=(comp_power_control > comp_power_control2), alpha=1, linewidth=0, color='lightpink')

# Labels and formatting
ax_control.set_ylabel("Power [kW]", fontsize=A, color='black')
ax_control.set_xlabel("Time [H:M]", fontsize=A, color='black')
ax_control.tick_params(axis='y', labelcolor='black', labelsize=A)
ax_control.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax_control.grid(True)

# X-axis formatting (time of day)
# time_format = mdates.DateFormatter('%H:%M')
time_format = mdates.DateFormatter('%H')
ax_control.xaxis.set_major_formatter(time_format)
ax_control.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax_control.tick_params(axis='x', labelrotation=65, labelsize=A) # labelrotation = 65 normal
ax_control.set_xlim(ref_date, ref_date + timedelta(days=1))

# # -------------------------------
# # Shade schedule periods
# # -------------------------------
# for key, info in periods.items():
#     start, end = get_time_range(ref_date, key)
#     ax_control.axvspan(start, end, color=info['color'], alpha=0.2)

# Y-limits (optional)
max_power = max(comp_power_control.max(), comp_power_control2.max())
ax_control.set_ylim(0, max_power * 1.05)

plt.tight_layout()
plt.show()

