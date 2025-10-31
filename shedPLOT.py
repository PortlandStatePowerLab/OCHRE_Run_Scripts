# =============================================================================
# 
# =============================================================================
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import joypy
# 
# np.random.seed(123)
# 
# # -------------------------
# # 1) Simulate continuous sweep variable
# # -------------------------
# n_points_per_group = 100
# sweep_var = np.linspace(0, 10, 20)  # the variable you want on vertical axis
# 
# records = []
# 
# for s in sweep_var:
#     # For each sweep value, generate a distribution of %Diff
#     values = np.random.normal(loc=np.sin(s) * 10, scale=3, size=n_points_per_group)
#     for v in values:
#         records.append({
#             'SweepVar': s,   # vertical axis
#             '%Diff': v       # horizontal axis
#         })
# 
# df = pd.DataFrame(records)
# 
# # -------------------------
# # 2) Bin sweep variable to create groups for joypy
# # -------------------------
# # joypy needs categorical grouping
# df['SweepGroup'] = df['SweepVar'].round(1).astype(str)
# 
# # -------------------------
# # 3) Ridgeline plot
# # -------------------------
# fig, axes = joypy.joyplot(
#     df,
#     by="SweepGroup",
#     column="%Diff",
#     kind="kde",
#     range_style='own',
#     tails=0.05,
#     overlap=3,
#     linewidth=1.5,
#     colormap=cm.viridis,
#     grid='y',
#     figsize=(12,8),
#     title="%Diff distributions across sweep variable"
# )
# 
# plt.xlabel("%Diff")
# plt.ylabel("Sweep Variable")
# plt.show()
# 
# =============================================================================
# 
# 
# 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import joypy

# -------------------------
# 1) User settings
# -------------------------
WORKING_DIR = r"C:\Users\danap\OCHRE_Working\Input Files"
power_col = 'Water Heating Electric Power (kW)'
scenarios = ['Baseline', 'Deadband 1F', 'Deadband 2F', 'Deadband 3F', 'Deadband 4F', 'Deadband 5F']

# Shed periods
shed_periods = {
    "Morning Shed": ("06:00", 4),
    "Evening Shed": ("17:00", 3),
    "Full Day": ("00:00", 24)
}

# -------------------------
# 2) Helper functions
# -------------------------
def percent_diff(base, control):
    if base == 0:
        return np.nan
    return (control - base) / base * 100

def energy_from_power_series(series):
    return series.sum() if len(series) > 1 else 0.0

def get_shed_mask(index, day, start_time_str, duration_hours):
    start = pd.to_datetime(f"{day} {start_time_str}")
    end = start + pd.Timedelta(hours=duration_hours)
    return (index >= start) & (index < end)

# -------------------------
# 3) Collect data for all scenarios
# -------------------------
all_records = []

homes = [os.path.join(WORKING_DIR, d) for d in os.listdir(WORKING_DIR)
         if os.path.isdir(os.path.join(WORKING_DIR, d))]

for home in homes:
    results_dir = os.path.join(home, "Results")
    baseline_file = os.path.join(results_dir, "hpwh_baseline.csv")
    if not os.path.exists(baseline_file):
        continue
    
    # Load baseline
    df_base = pd.read_csv(baseline_file, index_col=0, parse_dates=True)[[power_col]]
    df_base[power_col] = df_base[power_col].replace(0.001, 0)
    
    # Loop over Deadband scenarios
    for scen in scenarios[1:]:  # skip Baseline itself
        control_file = os.path.join(results_dir, f"hpwh_{scen.lower().replace(' ', '_')}.csv")
        if not os.path.exists(control_file):
            continue
        
        df_ctrl = pd.read_csv(control_file, index_col=0, parse_dates=True)[[power_col]]
        df_ctrl[power_col] = df_ctrl[power_col].replace(0.001, 0)
        
        # Loop over days and periods
        unique_days = pd.to_datetime(df_ctrl.index.date).unique()
        for day in unique_days:
            for period_name, (start_time, duration) in shed_periods.items():
                mask_base = get_shed_mask(df_base.index, day, start_time, duration)
                mask_ctrl = get_shed_mask(df_ctrl.index, day, start_time, duration)
                
                energy_base = energy_from_power_series(df_base[power_col][mask_base])
                energy_ctrl = energy_from_power_series(df_ctrl[power_col][mask_ctrl])
                
                all_records.append({
                    'Scenario': scen,
                    'House': os.path.basename(home),
                    'Period': period_name,
                    '%Diff': percent_diff(energy_base, energy_ctrl)
                })

# -------------------------
# 4) Create DataFrame
# -------------------------
df_plot = pd.DataFrame(all_records)
df_plot = df_plot.dropna(subset=['%Diff'])

# Convert Scenario to categorical for ordered ridges
df_plot['ScenarioGroup'] = pd.Categorical(df_plot['Scenario'], categories=scenarios[1:], ordered=True)

# -------------------------
# 5) Ridgeline plot
# -------------------------
fig, axes = joypy.joyplot(
    df_plot,
    by='ScenarioGroup',
    column='%Diff',
    kind='kde',
    range_style='own',
    tails=0.05,
    overlap=3,
    linewidth=1.5,
    colormap=cm.viridis,
    grid='y',
    figsize=(12,8),
    title="%Diff distributions across Deadband Scenarios"
)

plt.xlabel("%Diff (Baseline vs Control)")
plt.ylabel("Scenario")
plt.tight_layout()
plt.show()

# -------------------------
# 6) Optional: Median + 95% CI
# -------------------------
for ax, scen in zip(axes, df_plot['ScenarioGroup'].cat.categories):
    sub = df_plot[df_plot['ScenarioGroup']==scen]['%Diff']
    if sub.empty:
        continue
    median = sub.median()
    lower = sub.quantile(0.025)
    upper = sub.quantile(0.975)
    y_min, y_max = ax.get_ylim()
    ax.vlines(median, y_min, y_max, color='black', lw=2)
    ax.vlines([lower, upper], y_min, y_max, color='black', lw=1, linestyle='--')

