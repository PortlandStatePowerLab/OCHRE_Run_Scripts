# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 17:39:26 2025

@author: danap
"""

import os
import shutil
import datetime as dt
import pandas as pd
from ochre import Dwelling
from ochre.utils.schedule import ALL_SCHEDULE_NAMES
import concurrent.futures

#########################################
# USER SETTINGS
#########################################

filename = '180111_1_15_NR'

# Original OCHRE defaults folder
DEFAULT_INPUT = r"C:\Users\danap\anaconda3\Lib\site-packages\ochre\defaults\Input Files"
DEFAULT_WEATHER = r"C:\Users\danap\anaconda3\Lib\site-packages\ochre\defaults\Weather\USA_OR_Portland.Intl.AP.726980_TMY3.epw"

# Safe working folder (writable)
WORKING_DIR = r"C:\Users\danap\OCHRE_Working"
INPUT_DIR = os.path.join(WORKING_DIR, "Input Files")
WEATHER_DIR = os.path.join(WORKING_DIR, "Weather")
WEATHER_FILE = os.path.join(WEATHER_DIR, "USA_OR_Portland.Intl.AP.726980_TMY3.epw")


# Simulation parameters
Start = dt.datetime(2018, 1, 11, 0, 0)
Duration = 2  # days
t_res = 15  # minutes

# HPWH control parameters (°F)
Tcontrol_SHEDF = 126
Tcontrol_deadbandF = 10
Tcontrol_LOADF = 130
Tcontrol_LOADdeadbandF = 2
TbaselineF = 130
TdeadbandF = 7
Tinit = 128

# Schedule variant
my_schedule = {
    'M_LU_time': '05:30',
    'M_LU_duration': 0.5,
    'M_S_time': '06:00',
    'M_S_duration': 4,
    'E_ALU_time': '16:30',
    'E_ALU_duration': 0.5,
    'E_S_time': '17:00',
    'E_S_duration': 3
}

#########################################
# TEMPERATURE CONVERSIONS F to C
#########################################

def f_to_c(temp_f): 
    return (temp_f - 32) * 5/9

def f_to_c_DB(temp_f):
    return 5/9 * temp_f

Tcontrol_SHEDC = f_to_c(Tcontrol_SHEDF)
Tcontrol_deadbandC = f_to_c_DB(Tcontrol_deadbandF)
Tcontrol_LOADC = f_to_c(Tcontrol_LOADF)
Tcontrol_LOADdeadbandC = f_to_c_DB(Tcontrol_LOADdeadbandF)
TbaselineC = f_to_c(TbaselineF)
TdeadbandC = f_to_c_DB(TdeadbandF)
TinitC = f_to_c(Tinit)

#########################################
# HPWH CONTROL FUNCTION
#########################################

def determine_hpwh_control(sim_time, current_temp_c, sched_cfg, **kwargs):
    ctrl_signal = {
        'Water Heating': {
            'Setpoint': TbaselineC,
            'Deadband': TdeadbandC,
            'Load Fraction': 1,
        }
    }

    base_date = sim_time.date()
    def get_time_range(key_prefix):
        start = pd.to_datetime(f"{base_date} {sched_cfg[f'{key_prefix}_time']}")
        end = start + pd.Timedelta(hours=sched_cfg[f'{key_prefix}_duration'])
        return start, end

    ranges = {
        'M_LU': get_time_range('M_LU'),
        'M_S': get_time_range('M_S'),
        'E_ALU': get_time_range('E_ALU'),
        'E_S': get_time_range('E_S'),
    }

    if ranges['M_LU'][0] <= sim_time < ranges['M_LU'][1] or ranges['E_ALU'][0] <= sim_time < ranges['E_ALU'][1]:
        ctrl_signal['Water Heating'].update({
            'Setpoint': Tcontrol_LOADC,
            'Deadband': Tcontrol_LOADdeadbandC
        })
    elif ranges['M_S'][0] <= sim_time < ranges['M_S'][1] or ranges['E_S'][0] <= sim_time < ranges['E_S'][1]:
        ctrl_signal['Water Heating'].update({
            'Setpoint': Tcontrol_SHEDC,
            'Deadband': Tcontrol_deadbandC
        })

    return ctrl_signal

#########################################
# SCHEDULE FILTERING
#########################################

def filter_schedules(home_path):
    orig_sched_file = os.path.join(home_path, 'schedules.csv')
    filtered_sched_file = os.path.join(home_path, 'filtered_schedules.csv')

    df_sched = pd.read_csv(orig_sched_file)
    valid_schedule_names = set(ALL_SCHEDULE_NAMES.keys())
    filtered_columns = [col for col in df_sched.columns if col in valid_schedule_names]
    dropped_columns = [col for col in df_sched.columns if col not in filtered_columns]
    if dropped_columns:
        print(f"Dropped invalid schedules for {home_path}: {dropped_columns}")

    df_sched_filtered = df_sched[filtered_columns]
    df_sched_filtered.to_csv(filtered_sched_file, index=False)
    return filtered_sched_file

#########################################
# SIMULATION FUNCTION
#########################################

def simulate_home(home_path, weather_file_path, schedule_cfg):

    filtered_sched_file = filter_schedules(home_path)
    hpxml_file = os.path.join(home_path, 'in.XML')
    results_dir = os.path.join(home_path, "Results")
    os.makedirs(results_dir, exist_ok=True)

    dwelling_args_local = {
        "start_time": Start,
        "time_res": dt.timedelta(minutes=t_res),
        "duration": dt.timedelta(days=Duration),
        "hpxml_file": hpxml_file,
        "hpxml_schedule_file": filtered_sched_file,
        "weather_file": weather_file_path,
        "verbosity": 7,
        "Equipment": {
            "Water Heating": {
                "Initial Temperature (C)": TinitC, 
                "hp_only_mode": True,
                "Max Tank Temperature": 70,
                "Upper Node": 3,
                "Lower Node": 10,
                "Upper Node Weight": 0.75,
            },
        }
    }

    # Baseline
    base_dwelling = Dwelling(name="HPWH Baseline", **dwelling_args_local)
    for t_base in base_dwelling.sim_times:
        base_ctrl = {"Water Heating": {"Setpoint": TbaselineC, "Deadband": TdeadbandC, "Load Fraction": 1}}
        base_dwelling.update(control_signal=base_ctrl)
    df_base, _, _ = base_dwelling.finalize()

    # Controlled
    sim_dwelling = Dwelling(name="HPWH Controlled", **dwelling_args_local)
    hpwh_unit = sim_dwelling.get_equipment_by_end_use('Water Heating')
    for sim_time in sim_dwelling.sim_times:
        current_setpt = hpwh_unit.schedule.loc[sim_time, 'Water Heating Setpoint (C)']
        control_cmd = determine_hpwh_control(sim_time=sim_time, current_temp_c=current_setpt, sched_cfg=schedule_cfg)
        sim_dwelling.update(control_signal=control_cmd)
    df_ctrl, _, _ = sim_dwelling.finalize()


    df_ctrl = remove_first_day(df_ctrl, Start)
    df_base = remove_first_day(df_base, Start)
    

    CTRL_COLS = ["Time", "Total Electric Power (kW)",
                 "Total Electric Energy (kWh)",
                 "Water Heating Electric Power (kW)",
                 "Water Heating COP (-)",
                 "Water Heating Deadband Upper Limit (C)",
                 "Water Heating Deadband Lower Limit (C)",
                 "Water Heating Heat Pump COP (-)",
                 "Water Heating Control Temperature (C)",
                 "Hot Water Outlet Temperature (C)",
                 "Temperature - Indoor (C)"]
    BASE_COLS = CTRL_COLS
    

    df_ctrl = df_ctrl[[c for c in CTRL_COLS if c in df_ctrl.columns]]

    df_base = df_base[[c for c in BASE_COLS if c in df_base.columns]]
        
    
    df_ctrl.to_csv(os.path.join(results_dir, 'hpwh_controlled.csv'), index=False)
    df_base.to_csv(os.path.join(results_dir, 'hpwh_baseline.csv'), index=False)


    return df_ctrl, df_base

#########################################
# FIND ALL HOMES
#########################################

def find_all_homes(base_dir):
    homes = []
    for item in os.listdir(base_dir):
        home_path = os.path.join(base_dir, item)
        if os.path.isdir(home_path):
            # Only add folders with required files
            if os.path.isfile(os.path.join(home_path, 'in.XML')) and \
               os.path.isfile(os.path.join(home_path, 'schedules.csv')):
                homes.append(home_path)
    return homes

#########################################
# DELETE FIRST DAY ONLY
#########################################

def remove_first_day(df, start_date):
    """
    Remove the first day of simulation results.
    Works whether 'Time' is a column or the index.
    """
    # If 'Time' column doesn't exist, try using the index
    if 'Time' not in df.columns:
        df = df.reset_index()
        if 'index' in df.columns:
            df.rename(columns={'index': 'Time'}, inplace=True)

    # Ensure Time is datetime
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

    # Remove first day
    first_day_end = start_date + pd.Timedelta(days=1)
    return df[df['Time'] >= first_day_end].copy()



#########################################
# MAIN EXECUTION
#########################################

if __name__ == "__main__":
    # Ensure working folders exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(WEATHER_DIR, exist_ok=True)

    # Copy all homes from defaults (if not already copied)
    for item in os.listdir(DEFAULT_INPUT):
        src = os.path.join(DEFAULT_INPUT, item)
        dst = os.path.join(INPUT_DIR, item)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)

    # Copy weather file
    if not os.path.exists(WEATHER_FILE):
        shutil.copy(DEFAULT_WEATHER, WEATHER_FILE)

    # Discover homes
    homes = find_all_homes(INPUT_DIR)
    print(f"Found {len(homes)} homes")

    # Parallel simulations (threads are Windows-safe)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(simulate_home, home, WEATHER_FILE, my_schedule) for home in homes]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()  # forces execution and raises exceptions if any
            except Exception as e:
                print("Simulation failed:", e)

    print("All simulations complete!")




# def aggregate_results(homes, work_dir, ctrl_cols=None, base_cols=None):
#     all_ctrl, all_base = [], []

#     for home in homes:
#         results_dir = os.path.join(home, "Results")
#         ctrl_file = os.path.join(results_dir, "hpwh_controlled.csv")
#         base_file = os.path.join(results_dir, "hpwh_baseline.csv")

#         if os.path.exists(ctrl_file):
#             df_ctrl = pd.read_csv(ctrl_file)
#             if ctrl_cols:  # filter only selected columns
#                 df_ctrl = df_ctrl[[c for c in ctrl_cols if c in df_ctrl.columns]]
#             df_ctrl["Home"] = os.path.basename(home)
#             all_ctrl.append(df_ctrl)

#         if os.path.exists(base_file):
#             df_base = pd.read_csv(base_file)
#             if base_cols:
#                 df_base = df_base[[c for c in base_cols if c in df_base.columns]]
#             df_base["Home"] = os.path.basename(home)
#             all_base.append(df_base)

#     if all_ctrl:
#         df_ctrl_all = pd.concat(all_ctrl, ignore_index=True)
#         df_ctrl_all.to_csv(os.path.join(work_dir, "hpwh_controlled_all.csv"), index=False)

#     if all_base:
#         df_base_all = pd.concat(all_base, ignore_index=True)
#         df_base_all.to_csv(os.path.join(work_dir, "hpwh_baseline_all.csv"), index=False)

#     print("Aggregated CSVs written!")

def aggregate_results(homes, work_dir):
    all_ctrl, all_base = [], []

    for home in homes:
        results_dir = os.path.join(home, "Results")
        ctrl_file = os.path.join(results_dir, "hpwh_controlled.csv")
        base_file = os.path.join(results_dir, "hpwh_baseline.csv")

        if os.path.exists(ctrl_file):
            df_ctrl = pd.read_csv(ctrl_file)
            df_ctrl["Home"] = os.path.basename(home)
            all_ctrl.append(df_ctrl)

        if os.path.exists(base_file):
            df_base = pd.read_csv(base_file)
            df_base["Home"] = os.path.basename(home)
            all_base.append(df_base)

    if all_ctrl:
        df_ctrl_all = pd.concat(all_ctrl, ignore_index=True)
        df_ctrl_all.to_csv(os.path.join(work_dir, filename + "_controlled.csv"), index=False)

    if all_base:
        df_base_all = pd.concat(all_base, ignore_index=True)
        df_base_all.to_csv(os.path.join(work_dir, filename + "_baseline.csv"), index=False)

    print("Aggregated CSVs written!")




# CTRL_COLS = ["Time", "Total Electric Power (kW)",
#              "Total Electric Energy (kWh)",
#              "Water Heating Electric Power (kW)",
#              "Water Heating COP (-)",
#              "Water Heating Deadband Upper Limit (C)",
#              "Water Heating Deadband Lower Limit (C)",
#              "Water Heating Heat Pump COP (-)",
#              "Water Heating Control Temperature (C)",
#              "Hot Water Outlet Temperature (C)",
#              "Temperature - Indoor (C)"]
# BASE_COLS = CTRL_COLS

aggregate_results(homes, WORKING_DIR)

