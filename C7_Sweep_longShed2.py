# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:06:45 2025

This script is used to determine Shed duration capabilities. 
The script runs for an extra simulation day to initialize water heater
state randomness. No control schedule is used for Day1.
For Day 2, the HPWHs are coordinated with a schedule. In the morning,
all WHs are sent a coordinated Load Up between 3AM and 6AM, and then
placed in Shed from 6AM to Midnight. 

This script sweeps through different Shed deadbands 5, 10, and 15F

The output of this script will be an aggregation of all deadbands at
the Shed set point in parquet format.

@author: danap
"""

import os
import shutil
import datetime as dt
import pandas as pd
from ochre import Dwelling
from ochre.utils.schedule import ALL_SCHEDULE_NAMES
import concurrent.futures
import random
import time
import datetime
import numpy as np


print(datetime.datetime.fromtimestamp(time.time(), datetime.timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')) 
start_time = time.time()

#########################################
# USER SETTINGS
#########################################

filename = '180110_1_3_ShedLong118' # date that's thrown away, num of simulation days, data res, ramp or no ramp control
# 04 / 07 


# Paths
DEFAULT_INPUT = r"C:\Users\danap\anaconda3\Lib\site-packages\ochre\defaults\Input Files"
DEFAULT_WEATHER = r"C:\Users\danap\anaconda3\Lib\site-packages\ochre\defaults\Weather\USA_OR_Portland.Intl.AP.726980_TMY3.epw"
WORKING_DIR = r"C:\Users\danap\OCHRE_Working"
INPUT_DIR = os.path.join(WORKING_DIR, "Input Files")
WEATHER_DIR = os.path.join(WORKING_DIR, "Weather")
WEATHER_FILE = os.path.join(WEATHER_DIR, "USA_OR_Portland.Intl.AP.726980_TMY3.epw")

# Simulation parameters
Start = dt.datetime(2018, 1, 10, 0, 0)
Duration = 2  # days
t_res = 3  # minutes
jitter_min = 5

# HPWH control parameters (°F)
Tcontrol_SHEDF = 118
step = 7
Tcontrol_dbF = np.arange(7, 7 + step, step)  # Deadband sweep list (°F)
# Tcontrol_deadbandF = 10
Tcontrol_LOADF = 126
Tcontrol_LOADdeadbandF = 2
TbaselineF = Tcontrol_LOADF
TdeadbandF = 7
Tinit = 128

print(f'Tset = {Tcontrol_LOADF}')

# Base schedule template
my_schedule = {
    'M_LU_time': '03:00',
    'M_LU_duration': 3,
    'M_S_time': '06:00',
    'M_S_duration': 18,
}

# Randomization bins
M_LU_weights = [14, 28, 34, 41, 46, 46, 41, 33, 30, 31, 35, 30]
M_LU_bins = pd.date_range("03:00", periods=len(M_LU_weights), freq="15min").strftime("%H:%M").tolist()


#########################################
# TEMPERATURE CONVERSIONS F to C
#########################################

def f_to_c(temp_f): 
    return (temp_f - 32) * 5/9

def f_to_c_DB(temp_f):
    return 5/9 * temp_f

Tcontrol_SHEDC = f_to_c(Tcontrol_SHEDF)
Tcontrol_deadbandC_list = Tcontrol_dbF * 5/9  
Tcontrol_LOADC = f_to_c(Tcontrol_LOADF)
Tcontrol_LOADdeadbandC = f_to_c_DB(Tcontrol_LOADdeadbandF)
TbaselineC = f_to_c(TbaselineF)
TdeadbandC = f_to_c_DB(TdeadbandF)
TinitC = f_to_c(Tinit)

#########################################
# HPWH CONTROL FUNCTION
#########################################

def determine_hpwh_control(sim_time, current_temp_c, sched_cfg, deadband_C, **kwargs):  
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
    }

    if ranges['M_LU'][0] <= sim_time < ranges['M_LU'][1]:
        ctrl_signal['Water Heating'].update({
            'Setpoint': Tcontrol_LOADC,
            'Deadband': Tcontrol_LOADdeadbandC
        })
    elif ranges['M_S'][0] <= sim_time < ranges['M_S'][1]:
        ctrl_signal['Water Heating'].update({
            'Setpoint': Tcontrol_SHEDC,
            'Deadband': deadband_C  # <<< use the current sweep value
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

    hpwh_cols = ['M_LU_time','M_LU_duration','M_S_time','M_S_duration']

    filtered_columns = [col for col in df_sched.columns if col in valid_schedule_names or col in hpwh_cols]

    dropped_columns = [col for col in df_sched.columns if col not in filtered_columns]
    if dropped_columns:
        print(f"Dropped invalid schedules for {home_path}: {dropped_columns}")

    df_sched_filtered = df_sched[filtered_columns]
    df_sched_filtered.to_csv(filtered_sched_file, index=False)
    return filtered_sched_file

#########################################
# SIMULATION FUNCTION
#########################################

def simulate_home(home_path, weather_file_path, schedule_cfg, deadband_C):  # <<< CHANGED
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

    sim_dwelling = Dwelling(name="HPWH Controlled", **dwelling_args_local)
    hpwh_unit = sim_dwelling.get_equipment_by_end_use('Water Heating')

    # for sim_time in sim_dwelling.sim_times:
    #     current_setpt = hpwh_unit.schedule.loc[sim_time, 'Water Heating Setpoint (C)']
    #     control_cmd = determine_hpwh_control(sim_time=sim_time,
    #                                          current_temp_c=current_setpt,
    #                                          sched_cfg=schedule_cfg,
    #                                          deadband_C=deadband_C)
    #     sim_dwelling.update(control_signal=control_cmd)
    
    
    for sim_time in sim_dwelling.sim_times:
    
        # --- NEW: Day 1 = no control -----------------------------------------
        if sim_time < Start + pd.Timedelta(days=1):
            # FORCE baseline control explicitly
            control_cmd = {
                'Water Heating': {
                    'Setpoint': TbaselineC,
                    'Deadband': TdeadbandC,
                    'Load Fraction': 1,
                }
            }
            sim_dwelling.update(control_signal=control_cmd)
            continue

        # ----------------------------------------------------------------------
    
        # Day 2 = controlled as before
        current_setpt = hpwh_unit.schedule.loc[sim_time, 'Water Heating Setpoint (C)']
    
        control_cmd = determine_hpwh_control(sim_time=sim_time,
                                             current_temp_c=current_setpt,
                                             sched_cfg=schedule_cfg,
                                             deadband_C=deadband_C)
    
        sim_dwelling.update(control_signal=control_cmd)

    df_ctrl, _, _ = sim_dwelling.finalize()

    df_ctrl = remove_first_day(df_ctrl, Start)

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

    df_ctrl = df_ctrl[[c for c in CTRL_COLS if c in df_ctrl.columns]]

    suffix = f"_DB{round(deadband_C * 9/5)}F"  # <<< ADDED: file suffix
    out_path = os.path.join(results_dir, f"hpwh_controlled{suffix}.parquet")
    df_ctrl.to_parquet(out_path, index=False)

    # <<< CHANGED >>> Do NOT delete other sweep results. Keep all _DB*.parquet files.
    # If you want to remove non-parquet junk, do that safely here (example below).
    for item in os.listdir(results_dir):
        path = os.path.join(results_dir, item)
        if os.path.isfile(path) and not item.endswith(".parquet"):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Could not delete {path}: {e}")

    return df_ctrl

#########################################
# FIND ALL HOMES
#########################################

def find_all_homes(base_dir):
    homes = []
    for item in os.listdir(base_dir):
        home_path = os.path.join(base_dir, item)
        if os.path.isdir(home_path):
            if os.path.isfile(os.path.join(home_path, 'in.XML')) and \
               os.path.isfile(os.path.join(home_path, 'schedules.csv')):
                homes.append(home_path)
    return homes

#########################################
# DELETE FIRST DAY ONLY
#########################################

def remove_first_day(df, start_date):
    if 'Time' not in df.columns:
        df = df.reset_index()
        if 'index' in df.columns:
            df.rename(columns={'index': 'Time'}, inplace=True)
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    first_day_end = start_date + pd.Timedelta(days=1)
    return df[df['Time'] >= first_day_end].copy()

#########################################
# CLEAN UP FILES
#########################################

def cleanup_results_dir(results_dir, keep_files=None):
    if keep_files is None:
        keep_files = []
    for item in os.listdir(results_dir):
        path = os.path.join(results_dir, item)
        if os.path.isfile(path) and item not in keep_files:
            try:
                os.remove(path)
            except Exception as e:
                print(f"Could not delete {path}: {e}")
        elif os.path.isdir(path):
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"Could not delete folder {path}: {e}")

#########################################
# MAIN EXECUTION
#########################################

if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(WEATHER_DIR, exist_ok=True)

    for item in os.listdir(DEFAULT_INPUT):
        src = os.path.join(DEFAULT_INPUT, item)
        dst = os.path.join(INPUT_DIR, item)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)

    if not os.path.exists(WEATHER_FILE):
        shutil.copy(DEFAULT_WEATHER, WEATHER_FILE)

    homes = find_all_homes(INPUT_DIR)
    print(f"Found {len(homes)} homes")

    # Weighted pools setup (same as before)
    M_LU_weighted_pool = [bin_time for bin_time, weight in zip(M_LU_bins, M_LU_weights) for _ in range(weight)]

    
    home_schedules = {}
    fmt = "%H:%M"
    
    for home in homes:
        sched = my_schedule.copy()
        
        # -----------------------------
        # Randomize morning usage start
        # -----------------------------
        if M_LU_weighted_pool:
            M_LU_base = M_LU_weighted_pool.pop()
        else:
            M_LU_base = random.choice(M_LU_bins)
        t_base = pd.to_datetime(M_LU_base, format=fmt)
        jitter = pd.Timedelta(minutes=random.uniform(-jitter_min, jitter_min))
        t_jittered = t_base + jitter
        sched['M_LU_time'] = t_jittered.strftime(fmt)
        
        # -----------------------------
        # Randomize midday start
        # -----------------------------
        t_MS_start = pd.to_datetime(my_schedule['M_S_time'], format=fmt)
        t_MS_start += pd.Timedelta(minutes=random.uniform(-jitter_min, jitter_min))
        sched['M_S_time'] = t_MS_start.strftime(fmt)
        
        # -----------------------------
        # Calculate M_LU_duration
        # -----------------------------
        t_MLU_start = pd.to_datetime(sched['M_LU_time'], format=fmt)
        min_end = pd.to_datetime("06:00", format=fmt)  # minimum end at 6 AM
        t_MLU_end = max(min_end, t_MS_start)           # ensure ends at least 6 AM
        
        # Handle crossing midnight
        if t_MLU_end <= t_MLU_start:
            t_MLU_end += pd.Timedelta(days=1)
        
        # Duration in hours, minimum 1 hour
        sched['M_LU_duration'] = max(1, (t_MLU_end - t_MLU_start).total_seconds() / 3600)
        
        # -----------------------------
        # Save schedule
        # -----------------------------
        home_schedules[home] = sched


    #########################################
    # SWEEP DEADBANDBAND VALUES
    #########################################

    for deadband_C in Tcontrol_deadbandC_list:
        print(f"\n=== Running simulations for deadband {deadband_C:.2f} °C ({round(deadband_C*9/5)} °F) ===")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(simulate_home, home, WEATHER_FILE, home_schedules[home], deadband_C)
                for home in homes
            ]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print("Simulation failed:", e)

    #########################################
    # AGGREGATE RESULTS
    #########################################

    def aggregate_results(homes, work_dir):
        for deadband_C in Tcontrol_deadbandC_list:
            suffix = f"_DB{round(deadband_C * 9/5)}F"
            all_ctrl = []
            for home in homes:
                results_dir = os.path.join(home, "Results")
                ctrl_file = os.path.join(results_dir, f"hpwh_controlled{suffix}.parquet")
                if os.path.exists(ctrl_file):
                    df_ctrl = pd.read_parquet(ctrl_file)
                    df_ctrl["Home"] = os.path.basename(home)
                    df_ctrl["Deadband_C"] = deadband_C
                    all_ctrl.append(df_ctrl)
            if all_ctrl:
                df_ctrl_all = pd.concat(all_ctrl, ignore_index=True)
                outp = os.path.join(work_dir, filename + f"{suffix}_Control.parquet")
                df_ctrl_all.to_parquet(outp, index=False)
                print(f"Aggregated results written for {suffix}")

    aggregate_results(homes, WORKING_DIR)



    def aggregate_current_setpoint(work_dir, prefix):
        """
        Aggregate all per-deadband HPWH control files produced by aggregate_results
        for the same simulation prefix (date, duration, setpoint).
    
        Example:
            prefix = '180110_1_3_Shed110'
            Input files: 180110_1_3_Shed110_DB5F_Control.parquet,
                         180110_1_3_Shed110_DB10F_Control.parquet, ...
            Output file: 180110_1_3_Shed110_Control.parquet
        """
        # Find all _DB*.parquet files for this prefix
        all_files = [
            f for f in os.listdir(work_dir)
            if f.endswith("_Control.parquet") and f.startswith(prefix + "_DB")
        ]
    
        if not all_files:
            print(f"No deadband files found for prefix {prefix}")
            return
    
        all_dfs = []
        for f in all_files:
            path = os.path.join(work_dir, f)
            df = pd.read_parquet(path)
            df["SourceFile"] = f  # track which deadband this came from
            all_dfs.append(df)
    
        df_master = pd.concat(all_dfs, ignore_index=True)
        master_file = os.path.join(work_dir, f"{prefix}_Control.parquet")
        df_master.to_parquet(master_file, index=False)
        print(f"Aggregated {len(all_files)} deadband files for {prefix} → {master_file}")


    aggregate_current_setpoint(WORKING_DIR, filename)


end_time = time.time()
execution_time = end_time - start_time
execution_min = execution_time/60
print(f"Execution time: {execution_min} minutes")
