# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:06:45 2025

Updated by ChatGPT: Full HPWH simulation with LEVELS efficiency logic
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
import numpy as np

print(dt.datetime.fromtimestamp(time.time(), dt.timezone.utc)
      .astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))


start_time = time.time()

#########################################
# USER SETTINGS
#########################################

filename = '180407_1_15_xx'  # simulation output filename

# Paths
DEFAULT_INPUT = r"C:\Users\danap\anaconda3\Lib\site-packages\ochre\defaults\Input Files"
DEFAULT_WEATHER = r"C:\Users\danap\anaconda3\Lib\site-packages\ochre\defaults\Weather\USA_OR_Portland.Intl.AP.726980_TMY3.epw"
WORKING_DIR = r"C:\Users\danap\OCHRE_Working"
INPUT_DIR = os.path.join(WORKING_DIR, "Input Files")
WEATHER_DIR = os.path.join(WORKING_DIR, "Weather")
WEATHER_FILE = os.path.join(WEATHER_DIR, "USA_OR_Portland.Intl.AP.726980_TMY3.epw")

# Simulation parameters
Start = dt.datetime(2018, 4, 7, 0, 0)
Duration = 2  # days
t_res = 3  # minutes
jitter_min = 5

# HPWH control parameters (°F)
Tcontrol_SHEDF = 126
Tcontrol_dbF = 10
Tcontrol_deadbandF = 10
Tcontrol_LOADF = 130
Tcontrol_LOADdeadbandF = 2
TbaselineF = 130
TdeadbandF = 7
Tinit = 128

# Base schedule template
my_schedule = {
    'M_LU_time': '03:00', 'M_LU_duration': 3,
    'M_S_time': '06:00', 'M_S_duration': 4,
    'E_ALU_time': '16:00', 'E_ALU_duration': 1,
    'E_S_time': '17:00', 'E_S_duration': 3
}

# Randomization bins
M_LU_weights = [14, 28, 34, 41, 46, 46, 41, 33, 30, 31, 35, 30]
M_LU_bins = pd.date_range("03:00", periods=len(M_LU_weights), freq="15min").strftime("%H:%M").tolist()

E_ALU_weights = [17, 21, 27, 37, 40, 46, 40, 42, 36, 32, 33, 38]
E_ALU_bins = pd.date_range("14:00", periods=len(E_ALU_weights), freq="15min").strftime("%H:%M").tolist()

MS_bins = pd.date_range("10:00", "13:45", freq="15min")
MS_weights = [20, 23, 24, 23, 22, 22, 25, 26, 26, 29, 29, 29, 29, 27, 28, 27]
MS_offsets = [(t - pd.Timestamp("10:00")).total_seconds()/3600 for t in MS_bins]

ES_bins = pd.date_range("20:00", "23:45", freq="15min")
ES_weights = [17, 21, 24, 25, 26, 24, 24, 23, 23, 23, 23, 25, 28, 30, 33, 40]
ES_offsets = [(t - pd.Timestamp("20:00")).total_seconds()/3600 for t in ES_bins]

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
# LEVELS definition
#########################################
LEVELS = {
    1: {"NORMAL": {"ER": [0,130], "HP": [0,130]},
        "SHED": {"ER": [0,126], "HP": [0,126]},
        "LOAD": {"ER": [0,130], "HP": [0,130]}},
    2: {"NORMAL": {"ER": [0,130], "HP": [0,60]},
        "SHED": {"ER": [0,126], "HP": [0,60]},
        "LOAD": {"ER": [0,130], "HP": [0,60]}},
    3: {"NORMAL": {"ER": [0,128], "HP": [128,130]},
        "SHED": {"ER": [0,124], "HP": [124,126]},
        "LOAD": {"ER": [0,61], "HP": [62,130]}},
    4: {"NORMAL": {"ER": [0,126], "HP": [126,130]},
        "SHED": {"ER": [0,122], "HP": [122,126]},
        "LOAD": {"ER": [0,61], "HP": [62,130]}},
    5: {"NORMAL": {"ER": [0,124], "HP": [124,130]},
        "SHED": {"ER": [0,120], "HP": [120,126]},
        "LOAD": {"ER": [0,61], "HP": [62,130]}},
    6: {"NORMAL": {"ER": [0,122], "HP": [122,130]},
        "SHED": {"ER": [0,118], "HP": [118,126]},
        "LOAD": {"ER": [0,61], "HP": [62,130]}},
    7: {"NORMAL": {"ER": [0,120], "HP": [120,130]},
        "SHED": {"ER": [0,116], "HP": [116,126]},
        "LOAD": {"ER": [0,61], "HP": [62,130]}},
    8: {"NORMAL": {"ER": [0,118], "HP": [118,130]},
        "SHED": {"ER": [0,114], "HP": [114,126]},
        "LOAD": {"ER": [0,61], "HP": [62,130]}},
    9: {"NORMAL": {"ER": [0,61], "HP": [62,130]},
        "SHED": {"ER": [0,61], "HP": [62,126]},
        "LOAD": {"ER": [0,61], "HP": [62,130]}},
}

# Convert all LEVELS temperatures to Celsius for simulation
for level in LEVELS:
    for mode in LEVELS[level]:
        for heater in LEVELS[level][mode]:
            LEVELS[level][mode][heater] = [f_to_c(x) for x in LEVELS[level][mode][heater]]




#########################################
# HPWH CONTROL FUNCTION (with LEVELS)
#########################################

def determine_hpwh_control(sim_time, current_temp_c, sched_cfg, level):
    """
    Determine HPWH control signal with LEVELS efficiency and correct deadband logic.
    """
    ctrl_signal = {'Water Heating': {'Setpoint': TbaselineC, 'Deadband': TdeadbandC, 'Load Fraction': 1}}
    base_date = sim_time.date()

    # Helper to get start/end times of schedules
    def get_time_range(key_prefix):
        start = pd.to_datetime(f"{base_date} {sched_cfg[f'{key_prefix}_time']}")
        end = start + pd.Timedelta(hours=sched_cfg[f'{key_prefix}_duration'])
        return start, end

    ranges = {k: get_time_range(k) for k in ['M_LU', 'M_S', 'E_ALU', 'E_S']}

    # Determine current mode
    if ranges['M_LU'][0] <= sim_time < ranges['M_LU'][1] or ranges['E_ALU'][0] <= sim_time < ranges['E_ALU'][1]:
        mode = 'LOAD'
        setpoint = Tcontrol_LOADC
        deadband = Tcontrol_LOADdeadbandC
    elif ranges['M_S'][0] <= sim_time < ranges['M_S'][1] or ranges['E_S'][0] <= sim_time < ranges['E_S'][1]:
        mode = 'SHED'
        setpoint = Tcontrol_SHEDC
        deadband = Tcontrol_deadbandC
    else:
        mode = 'NORMAL'
        setpoint = TbaselineC
        deadband = TdeadbandC

    # Apply LEVELS efficiency
    level_dict = LEVELS[level][mode]
    ER_range = level_dict['ER']
    HP_range = level_dict['HP']

    lower_limit = setpoint - deadband
    upper_limit = setpoint

    # Determine ON/OFF status respecting both deadband and LEVELS ranges
    ER_on = (current_temp_c < upper_limit) and (current_temp_c < ER_range[1]) and (current_temp_c >= ER_range[0])
    HP_on = (current_temp_c < upper_limit) and (current_temp_c < HP_range[1]) and (current_temp_c >= HP_range[0])

    # Set control temperature as upper limit if any heater is on
    if ER_on or HP_on:
        control_temp = upper_limit
    else:
        control_temp = current_temp_c  # no heating

    ctrl_signal['Water Heating'].update({
        'Setpoint': control_temp,
        'Deadband': deadband,
        'ER ON': ER_on,
        'HP ON': HP_on
    })

    return ctrl_signal







#########################################
# SCHEDULE FILTERING, HOME DISCOVERY, CLEANUP
#########################################
def filter_schedules(home_path):
    orig_sched_file = os.path.join(home_path, 'schedules.csv')
    filtered_sched_file = os.path.join(home_path, 'filtered_schedules.csv')
    df_sched = pd.read_csv(orig_sched_file)
    valid_schedule_names = set(ALL_SCHEDULE_NAMES.keys())
    hpwh_cols = ['M_LU_time','M_LU_duration','M_S_time','M_S_duration',
                 'E_ALU_time','E_ALU_duration','E_S_time','E_S_duration']
    filtered_columns = [col for col in df_sched.columns if col in valid_schedule_names or col in hpwh_cols]
    df_sched_filtered = df_sched[filtered_columns]
    df_sched_filtered.to_csv(filtered_sched_file, index=False)
    return filtered_sched_file

def find_all_homes(base_dir):
    homes = []
    for item in os.listdir(base_dir):
        home_path = os.path.join(base_dir, item)
        if os.path.isdir(home_path):
            if os.path.isfile(os.path.join(home_path, 'in.XML')) and \
               os.path.isfile(os.path.join(home_path, 'schedules.csv')):
                homes.append(home_path)
    return homes

def remove_first_day(df, start_date):
    if 'Time' not in df.columns:
        df = df.reset_index()
        if 'index' in df.columns:
            df.rename(columns={'index':'Time'}, inplace=True)
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    return df[df['Time'] >= start_date + pd.Timedelta(days=1)].copy()

def cleanup_results_dir(results_dir, keep_files=None):
    if keep_files is None:
        keep_files = []
    for item in os.listdir(results_dir):
        path = os.path.join(results_dir, item)
        if os.path.isfile(path) and item not in keep_files:
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

#########################################
# SIMULATION FUNCTION
#########################################
def simulate_home(home_path, weather_file_path, schedule_cfg, level):
    """
    Run a simulation for a single home at a given efficiency level.
    """
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

    sim_dwelling = Dwelling(name=f"HPWH_Level{level}", **dwelling_args_local)
    hpwh_unit = sim_dwelling.get_equipment_by_end_use('Water Heating')

    for sim_time in sim_dwelling.sim_times:
        current_temp_c = hpwh_unit.schedule.loc[sim_time, 'Water Heating Setpoint (C)']
        control_cmd = determine_hpwh_control(
            sim_time=sim_time,
            current_temp_c=current_temp_c,
            sched_cfg=schedule_cfg,
            level=level  # pass level into control function
        )
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
    df_ctrl.to_parquet(os.path.join(results_dir, 'hpwh_controlled.parquet'), index=False)

    cleanup_results_dir(results_dir, keep_files=['hpwh_controlled.parquet'])
    return df_ctrl




#########################################
# MAIN EXECUTION
#########################################

if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(WEATHER_DIR, exist_ok=True)
    
    # Copy homes and weather file
    for item in os.listdir(DEFAULT_INPUT):
        src = os.path.join(DEFAULT_INPUT, item)
        dst = os.path.join(INPUT_DIR, item)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
    if not os.path.exists(WEATHER_FILE):
        shutil.copy(DEFAULT_WEATHER, WEATHER_FILE)
    
    # Discover homes
    homes = find_all_homes(INPUT_DIR)
    print(f"Found {len(homes)} homes")
    
    # Precompute weighted schedule pools (original logic)
    M_LU_weighted_pool = [bin_time for bin_time, weight in zip(M_LU_bins, M_LU_weights) for _ in range(weight)]
    random.shuffle(M_LU_weighted_pool)
    
    MS_bins = pd.date_range("10:00", "13:45", freq="15min")
    MS_weights = [20, 23, 24, 23, 22, 22, 25, 26, 26, 29, 29, 29, 29, 27, 28, 27]
    MS_offsets = [(t - pd.Timestamp("10:00")).total_seconds()/3600 for t in MS_bins]
    MS_weighted_pool = [offset for offset, w in zip(MS_offsets, MS_weights) for _ in range(w)]
    random.shuffle(MS_weighted_pool)
    
    E_ALU_weighted_pool = [bin_time for bin_time, weight in zip(E_ALU_bins, E_ALU_weights) for _ in range(weight)]
    random.shuffle(E_ALU_weighted_pool)
    
    ES_bins = pd.date_range("20:00", "23:45", freq="15min")
    ES_weights = [17, 21, 24, 25, 26, 24, 24, 23, 23, 23, 23, 25, 28, 30, 33, 40]
    ES_offsets = [(t - pd.Timestamp("20:00")).total_seconds()/3600 for t in ES_bins]
    ES_weighted_pool = [offset2 for offset2, m in zip(ES_offsets, ES_weights) for _ in range(m)]
    random.shuffle(ES_weighted_pool)
    
    # Loop through all levels
    for level in range(1, 10):
        print(f"Running simulations for LEVEL {level}...")
        
        # Assign schedules per home
        home_schedules = {}
        fmt = "%H:%M"
        for home in homes:
            sched = my_schedule.copy()
            
            # Morning Load Use (M_LU)
            if M_LU_weighted_pool:
                M_LU_base = M_LU_weighted_pool.pop()
            else:
                M_LU_base = random.choice(M_LU_bins)
            t_base = pd.to_datetime(M_LU_base, format=fmt)
            jitter = dt.timedelta(minutes=random.uniform(-jitter_min, jitter_min))
            t_jittered = t_base + jitter
            sched['M_LU_time'] = t_jittered.strftime(fmt)
            
            # M_S start time and duration
            t_MS_start = pd.to_datetime(my_schedule['M_S_time'], format=fmt)
            t_MS_start += dt.timedelta(minutes=random.uniform(-jitter_min, jitter_min))
            sched['M_S_time'] = t_MS_start.strftime(fmt)
            t_MLU_start = pd.to_datetime(sched['M_LU_time'], format=fmt)
            t_MLU_end = t_MS_start
            if t_MLU_end <= t_MLU_start:
                t_MLU_end += dt.timedelta(days=1)
            sched['M_LU_duration'] = max(1, (t_MLU_end - t_MLU_start).total_seconds()/3600)
            if MS_weighted_pool:
                n = MS_weighted_pool.pop()
            else:
                n = random.choice(MS_offsets)
            sched['M_S_duration'] = 4 + n
            
            # Evening schedules
            if E_ALU_weighted_pool:
                E_ALU_base = E_ALU_weighted_pool.pop()
            else:
                E_ALU_base = random.choice(E_ALU_bins)
            t_E_ALU_start = pd.to_datetime(E_ALU_base, format=fmt)
            t_E_ALU_start += dt.timedelta(minutes=random.uniform(-jitter_min, jitter_min))
            sched['E_ALU_time'] = t_E_ALU_start.strftime(fmt)
            
            t_ES_start = pd.to_datetime(my_schedule['E_S_time'], format=fmt)
            t_ES_start += dt.timedelta(minutes=random.uniform(-jitter_min, jitter_min))
            sched['E_S_time'] = t_ES_start.strftime(fmt)
            if t_ES_start <= t_E_ALU_start:
                t_ES_start += dt.timedelta(days=1)
            sched['E_ALU_duration'] = max(1, (t_ES_start - t_E_ALU_start).total_seconds()/3600)
            
            if ES_weighted_pool:
                n = ES_weighted_pool.pop()
            else:
                n = random.choice(ES_offsets)
            sched['E_S_duration'] = 3 + n
            
            home_schedules[home] = sched
        
        # Parallel simulation per home
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(simulate_home, home, WEATHER_FILE, home_schedules[home], level)
                for home in homes
            ]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print("Simulation failed:", e)
        
        # Aggregate results per level
        def aggregate_results(homes, work_dir, level):
            all_ctrl = []
            for home in homes:
                results_dir = os.path.join(home, "Results")
                ctrl_file = os.path.join(results_dir, "hpwh_controlled.parquet")
                if os.path.exists(ctrl_file):
                    df_ctrl = pd.read_parquet(ctrl_file)
                    df_ctrl["Home"] = os.path.basename(home)
                    df_ctrl["Level"] = level
                    all_ctrl.append(df_ctrl)
            if all_ctrl:
                df_ctrl_all = pd.concat(all_ctrl, ignore_index=True)
                df_ctrl_all.to_parquet(os.path.join(work_dir, f"{filename}_Level{level}_Control.parquet"), index=False)
            print(f"Aggregated results written for LEVEL {level}!")
        
        aggregate_results(homes, WORKING_DIR, level)
    
    print("All simulations complete!")




end_time = time.time()
print(f"Execution time: {(end_time-start_time)/60:.2f} minutes")
