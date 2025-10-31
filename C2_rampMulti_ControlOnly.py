# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:52:01 2025

@author: danap
"""

"""
HPWH Simulation with Coordinated Shed Periods and Randomized E_ALU_time
Controlled simulation only (baseline removed)
"""

import os
import shutil
import datetime as dt
import pandas as pd
from ochre import Dwelling
from ochre.utils.schedule import ALL_SCHEDULE_NAMES
import concurrent.futures
import random

#########################################
# USER SETTINGS
#########################################

filename = '171231_30_15_BL'  # yr, month, date _ duration _ tres _ RampControl or Baseline

# Paths
DEFAULT_INPUT = r"C:\Users\danap\anaconda3\Lib\site-packages\ochre\defaults\Input Files"
DEFAULT_WEATHER = r"C:\Users\danap\anaconda3\Lib\site-packages\ochre\defaults\Weather\USA_OR_Portland.Intl.AP.726980_TMY3.epw"
WORKING_DIR = r"C:\Users\danap\OCHRE_Working"
INPUT_DIR = os.path.join(WORKING_DIR, "Input Files")
WEATHER_DIR = os.path.join(WORKING_DIR, "Weather")
WEATHER_FILE = os.path.join(WEATHER_DIR, "USA_OR_Portland.Intl.AP.726980_TMY3.epw")

# Simulation parameters
Start = dt.datetime(2018, 1, 1, 0, 0)
Duration = 30  # days
t_res = 15  # minutes

# HPWH control parameters (°F) # this is AOSMITH2HPWH
Tcontrol_SHEDF = 126
Tcontrol_deadbandF = 10
Tcontrol_LOADF = 130
Tcontrol_LOADdeadbandF = 2
TbaselineF = 130
TdeadbandF = 7
Tinit = 128

# # this is normal baseline setting
# Tcontrol_SHEDF = 130
# Tcontrol_deadbandF = 7
# Tcontrol_LOADF = 130
# Tcontrol_LOADdeadbandF = 7
# TbaselineF = 130
# TdeadbandF = 7
# Tinit = 128

# Base schedule template
my_schedule = {
    'M_LU_time': '03:00',
    'M_LU_duration': 3,
    'M_S_time': '06:00',
    'M_S_duration': 4,
    'E_ALU_time': '16:00',
    'E_ALU_duration': 1,
    'E_S_time': '17:00',
    'E_S_duration': 3
}

# Randomization bins
M_LU_weights = [14, 28, 34, 41, 46, 46, 41, 33, 30, 31, 35, 30]
M_LU_bins = pd.date_range("03:00", periods=len(M_LU_weights), freq="15min").strftime("%H:%M").tolist()

E_ALU_weights = [17, 21, 27, 37, 40, 46, 40, 42, 36, 32, 33, 38]
E_ALU_bins = pd.date_range("14:00", periods=len(E_ALU_weights), freq="15min").strftime("%H:%M").tolist()

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
# SIMULATION FUNCTION (CONTROL ONLY)
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

    # Controlled simulation only
    sim_dwelling = Dwelling(name="HPWH Controlled", **dwelling_args_local)
    hpwh_unit = sim_dwelling.get_equipment_by_end_use('Water Heating')
    for sim_time in sim_dwelling.sim_times:
        current_setpt = hpwh_unit.schedule.loc[sim_time, 'Water Heating Setpoint (C)']
        control_cmd = determine_hpwh_control(sim_time=sim_time, current_temp_c=current_setpt, sched_cfg=schedule_cfg)
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
    df_ctrl.to_csv(os.path.join(results_dir, 'hpwh_controlled.csv'), index=False)
    cleanup_results_dir(results_dir, keep_files=['hpwh_controlled.csv', 'hpwh_baseline.csv'])

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

    # Weighted pools
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

    # Assign schedules per home
    home_schedules = {}
    fmt = "%H:%M"
    for home in homes:
        sched = my_schedule.copy()

        # M_LU_time
        M_LU_start = M_LU_weighted_pool.pop() if M_LU_weighted_pool else random.choice(M_LU_bins)
        sched['M_LU_time'] = M_LU_start

 

        sched['M_S_time'] = (
            '06:30' if M_LU_start == '05:45' 
            else '6:15' if M_LU_start == '5:30' 
            else my_schedule['M_S_time']
        )


        t_start = pd.to_datetime(M_LU_start, format=fmt)
        t_end = pd.to_datetime(sched['M_S_time'], format=fmt)
        if t_end <= t_start:
            t_end += pd.Timedelta(days=1)
        sched['M_LU_duration'] = (t_end - t_start).total_seconds() / 3600

        # M_S duration
        initial_duration = 4
        n = MS_weighted_pool.pop() if MS_weighted_pool else random.choice(MS_offsets)
        sched['M_S_duration'] = initial_duration + n

        # E_ALU_time
        E_ALU_start = E_ALU_weighted_pool.pop() if E_ALU_weighted_pool else random.choice(E_ALU_bins)
        sched['E_ALU_time'] = E_ALU_start
        sched['E_ALU_duration'] = my_schedule['E_ALU_duration']

        # E_S duration
        n = ES_weighted_pool.pop() if ES_weighted_pool else random.choice(ES_offsets)
        sched['E_S_duration'] = 3 + n
        sched['E_S_time'] = my_schedule['E_S_time']

        home_schedules[home] = sched

    # Run parallel simulations
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(simulate_home, home, WEATHER_FILE, home_schedules[home])
            for home in homes
        ]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print("Simulation failed:", e)

    print("All simulations complete!")

    # Aggregate results (controlled only)
    all_ctrl = []
    for home in homes:
        results_dir = os.path.join(home, "Results")
        ctrl_file = os.path.join(results_dir, "hpwh_controlled.csv")
        if os.path.exists(ctrl_file):
            df_ctrl = pd.read_csv(ctrl_file)
            df_ctrl["Home"] = os.path.basename(home)
            all_ctrl.append(df_ctrl)

    if all_ctrl:
        df_ctrl_all = pd.concat(all_ctrl, ignore_index=True)
        df_ctrl_all.to_csv(os.path.join(WORKING_DIR, filename + "whAgg_contr.csv"), index=False)
        print("Aggregated controlled CSV written!")
