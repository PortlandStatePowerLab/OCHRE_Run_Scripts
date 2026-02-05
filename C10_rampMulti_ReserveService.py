# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:06:45 2025

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
import re

print(datetime.datetime.fromtimestamp(time.time(), datetime.timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))

start_time = time.time()

#########################################
# USER SETTINGS
#########################################

filename = '180113_1_3_Reserve20'

# Paths
DEFAULT_INPUT = r"C:\Users\danap\anaconda3\Lib\site-packages\ochre\defaults\Input Files"
DEFAULT_WEATHER = r"C:\Users\danap\anaconda3\Lib\site-packages\ochre\defaults\Weather\USA_OR_Portland.Intl.AP.726980_TMY3.epw"
WORKING_DIR = r"C:\Users\danap\OCHRE_Working"
INPUT_DIR = os.path.join(WORKING_DIR, "Input Files")
WEATHER_DIR = os.path.join(WORKING_DIR, "Weather")
WEATHER_FILE = os.path.join(WEATHER_DIR, "USA_OR_Portland.Intl.AP.726980_TMY3.epw")

# Simulation parameters
Start = dt.datetime(2018, 1, 13, 0, 0)
Duration = 2  # days
t_res = 3  # minutes
jitter_min = 5

# HPWH control parameters (°F)
Tcontrol_SHEDF = 145 # 145 this is the Reserve temperature
step = 2 # 2F
Tcontrol_dbF = np.arange(2, 2 + step, step) #2F
Tcontrol_LOADF = 123
Tcontrol_LOADdeadbandF = 10
TbaselineF = 130
TdeadbandF = 7
Tinit = 128

# Base schedule template
my_schedule = {
    'M_LU_time': '10:00',
    'M_LU_duration': 2,
    'M_S_time': '14:00',
    'M_S_duration': 4,
}

# Randomization bins
M_LU_weights = [10, 13, 14, 16, 16, 13]  # 82 participating homes 20%
M_LU_bins = pd.date_range("10:00", periods=len(M_LU_weights), freq="30min").strftime("%H:%M").tolist()

#########################################
# TEMPERATURE CONVERSIONS F to C
#########################################

def f_to_c(temp_f): 
    return (temp_f - 32) * 5/9

def f_to_c_DB(temp_f):
    return 5/9 * temp_f

Tcontrol_SHEDC = f_to_c(Tcontrol_SHEDF)
Tcontrol_LOADC = f_to_c(Tcontrol_LOADF)
Tcontrol_LOADdeadbandC = f_to_c_DB(Tcontrol_LOADdeadbandF)
TbaselineC = f_to_c(TbaselineF)
TdeadbandC = f_to_c_DB(TdeadbandF)
TinitC = f_to_c(Tinit)

#########################################
# HPWH CONTROL FUNCTION
#########################################

def determine_hpwh_control(sim_time, current_temp_c, sched_cfg, shed_deadbandC, **kwargs):
    ctrl_signal = {
        'Water Heating': {
            'Setpoint': TbaselineC,
            'Deadband': TdeadbandC,
            'Load Fraction': 1,
        }
    }

    base_date = sim_time.date()
    
    # Load-up
    if sched_cfg.get('M_LU_time') is not None:
        start_LU = pd.to_datetime(f"{base_date} {sched_cfg['M_LU_time']}")
        end_LU = start_LU + pd.Timedelta(hours=sched_cfg['M_LU_duration'])
        if start_LU <= sim_time < end_LU:
            ctrl_signal['Water Heating'].update({
                'Setpoint': Tcontrol_LOADC,
                'Deadband': Tcontrol_LOADdeadbandC
            })

    # Shed
    if sched_cfg.get('M_S_time') is not None:
        start_S = pd.to_datetime(f"{base_date} {sched_cfg['M_S_time']}")
        end_S = start_S + pd.Timedelta(hours=sched_cfg['M_S_duration'])
        if start_S <= sim_time < end_S:
            ctrl_signal['Water Heating'].update({
                'Setpoint': Tcontrol_SHEDC,
                'Deadband': shed_deadbandC
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

def simulate_home(home_path, weather_file_path, schedule_cfg, shed_deadbandF):
    shed_deadbandC = f_to_c_DB(shed_deadbandF)

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

    # Controlled
    sim_dwelling = Dwelling(name="HPWH Controlled", **dwelling_args_local)
    hpwh_unit = sim_dwelling.get_equipment_by_end_use('Water Heating')
    
    for sim_time in sim_dwelling.sim_times:
        # Day 1: no control
        if sim_time < Start + pd.Timedelta(days=1):
            control_cmd = {
                'Water Heating': {
                    'Setpoint': TbaselineC,
                    'Deadband': TdeadbandC,
                    'Load Fraction': 1,
                }
            }
            sim_dwelling.update(control_signal=control_cmd)
            continue

        # Day 2: controlled
        current_setpt = hpwh_unit.schedule.loc[sim_time, 'Water Heating Setpoint (C)']
        control_cmd = determine_hpwh_control(sim_time=sim_time,
                                             current_temp_c=current_setpt,
                                             sched_cfg=schedule_cfg,
                                             shed_deadbandC=shed_deadbandC)
        sim_dwelling.update(control_signal=control_cmd)
    
    df_ctrl, _, _ = sim_dwelling.finalize()
    df_ctrl = remove_first_day(df_ctrl, Start)
    df_ctrl["Shed Deadband (F)"] = shed_deadbandF

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
    df_ctrl.to_parquet(os.path.join(results_dir, f'hpwh_controlled.parquet'), index=False)

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
# CROSS-DEADBAND AGGREGATION
#########################################

def aggregate_across_deadbands(work_dir, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}_Control_DB(\d+)\.parquet$")
    matches = []
    for fname in os.listdir(work_dir):
        m = pattern.match(fname)
        if m:
            matches.append((fname, int(m.group(1))))
    if not matches:
        print(f"⚠️ No deadband files found for {prefix}")
        return
    dfs = []
    for fname, dbF in sorted(matches, key=lambda x: x[1]):
        path = os.path.join(work_dir, fname)
        df = pd.read_parquet(path)
        df["Shed Deadband (F)"] = dbF
        df["SourceFile"] = fname
        dfs.append(df)
    df_master = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(work_dir, f"{prefix}_Control.parquet")
    df_master.to_parquet(out_path, index=False)
    print(f"\n✅ Cross-deadband aggregation complete\n   Deadbands: {[db for _, db in matches]}\n   Rows: {len(df_master):,}\n   Output: {out_path}")

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

    # -----------------------------
    # Assign schedules to homes
    # -----------------------------
    home_schedules = {}
    fmt = "%H:%M"
    NUM_PARTICIPATING = 76
    MIN_SHED_HOURS = 1
    SLOW_DROP_HOURS = 2

    # Weighted pool for load-up
    M_LU_weighted_pool = [bin_time for bin_time, weight in zip(M_LU_bins, M_LU_weights) for _ in range(weight)]
    random.shuffle(M_LU_weighted_pool)

    # Generate stagger offsets for slow drop-off
    stagger_offsets = np.linspace(0, SLOW_DROP_HOURS, NUM_PARTICIPATING)
    stagger_offsets = [o + random.uniform(-10/60, 10/60) for o in stagger_offsets]  # ±10 min jitter
    random.shuffle(stagger_offsets)

    for idx, home in enumerate(homes):
        sched = my_schedule.copy()

        # -----------------------------
        # Participating homes: first 76
        # -----------------------------
        if idx < NUM_PARTICIPATING:
            # Load-up
            M_LU_base = M_LU_weighted_pool.pop()
            t_base = pd.to_datetime(M_LU_base, format=fmt)
            jitter = pd.Timedelta(minutes=random.uniform(-jitter_min, jitter_min))
            sched['M_LU_time'] = (t_base + jitter).strftime(fmt)
            sched['M_LU_duration'] = max(1.5, random.uniform(1.5, 3.0))

            # Shed: 2h minimum + staggered slow drop-off
            t_MS_start = pd.Timestamp("14:00") + pd.Timedelta(hours=stagger_offsets[idx])
            t_MS_start += pd.Timedelta(minutes=random.uniform(-10, 10))
            sched['M_S_time'] = t_MS_start.strftime(fmt)
            sched['M_S_duration'] = MIN_SHED_HOURS + stagger_offsets[idx]  # gradual drop-off
        else:
            # Non-participating: no load-up or shed
            sched['M_LU_time'] = None
            sched['M_LU_duration'] = 0
            sched['M_S_time'] = None
            sched['M_S_duration'] = 0

        home_schedules[home] = sched

    # -----------------------------
    # Sweep deadbands
    # -----------------------------
    for shed_dbF in Tcontrol_dbF:
        print(f"\nRunning shed deadband = {shed_dbF} F")
        all_ctrl = []

        def simulate_home_safe(home_path, weather_file, sched_cfg, shed_dbF):
            try:
                return simulate_home(home_path, weather_file, sched_cfg, shed_dbF)
            except Exception as e:
                print(f"⚠️ Simulation failed for {home_path} (DB={shed_dbF}): {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(simulate_home_safe, home, WEATHER_FILE, home_schedules[home], shed_dbF)
                       for home in homes]
            for f in concurrent.futures.as_completed(futures):
                df_result = f.result()
                if df_result is not None:
                    all_ctrl.append(df_result)

        # Aggregate
        if all_ctrl:
            df_all = pd.concat(all_ctrl, ignore_index=True)
            df_all["Home"] = df_all.get("Home", "Unknown")
            df_all["Shed Deadband (F)"] = shed_dbF
            out_file = os.path.join(WORKING_DIR, f"{filename}_Control_DB{int(shed_dbF)}.parquet")
            df_all.to_parquet(out_file, index=False)
            print(f"Aggregated DB{shed_dbF}: {len(df_all):,} rows, {df_all['Home'].nunique()} homes")
        else:
            print(f"⚠️ No successful homes to aggregate for DB{shed_dbF}")

    # Cross-deadband aggregation
    aggregate_across_deadbands(WORKING_DIR, filename)

    end_time = time.time()
    print(f"Execution time: {(end_time - start_time)/60:.2f} minutes")
