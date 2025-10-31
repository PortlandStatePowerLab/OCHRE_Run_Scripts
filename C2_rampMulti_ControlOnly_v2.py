# -*- coding: utf-8 -*-
"""
Controlled-only HPWH Simulation with Coordinated Shed Periods, Randomized Load-up Times, and Per-Bin Jitter
"""

import os
import shutil
import datetime as dt
import pandas as pd
from ochre import Dwelling
from ochre.utils.schedule import ALL_SCHEDULE_NAMES
import concurrent.futures
import random
import threading
import time

start_time = time.time()
print(start_time)

#########################################
# USER SETTINGS
#########################################

filename = '180109_2_3_RCJ_control_only2'

# Paths
DEFAULT_INPUT = r"C:\Users\danap\anaconda3\Lib\site-packages\ochre\defaults\Input Files"
DEFAULT_WEATHER = r"C:\Users\danap\anaconda3\Lib\site-packages\ochre\defaults\Weather\USA_OR_Portland.Intl.AP.726980_TMY3.epw"
WORKING_DIR = r"C:\Users\danap\OCHRE_Working"
INPUT_DIR = os.path.join(WORKING_DIR, "Input Files")
WEATHER_DIR = os.path.join(WORKING_DIR, "Weather")
WEATHER_FILE = os.path.join(WEATHER_DIR, "USA_OR_Portland.Intl.AP.726980_TMY3.epw")

# Simulation parameters
Start = dt.datetime(2018, 1, 9, 0, 0)
Duration = 3  # days
t_res = 3  # minutes
TIME_JITTER_MIN = 5

# HPWH control parameters (°F)
Tcontrol_SHEDF = 126
Tcontrol_deadbandF = 10
Tcontrol_LOADF = 130
Tcontrol_LOADdeadbandF = 2
TbaselineF = 130
TdeadbandF = 7
Tinit = 128

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

# Randomization bins + weights
M_LU_weights = [14, 28, 34, 41, 46, 46, 41, 33, 30, 31, 35, 30]
M_LU_bins = pd.date_range("03:00", periods=len(M_LU_weights), freq="15min").strftime("%H:%M").tolist()

E_ALU_weights = [17, 21, 27, 37, 40, 46, 40, 42, 36, 32, 33, 38]
E_ALU_bins = pd.date_range("14:00", periods=len(E_ALU_weights), freq="15min").strftime("%H:%M").tolist()

_daily_pool_lock = threading.Lock()
_daily_schedule_cache = {}
_daily_time_pools = {}

#########################################
# TEMPERATURE CONVERSIONS
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
# HELPER FUNCTIONS
#########################################

def build_weighted_pool_from_bins(bins, weights):
    pool = [b for b, w in zip(bins, weights) for _ in range(max(1, int(w)))]
    random.shuffle(pool)
    return pool

def add_time_jitter(base_time_str, jitter_minutes=TIME_JITTER_MIN):
    base_time = pd.to_datetime(base_time_str, format="%H:%M")
    jitter = random.randint(-jitter_minutes, jitter_minutes)
    return (base_time + pd.Timedelta(minutes=jitter)).strftime("%H:%M")

def _ensure_daily_time_pool_for_date(base_date):
    if base_date not in _daily_time_pools:
        _daily_time_pools[base_date] = {
            'M_LU': build_weighted_pool_from_bins(M_LU_bins, M_LU_weights),
            'E_ALU': build_weighted_pool_from_bins(E_ALU_bins, E_ALU_weights)
        }

def _draw_time_for_date(base_date, key):
    _ensure_daily_time_pool_for_date(base_date)
    with _daily_pool_lock:
        pool = _daily_time_pools[base_date].get(key, [])
        if not pool:
            if key == 'M_LU':
                pool[:] = build_weighted_pool_from_bins(M_LU_bins, M_LU_weights)
            elif key == 'E_ALU':
                pool[:] = build_weighted_pool_from_bins(E_ALU_bins, E_ALU_weights)
            random.shuffle(pool)
        base_time = pool.pop()
    return add_time_jitter(base_time)

#########################################
# HPWH CONTROL FUNCTION
#########################################

def determine_hpwh_control(sim_time, current_temp_c, sched_cfg, home_id=None):
    ctrl_signal = {
        'Water Heating': {'Setpoint': TbaselineC, 'Deadband': TdeadbandC, 'Load Fraction': 1}
    }
    base_date = sim_time.date()
    cache_key = (home_id, base_date)

    if cache_key not in _daily_schedule_cache:
        # Draw randomized times
        M_LU_time = _draw_time_for_date(base_date, 'M_LU')
        E_ALU_time = _draw_time_for_date(base_date, 'E_ALU')

        scheduled = {
            'M_LU_time': M_LU_time,
            'M_S_time': sched_cfg.get('M_S_time', my_schedule['M_S_time']),
            'E_ALU_time': E_ALU_time,
            'E_S_time': sched_cfg.get('E_S_time', my_schedule['E_S_time'])
        }

        # Morning duration
        mlu_dt = pd.to_datetime(M_LU_time, format="%H:%M")
        m_end_dt = pd.to_datetime(scheduled['M_S_time'], format="%H:%M")
        if m_end_dt <= mlu_dt:
            m_end_dt += pd.Timedelta(days=1)
        scheduled['M_LU_duration'] = (m_end_dt - mlu_dt).total_seconds()/3600
        scheduled['M_S_duration'] = sched_cfg.get('M_S_duration', my_schedule['M_S_duration'])

        # Evening duration
        ealu_dt = pd.to_datetime(E_ALU_time, format="%H:%M")
        e_end_dt = pd.to_datetime(scheduled['E_S_time'], format="%H:%M")
        if e_end_dt <= ealu_dt:
            e_end_dt += pd.Timedelta(days=1)
        scheduled['E_ALU_duration'] = (e_end_dt - ealu_dt).total_seconds()/3600
        scheduled['E_S_duration'] = sched_cfg.get('E_S_duration', my_schedule['E_S_duration'])

        _daily_schedule_cache[cache_key] = scheduled
        print(f"[{home_id}] {base_date}: M_LU={M_LU_time} ({scheduled['M_LU_duration']:.2f}h), "
              f"E_ALU={E_ALU_time} ({scheduled['E_ALU_duration']:.2f}h)")

    day_sched = _daily_schedule_cache[cache_key]

    def get_range(key):
        start = pd.to_datetime(f"{base_date} {day_sched[f'{key}_time']}")
        end = start + pd.Timedelta(hours=day_sched[f'{key}_duration'])
        return start, end

    ranges = {k: get_range(k) for k in ['M_LU','M_S','E_ALU','E_S']}

    if ranges['M_LU'][0] <= sim_time < ranges['M_LU'][1] or ranges['E_ALU'][0] <= sim_time < ranges['E_ALU'][1]:
        ctrl_signal['Water Heating'].update({'Setpoint': Tcontrol_LOADC, 'Deadband': Tcontrol_LOADdeadbandC})
    elif ranges['M_S'][0] <= sim_time < ranges['M_S'][1] or ranges['E_S'][0] <= sim_time < ranges['E_S'][1]:
        ctrl_signal['Water Heating'].update({'Setpoint': Tcontrol_SHEDC, 'Deadband': Tcontrol_deadbandC})

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
    df_sched[filtered_columns].to_csv(filtered_sched_file, index=False)
    return filtered_sched_file

#########################################
# SIMULATION FUNCTION
#########################################

def simulate_home(home_path, weather_file_path, schedule_cfg):
    filtered_sched_file = filter_schedules(home_path)
    hpxml_file = os.path.join(home_path, "in.XML")
    results_dir = os.path.join(home_path, "Results")
    os.makedirs(results_dir, exist_ok=True)

    dwelling_args = {
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

    sim_dwelling = Dwelling(name="HPWH Controlled", **dwelling_args)
    hpwh_unit = sim_dwelling.get_equipment_by_end_use('Water Heating')
    home_id = os.path.basename(home_path)
    for sim_time in sim_dwelling.sim_times:
        current_setpt = hpwh_unit.schedule.loc[sim_time, 'Water Heating Setpoint (C)']
        control_cmd = determine_hpwh_control(sim_time, current_setpt, schedule_cfg, home_id)
        sim_dwelling.update(control_signal=control_cmd)

    df_ctrl, _, _ = sim_dwelling.finalize()
    df_ctrl = remove_first_day(df_ctrl, Start)
    df_ctrl.to_csv(os.path.join(results_dir, 'hpwh_controlled.csv'), index=False)

    return df_ctrl

#########################################
# FIND HOMES
#########################################

def find_all_homes(base_dir):
    homes = []
    for item in os.listdir(base_dir):
        home_path = os.path.join(base_dir, item)
        if os.path.isdir(home_path) and os.path.isfile(os.path.join(home_path, 'in.XML')) and os.path.isfile(os.path.join(home_path, 'schedules.csv')):
            homes.append(home_path)
    return homes

def remove_first_day(df, start_date):
    if 'Time' not in df.columns:
        df = df.reset_index()
        if 'index' in df.columns:
            df.rename(columns={'index': 'Time'}, inplace=True)
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    first_day_end = start_date + pd.Timedelta(days=1)
    return df[df['Time'] >= first_day_end].copy()

def aggregate_results(homes, work_dir):
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
        df_ctrl_all.to_csv(os.path.join(work_dir, filename + "_Control.csv"), index=False)
    print("Aggregated CSV written!")

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

    home_schedules = {home: my_schedule.copy() for home in homes}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(simulate_home, home, WEATHER_FILE, home_schedules[home]) for home in homes]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print("Simulation failed:", e)

    print("All simulations complete!")

    aggregate_results(homes, WORKING_DIR)

    end_time = time.time()
    execution_min = (end_time - start_time)/60
    print(f"Execution time: {execution_min:.2f} minutes")
