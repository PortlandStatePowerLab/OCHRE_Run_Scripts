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

print(dt.datetime.fromtimestamp(time.time(), dt.timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))
start_time = time.time()

#########################################
# USER SETTINGS
#########################################

filename = '180110_1_3_Efficiencyv3'  # simulation identifier

# Paths
DEFAULT_INPUT = r"C:\Users\danap\anaconda3\Lib\site-packages\ochre\defaults\Input Files"
DEFAULT_WEATHER = r"C:\Users\danap\anaconda3\Lib\site-packages\ochre\defaults\Weather\USA_OR_Portland.Intl.AP.726980_TMY3.epw"
WORKING_DIR = r"C:\Users\danap\OCHRE_Working"
INPUT_DIR = os.path.join(WORKING_DIR, "Input Files")
WEATHER_DIR = os.path.join(WORKING_DIR, "Weather")
WEATHER_FILE = os.path.join(WEATHER_DIR, "USA_OR_Portland.Intl.AP.726980_TMY3.epw")

# Simulation parameters
Start = dt.datetime(2018, 1, 11, 0, 0) # 4/7
Duration = 2  # days
t_res = 3  # minutes
jitter_min = 5

# HPWH control parameters (°F)
Tcontrol_SHEDF = 125
Tcontrol_dbF = 10
Tcontrol_LOADF = 130
Tcontrol_LOADdeadbandF = 2
TbaselineF = 130
TdeadbandF = 7
Tinit = 128

# LEVELS = {
#     1: {"Normal": {"ER_ON": 120, "ER_OFF": 130, "HP_ON": 120, "HP_OFF": 130},
#         "Shed": {"ER_ON": 116, "ER_OFF": 126, "HP_ON": 116, "HP_OFF": 126}},
#     2: {"Normal": {"ER_ON": 120, "ER_OFF": 130, "HP_ON": 70, "HP_OFF": 70},
#         "Shed": {"ER_ON": 116, "ER_OFF": 126, "HP_ON": 70, "HP_OFF": 70}},
#     3: {"Normal": {"ER_ON": 120, "ER_OFF": 128, "HP_ON": 128, "HP_OFF": 130},
#         "Shed": {"ER_ON": 116, "ER_OFF": 124, "HP_ON": 124, "HP_OFF": 126}},
#     4: {"Normal": {"ER_ON": 120, "ER_OFF": 125, "HP_ON": 125, "HP_OFF": 130},
#         "Shed": {"ER_ON": 116, "ER_OFF": 121, "HP_ON": 121, "HP_OFF": 126}},
#     5: {"Normal": {"ER_ON": 120, "ER_OFF": 123, "HP_ON": 123, "HP_OFF": 130},
#         "Shed": {"ER_ON": 116, "ER_OFF": 119, "HP_ON": 119, "HP_OFF": 126}},
#     6: {"Normal": {"ER_ON": 70, "ER_OFF": 117, "HP_ON": 117, "HP_OFF": 130},
#         "Shed": {"ER_ON": 70, "ER_OFF": 113, "HP_ON": 113, "HP_OFF": 126}},
#     7: {"Normal": {"ER_ON": 70, "ER_OFF": 115, "HP_ON": 115, "HP_OFF": 130},
#         "Shed": {"ER_ON": 70, "ER_OFF": 111, "HP_ON": 111, "HP_OFF": 126}},
#     8: {"Normal": {"ER_ON": 70, "ER_OFF": 112, "HP_ON": 112, "HP_OFF": 130},
#         "Shed": {"ER_ON": 70, "ER_OFF": 108, "HP_ON": 108, "HP_OFF": 126}},
#     9: {"Normal": {"ER_ON": 70, "ER_OFF": 70, "HP_ON": 70, "HP_OFF": 130},
#         "Shed": {"ER_ON": 70, "ER_OFF": 70, "HP_ON": 70, "HP_OFF": 126}},
# }

LEVELS = {
    1: {
        "Normal": {"ER_ON": 50, "ER_OFF": 130, "HP_ON": 50, "HP_OFF": 130},
        "Shed":   {"ER_ON": 50, "ER_OFF": 126, "HP_ON": 50, "HP_OFF": 126},},
    2: {
        "Normal": {"ER_ON": 50, "ER_OFF": 130, "HP_ON": 50, "HP_OFF": 50},
        "Shed":   {"ER_ON": 50, "ER_OFF": 126, "HP_ON": 50, "HP_OFF": 50},},
    3: {
        "Normal": {"ER_ON": 50, "ER_OFF": 128, "HP_ON": 128, "HP_OFF": 130},
        "Shed":   {"ER_ON": 50, "ER_OFF": 124, "HP_ON": 124, "HP_OFF": 126},},
    4: {
        "Normal": {"ER_ON": 50, "ER_OFF": 126, "HP_ON": 126, "HP_OFF": 130},
        "Shed":   {"ER_ON": 50, "ER_OFF": 122, "HP_ON": 122, "HP_OFF": 126},    },
    5: {
        "Normal": {"ER_ON": 50, "ER_OFF": 124, "HP_ON": 124, "HP_OFF": 130},
        "Shed":   {"ER_ON": 50, "ER_OFF": 120, "HP_ON": 120, "HP_OFF": 126},    },
    6: {
        "Normal": {"ER_ON": 50, "ER_OFF": 122, "HP_ON": 122, "HP_OFF": 130},
        "Shed":   {"ER_ON": 50, "ER_OFF": 118, "HP_ON": 118, "HP_OFF": 126},    },
    7: {
        "Normal": {"ER_ON": 50, "ER_OFF": 120, "HP_ON": 120, "HP_OFF": 130},
        "Shed":   {"ER_ON": 50, "ER_OFF": 116, "HP_ON": 116, "HP_OFF": 126},    },
    8: {
        "Normal": {"ER_ON": 50, "ER_OFF": 118, "HP_ON": 118, "HP_OFF": 130},
        "Shed":   {"ER_ON": 50, "ER_OFF": 114, "HP_ON": 114, "HP_OFF": 126},    },
    9: {
        "Normal": {"ER_ON": 50, "ER_OFF": 50, "HP_ON": 50, "HP_OFF": 130},
        "Shed":   {"ER_ON": 50, "ER_OFF": 50, "HP_ON": 50, "HP_OFF": 126},
    },
}


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

#########################################
# TEMPERATURE CONVERSIONS
#########################################

def f_to_c(temp_f):
    return (temp_f - 32) * 5 / 9

def f_to_c_DB(temp_f):
    return 5 / 9 * temp_f

def safe_f_to_c(val):
    return f_to_c(val) if val is not None else None

Tcontrol_SHEDC = f_to_c(Tcontrol_SHEDF)
Tcontrol_deadbandC = f_to_c_DB(Tcontrol_dbF)
Tcontrol_LOADC = f_to_c(Tcontrol_LOADF)
Tcontrol_LOADdeadbandC = f_to_c_DB(Tcontrol_LOADdeadbandF)
TbaselineC = f_to_c(TbaselineF)
TdeadbandC = f_to_c_DB(TdeadbandF)
TinitC = f_to_c(Tinit)

#########################################
# SCHEDULE GENERATION
#########################################

def generate_schedules_for_homes(homes):
    home_schedules = {}
    fmt = "%H:%M"

    # ---------- Weighted pools ----------
    M_LU_weighted_pool = [
        t for t, w in zip(M_LU_bins, M_LU_weights) for _ in range(w)
    ]
    random.shuffle(M_LU_weighted_pool)

    E_ALU_weighted_pool = [
        t for t, w in zip(E_ALU_bins, E_ALU_weights) for _ in range(w)
    ]
    random.shuffle(E_ALU_weighted_pool)

    # Morning shed offsets (hours)
    MS_bins = pd.date_range("10:00", "13:45", freq="15min")
    MS_offsets = [(t - pd.Timestamp("10:00")).total_seconds() / 3600 for t in MS_bins]
    MS_weights = [20, 23, 24, 23, 22, 22, 25, 26, 26, 29, 29, 29, 29, 27, 28, 27]
    MS_weighted_pool = [o for o, w in zip(MS_offsets, MS_weights) for _ in range(w)]
    random.shuffle(MS_weighted_pool)

    # Evening shed offsets
    ES_bins = pd.date_range("20:00", "23:45", freq="15min")
    ES_offsets = [(t - pd.Timestamp("20:00")).total_seconds() / 3600 for t in ES_bins]
    ES_weights = [17, 21, 24, 25, 26, 24, 24, 23, 23, 23, 23, 25, 28, 30, 33, 40]
    ES_weighted_pool = [o for o, w in zip(ES_offsets, ES_weights) for _ in range(w)]
    random.shuffle(ES_weighted_pool)

    # ---------- Assign schedules ----------
    for home in homes:
        sched = my_schedule.copy()

        # ===== Morning Load =====
        M_LU_base = M_LU_weighted_pool.pop() if M_LU_weighted_pool else random.choice(M_LU_bins)
        t_MLU = pd.to_datetime(M_LU_base, format=fmt)
        t_MLU += pd.Timedelta(minutes=random.uniform(-jitter_min, jitter_min))
        sched['M_LU_time'] = t_MLU.strftime(fmt)

        # ===== Morning Shed start =====
        t_MS = pd.to_datetime(my_schedule['M_S_time'], format=fmt)
        t_MS += pd.Timedelta(minutes=random.uniform(-jitter_min, jitter_min))
        sched['M_S_time'] = t_MS.strftime(fmt)

        # ===== Morning Load duration =====
        if t_MS <= t_MLU:
            t_MS += pd.Timedelta(days=1)
        sched['M_LU_duration'] = max(
            0.5, (t_MS - t_MLU).total_seconds() / 3600
        )

        # ===== Morning Shed duration =====
        ms_offset = MS_weighted_pool.pop() if MS_weighted_pool else random.choice(MS_offsets)
        sched['M_S_duration'] = 4 + ms_offset

        # ===== Evening Load =====
        E_ALU_base = E_ALU_weighted_pool.pop() if E_ALU_weighted_pool else random.choice(E_ALU_bins)
        t_EALU = pd.to_datetime(E_ALU_base, format=fmt)
        t_EALU += pd.Timedelta(minutes=random.uniform(-jitter_min, jitter_min))
        sched['E_ALU_time'] = t_EALU.strftime(fmt)

        # ===== Evening Shed start =====
        t_ES = pd.to_datetime(my_schedule['E_S_time'], format=fmt)
        t_ES += pd.Timedelta(minutes=random.uniform(-jitter_min, jitter_min))
        sched['E_S_time'] = t_ES.strftime(fmt)

        # ===== Evening Load duration =====
        if t_ES <= t_EALU:
            t_ES += pd.Timedelta(days=1)
        sched['E_ALU_duration'] = max(
            0.5, (t_ES - t_EALU).total_seconds() / 3600
        )

        # ===== Evening Shed duration =====
        es_offset = ES_weighted_pool.pop() if ES_weighted_pool else random.choice(ES_offsets)
        sched['E_S_duration'] = 3 + es_offset

        home_schedules[home] = sched

    return home_schedules



#########################################
# HPWH CONTROL FUNCTION
#########################################

def determine_hpwh_control(sim_time, current_temp_c, sched_cfg, level_dict):
    mode_vals = level_dict["Normal"]
    ctrl_signal = {
        'Water Heating': {
            'Setpoint': TbaselineC,
            'Deadband': TdeadbandC,
            'Load Fraction': 1,
            'ER_ON': mode_vals["ER_ON"],
            'ER_OFF': mode_vals["ER_OFF"],
            'HP_ON': mode_vals["HP_ON"],
            'HP_OFF': mode_vals["HP_OFF"]
        }
    }

    base_date = sim_time.date()
    def get_time_range(key):
        start = pd.to_datetime(f"{base_date} {sched_cfg[f'{key}_time']}")
        end = start + pd.Timedelta(hours=sched_cfg[f'{key}_duration'])
        return start, end

    ranges = {k: get_time_range(k) for k in ['M_LU','M_S','E_ALU','E_S']}

    # Adjust for load periods
    if ranges['M_LU'][0] <= sim_time < ranges['M_LU'][1] or ranges['E_ALU'][0] <= sim_time < ranges['E_ALU'][1]:
        ctrl_signal['Water Heating']['Setpoint'] = Tcontrol_LOADC
        ctrl_signal['Water Heating']['Deadband'] = Tcontrol_LOADdeadbandC
    elif ranges['M_S'][0] <= sim_time < ranges['M_S'][1] or ranges['E_S'][0] <= sim_time < ranges['E_S'][1]:
        shed_vals = level_dict["Shed"]
        ctrl_signal['Water Heating'].update({
            'Setpoint': Tcontrol_SHEDC,
            'Deadband': Tcontrol_deadbandC,
            'ER_ON': shed_vals["ER_ON"],
            'ER_OFF': shed_vals["ER_OFF"],
            'HP_ON': shed_vals["HP_ON"],
            'HP_OFF': shed_vals["HP_OFF"]
        })

    return ctrl_signal

#########################################
# SIMULATION FUNCTION
#########################################

def simulate_home(home_path, weather_file_path, schedule_cfg, level_dict):
    """
    Simulate a single home with HPWH control, keeping all original OCHRE output columns
    and adding ER_ON, ER_OFF, HP_ON, HP_OFF columns. Saves results to Results folder.
    """


    # Paths
    hpxml_file = os.path.join(home_path, 'in.XML')
    results_dir = os.path.join(home_path, "Results")
    os.makedirs(results_dir, exist_ok=True)

    # Filter schedules safely
    filtered_sched_file = os.path.join(home_path, 'schedules.csv')
    if os.path.exists(filtered_sched_file):
        filtered_sched_file = filter_schedules(home_path)

    # Dwelling setup
    dwelling_args = {
        "start_time": Start,
        "time_res": dt.timedelta(minutes=t_res),
        "duration": dt.timedelta(days=Duration),
        "hpxml_file": hpxml_file,
        "hpxml_schedule_file": filtered_sched_file,
        "weather_file": weather_file_path,
        "verbosity": 7,
        "Equipment": {"Water Heating": {
            "Initial Temperature (C)": TinitC,
            "hp_only_mode": False,
            "Max Tank Temperature": 70,
            "Upper Node": 3,
            "Lower Node": 10,
            "Upper Node Weight": 0.75
        }},
    }

    # Initialize dwelling
    sim_dwelling = Dwelling(name="HPWH Controlled", **dwelling_args)
    hpwh_unit = sim_dwelling.get_equipment_by_end_use('Water Heating')

    # Store per-timestep control values
    df_ctrl_all = []

    for sim_time in sim_dwelling.sim_times:
        # Get current water heating temperature
        current_temp = hpwh_unit.schedule.loc[sim_time, 'Water Heating Setpoint (C)']

        # Determine control setpoints
        control_cmd = determine_hpwh_control(sim_time, current_temp, schedule_cfg, level_dict=level_dict)

        # Update HPWH ON/OFF temps
        hpwh_unit.er_on_temp = safe_f_to_c(control_cmd['Water Heating'].get('ER_ON'))
        hpwh_unit.er_off_temp = safe_f_to_c(control_cmd['Water Heating'].get('ER_OFF'))
        hpwh_unit.hp_on_temp = safe_f_to_c(control_cmd['Water Heating'].get('HP_ON'))
        hpwh_unit.hp_off_temp = safe_f_to_c(control_cmd['Water Heating'].get('HP_OFF'))

        # Update dwelling
        sim_dwelling.update(control_signal={'Water Heating': {
            'Setpoint': control_cmd['Water Heating']['Setpoint'],
            'Deadband': control_cmd['Water Heating']['Deadband'],
            'Load Fraction': 1
        }})

        # Append timestep control values
        df_ctrl_all.append({
            'Time': sim_time,
            'ER_ON': hpwh_unit.er_on_temp,
            'ER_OFF': hpwh_unit.er_off_temp,
            'HP_ON': hpwh_unit.hp_on_temp,
            'HP_OFF': hpwh_unit.hp_off_temp
        })

    # Convert to dataframe
    df_hp_er = pd.DataFrame(df_ctrl_all)

    # Get OCHRE outputs
    df_ochre, _, _ = sim_dwelling.finalize()

    # Ensure Time column exists and is datetime
    if 'Time' not in df_ochre.columns:
        df_ochre = df_ochre.reset_index()
        if 'index' in df_ochre.columns:
            df_ochre.rename(columns={'index': 'Time'}, inplace=True)
    df_ochre['Time'] = pd.to_datetime(df_ochre['Time'], errors='coerce')

    # Remove first day
    df_ochre = remove_first_day(df_ochre, Start)
    df_hp_er = df_hp_er[df_hp_er['Time'] >= Start + dt.timedelta(days=1)]

    # Merge HP/ER data safely
    df_ctrl = pd.merge(df_ochre, df_hp_er, on='Time', how='left')

    # Keep only desired columns
    CTRL_COLS = [
        "Time",
        "Total Electric Power (kW)",
        "Total Electric Energy (kWh)",
        "Water Heating Electric Power (kW)",
        "Water Heating COP (-)",
        "Water Heating Deadband Upper Limit (C)",
        "Water Heating Deadband Lower Limit (C)",
        "Water Heating Heat Pump COP (-)",
        "Water Heating Control Temperature (C)",
        "Hot Water Outlet Temperature (C)",
        "Temperature - Indoor (C)",
        "ER_ON",
        "ER_OFF",
        "HP_ON",
        "HP_OFF"
    ]
    df_ctrl = df_ctrl[[c for c in CTRL_COLS if c in df_ctrl.columns]]

    # Save results
    results_file = os.path.join(results_dir, 'hpwh_controlled.parquet')
    df_ctrl.to_parquet(results_file, index=False)

    # Cleanup intermediate files
    cleanup_results_dir(results_dir, keep_files=['hpwh_controlled.parquet'])

    return df_ctrl



#########################################
# HELPER FUNCTIONS
#########################################

def filter_schedules(home_path):
    orig_sched_file = os.path.join(home_path, 'schedules.csv')
    filtered_sched_file = os.path.join(home_path, 'filtered_schedules.csv')
    df_sched = pd.read_csv(orig_sched_file)
    valid_cols = list(ALL_SCHEDULE_NAMES.keys()) + ['M_LU_time','M_LU_duration','M_S_time','M_S_duration','E_ALU_time','E_ALU_duration','E_S_time','E_S_duration']
    filtered_cols = [c for c in valid_cols if c in df_sched.columns]
    df_sched[filtered_cols].to_csv(filtered_sched_file, index=False)
    return filtered_sched_file

def remove_first_day(df, start_date):
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    return df[df['Time'] >= start_date + dt.timedelta(days=1)].copy()

def cleanup_results_dir(results_dir, keep_files=None):
    if keep_files is None: keep_files = []
    for f in os.listdir(results_dir):
        path = os.path.join(results_dir, f)
        if os.path.isfile(path) and f not in keep_files: os.remove(path)
        elif os.path.isdir(path): shutil.rmtree(path)

def find_all_homes(base_dir):
    return [os.path.join(base_dir, d) for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
            and os.path.isfile(os.path.join(base_dir, d, 'in.XML'))
            and os.path.isfile(os.path.join(base_dir, d, 'schedules.csv'))]

def aggregate_results(homes, work_dir, filename):
    all_ctrl = []
    for home in homes:
        ctrl_file = os.path.join(home, "Results", "hpwh_controlled.parquet")
        if os.path.exists(ctrl_file):
            df = pd.read_parquet(ctrl_file)
            df["Home"] = os.path.basename(home)
            all_ctrl.append(df)
    if all_ctrl:
        df_all = pd.concat(all_ctrl, ignore_index=True)
        df_all.to_parquet(os.path.join(work_dir, filename + "_Control.parquet"), index=False)
    print("Aggregated parquet files written!")

#########################################
# MAIN EXECUTION
#########################################

if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(WEATHER_DIR, exist_ok=True)
    
    for item in os.listdir(DEFAULT_INPUT):
        src, dst = os.path.join(DEFAULT_INPUT, item), os.path.join(INPUT_DIR, item)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
    
    if not os.path.exists(WEATHER_FILE):
        shutil.copy(DEFAULT_WEATHER, WEATHER_FILE)

    homes = find_all_homes(INPUT_DIR)
    print(f"Found {len(homes)} homes")

    home_schedules = generate_schedules_for_homes(homes)

    for level in range(1, 10):
        print(f"Running simulations for LEVEL {level}")
        level_dict = LEVELS[level]
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(simulate_home, home, WEATHER_FILE, home_schedules[home], level_dict) for home in homes]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print("Simulation failed:", e)
        aggregate_results(homes, WORKING_DIR, f"{filename}_Level{level}")

end_time = time.time()
print(f"Execution time: {(end_time - start_time)/60:.2f} minutes")
