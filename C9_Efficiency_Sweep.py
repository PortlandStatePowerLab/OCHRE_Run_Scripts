# -*- coding: utf-8 -*-
"""
Integrated OCHRE Efficiency Sweep
Levels 1-9 across specified Deadbands
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
from ochre.utils.base import OCHREException

print(datetime.datetime.fromtimestamp(time.time(), datetime.timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))

start_time = time.time()

#########################################
# USER SETTINGS
#########################################
filename_base = '180113_1_3_EffSweep' 

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
t_res = 3     # minutes
jitter_min = 5

# HPWH control parameters (°F)
Tcontrol_SHEDF = 130
step = 7
Tcontrol_dbF = np.arange(7, 7 + step, step) 
Tcontrol_LOADF = 130
Tcontrol_LOADdeadbandF = 7
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

# Efficiency Map
# LVL = {1:0, 2:0.14, 3:0.29, 4:0.43, 5:0.57, 6:0.71, 7:0.857, 8:1, 9:10} # linear db width
LVL = {1:0, 3:0.34, 4:0.6, 5:0.88, 6:1.02, 7:1.05, 8:1.35, 9:10} # linear marginal load 

# Global Placeholders
EFF_BASELINE = 0
EFF_SHED = 0
EFF_LOAD = 0

# Randomization bins
M_LU_weights = [14, 28, 34, 41, 46, 46, 41, 33, 30, 31, 35, 30]
M_LU_bins = pd.date_range("03:00", periods=len(M_LU_weights), freq="15min").strftime("%H:%M").tolist()
E_ALU_weights = [17, 21, 27, 37, 40, 46, 40, 42, 36, 32, 33, 38]
E_ALU_bins = pd.date_range("14:00", periods=len(E_ALU_weights), freq="15min").strftime("%H:%M").tolist()

#########################################
# UTILITIES
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

def determine_hpwh_control(sim_time, current_temp_c, sched_cfg, shed_deadbandC, **kwargs):
    ctrl_signal = {
        'Water Heating': {
            'Setpoint': TbaselineC,
            'Deadband': TdeadbandC,
            'Load Fraction': 1,
            'Efficiency Coefficient': EFF_BASELINE,
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
            'Deadband': Tcontrol_LOADdeadbandC,
            'Efficiency Coefficient': EFF_LOAD
        })
    elif ranges['M_S'][0] <= sim_time < ranges['M_S'][1] or ranges['E_S'][0] <= sim_time < ranges['E_S'][1]:
        ctrl_signal['Water Heating'].update({
            'Setpoint': Tcontrol_SHEDC,
            'Deadband': shed_deadbandC,
            'Efficiency Coefficient': EFF_SHED
        })

    return ctrl_signal

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

def remove_first_day(df, start_date):
    if 'Time' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'Time'})
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    first_day_end = start_date + pd.Timedelta(days=1)
    return df[df['Time'] >= first_day_end].copy()

def find_all_homes(base_dir):
    homes = []
    for item in os.listdir(base_dir):
        home_path = os.path.join(base_dir, item)
        if os.path.isdir(home_path):
            if os.path.isfile(os.path.join(home_path, 'in.XML')) and \
               os.path.isfile(os.path.join(home_path, 'schedules.csv')):
                homes.append(home_path)
    return homes

def simulate_home(home_path, weather_file_path, schedule_cfg, shed_deadbandF):
    shed_deadbandC = f_to_c_DB(shed_deadbandF)
    filtered_sched_file = filter_schedules(home_path)
    hpxml_file = os.path.join(home_path, 'in.XML')
    
    dwelling_args_local = {
        "start_time": Start,
        "time_res": dt.timedelta(minutes=t_res),
        "duration": dt.timedelta(days=Duration),
        "hpxml_file": hpxml_file,
        "hpxml_schedule_file": filtered_sched_file,
        "weather_file": weather_file_path,
        "verbosity": 1,
        "Equipment": {
            "Water Heating": {
                "Initial Temperature (C)": TinitC, 
                "hp_only_mode": False,
                "Max Tank Temperature": 70,
                "Upper Node": 3,
                "Lower Node": 10,
                "Upper Node Weight": 0.75,   
            },
        }
    }

    # Catching generic Exception covers internal XML formatting issues (like missing Pool Pumps)
    try:
        sim_dwelling = Dwelling(name="HPWH Controlled", **dwelling_args_local)
    except Exception as e:
        print(f"❌ Skipping building {os.path.basename(home_path)} due to error: {e}")
        return None
    
    hpwh_unit = sim_dwelling.get_equipment_by_end_use('Water Heating')
    
    for sim_time in sim_dwelling.sim_times:
        if sim_time < Start + pd.Timedelta(days=1):
            control_cmd = {'Water Heating': {'Setpoint': TbaselineC, 'Deadband': TdeadbandC, 'Load Fraction': 1}}
        else:
            current_setpt = hpwh_unit.schedule.loc[sim_time, 'Water Heating Setpoint (C)']
            control_cmd = determine_hpwh_control(sim_time, current_setpt, schedule_cfg, shed_deadbandC)
        
        sim_dwelling.update(control_signal=control_cmd)
        
    df_ctrl, _, _ = sim_dwelling.finalize()
    df_ctrl = remove_first_day(df_ctrl, Start)
    df_ctrl["Shed Deadband (F)"] = shed_deadbandF
    df_ctrl["Home"] = os.path.basename(home_path)

    cols_to_keep = ["Time", "Home", "Total Electric Power (kW)", "Water Heating Electric Power (kW)", "Shed Deadband (F)"]
    return df_ctrl[[c for c in cols_to_keep if c in df_ctrl.columns]]

def aggregate_across_deadbands(work_dir, prefix):
    # Regex updated to explicitly match the pattern outputted by our loops below
    pattern = re.compile(rf"^EfficiencyLevel{re.escape(prefix)}_DB(\d+)_LinearMarginal\.parquet$")
    
    matches = []
    for f in os.listdir(work_dir):
        match = pattern.match(f)
        if match:
            matches.append((f, int(match.group(1))))
            
    if not matches: 
        print(f"⚠️ No deadband files found to aggregate for prefix: {prefix}")
        return

    # Sort files sequentially by deadband value
    dfs = [pd.read_parquet(os.path.join(work_dir, f)) for f, db in sorted(matches, key=lambda x: x[1])]
    df_master = pd.concat(dfs, ignore_index=True)
    
    out_path = os.path.join(work_dir, f"EfficiencyLevel{prefix}_FULL_LEVEL.parquet")
    df_master.to_parquet(out_path, index=False)
    print(f"✅ Aggregated Level {prefix} to {out_path}")

#########################################
# MAIN EXECUTION
#########################################

if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(WEATHER_DIR, exist_ok=True)

    for item in os.listdir(DEFAULT_INPUT):
        src, dst = os.path.join(DEFAULT_INPUT, item), os.path.join(INPUT_DIR, item)
        if os.path.isdir(src) and not os.path.exists(dst): shutil.copytree(src, dst)
    if not os.path.exists(WEATHER_FILE): shutil.copy(DEFAULT_WEATHER, WEATHER_FILE)

    homes = find_all_homes(INPUT_DIR)
    
    home_schedules = {}
    fmt = "%H:%M"
    
    M_LU_weighted_pool = [b for b, w in zip(M_LU_bins, M_LU_weights) for _ in range(w)]
    E_ALU_weighted_pool = [b for b, w in zip(E_ALU_bins, E_ALU_weights) for _ in range(w)]
    random.shuffle(M_LU_weighted_pool)
    random.shuffle(E_ALU_weighted_pool)

    for home in homes:
        sched = my_schedule.copy()
        t_base = pd.to_datetime(M_LU_weighted_pool.pop() if M_LU_weighted_pool else random.choice(M_LU_bins), format=fmt)
        sched['M_LU_time'] = (t_base + pd.Timedelta(minutes=random.uniform(-jitter_min, jitter_min))).strftime(fmt)
        
        t_ms_start = pd.to_datetime(my_schedule['M_S_time'], format=fmt) + pd.Timedelta(minutes=random.uniform(-jitter_min, jitter_min))
        sched['M_S_time'] = t_ms_start.strftime(fmt)
        
        t_mlu_start = pd.to_datetime(sched['M_LU_time'], format=fmt)
        diff = (t_ms_start - t_mlu_start).total_seconds() / 3600
        sched['M_LU_duration'] = max(1, diff if diff > 0 else diff + 24)
        sched['M_S_duration'] = 4 + random.uniform(0, 3)

        t_e_alu_base = pd.to_datetime(E_ALU_weighted_pool.pop() if E_ALU_weighted_pool else random.choice(E_ALU_bins), format=fmt)
        sched['E_ALU_time'] = (t_e_alu_base + pd.Timedelta(minutes=random.uniform(-jitter_min, jitter_min))).strftime(fmt)
        home_schedules[home] = sched

    # 3. SWEEP LEVELS 1-9
    for current_lvl in sorted(LVL.keys()):
        print(f"\n>>> LEVEL {current_lvl} (Coeff: {LVL[current_lvl]})")
        
        EFF_BASELINE = EFF_SHED = EFF_LOAD = LVL[current_lvl]
        lvl_prefix = f"{filename_base}_L{current_lvl}"

        # 4. SWEEP DEADBANDS
        for shed_dbF in Tcontrol_dbF:
            all_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(simulate_home, h, WEATHER_FILE, home_schedules[h], shed_dbF) for h in homes]
                for f in concurrent.futures.as_completed(futures):
                    res = f.result()
                    if res is not None: all_results.append(res)
            
            if all_results:
                df_lvl_db = pd.concat(all_results, ignore_index=True)
                # Modified naming convention here to distinctively match the deadband number
                out_name = os.path.join(WORKING_DIR, f"EfficiencyLevel{lvl_prefix}_DB{shed_dbF}_LinearMarginal.parquet")
                df_lvl_db.to_parquet(out_name, index=False)

        # 5. Aggregate Level
        aggregate_across_deadbands(WORKING_DIR, lvl_prefix)

    print(f"\n✅ COMPLETE. Total time: {(time.time() - start_time)/60:.2f} min")