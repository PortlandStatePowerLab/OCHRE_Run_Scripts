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

filename = '180113_1_3_EfficiencyControl4C2' # date that's thrown away, num of simulation days, data res, ramp or no ramp control

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
Tcontrol_SHEDF = 123 #F
step = 10 #F
Tcontrol_dbF = np.arange(7, 7 + step, step) #<------------------------------------------
Tcontrol_LOADF = 130 #F
Tcontrol_LOADdeadbandF = 2 #F
TbaselineF = 130 #F
TdeadbandF = 7 #F
Tinit = 128 #F

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
# Tcontrol_deadbandC = Tcontrol_dbF * 5/9
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

    # Keep all HPWH custom columns too
    hpwh_cols = ['M_LU_time','M_LU_duration','M_S_time','M_S_duration',
                 'E_ALU_time','E_ALU_duration','E_S_time','E_S_duration']
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
                "hp_only_mode": False,# can set to True for HP only. 
                "Max Tank Temperature": 70,
                "Upper Node": 3,
                "Lower Node": 10,
                "Upper Node Weight": 0.75,
                "Efficiency Coefficient": 0.57, # 50 = HP only, 0 = ER only. --->  HP_DB = Tset - TDB * efficiency_coefficient 
            },
        }
    }

    # # Baseline
    # base_dwelling = Dwelling(name="HPWH Baseline", **dwelling_args_local)
    # for t_base in base_dwelling.sim_times:
    #     base_ctrl = {"Water Heating": {"Setpoint": TbaselineC, "Deadband": TdeadbandC, "Load Fraction": 1}}
    #     base_dwelling.update(control_signal=base_ctrl)
    # df_base, _, _ = base_dwelling.finalize()

    # Controlled
    sim_dwelling = Dwelling(name="HPWH Controlled", **dwelling_args_local)
    hpwh_unit = sim_dwelling.get_equipment_by_end_use('Water Heating')
    for sim_time in sim_dwelling.sim_times:
        current_setpt = hpwh_unit.schedule.loc[sim_time, 'Water Heating Setpoint (C)']
        control_cmd = determine_hpwh_control(sim_time=sim_time, current_temp_c=current_setpt, sched_cfg=schedule_cfg, shed_deadbandC=shed_deadbandC)
        sim_dwelling.update(control_signal=control_cmd)
    df_ctrl, _, _ = sim_dwelling.finalize()

    df_ctrl = remove_first_day(df_ctrl, Start)
    # df_base = remove_first_day(df_base, Start)
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
    # BASE_COLS = CTRL_COLS

    df_ctrl = df_ctrl[[c for c in CTRL_COLS if c in df_ctrl.columns]]
    # df_base = df_base[[c for c in BASE_COLS if c in df_base.columns]]
        
    df_ctrl.to_parquet(
        os.path.join(results_dir, f'hpwh_controlled.parquet'),
        index=False
    )

    # cleanup_results_dir(results_dir, keep_files=['hpwh_baseline.parquet', 'hpwh_controlled.parquet'])

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
                
                


def aggregate_across_deadbands(work_dir, prefix):
    """
    Combine <prefix>_Control_DB*.parquet into <prefix>_Control.parquet
    """

    pattern = re.compile(
        rf"^{re.escape(prefix)}_Control_DB(\d+)\.parquet$"
    )

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

        # Enforce deadband metadata
        df["Shed Deadband (F)"] = dbF
        df["SourceFile"] = fname

        dfs.append(df)

    df_master = pd.concat(dfs, ignore_index=True)

    out_path = os.path.join(work_dir, f"{prefix}_Control.parquet")
    df_master.to_parquet(out_path, index=False)

    print(
        f"\n✅ Cross-deadband aggregation complete\n"
        f"   Deadbands: {[db for _, db in matches]}\n"
        f"   Rows: {len(df_master):,}\n"
        f"   Output: {out_path}"
    )


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

    # Assign schedules
    for home in homes:
        sched = my_schedule.copy()

        # -----------------------------
        # M_LU_time with jitter
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
        # M_S_time and M_LU_duration with jitter
        # -----------------------------
        t_MS_start = pd.to_datetime(my_schedule['M_S_time'], format=fmt)
        t_MS_start += pd.Timedelta(minutes=random.uniform(-jitter_min, jitter_min))
        sched['M_S_time'] = t_MS_start.strftime(fmt)

        t_MLU_start = pd.to_datetime(sched['M_LU_time'], format=fmt)
        t_MLU_end = t_MS_start
        if t_MLU_end <= t_MLU_start:
            t_MLU_end += pd.Timedelta(days=1)
        sched['M_LU_duration'] = max(1, (t_MLU_end - t_MLU_start).total_seconds() / 3600)

        if MS_weighted_pool:
            n = MS_weighted_pool.pop()
        else:
            n = random.choice(MS_offsets)
        sched['M_S_duration'] = 4 + n

        # -----------------------------
        # Evening Schedule Assignment
        # -----------------------------
        if E_ALU_weighted_pool:
            E_ALU_base = E_ALU_weighted_pool.pop()
        else:
            E_ALU_base = random.choice(E_ALU_bins)
        t_E_ALU_start = pd.to_datetime(E_ALU_base, format=fmt)
        t_E_ALU_start += pd.Timedelta(minutes=random.uniform(-jitter_min, jitter_min))
        sched['E_ALU_time'] = t_E_ALU_start.strftime(fmt)

        t_ES_start = pd.to_datetime(my_schedule['E_S_time'], format=fmt)
        t_ES_start += pd.Timedelta(minutes=random.uniform(-jitter_min, jitter_min))
        sched['E_S_time'] = t_ES_start.strftime(fmt)

        if t_ES_start <= t_E_ALU_start:
            t_ES_start += pd.Timedelta(days=1)
        sched['E_ALU_duration'] = max(1, (t_ES_start - t_E_ALU_start).total_seconds() / 3600)

        if ES_weighted_pool:
            n = ES_weighted_pool.pop()
        else:
            n = random.choice(ES_offsets)
        sched['E_S_duration'] = 3 + n

        # Save schedule
        home_schedules[home] = sched

    # -----------------------------
    # Sweep deadbands
    # -----------------------------

    # -----------------------------
    # Sweep deadbands (safe aggregation)
    # -----------------------------
    for shed_dbF in Tcontrol_dbF:
        print(f"\nRunning shed deadband = {shed_dbF} F")
    
        all_ctrl = []
    
        # -----------------------------
        # Run all homes in parallel safely
        # -----------------------------
        def simulate_home_safe(home_path, weather_file, sched_cfg, shed_dbF):
            try:
                return simulate_home(home_path, weather_file, sched_cfg, shed_dbF)
            except Exception as e:
                print(f"⚠️ Simulation failed for {home_path} (DB={shed_dbF}): {e}")
                return None
    
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(simulate_home_safe, home, WEATHER_FILE, home_schedules[home], shed_dbF)
                for home in homes
            ]
    
            for f in concurrent.futures.as_completed(futures):
                df_result = f.result()
                if df_result is not None:
                    all_ctrl.append(df_result)
    
        # -----------------------------
        # Aggregate immediately
        # -----------------------------
        if all_ctrl:  # only if at least one home succeeded
            df_all = pd.concat(all_ctrl, ignore_index=True)
            df_all["Home"] = df_all.get("Home", "Unknown")
            df_all["Shed Deadband (F)"] = shed_dbF
    
            out_file = os.path.join(
                WORKING_DIR,
                f"{filename}_Control_DB{int(shed_dbF)}.parquet"
            )
            df_all.to_parquet(out_file, index=False)
    
            print(
                f"Aggregated DB{shed_dbF}: "
                f"{len(df_all):,} rows, "
                f"{df_all['Home'].nunique()} homes"
            )
        else:
            print(f"⚠️ No successful homes to aggregate for DB{shed_dbF}")
            
            
# -----------------------------
# Cross-deadband aggregation
# -----------------------------
aggregate_across_deadbands(
    work_dir=WORKING_DIR,
    prefix=filename
)




end_time = time.time()
execution_time = end_time - start_time
execution_min = execution_time/60
print(f"Execution time: {execution_min} minutes")
