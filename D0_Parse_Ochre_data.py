# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 14:46:47 2025

@author: Joe_admin
"""


import pandas as pd
from datetime import datetime
import os

# Converts the datetime information in the HEMS data to usable datetimes
def convert_custom_datetime(series):
    return series.apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M"))


############################################################################
#                           Enter inputs here                              #
############################################################################

# enter in the input and output file names. 

WORKING_DIR = r"C:\Users\danap\OCHRE_Working"
  
input_file_name  = "180117_2_15_RCJ_Baseline.csv"
output_file_name  = "D0_180117_2_15_RCJ_Baseline.csv"

input_file_name = os.path.join(WORKING_DIR, input_file_name)
output_file_name = os.path.join(WORKING_DIR, output_file_name)

############################################################################
#                             Program Start                                #
############################################################################


# read data 
df = pd.read_csv(input_file_name)

df['time'] = pd.to_datetime(df['Time'])


# Create column that contains hour and minute data
df['hr_min'] = df['time'].dt.strftime('%H:%M')

# drop unwanted columns
df = df.drop(['Time', 'Total Electric Power (kW)', 'Total Electric Energy (kWh)', 'Water Heating COP (-)', 
              'Water Heating Deadband Upper Limit (C)', 'Water Heating Deadband Lower Limit (C)', 'Water Heating Heat Pump COP (-)', 
              'Water Heating Control Temperature (C)', 'Hot Water Outlet Temperature (C)', 'Temperature - Indoor (C)', 'time'], axis=1)


df_pivot = df.pivot_table(index = 'Home', columns = 'hr_min', values = 'Water Heating Electric Power (kW)')

# write data to csv
df_pivot.to_csv(output_file_name, index=True)
