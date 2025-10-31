# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 11:54:29 2025

@author: Joe_admin
"""

import pandas as pd
import time
import numpy as np
import os
import datetime

now = datetime.datetime.now()
now = now.time()
print(now)

start_time = time.time()
print('Processing...')


############################################################################
#                           Enter inputs here                              #
############################################################################


# enter in the input and output file names. 

WORKING_DIR = r"C:\Users\danap\OCHRE_Working"
  
input_file_name  = "D0_180117_2_15_RCJ.csv"
ninety_fifth_output_file  = "D1b_180117_2_15_RCJ_95.csv"
mean_output_file  = "D1b_180117_2_15_RCJ_Mean.csv"
fifth_output_file  = "D1b_180117_2_15_RCJ_5.csv"


input_file_name = os.path.join(WORKING_DIR, input_file_name)
ninety_fifth_output_file = os.path.join(WORKING_DIR, ninety_fifth_output_file)
mean_output_file = os.path.join(WORKING_DIR, mean_output_file)
fifth_output_file = os.path.join(WORKING_DIR, fifth_output_file)



unit_runs = 500
MCS_runs = 1000  # this method 02 is much faster

############################################################################
#                        FUNCTIONS                                         #
############################################################################


def sample_data(input_df, units):
    # Randomly sample N rows with replacement
    df_sampled = input_df.sample(n=units, replace=True) # remove the random state when done testing! 
    
    #before returning, remove the site ID column and sort
    df_sampled = df_sampled.drop(['Home'], axis=1)
    return df_sampled
    
def get_MCS_run(N, input_df):
    
    for j, M in enumerate(np.arange(1, MCS_runs+1)):
        # sample the data
        df_sampled = sample_data(input_df, N)
        
        # get the aggragate load of the sample set
        agg_sample = df_sampled.sum()
        
                
        # add the agg load to the MSC_table
        MCS_table.loc[j] = agg_sample # this is one row of the MCS table!
    
    return MCS_table

# def get_stats(input_df):
#     # Compute the statistics
#     summary_df = pd.DataFrame({
#         '95th Percentile': input_df.quantile(0.975),
#         'Mean': input_df.mean(),
#         '5th Percentile': input_df.quantile(0.025)
#         }).T  # Transpose to get rows as statistics
    
#     return summary_df

def get_stats(input_df):
    """
    Compute 5th, mean, and 95th *profiles* (preserving 24-hour shape)
    by ranking the 1000 Monte Carlo runs as a whole.
    """
    # Each row of input_df = one MC run (already a full 24-hour profile)
    # Compute a metric to sort by (e.g., total daily sum)
    total_load = input_df.sum(axis=1)

    # Sort by total load
    sorted_idx = total_load.sort_values().index

    # Compute indices for 5th and 95th percentiles
    i5 = int(0.025 * len(sorted_idx))
    i95 = int(0.975 * len(sorted_idx)) - 1

    # Select full 24-hour profiles at those positions
    prof5 = input_df.loc[sorted_idx[i5]]
    prof_mean = input_df.mean(axis=0)
    prof95 = input_df.loc[sorted_idx[i95]]

    # Return as a DataFrame like before (so the rest of your code still works)
    summary_df = pd.DataFrame({
        '95th Percentile': prof95,
        'Mean': prof_mean,
        '5th Percentile': prof5
    }).T

    return summary_df

    

############################################################################
#                             Program Start                                #
############################################################################


# read data 
df = pd.read_csv(input_file_name)

# Randomly sample 50 rows with replacement
# df_sampled = df.sample(n=50, replace=True, random_state=42)

units_arr = np.arange(1, unit_runs+1)

# get the times 
times = df.drop(['Home'], axis=1).columns # this was changed from ee_site)id

# initialize MSC table
MCS_table = pd.DataFrame(np.nan, index=range(MCS_runs), columns=times)

# initialize stats tables
ninety_fifth_df = pd.DataFrame(np.nan, index=range(unit_runs), columns=times)
mean_df         = pd.DataFrame(np.nan, index=range(unit_runs), columns=times)
fifth_df        = pd.DataFrame(np.nan, index=range(unit_runs), columns=times)

for i, N in enumerate(np.arange(1, unit_runs+1)):
    # get the table that contains each MCS run 
    MCS_table = get_MCS_run(N, df)
    MCS_table = MCS_table.div(0.5 * N) 
    
    # find the 95th, mean, 5th percentile values at each time step
    stats_df = get_stats(MCS_table)

    
    # save those stats to three seperate tables. 
    ninety_fifth_df.loc[i] = stats_df.loc['95th Percentile']
    mean_df.loc[i] = stats_df.loc['Mean']
    fifth_df.loc[i] = stats_df.loc['5th Percentile']


# results_df.to_csv(output_file_name, index=True)
ninety_fifth_df.to_csv(ninety_fifth_output_file, index=True)
mean_df.to_csv(mean_output_file, index=True)
fifth_df.to_csv(fifth_output_file, index=True)



# print out the time it took to run the program
end_time = time.time()
execution_time = end_time - start_time

now = datetime.datetime.now()
now = now.time()
print(now)
print('Processing Complete.\n')
execution_min = execution_time/60
print(f"Execution time: {execution_min} minutes")

'''
# get a set of the sampled data
df_sampled = sample_data(df, 50)

# drop the site ids
df_sampled = df_sampled.drop(['ee_site_id'], axis=1)

# get the aggragate load of the sample set
agg_sample = df_sampled.sum()

# save the series above into the MCS table


'''


