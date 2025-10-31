# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 08:57:35 2025

@author: Joe_admin
"""

import pandas as pd
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from scipy.ndimage import maximum_filter, uniform_filter, zoom
import os

start_time = time.time()

############################################################################
#                           Enter inputs here                              #
############################################################################

WORKING_DIR = r"C:\Users\danap\OCHRE_Working"
  
ninety_fifth_output_file  = "D1_180111_ADMD_95.csv"
mean_output_file  = "D1_180111_ADMD_mean.csv"
fifth_output_file  = "D1_180111_ADMD_5.csv"


ninety_fifth_file_name = os.path.join(WORKING_DIR, ninety_fifth_output_file)
mean_file_name = os.path.join(WORKING_DIR, mean_output_file)
fifth_file_name = os.path.join(WORKING_DIR, fifth_output_file)




write_percent_error = False
############################################################################
#                             Program Start                                #
############################################################################

# read data 
ninety_fifth_df = pd.read_csv(ninety_fifth_file_name)
mean_df         = pd.read_csv(mean_file_name)
fifth_df        = pd.read_csv(fifth_file_name)

# remove the columns we dont need
ninety_fifth_df = ninety_fifth_df.drop(['Unnamed: 0'], axis=1)
mean_df         = mean_df.drop(['Unnamed: 0'], axis=1)
fifth_df        = fifth_df.drop(['Unnamed: 0'], axis=1)

# then later apply the strings as x tick labels. 
x_labels = ninety_fifth_df.columns.tolist()
x = np.arange(len(x_labels))

y = ninety_fifth_df.index
X,Y = np.meshgrid(x,y)

# get values for 95th percentile
Z_95th = ninety_fifth_df.values

# get values for the mean
Z_Mean = mean_df.values

# get values for 5th percentile
Z_5th = fifth_df.values


# =============================================================================
# Dana added this
# =============================================================================
'''
size = 5
Z_max_smooth = maximum_filter(Z, size=size)

Z_uniform_filter = uniform_filter(Z_max_smooth, size=size)

factor = 1
Zi = zoom(Z, factor) # Change this to change the filter type
Xi = zoom(X, factor)
Yi = zoom(Y, factor)

'''
# =============================================================================
# End of Dana added this
# =============================================================================

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

# Plot the 95th percentile
ax.plot_surface(X, Y, Z_95th, cmap='viridis', rcount=48, ccount=96) 

# plot the mean
# ax.plot_surface(X, Y, Z_Mean, cmap='viridis', rcount=48, ccount=96) 

# plot the 5the percentile
ax.plot_surface(X, Y, Z_5th, cmap='viridis', rcount=48, ccount=96) 

# plot the % error
# ax.plot_surface(X, Y, ((Z_95th - Z_5th) / Z_Mean)*100, cmap='viridis', rcount=100, ccount=96) 

# Plot the maximum line
# ax.plot(96, y, peak_load, color='r', linewidth=1)

# Flip the Y axis
ax.set_xlim(ax.get_xlim()[::-1])

# update the x ticks
tick_labels   = ninety_fifth_df.columns[::40]
ax.set_xticks(range(480)[::40])
ax.set_xticklabels(tick_labels , rotation=-45, ha='left')

# Labels and colorbar
# ax.set_xlabel('X axis')
ax.set_ylabel('Units')
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('Power [pu]', rotation=90)
# ax.set_zlim(0, 25)

# set the view
ax.view_init(elev=12, azim=67)
plt.title('HPWH ADMD')
plt.show()


# save the error results
if write_percent_error:
    percent_error = ((ninety_fifth_df - mean_df) / mean_df) * 100
    percent_error['max error'] = percent_error.max(axis=1)
    percent_error['min error'] = percent_error.min(axis=1)
    percent_error.to_csv("C:/Users/Joe_admin/Documents/Thesis PGE/HEMS/OCHRE_HPWH_95_mean_perc_error.csv")


# print out the time it took to run the program
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
execution_min = execution_time/60
print(f"Execution time: {execution_min} minutes")
