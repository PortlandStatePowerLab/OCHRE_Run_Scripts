## OCHRE Testing Scripts Using CTA-2045 HPWH Settings:
- These scripts can be used to simulate HPWHs in residential buildings.
- These scripts were developed to generate a CTA-2045 Implementation Guide for residential HPWHs.

### Requirements:

- OCHRE

### Working Multi-Home Simulation Run Scripts (Dana):
1. C5_Control_Sweep.py
    - This script can run as an uncontrolled baseline, where HPWHs are acting as they normally would. 
        - To do this, you set Tset = Tshed = Tloadup and each of their deadbands to your baseline deadband width.
    - This script acts as an _Energy_ grid service, using PGE's commercial ToU schedule, or using user input.
    - You can generate a list of deadband sweeps and it will aggregate results into a parquet file.
    - Users can set Tset, Tset_DB, Tshed, Tshed_DB, Loadup_DB, and mode durations. 
    - This script coordinates loads to avoid impulse-like cold-load-pickups.
    - Coordination is currently randomized using weighted start-times/bins and time jitter.
 
2. C6_LoadupSweep.py
    - This script performs a very long LoadUp or Advanced LoadUp and sweeps through deadbands.
    - You can generate a list of deadbands to iterate through and it will aggregate results by deadband and one with all data into a parquet file.
    - This script requires updated WaterHeater.py and Water.py files to avoid crashing simulation and missing house data. 

3. C7_longShed_Sweep.py
    - This script sweeps _Shed_-type temperature deadbands in a loop and aggregates data into one parquet file.
    - This is not analogous to an _Energy_ grid service. It is an extended _Shed_ period from 6am - midnight.
    - This script was used to stress-test configurations to measure approximate cumulative energy shifted at hour t.

4. C8_Blackstart.py
    - This script simulates a _Blackstart Support_ grid service. 
    - WHs will have a very high cold-load-pickup and need to be carefully energized over a long period of time.
    - This script assumes a very long _Grid Service_ event, and energizes over many hours by staggering _Normal_ modes.
    - Staggered energization is randomized using weighted start-times/bins and time jitter.

5. C9_Efficiency_Sweep.py
    - This script will sweep through _Efficiency Levels_ 1-9. 
    - Users can set _efficiency coefficients_ in the LVL library. 
    - _Efficiency coefficients_ are used to change the HP-operating deadband of the HPWH
    - HP deadband:
        - Upper temperature threshold: Tset
        - Lower temperature threshold: Tset - Tset_DB * efficiency_coefficient
    - As the _efficiency coefficient_ increases, the HP operating deadband widens, making the unit more efficient.

6. C10_Reserve.py
    - A _Reserve_ service can be simulated by asking a small percentage of aggregated loads to anticipate...the unaticipated! 
    - Most loads will run as normal for the full simulation period. 
    - Some loads will consume (_Load Up, ALU_) during a midday, hypothetical, overgeneration scenario. 
    - This code needs work to better coordinate...

### ADMD scripts (Joe), Results plotter (Dana)

These scripts are used to run a Monte Carlo Simulation for any aggregated load CSV file (can be changed to parquet).
    - D0_Parse_Ochre_data.py
    - D1_Get_ADMD_data.py
    - D2_Plot_3D_ADMD.py or
    - D3_Plot_3D_ADMD_LtdTime.py or
    - D3_Plot_MCS_N.py or
    - D3_Plot_Time_N.py


1. Upload original results CSV file to D0
2. Upload results of D0 and run it in D1. This will take a while (10s of minutes) depending on Unit Runs (N-loads) and MCS Runs (currently set to 1000).
3. The resultant file can be used in any of the D2/D3 files for plotting the results. Plotting files could use some work to make data input less hardcode-y.


### Plotting scripts (Dana)

These are some plotting scripts I used to make figures for reports/thesis. 

1. X1_PlotAggregated
2. X2_PlotViolin
3. X3_3DPlot_95CI
4. X4_analyzeDB
5. X5_singleHouse_withEfficiency


