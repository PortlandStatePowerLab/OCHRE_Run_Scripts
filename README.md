## OCHRE Testing Scripts Using CTA-2045 HPWH Settings:
- These scripts can be used to simulate HPWHs in residential buildings.
- These scripts were developed to generate a CTA-2045 Implementation Guide for residential HPWHs.

### Requirements:

- OCHRE

### Working Scripts:
- C5_Control_Sweep.py
    - This script acts as an _Energy_ grid service, using PGE's commercial ToU schedule, or using user input.
    - You can generate a list of deadband sweeps and it will aggregate results into a parquet file.
    - Users can set Tset, Tset_DB, Tshed, Tshed_DB, Loadup_DB, and mode durations. 
    - This script coordinates loads to avoid impulse-like cold-load-pickups.
    - Coordination is currently randomized using weighted start-times/bins and time jitter.
 
- C6_LoadupSweep.py
    - This script performs a very long LoadUp or Advanced LoadUp and sweeps through deadbands.
    - You can generate a list of deadbands to iterate through and it will aggregate results by deadband and one with all data into a parquet file.
    - This script requires updated WaterHeater.py and Water.py files to avoid crashing simulation and missing house data. 

- C7_longShed_Sweep.py
    - This script sweeps _Shed_-type temperature deadbands in a loop and aggregates data into one parquet file.
    - This is not analogous to an _Energy_ grid service. It is an extended _Shed_ period from 6am - midnight.
    - This script was used to stress-test configurations to measure approximate cumulative energy shifted at hour t.

- C8_Blackstart.py
    - This script simulates a _Blackstart Support_ grid service. 
    - WHs will have a very high cold-load-pickup and need to be carefully energized over a long period of time.
    - This script assumes a very long _Grid Service_ event, and energizes over many hours by staggering _Normal_ modes.
    - Staggered energization is randomized using weighted start-times/bins and time jitter.

- C9_Efficiency_Sweep.py
    - This script will sweep through _Efficiency Levels_ 1-9. 
    - Users can set _efficiency coefficients_ in the LVL library. 
    - _Efficiency coefficients_ are used to change the HP-operating deadband of the HPWH
    - HP deadband:
        - Upper temperature threshold: Tset
        - Lower temperature threshold: Tset - Tset_DB * efficiency_coefficient
    - As the _efficiency coefficient_ increases, the HP operating deadband widens, making the unit more efficient.