## OCHRE Testing Scripts Using CTA-2045 HPWH Settings:
- These scripts can be used to simulate HPWHs in residential buildings.
- These scripts were developed to generate a CTA-2045 Implementation Guide for residential HPWHs.

### Requirements:

- OCHRE

### Working Scripts:
- C4_Sweep_Test.py
    - This script acts as an _Energy_ grid service, using PGE's commercial ToU schedule.
    - You can generate a list of deadband sweeps and it will aggregate results as a parquet file.
- C7_Sweep_longShedv2.py
    - This script sweeps _Shed_ temperature deadbands in a loop and aggregates data into one parquet file.
    - This is not analogous to an _Energy_ grid service. It is an extended _Shed_ period from 6am - midnight.
 
- E0_LoadUpSweep
    - This script performs a very long LoadUp or Advanced LoadUp and sweeps through deadbands.
    - You can generate a list of deadbands to iterate through and it will aggregate results by deadband and one with all data as a parquet file.
    - This script requires updated WaterHeater.py and Water.py files to avoid crashing simulation and missing house data. 



