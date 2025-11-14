## OCHRE Testing Scripts Using CTA-2045 HPWH Settings:
- These scripts can be used to simulate HPWHs in residential buildings. 

### Requirements:

- OCHRE

### Working Scripts:
- C4_Sweep_Test.py
    - This script acts as an _Energy_ grid service, using PGE's commercial ToU schedule.
    - You can generate a list of deadband sweeps and it will aggregate results as a parquet file.
- C6_Sweep_longShed.py
    - This script sweeps _Shed_ temperature deadbands in a loop and aggregates data into one parquet file.
    - This is not analogous to an _Energy_ grid service. It is an extended _Shed_ period from 6am - midnight.



