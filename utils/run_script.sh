#!/bin/bash

for fileno in $(seq 1 1 10)
do
    echo ${fileno}
    
   # python analysis/steer_analysis.py --generate --write -o output_${fileno} -f output_file.h5 &
    #python analysis/steer_analysis.py --read ./Output/output_file${fileno}.h5 --analyze -o Output -f output_file${fileno}.h5 &
    sleep 5
done
