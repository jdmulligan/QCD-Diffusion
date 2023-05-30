#!/usr/bin/env python
'''
This module contains functions to read and write data structures relevant for the analysis
'''

import os
import numpy as np
from silx.io.dictdump import dicttoh5, h5todict
from collections import defaultdict

#---------------------------------------------------------------
# Write nested dictionary of ndarray to hdf5 file
# Note: all keys should be strings
#---------------------------------------------------------------
def write_data(results, output_dir, filename = 'results.h5'):
    print()

    print(f'Writing results to {output_dir}/{filename}...')
    dicttoh5(results, os.path.join(output_dir, filename), overwrite_data=True)

    print('done.')
    print()

#---------------------------------------------------------------
# Read dictionary of ndarrays from hdf5
# Note: all keys should be strings
#---------------------------------------------------------------
def read_data(input_file):
    print()
    print(f'Loading results from {input_file}...')

    results = h5todict(input_file)

    print('done.')
    print()

    return results
    
#---------------------------------------------------------------
# Create a nested defaultdict
#---------------------------------------------------------------
def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)

#---------------------------------------------------------------
# Transform list of event dictionaries to dictionary of ndarrays
#
# The per-event dictionaries are structured as:
#    event[f'jet__{jetR}__{observable}']      # n-dimensional numpy array      
# 
# We transform these to:
#   results[f'jet__{jetR}__{observable}']     # (n+1)d numpy array, where the first dimension is the event index
#
# Note that the number of particles per jet are zero-padded.
#---------------------------------------------------------------
def event_list_to_results_dict(event_output_list): 

    results = {}
    event0 = event_output_list[0]

    # Loop over observables
    for key in event0.keys():

        # Create list of numpy arrays for each key 
        list_of_arrays = [event[key] for event in event_output_list] 

        # Then convert to final ndarray, such that the first dimension is the event index
        results[key] = np.stack(list_of_arrays, axis=0)

    return results