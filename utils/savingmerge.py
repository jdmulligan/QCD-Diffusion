'''
Simple script to merge multiple output files into one
'''

import os
import numpy as np
from silx.io.dictdump import dicttoh5
import data_IO

output_dir = 'Output'
output_file = 'out_merged.h5'
filename = 'goodmerged.h5'

# Load training data
training_data_path = os.path.join(output_dir, output_file)
results = data_IO.read_data(training_data_path)
print(type(results))
print('The following observables are available to plot:')
for key in results['output_file1'].keys():
    print(f'  {key}', type(results['output_file1'][key]), results['output_file1'][key].shape )
    print()

mergeresults = {}
for skey in results['output_file1'].keys():
    mergeresults[skey] =  np.concatenate((results['output_file1'][skey], results['output_file2'][skey],
                                          results['output_file3'][skey],results['output_file4'][skey],
                                          results['output_file5'][skey],results['output_file6'][skey],
                                          results['output_file7'][skey],results['output_file8'][skey],
                                          results['output_file9'][skey],results['output_file10'][skey],
                                          ),axis=0)

for key in mergeresults.keys():
    print(f'  {key}', type(mergeresults[key]), mergeresults[key].shape )
print()
    
print(f'Writing results to {output_dir}/{filename}...')
dicttoh5(mergeresults, os.path.join(output_dir, filename), overwrite_data=True)
print('done.')
    