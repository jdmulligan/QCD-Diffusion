'''
Qukck script to do additional filtering of the jet sample
'''

import os
import numpy as np
import os
import numpy as np
from silx.io.dictdump import dicttoh5
import data_IO

jetR = 0.4
min_pt = 35
min_numparticles = 2
output_dir = 'Output'
output_file = 'goodmerged.h5'
filename = 'goodandfiltered.h5'

# Load training data
training_data_path = os.path.join(output_dir, output_file)
results = data_IO.read_data(training_data_path)
print('I read')
print(type(results))
print('The following observables are available to plot:')
for key in results.keys():
    print(f'  {key}', type(results[key]), results[key].shape )
    print()

fresults = {}
num_events = results['jet_dR'].shape[0]

# Filtering conditions
def Matched(i):
    if results['jet_dR'][i]<= jetR/2:
        return True
    else:
        return False
def Flavor(i):
    id = results['leadingparticle_id'][i]
    if (id == 1 or id == -1 or id == 2 or id == -2 or id == 3 or id == -3):
        return True
    else:
        return False
def Energy(i):
    if min_pt <= results['jet__0.4__partonjet_pt'][i]:
        return True
    else:
        return False
def Num(i):
    if min_numparticles <= results['jet__0.4__partonnumparticlesperjet'][i]:
        return True
    else:
        return False

# Get good indexes and fill fresults
findex = [a for a in range(num_events) if (Matched(a) and Flavor(a) and Energy(a) and Num(a))]

for key in results.keys():
    fresults[key] = np.array([results[key][i] for i in findex])
for key in fresults.keys():
    print(f'  {key}', type(fresults[key]), fresults[key].shape )
    print()

# Count
light_q = [i for i in range(num_events) if Flavor(i)]
other_q = [j for j in range(num_events) if (results['leadingparticle_isquark'][j] and (not Flavor(j)))]
charm_q = [j for j in range(num_events) if (results['leadingparticle_id'][j] == 4 or results['leadingparticle_id'][j] == -4)]
botom_q = [j for j in range(num_events) if (results['leadingparticle_id'][j] == 5 or results['leadingparticle_id'][j] == -5)]
top_q = [j for j in range(num_events) if (results['leadingparticle_id'][j] == 6 or results['leadingparticle_id'][j] == -6)]
matched =  [i for i in range(num_events) if Matched(i)]
energy =  [i for i in range(num_events) if Energy(i)]
num  = [i for i in range(num_events) if Num(i)]
quark_index = [i for i in range(results['jet_dR'].shape[0]) if results['leadingparticle_isquark'][i] ]
print('You filtered out ', 100-len(findex)/num_events * 100 , '%' +' of the data you left ', fresults['jet_dR'].shape[0], ' events')
print()
print('Total number of quark events: ', len(quark_index))
print('Number of light quark events (uds): ', len(light_q), 'which is ', len(light_q)/len(quark_index)*100 , '%')
print('Number of heavy quark events: ', len(other_q), 'which is ', len(other_q)/len(quark_index)*100 , '%')
print('Number of charm quark events: ', len(charm_q), 'which is ', len(charm_q)/len(other_q)*100 , '%', 'of the heavy quark events')
print('Number of bottom quark events: ', len(botom_q), 'which is ', len(botom_q)/len(other_q)*100 , '%', 'of the heavy quark events')
print('Number of top quark events: ', len(top_q), 'which is ', len(top_q)/len(other_q)*100 , '%', 'of the heavy quark events')
print()
print('The data with pt (parton)>= 35 Gev is: ', len(energy)/num_events*100 , '% ','of the total data')
print()
print('The data with match angles (dR<R/2) is: ', len(matched)/num_events*100 , '% ','of the total data')
print()
print('The data with at least two particles (parton level) is: ', len(num)/num_events*100 , '% ','of the total data')

# Save
print(f'Writing results to {output_dir}/{filename}...')
dicttoh5(fresults, os.path.join(output_dir, filename), overwrite_data=True)
print('All done.')