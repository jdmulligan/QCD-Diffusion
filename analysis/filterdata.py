import sys
import os
import yaml
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})
import os
import numpy as np
from silx.io.dictdump import dicttoh5, h5todict
from collections import defaultdict
import data_IO
##Data cutoffs
jetR = 0.4
#Matching condition dR<R/2
#Flavor uds only
min_pt = 35
min_numparticles = 2




#---------------------------------------------------------------
# Main processing function
#---------------------------------------------------------------
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
   # print(results['output_file1'][key][0])
    print()
#
fresults = {}
num_events = results['jet_dR'].shape[0]
#Conditions
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
#Get good indexes and fill fresults
findex =[a for a in range(num_events) if (Matched(a) and Flavor(a) and Energy(a) and Num(a))]

for key in results.keys():
    fresults[key] = np.array([results[key][i] for i in findex])
for key in fresults.keys():
    print(f'  {key}', type(fresults[key]), fresults[key].shape )
    print()

# Center and scale
def center_and_scale(x, jetR):
    # If the jet is spread across the 0/2pi boundary, shift one side before centering
    phi_values = x[:,2]
    if np.min(phi_values) < np.pi - 2*jetR and np.max(phi_values) > np.pi + 2*jetR:
        phi_shifted = np.subtract(x[:,2], 2*np.pi)
        x[:,2] = np.where(x[:,2] < np.pi + 2*jetR, x[:,2], phi_shifted)

    # Center and scale
    yphi_avg = np.average(x[:,1:3], weights=x[:,0], axis=0)
    x[:,1:3] -= yphi_avg
    x[:,0] /= x[:,0].sum()



#Count
light_q = [i for i in range(num_events) if Flavor(i)]
other_q = [j for j in range(num_events) if (results['leadingparticle_isquark'][j] and (not Flavor(j)))]
matched =  [i for i in range(num_events) if Matched(i)]
energy =  [i for i in range(num_events) if Energy(i)]
num  = [i for i in range(num_events) if Num(i)]
quark_index = [i for i in range(results['jet_dR'].shape[0]) if results['leadingparticle_isquark'][i] ]
print('You filtered out ', 100-len(findex)/num_events * 100 , '%' +' of the data you left ', fresults['jet_dR'].shape[0], ' events')
print()
print('Total number of quark events: ', len(quark_index))
print('Number of light quark events (uds): ', len(light_q), 'which is ', len(light_q)/len(quark_index)*100 , '%')
print('Number of heavy quark events: ', len(other_q), 'which is ', len(other_q)/len(quark_index)*100 , '%')
print()
print('The data with pt (parton)>= 35 Gev is: ', len(energy)/num_events*100 , '% ','of the total data')
print()
print('The data with match angles (dR<R/2) is: ', len(matched)/num_events*100 , '% ','of the total data')
print()
print('The data with at least two particles (parton level) is: ', len(num)/num_events*100 , '% ','of the total data')

for j in range(fresults['jet_dR'].shape[0]):

    x_parton = fresults[f'jet__{jetR}__partonfour_vector'][j]
    x_hadron = fresults[f'jet__{jetR}__hadronfour_vector'][j]
    center_and_scale(x_parton,jetR)
    center_and_scale(x_hadron,jetR)

print('I centered jets')
#Save
print(f'Writing results to {output_dir}/{filename}...')
dicttoh5(fresults, os.path.join(output_dir, filename), overwrite_data=True)
print('All done.')