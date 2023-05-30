#!/usr/bin/env python3

"""
Plot histograms
"""

import os
import yaml
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

import data_IO

#---------------------------------------------------------------
# Main processing function
#---------------------------------------------------------------
def plot_results(config_file, output_dir):

    # Load training data
    training_data_path = os.path.join(output_dir, 'training_data.h5')
    results = data_IO.read_data(training_data_path)

    print('The following observables are available to plot:')
    for key in results.keys():
        print(f'  {key}')
    print()

    # Initialize info
    with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
    jetR = config['jetR']

    #------------------------
    # Plot pt of jets
    x = results[f'jet__{jetR}__jet_pt'][:,0].flatten()
    bins = np.linspace(0, 50, 50)
    plot_histogram_1d(x_list=[x], label_list=['hadrons'],
                        bins=bins, logy=True,
                        xlabel='pt,jet', ylabel='dN/dpt,jet',
                        filename = f'jet_pt_R{jetR}.pdf', output_dir=output_dir)
    
    #--------
    # Plot multiplicity of particles in jet
    x = results[f'jet__{jetR}__four_vector'][:,:,0].flatten()
    x = x[x>0.0]
    bins = np.linspace(0, 200, 200)
    plot_histogram_1d(x_list=[x], label_list=['hadrons'],
                            bins=bins, logy=True,
                            xlabel='multiplicity', ylabel='N jets',
                            filename = f'multiplicity_R{jetR}.pdf', output_dir=output_dir)
    
    #--------
    # ...

#---------------------------------------------------------------
# Function to plot 1D histograms
#---------------------------------------------------------------
def plot_histogram_1d(x_list=[], label_list=[],
                      bins=np.array([]), logy=False,
                      xlabel='', ylabel='', xfontsize=12, yfontsize=16, 
                      filename='', output_dir=''):

    if not bins.any():
        bins = np.linspace(np.amin(x_list[0]), np.amax(x_list[0]), 50)

    for i,x in enumerate(x_list):
        plt.hist(x,
                    bins,
                    histtype='step',
                    density=False,
                    label = label_list[i],
                    linewidth=2,
                    linestyle='-',
                    alpha=0.5,
                    log=logy)
    
    plt.legend(loc='best', fontsize=10, frameon=False)

    plt.xlabel(xlabel, fontsize=xfontsize)
    plt.ylabel(ylabel, fontsize=yfontsize)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()