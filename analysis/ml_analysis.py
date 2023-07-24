#!/usr/bin/env python

import yaml
import os

import common_base
import plot_results

# Pytorch
import torch

####################################################################################################################
class MLAnalysis(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file='', output_dir='', **kwargs):
        super(MLAnalysis, self).__init__(config_file=config_file, output_dir=output_dir, **kwargs)

        self.output_dir = output_dir

        self.initialize_config(config_file)

        # Set torch device
        os.environ['TORCH'] = torch.__version__
        print()
        print(f'pytorch version: {torch.__version__}')
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.torch_device)
        if self.torch_device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        print()

        print(self)

    #---------------------------------------------------------------
    # Initialize config
    #---------------------------------------------------------------
    def initialize_config(self, config_file):

        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.n_events = config['n_events']
        self.jetR = config['jetR']
        self.n_train = config['n_train']
        self.n_val = config['n_val']
        self.n_test = config['n_test']
        self.n_total = self.n_train + self.n_val + self.n_test
        self.test_frac = 1. * self.n_test / self.n_total
        self.val_frac = 1. * self.n_val / (self.n_train + self.n_val)

    #---------------------------------------------------------------
    # Do ML analysis and run plotting script
    #---------------------------------------------------------------
    def do_analysis(self, results):

        # Loop over jet radii
        print()
       # print('Implement ML analysis here...')
       # print()
       # print(f'The results dictionary contains the following keys: {results.keys()}')
       # for key in results.keys():
       #     print(f'  {key}', type(results[key]),results[key].shape )
        #    print()
       # print('Leadingid',[results['leadingparticle_id'][i] for i in range(10)] )
       # print('Is quark', [results['leadingparticle_isquark'][i] for i in range(10)])
       # print( 'Is gluon', [results['leadingparticle_isgluon'][i] for i in range(10)] )
       # print('leading pt')
       # print([results['leadingparticle_pt'][i] for i in range(10)])
       # print('parton jet pt')
       # print([results['jet__0.4__partonjet_pt'][i] for i in range(10)])

