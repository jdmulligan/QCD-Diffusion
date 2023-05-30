#!/usr/bin/env python
'''
Class to analyze PYTHIA events (do jet finding, etc.)
'''

import sys
import yaml
import numpy as np
from collections import defaultdict

import fastjet as fj
import fjext

import common_base

####################################################################################################################
class EventAnalyzer(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file='', **kwargs):
        super(EventAnalyzer, self).__init__(config_file=config_file, **kwargs)

        self.initialize_config(config_file)

        print()
        print(self)

    #---------------------------------------------------------------
    # Initialize config
    #---------------------------------------------------------------
    def initialize_config(self, config_file):

        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.jetR = config['jetR']
        self.n_particles_max_per_jet = config['n_particles_max_per_jet']

    #---------------------------------------------------------------
    # Analyze event -- return a dictionary of numpy arrays
    #---------------------------------------------------------------
    def analyze_event(self, df_event):

        # Store all output as dictionary of ndarrays
        self.event_output = {}

        # Filter the event dataframe for hadrons (rather than partons)
        df_event_hadron = df_event[df_event['is_parton']==False]

        # Do jet finding and compute some jet observables
        self.analyze_jets(df_event_hadron)

        return self.event_output
    
    #---------------------------------------------------------------
    # Analyze jets -- find jets and compute observables
    #---------------------------------------------------------------
    def analyze_jets(self, df_event):

        # Convert four-vectors to fastjet::PseudoJets
        fj_particles = self.get_fjparticles(df_event)

        # Set jet definition and a jet selector
        jet_def = fj.JetDefinition(fj.antikt_algorithm, self.jetR)

        # Do jet finding (hadron level)
        jets = None
        cs = fj.ClusterSequence(fj_particles, jet_def)
        jets = fj.sorted_by_pt(cs.inclusive_jets())

        # Let's get the leading jet (they are ordered by pt)
        jet = jets[0]
        
        # Compute any information we want to save to file
        self.compute_jet_observables(jet)

    #---------------------------------------------------------------
    # Compute observables for a given jet and store in output dict
    # Each observable should be stored as a numpy array in the output dict
    #---------------------------------------------------------------
    def compute_jet_observables(self, jet):

        # Define a convention for labeling the output observables
        key_prefix = f'jet__{self.jetR}__'

        # (1) Jet constituent four-vectors (zero-padded such that each jet has n_particles_max_per_jet)
        constituents = []
        for constituent in jet.constituents():
            constituents.append(np.array([constituent.pt(), constituent.eta(), constituent.phi_02pi(), constituent.m()])) 
        constituents_zero_padded = self.zero_pad(np.array(constituents), self.n_particles_max_per_jet)
        self.event_output[f'{key_prefix}four_vector'] = constituents_zero_padded

        # (2) Jet axis kinematics
        self.event_output[f'{key_prefix}jet_pt'] = np.array([jet.pt()])
        self.event_output[f'{key_prefix}jet_eta'] = np.array([jet.eta()])
        self.event_output[f'{key_prefix}jet_phi'] = np.array([jet.phi_02pi()])
        self.event_output[f'{key_prefix}jet_m'] = np.array([jet.m()])

        # (3) ...

    #---------------------------------------------------------------
    # zero pad a 2D array of four-vectors
    #---------------------------------------------------------------
    def zero_pad(self, a, n_max):

        if len(a) > n_max or len(a) == 0:
            sys.exit(f'ERROR: particle list has {len(a)} entries before zero-padding')

        if len(a.shape) == 1:
            return np.pad(a, [(0, n_max-a.shape[0])])
        elif len(a.shape) == 2:
            return np.pad(a, [(0, n_max-a.shape[0]), (0, 0)])

    #---------------------------------------------------------------
    # Return fastjet:PseudoJets from a given track dataframe
    #---------------------------------------------------------------
    def get_fjparticles(self, df_particles):

        # Use swig'd function to create a vector of fastjet::PseudoJets from numpy arrays of pt,eta,phi
        user_index_offset = 0
        fj_particles = fjext.vectorize_pt_eta_phi_m(df_particles['particle_pt'].to_numpy(), 
                                                    df_particles['particle_eta'].to_numpy(),
                                                    df_particles['particle_phi'].to_numpy(), 
                                                    df_particles['particle_m'], 
                                                    user_index_offset)
        return fj_particles