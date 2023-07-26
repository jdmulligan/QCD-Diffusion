#!/usr/bin/env python
'''
Class to analyze PYTHIA events (do jet finding, etc.)
'''

import sys
import yaml
import numpy as np
from collections import defaultdict
import fastjet as fj
import pythia8
import pythiafjext
import pythiaext
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
        self.image_dims = config['image_dims']

    #---------------------------------------------------------------
    # Analyze event -- return a dictionary of numpy arrays
    #---------------------------------------------------------------
    def analyze_event(self, df_event):

        # Store all output as dictionary of ndarrays
        self.event_output = {}

        # Filter the event dataframe for hadrons (rather than partons)
        df_event_hadron = df_event[df_event['is_parton']==False]
        df_event_parton = df_event[df_event['is_parton']==True]
        # Do jet finding and compute observables
        self.analyze_jets(df_event_hadron, df_event_parton)
        return self.event_output
    
    #---------------------------------------------------------------
    # Analyze jets -- find jets and compute observables
    #---------------------------------------------------------------
    def analyze_jets(self, df_event_hadron, df_event_parton):

    # Convert four-vectors to fastjet::PseudoJets
        fj_particles_hadron = self.get_fjparticles(df_event_hadron)
        fj_particles_parton = self.get_fjparticles(df_event_parton)

    # Set jet definition and a jet selector
        jet_def = fj.JetDefinition(fj.antikt_algorithm, self.jetR)

    # Do jet finding (hadron level)
        jets_hadron = None
        cs_hadron = fj.ClusterSequence(fj_particles_hadron, jet_def)
        jets_hadron = fj.sorted_by_pt(cs_hadron.inclusive_jets())

    # Do jet finding (parton level)
        jets_parton = None
        cs_parton = fj.ClusterSequence(fj_particles_parton, jet_def)
        jets_parton = fj.sorted_by_pt(cs_parton.inclusive_jets())

    # Let's get the leading jet from hadron level (they are ordered by pt)
        jet_hadron = jets_hadron[0]

    # Let's get the leading jet from parton level (they are ordered by pt)
        jet_parton = jets_parton[0]

        #leading_particle = fj.sorted_by_pt(jet_parton.constituents())[0]
       
       # print('[i] leading particle index', leading_particle.user_index())
        #print('    ', df_event_parton['particle_id'][leading_particle.user_index()])
        #print('    ', df_event_parton['particle_pt'][leading_particle.user_index()], '=?=', leading_particle.perp())
    #ID
        self.Save_jetid(df_event_parton,jet_parton)
    #Compute observables
        self.compute_jet_observables(jet_hadron,'hadron')
        self.compute_jet_observables(jet_parton,'parton')
    #Save Alignment Check

        self.Save_jetcheck(jet_hadron,jet_parton)

    #Check alignment
    def matcheck(self,jh,jp):
        deta = jh.eta()-jp.eta()
        x = jp.phi_02pi()
        y = jh.phi_02pi()
        dphi = min(np.abs(x-y),min(x,y)-max(x,y)+2*np.pi)
        dR = np.sqrt(deta**2+dphi**2)
        return dR
    #---------------------------------------------------------------
    # Compute observables for a given jet and store in output dict
    # Each observable should be stored as a numpy array in the output dict
    #---------------------------------------------------------------
    def compute_jet_observables(self, jet, particle_type):

        # Define a convention for labeling the output observables
        key_prefix = f'jet__{self.jetR}__'

        # observables
        # (1) Jet constituent four-vectors (zero-padded such that each jet has n_particles_max_per_jet)
        constituents = []
        num_constituents = [0]
        for constituent in jet.constituents():
            constituents.append(np.array([constituent.pt(), constituent.eta(), constituent.phi_02pi(), constituent.m()])) 
            # constituent_ids.append(pythiafjext.getPythia8Particle(constituent).id())
            # print(constituent.user_index())
        constituents_zero_padded = self.zero_pad(np.array(constituents), self.n_particles_max_per_jet)
        self.event_output[f'{key_prefix}{particle_type}four_vector'] = constituents_zero_padded
       # for constituent in jet.constituents():
        #    constituentid.append(np.array([pythiafjext.getPythia8Particle(constituent).id(), constituent.eta(), constituent.phi_02pi(), constituent.m()])) 
        #constituents_zero_padded = self.zero_pad(np.array(constituents), self.n_particles_max_per_jet)
        #self.event_output[f'{key_prefix}{particle_type}four_vector'] = constituents_zero_padded

        for constituent in jet.constituents():
            num_constituents[0] = num_constituents[0] +1
        self.event_output[f'{key_prefix}{particle_type}numparticlesperjet'] = num_constituents
        # (2) Jet axis kinematics
        self.event_output[f'{key_prefix}{particle_type}jet_pt'] = np.array([jet.pt()])
        self.event_output[f'{key_prefix}{particle_type}jet_eta'] = np.array([jet.eta()])
        self.event_output[f'{key_prefix}{particle_type}jet_phi'] = np.array([jet.phi_02pi()])
        self.event_output[f'{key_prefix}{particle_type}jet_m'] = np.array([jet.m()])
    
        # (3) Store pixelized jet images
        for image_dim in self.image_dims:
            self.event_output[f'{key_prefix}{particle_type}__jet_image__{image_dim}'] = self.pixelize_jet(jet, image_dim)

    #---------------------------------------------------------------
    #
    #---------------------------------------------------------------   
    def Save_jetcheck(self,jeth,jetp):
        key_prefix = f'jet__{self.jetR}__'
        self.event_output['jet_dR'] = self.matcheck(jeth,jetp)
 
    #---------------------------------------------------------------
    #
    #---------------------------------------------------------------
    def Save_jetid(self,dfp,jp):
        key_prefix = 'leadingparticle'
        leading_particle = fj.sorted_by_pt(jp.constituents())[0]
        self.event_output[f'{key_prefix}_id'] = dfp['particle_id'][leading_particle.user_index()]
        self.event_output[f'{key_prefix}_pt'] = dfp['particle_pt'][leading_particle.user_index()]
        self.event_output[f'{key_prefix}_eta'] = dfp['particle_eta'][leading_particle.user_index()]
        self.event_output[f'{key_prefix}_phi'] = dfp['particle_phi'][leading_particle.user_index()]
        self.event_output[f'{key_prefix}_m'] = dfp['particle_m'][leading_particle.user_index()]
        self.event_output[f'{key_prefix}_isgluon'] = dfp['is_gluon'][leading_particle.user_index()]
        self.event_output[f'{key_prefix}_isquark'] = dfp['is_quark'][leading_particle.user_index()]

    #---------------------------------------------------------------
    # Create "image" with jet information representation of (centered) eta * phi with z as pixel intensity
    #---------------------------------------------------------------
    def pixelize_jet(self, jet, image_dim):

        jet_pt = jet.pt() 
        jet_eta = jet.eta()
        jet_phi = jet.phi_02pi()

        image = np.zeros((image_dim, image_dim))
        edges = np.linspace(-self.jetR, self.jetR, image_dim+1)
        for constituent in jet.constituents():
            constituent_z = constituent.pt() / jet_pt
            constituent_eta = constituent.eta()
            constituent_phi = constituent.phi_02pi()

            # Center jet
            delta_eta = constituent_eta - jet_eta
            delta_phi = constituent_phi - jet_phi
            if delta_phi > np.pi:
                delta_phi -= 2*np.pi
            elif delta_phi < -np.pi:
                delta_phi += 2*np.pi

            # Convert to pixel coordinates
            i_eta = np.digitize(delta_eta, edges) - 1
            i_phi = np.digitize(delta_phi, edges) - 1
            if i_eta < image_dim and i_phi < image_dim: # For simplicity, ignore particles with eta,phi outside R
                image[i_eta, i_phi] += constituent_z
            
        return image

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
                                                    df_particles['particle_m'],user_index_offset)
       
        return fj_particles
    
    