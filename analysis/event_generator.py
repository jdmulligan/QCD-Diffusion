#!/usr/bin/env python
'''
Class to generate events with PYTHIA
'''

import numpy as np
import pandas as pd
import tqdm
import yaml
import sys

from heppy.pythiautils import configuration as pyconf
import fastjet as fj
import pythia8
import pythiafjext
import pythiaext

import common_base
import random

####################################################################################################################
class EventGenerator(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file='', **kwargs):
        super(EventGenerator, self).__init__(config_file=config_file, **kwargs)

        self.initialize_config(config_file)

        self.initialize_pythia()

        print()
        print(self)

    #---------------------------------------------------------------
    # Initialize config
    #---------------------------------------------------------------
    def initialize_config(self, config_file):

        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.parton_event_type = config['parton_event_type']

    #---------------------------------------------------------------
    # Initialize pythia
    #---------------------------------------------------------------
    def initialize_pythia(self):

        # Initialize PYTHIA for hadronic events LEP1 - main06.cc
        mZ = 91.188
        mycfg = ["PDF:lepton = off", # Allow no substructure in e+- beams: normal for corrected LEP data.
                 "WeakSingleBoson:ffbar2gmZ = on", # Process selection.
                 "23:onMode = off", # Switch off all Z0 decays and then switch back on those to quarks.
                 "23:onIfAny = 1 2 3 4 5", 
                 "Beams:idA =  11", 
                 "Beams:idB = -11", 
                 f"Beams:eCM={mZ}", # LEP1 initialization at Z0 mass.
                 "HadronLevel:all=off", # parton level first
                 "PhaseSpace:bias2Selection=off", # this is ON by default in pyconf - not OK for these settings
                 "Random:setSeed=on",
                 "Random:seed={}".format(random.randint(100000, 900000))] 

        self.pythia = pyconf.create_and_init_pythia(mycfg)
        if not self.pythia:
            return

    #---------------------------------------------------------------
    # Generator (in the pythonic sense) to loop over all events
    #---------------------------------------------------------------
    def __call__(self, n_events):

        fj.ClusterSequence.print_banner()
        pbar = tqdm.tqdm(range(n_events))

        # Event loop 
        #   - Generate a single parton-level event
        #   - Decide whether to accept it -- if not, try a new one
        #   - Hadronize the parton-level event repeatedly for n_event times
        accept_event = False
        while not accept_event:
        
            # Get next event
            if not self.pythia.next():
                return

            # Save the event, in case we want to hadronize the same parton-level even multiple times
            saved_event = pythia8.Event(self.pythia.event)

            # Select partons
            partons = pythiafjext.vectorize_select(self.pythia, [pythiafjext.kFinal], 0, True)
            accept_event = self.accept_parton_level_event()
            print(f'number of final-state partons in event: {len(partons)}')
            print(f'accept_parton_level_event: {accept_event}')
            print()

        # Loop through n_events, and independently hadronize the same parton-level event 
        print(f'Found suitable parton-level event. Hadronizing {n_events} times...')
        for i_event in range(n_events):
            pbar.update()

            yield self.next_event(i_event, saved_event, partons)

            # Print pythia stats for last event
            if i_event == n_events-1:
                self.pythia.stat()

    #---------------------------------------------------------------
    # Generate an event
    #---------------------------------------------------------------
    def accept_parton_level_event(self):

        # Check the outgoing particles from the hard process
        # Z boson is status -22
        # Decay products (q-qbar or neutrinos, typically) are status 23
        # Let's select the quark-antiquark pairs, for either light quarks or b quarks

        # Get the PID of the outgoing hard particles
        outgoing_hard_particles = [np.abs(p.id()) for p in self.pythia.process if p.status() == 23]
        print(f'outgoing_hard_particle IDs: {outgoing_hard_particles}')
        if len(outgoing_hard_particles) != 2:
            return False

        # Let's select the quark-antiquark pairs, for either light quarks or b quarks
        if self.parton_event_type == 'b':
            reference_set = {5}
        elif self.parton_event_type == 'uds':
            reference_set =  {1,2,3}
        else:
            sys.exit("ERROR: parton_event_type must be b or uds")

        return set(outgoing_hard_particles) == reference_set

    #---------------------------------------------------------------
    # Generate an event
    #---------------------------------------------------------------
    def next_event(self, i_event, saved_event, partons):

        # Hadronize the saved event
        self.pythia.event = saved_event
        #print('preH:', saved_event.size(), self.pythia.event.size())
        hstatus = self.pythia.forceHadronLevel()
        #print('postH:', saved_event.size(), self.pythia.event.size())
        if not hstatus:
            print(f'WARNING: forceHadronLevel false event: {i_event}')
            return

        # Select hadrons
        hadrons = pythiafjext.vectorize_select(self.pythia, [pythiafjext.kFinal], 0, True)

        # Write particles to a dataframe
        parton_df = self.fill_particle_info(partons, is_parton=True)
        hadron_df = self.fill_particle_info(hadrons, is_parton=False)
        event_dataframe = pd.concat([parton_df, hadron_df], ignore_index=True)
        
        # print()
        return event_dataframe

    #---------------------------------------------------------------
    # Create dataframe of particle info for list of particles
    #---------------------------------------------------------------
    def fill_particle_info(self, particles, is_parton=False): 

        row_list = []
        for p in particles:
            particle_dict = {
                'particle_pt': p.perp(),
                'particle_eta': p.eta(),
                'particle_phi': p.phi(), 
                'particle_m': p.m(),
                'particle_id': pythiafjext.getPythia8Particle(p).id(),
                'is_parton': is_parton,
                'is_gluon': pythiafjext.getPythia8Particle(p).isGluon(),
                'is_quark': pythiafjext.getPythia8Particle(p).isQuark()
            }
            row_list.append(particle_dict)
            
        return pd.DataFrame(row_list)