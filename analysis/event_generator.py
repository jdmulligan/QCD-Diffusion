#!/usr/bin/env python
'''
Class to generate events with PYTHIA
'''

import pandas as pd
import tqdm

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
    def __init__(self, **kwargs):
        super(EventGenerator, self).__init__(**kwargs)

        self.initialize_pythia()

        print()
        print(self)

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

        # Print progress bar (after fj banner)
      #  print()
        fj.ClusterSequence.print_banner()
       # print()
       # print('Generating events...')
        pbar = tqdm.tqdm(range(n_events))

        # Event loop
        for i_event in range(n_events):
            pbar.update()

            yield self.next_event(i_event)

            # Print pythia stats for last event
            if i_event == n_events-1:
                self.pythia.stat()

    #---------------------------------------------------------------
    # Generate an event
    #---------------------------------------------------------------
    def next_event(self, i_event):

        # Get next event
        if not self.pythia.next():
            return

        # Select partons
        partons = pythiafjext.vectorize_select(self.pythia, [pythiafjext.kFinal], 0, True)

        # Hadronize
        hstatus = self.pythia.forceHadronLevel()
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