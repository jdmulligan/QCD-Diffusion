#! /usr/bin/env python
'''
Main script to steer event generation, jet finding, and ML training
'''

import argparse
import os
import sys
import yaml
import time

import data_IO
import event_generator
import event_analyzer
import ml_analysis
import plot_results

import common_base

####################################################################################################################
class SteerAnalysis(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, generate=False, write=False, read=False, analyze=False, inspect=False,
                 input_file='', config_file='', output_dir='', output_file='', **kwargs):

        self.generate = generate
        self.write = write
        self.read = read
        self.analyze = analyze
        self.input_file = input_file
        self.config_file = config_file
        self.output_dir = output_dir
        self.output_file = output_file

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.initialize(config_file)

        print(self)

    #---------------------------------------------------------------
    # Initialize config
    #---------------------------------------------------------------
    def initialize(self, config_file):
        print('Initializing class objects')

        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.n_events = config['n_events']

    #---------------------------------------------------------------
    # Main function
    #---------------------------------------------------------------
    def run_analysis(self):

        # Generate events and construct input of ndarrays for ML
        if self.generate:

            # Generate events
            generator = event_generator.EventGenerator()
            analyzer = event_analyzer.EventAnalyzer(self.config_file)

            event_output_list = []
            for event in generator(self.n_events):

                # Analyze event -- compute observables and construct dictionary of output arrays for each event
                event_output = analyzer.analyze_event(event)
                event_output_list.append(event_output)
            print('Succesfull events: ', len(event_output_list) )
            # Construct dictionary of ndarrays from list of event analysis output, and write them to hdf5 file
            results = data_IO.event_list_to_results_dict(event_output_list)
            print(results.keys)
            if self.write:
                data_IO.write_data(results, self.output_dir, filename=self.output_file)

        # If generation is disabled, can read in ndarrays from hdf5 file
        elif self.read:
            results = data_IO.read_data(self.input_file)

        # Perform ML analysis
        if self.analyze:
            analysis = ml_analysis.MLAnalysis(self.config_file, self.output_dir)
            analysis.do_analysis(results)

            # Run plotting script
            print()
            print('Run plotting script here...')
            plot_results.plot_results(self.config_file, self.output_dir, self.output_file)

####################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Hadronization Analysis')
    parser.add_argument('--generate', 
                        help='generate events and do jet finding', 
                        action='store_true', default=False)
    parser.add_argument('--write', 
                        help='write ndarrays of jets/events to hdf5', 
                        action='store_true', default=False)
    parser.add_argument('--read', 
                        help='Path to hdf5 file with ndarrays for ML input. Specify only if --generate is not enabled.',
                        action='store', type=str,
                        default='', )
    parser.add_argument('--analyze', 
                        help='perform ML analysis', 
                        action='store_true', default=False)
    parser.add_argument('-c', '--configFile', 
                        help='Path of config file for analysis',
                        action='store', type=str,
                        default='config/config.yaml', )
    parser.add_argument('-o', '--outputDir',
                        help='Output directory for output to be written to',
                        action='store', type=str,
                        default='./Output', )
    parser.add_argument('-f', '--outputFile',
                        help='Output filename for hdf5',
                        action='store', type=str,
                        default='good_events_merged.h5', )
    args = parser.parse_args()

    print('Configuring...')
    print(f'  configFile: {args.configFile}')
    print(f'  ouputDir: {args.outputDir}')

    # If invalid inputFile is given, exit
    if args.read:
        print(f'  read input from: {args.read}')
        if not os.path.exists(args.read):
            print(f'File {args.read} does not exist! Exiting!')
            sys.exit(0)
    else:
        input_file = None

    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        print(f'File {args.configFile} does not exist! Exiting!')
        sys.exit(0)

    start_time = time.time()

    analysis = SteerAnalysis(generate=args.generate, write=args.write, read=args.read, analyze=args.analyze,
                             input_file=args.read, config_file=args.configFile, output_dir=args.outputDir, output_file=args.outputFile)
    analysis.run_analysis()

    print('--- {} minutes ---'.format((time.time() - start_time)/60.))