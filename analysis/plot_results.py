#!/usr/bin/env python3

"""
Plot histograms
"""
import sys
import os
import yaml
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

import data_IO

bbins = 300
#---------------------------------------------------------------
# Main processing function
#---------------------------------------------------------------
def plot_results(config_file, output_dir, output_file):

    # Load training data
    training_data_path = os.path.join(output_dir, output_file)
    results = data_IO.read_data(training_data_path)
    print(type(results))
    print('The following observables are available to plot:')
    for key in results.keys():
        print(f'  {key}', type(results[key]), results[key].shape )
        print(results[key][0])
    print()
    
    # Initialize info

    with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
    jetR = config['jetR']

    #Choose Plot
    Stat = True
    Split = False
    Plot_dR = True
    Plot_pt = True
    Plot_phieta = True
    Plot_m = True
    Plot_jets = True
    Plot_num = True
    Plot_z = True
    Plot_2d = False
    Flavor_plots = False
    Plot_esplits = False

    #Print statistics
    if Stat:
        num_events = results['jet_dR'].shape[0]
        num_quarks = 0
        num_gluons = 0 
        num_photons = 0
        num_epm = 0
        num_muons = 0
        num_tau = 0
        for i in range(results['jet_dR'].shape[0]):
            if results['leadingparticle_isquark'][i]:
                num_quarks = num_quarks + 1
            elif results['leadingparticle_isgluon'][i]:
                num_gluons = num_gluons + 1
            elif results['leadingparticle_id'][i]== 22:
                num_photons = num_photons + 1
            elif results['leadingparticle_id'][i]== 11 or results['leadingparticle_id'][i]== -11:
                num_epm = num_epm + 1
            elif results['leadingparticle_id'][i]== 13 or results['leadingparticle_id'][i]== -13:
                num_muons = num_muons + 1
            elif results['leadingparticle_id'][i]== 15 or results['leadingparticle_id'][i]== -15:
                num_tau = num_tau + 1
            else:
                print('Rare id: ',results['leadingparticle_id'][i])
        print()
        print('Total number of events: ', num_events)
        print( 'Number of quark jets: ', num_quarks,'= ',num_quarks*100/num_events, '%')
        print('Number of gluon jets: ', num_gluons,'= ',num_gluons*100/num_events, '%')
        print( 'Number of photon jets: ', num_photons,'= ',num_photons*100/num_events, '%')
        print( 'Number of electon/positron jets: ', num_epm,'= ',num_epm*100/num_events, '%')
        print( 'Number of muon/antimuon jets: ', num_muons,'= ',num_muons*100/num_events, '%')
        print( 'Number of tau jets: ', num_tau,'= ',num_tau*100/num_events, '%')
        print()   
    if Split:
    #Separate dictionaries acording to energy
        index_highenergy = [i for i in range(results['jet_dR'].shape[0]) if 33.4< results[f'jet__{jetR}__partonjet_pt'][i][0]<= 50.1 ]
        index_mediumenergy = [i for i in range(results['jet_dR'].shape[0]) if 16.7 < results[f'jet__{jetR}__partonjet_pt'][i][0]<= 33.4 ]
        index_lowenergy = [i for i in range(results['jet_dR'].shape[0]) if 0 < results[f'jet__{jetR}__partonjet_pt'][i][0]<= 16.7 ]
        print('Total number of events: ',results['jet_dR'].shape[0],'High energy results: ',100*len(index_highenergy)/results['jet_dR'].shape[0],'%','Middle energy results: ',100*len(index_mediumenergy)/results['jet_dR'].shape[0],'%', 'Low energy results: ',100*len(index_lowenergy)/results['jet_dR'].shape[0],'%' )
        results_high ={}
        results_middle = {}
        results_low = {}
        for key in results.keys():
            results_high[key] = np.array([results[key][i] for i in index_highenergy])
        for key in results.keys():
            results_middle[key] = np.array([results[key][i] for i in index_mediumenergy])
        for key in results.keys():
            results_low[key] = np.array([results[key][i] for i in index_lowenergy])   
        for key in results_high.keys():
                print(key,f'  {key}', type(results_high[key]),results_high[key].shape )
                print()
        for key in results_middle.keys():
                print(key,f'  {key}', type(results_middle[key]),results_middle[key].shape )
                print()
        for key in results_low.keys():
                print(key,f'  {key}', type(results_low[key]),results_low[key].shape )
                print()
        label_splitp = ['High energy parton jets (33.4<pt <50.1 )', 'Mid energy parton jets (16.7<pt <33.4)', 'Low energy parton jets (pt <16.7)']
        label_splith = ['High energy hadron jets (33.4<pt <50.1 )', 'Mid energy hadron jets (16.7<pt <33.4)', 'Low energy hadron jets (pt <16.7)']
    #------------------------


    if Flavor_plots:
        quark_index = [i for i in range(results['jet_dR'].shape[0]) if results['leadingparticle_isquark'][i] ]
        gluon_index = [i for i in range(results['jet_dR'].shape[0]) if results['leadingparticle_isgluon'][i] ]
        photon_index = [i for i in range(results['jet_dR'].shape[0]) if results['leadingparticle_id'][i]== 22 ]
        electron_index = [i for i in range(results['jet_dR'].shape[0]) if (results['leadingparticle_id'][i]== 11 or results['leadingparticle_id'][i]== -11)]
        muon_index = [i for i in range(results['jet_dR'].shape[0]) if (results['leadingparticle_id'][i]== 13 or results['leadingparticle_id'][i]== -13)]
        tau_index = [i for i in range(results['jet_dR'].shape[0]) if (results['leadingparticle_id'][i]== 15 or results['leadingparticle_id'][i]== -15)]
        print('Total number of events: ',len(quark_index)+len(gluon_index)+len(photon_index)+len(electron_index)+len(muon_index)+len(tau_index),
              'Quark results: ',100*len(quark_index)/results['jet_dR'].shape[0],'%',
              'Gluon results: ',100*len(gluon_index)/results['jet_dR'].shape[0],'%', 
              'Photon results: ',100*len(photon_index)/results['jet_dR'].shape[0],'%' 
              'Electron/positron results: ',100*len(electron_index)/results['jet_dR'].shape[0],'%',
              'Muon results: ',100*len(muon_index)/results['jet_dR'].shape[0],'%',
              'Tau results: ',100*len(tau_index)/results['jet_dR'].shape[0],'%')
        
        #Stat plot 
        # Generate some data for plotting
        #x = [-4,4]
        #y = [-4,4]
        #t1 = 'Total number of events: '+str(len(quark_index)+len(gluon_index)+len(photon_index)+len(electron_index)+len(muon_index)+len(tau_index))
        #t2 = 'Quark results: '+str(100*len(quark_index)/results['jet_dR'].shape[0])+'%     '+ str(len(quark_index))+' events'
        #t3 = 'Gluon results: '+str(100*len(gluon_index)/results['jet_dR'].shape[0])+'%     '+str(len(gluon_index))+' events'
        #t4 = 'Photon results: '+str(100*len(photon_index)/results['jet_dR'].shape[0])+'%     '+str(len(photon_index))+' events'
        #t5 = 'Electron results: '+str(100*len(electron_index)/results['jet_dR'].shape[0])+'%     '+str(len(electron_index))+' events'
        #t6 = 'Muon results: '+str(100*len(muon_index)/results['jet_dR'].shape[0])+'%     '+str(len(muon_index))+' events'
        #t7 = 'Tau results: '+str(100*len(tau_index)/results['jet_dR'].shape[0])+'%     '+str(len(tau_index))+' events'
        
        #plt.plot(x, y,color='white')
        # Add text to the plot
        #plt.text(0.0, 4,t1, fontsize=12, ha='center', va='center')
        #plt.text(-0.0, 3,t2 , fontsize=12, color = 'blue', ha='center', va='center')
        #plt.text(-0.0, 2,t3 , fontsize=12,  color = 'orange',ha='center', va='center')
        #plt.text(-0.0, 1, t4, fontsize=12, color = 'green', ha='center', va='center')
        #plt.text(-0.0, 0, t5, fontsize=12, color = 'red', ha='center', va='center')
        #plt.text(-0.0, -1, t6, fontsize=12, color = 'purple', ha='center', va='center')
        #plt.text(-0.0, -2, t7, fontsize=12, color = 'brown', ha='center', va='center')

        # Set labels and title
        #plt.xlabel(' ')
        #plt.ylabel(' ')
        #plt.title('Statistics in e+ e- collision leading jet flavor')
        #plt.show()
        #plt.savefig(os.path.join(output_dir, 'stats.pdf'))
        #print('I saved statplot')
        #plt.close()

        #Flavor dR plot
        #x_list = [[results['jet_dR'][i] for i in quark_index],
         #         [results['jet_dR'][i] for i in gluon_index],
          #        [results['jet_dR'][i] for i in photon_index],
           #       [results['jet_dR'][i] for i in electron_index],
            #      [results['jet_dR'][i] for i in muon_index],
             #     [results['jet_dR'][i] for i in tau_index],
        #]
        #bins = np.linspace(-1, 10, 200)
        #plot_histogram_1d(x_list=x_list, label_list=['Quark jets',
         #                                                       'Gluon jets',
          #                                                      'Photon jets',
           #                                                     'Electron jets',
            #                                                    'Muon iets',
             #                                                   'Tau jets'],
              #          bins=bins, logy=True,
               #         xlabel='dR_jet', ylabel='',
                #        filename = f'flavorjet_dR.pdf', output_dir=output_dir)
       # print('I saved flavordr')
        #Flavor Pt plot
        #x_list = [[ results[f'jet__{jetR}__partonjet_pt'][i] for i in quark_index],
         #         [ results[f'jet__{jetR}__partonjet_pt'][i] for i in gluon_index],
          #        [ results[f'jet__{jetR}__partonjet_pt'][i] for i in photon_index],
           #       [ results[f'jet__{jetR}__partonjet_pt'][i] for i in electron_index],
            #      [ results[f'jet__{jetR}__partonjet_pt'][i] for i in muon_index],
             #     [ results[f'jet__{jetR}__partonjet_pt'][i] for i in tau_index],
        #]
        #bins = np.linspace(-5, 50, 200)
        #plot_histogram_1d(x_list=x_list, label_list=['Quark jets',
         #                                                       'Gluon jets',
          #                                                      'Photon jets',
           #                                                     'Electron jets',
            #                                                    'Muon iets',
             #                                                   'Tau jets'],
              #          bins=bins, logy=True,
               #          xlabel='pt,jet (parton level)', ylabel='dN/dpt,jet',
                #        filename = f'flavorjet_pt.pdf', output_dir=output_dir)
        #print('I saved flavorpt')
         #Flavor m plot
        x_list = [[ results[f'jet__{jetR}__partonjet_m'][i][0] for i in quark_index],
                  [ results[f'jet__{jetR}__partonjet_m'][i][0] for i in gluon_index],
                  [ results[f'jet__{jetR}__partonjet_m'][i][0] for i in photon_index],
                  [ results[f'jet__{jetR}__partonjet_m'][i][0] for i in electron_index],
                  [ results[f'jet__{jetR}__partonjet_m'][i][0] for i in muon_index],
                  [ results[f'jet__{jetR}__partonjet_m'][i][0] for i in tau_index],
        ]
        bins = np.linspace(-5, 15, bbins)
        plot_histogram_1d(x_list=x_list, label_list=['Quark jets',
                                                                'Gluon jets',
                                                                'Photon jets',
                                                                'Electron jets',
                                                                'Muon iets',
                                                                'Tau jets'],
                        bins=bins, logy=True,
                         xlabel='m,jet (parton level)', ylabel='dN/dm,jet',
                        filename = 'flavorjet_m.pdf', output_dir=output_dir)
        print('I saved flavorm')
          #Flavor num plot
        x_list = [[ results[f'jet__{jetR}__partonnumparticlesperjet'][i][0] for i in quark_index],
                  [ results[f'jet__{jetR}__partonnumparticlesperjet'][i][0] for i in gluon_index],
                  [ results[f'jet__{jetR}__partonnumparticlesperjet'][i][0] for i in photon_index],
                  [ results[f'jet__{jetR}__partonnumparticlesperjet'][i][0] for i in electron_index],
                  [ results[f'jet__{jetR}__partonnumparticlesperjet'][i][0] for i in muon_index],
                  [ results[f'jet__{jetR}__partonnumparticlesperjet'][i][0] for i in tau_index],
        ]
        bins = np.linspace(-5, 50, 55)
        plot_histogram_1d(x_list=x_list, label_list=['Quark jets',
                                                                'Gluon jets',
                                                                'Photon jets',
                                                                'Electron jets',
                                                                'Muon iets',
                                                                'Tau jets'],
                        bins=bins, logy=True,
                         xlabel='numparticles,jet (parton level)', ylabel=' ',
                        filename = f'flavorjet_num.pdf', output_dir=output_dir)
        print('I saved flavornum')
               #Flavor eta plot
        x_list = [[ results[f'jet__{jetR}__partonjet_eta'][i][0] for i in quark_index],
                  [ results[f'jet__{jetR}__partonjet_eta'][i][0] for i in gluon_index],
                  [ results[f'jet__{jetR}__partonjet_eta'][i][0] for i in photon_index],
                  [ results[f'jet__{jetR}__partonjet_eta'][i][0] for i in electron_index],
                  [ results[f'jet__{jetR}__partonjet_eta'][i][0] for i in muon_index],
                  [ results[f'jet__{jetR}__partonjet_eta'][i][0] for i in tau_index],
        ]
        bins = np.linspace(-5, 7, bbins)
        plot_histogram_1d(x_list=x_list, label_list=['Quark jets',
                                                                'Gluon jets',
                                                                'Photon jets',
                                                                'Electron jets',
                                                                'Muon iets',
                                                                'Tau jets'],
                        bins=bins, logy=True,
                         xlabel='eta,jet (parton level)', ylabel=' ',
                        filename = f'flavorjet_eta.pdf', output_dir=output_dir)
        print('I saved flavoreta')
   
   
   
    if Plot_pt:
    # Plot pt of jets
        x_list = [results[f'jet__{jetR}__hadronjet_pt'][:, 0].flatten(), results[f'jet__{jetR}__partonjet_pt'][:, 0].flatten()]
   # print(type(x_list[0]))
        bins = np.linspace(-5, 50, 200)
        plot_histogram_1d(x_list=x_list, label_list=['hadrons', 'partons'],
                        bins=bins, logy=True,
                        xlabel='pt,jet', ylabel='dN/dpt,jet',
                        filename = f'jet_pt_R{jetR}.pdf', output_dir=output_dir)
        print('I saved pt')
    if Plot_esplits:
       # Plot pt of jets split parton 
        x_list = [results_high[f'jet__{jetR}__partonjet_pt'][:, 0].flatten(), results_middle[f'jet__{jetR}__partonjet_pt'][:, 0].flatten(), results_low[f'jet__{jetR}__partonjet_pt'][:, 0].flatten()]
   # print(type(x_list[0]))
        bins = np.linspace(-5, 50, 200)
        plot_histogram_1d(x_list=x_list, label_list=label_splitp,
                        bins=bins, logy=True,
                        xlabel='pt,jet', ylabel='dN/dpt,jet',
                        filename = f'jet_ptsplitp_R{jetR}.pdf', output_dir=output_dir)
           # Plot pt of jets split hadron 
        x_list = [results_high[f'jet__{jetR}__hadronjet_pt'][:, 0].flatten(), results_middle[f'jet__{jetR}__hadronjet_pt'][:, 0].flatten(), results_low[f'jet__{jetR}__hadronjet_pt'][:, 0].flatten()]
    #print(type(x_list[0]))
        bins = np.linspace(-5, 50, 200)
        plot_histogram_1d(x_list=x_list, label_list=label_splith,
                        bins=bins, logy=True,
                        xlabel='pt,jet', ylabel='dN/dpt,jet',
                        filename = f'jet_ptsplith_R{jetR}.pdf', output_dir=output_dir)
           #Split dR
        x_list = [results_high['jet_dR'],results_middle['jet_dR'],results_low['jet_dR']]
        #print(type(x_list[0]))
        bins = np.linspace(-1, 10, 200)
        plot_histogram_1d(x_list=x_list, label_list=['High energy jets','Middle energy jets','Low energy jets'],
                        bins=bins, logy=True,
                        xlabel='dR_jet', ylabel='',
                        filename = f'jet_dRsplit.pdf', output_dir=output_dir)
        #Num part split
        # Hadron
        x_list = [results_high[f'jet__{jetR}__hadronnumparticlesperjet'][:, 0].flatten(), results_middle[f'jet__{jetR}__hadronnumparticlesperjet'][:, 0].flatten(), results_low[f'jet__{jetR}__hadronnumparticlesperjet'][:, 0].flatten() ]
        bins = np.linspace(-5, 30, 35)
        plot_histogram_1d(x_list=x_list, label_list=label_splith,
                        bins=bins, logy=True,
                        xlabel='numberofparticles,jet', ylabel='dN/dnumpart,jet',
                        filename = f'numparticlesperjetsplith{jetR}.pdf', output_dir=output_dir)
        # Parton
        x_list = [results_high[f'jet__{jetR}__partonnumparticlesperjet'][:, 0].flatten(), results_middle[f'jet__{jetR}__partonnumparticlesperjet'][:, 0].flatten(), results_low[f'jet__{jetR}__partonnumparticlesperjet'][:, 0].flatten() ]
        bins = np.linspace(-5, 30, 35)
        plot_histogram_1d(x_list=x_list, label_list=label_splitp,
                        bins=bins, logy=True,
                        xlabel='numberofparticles,jet', ylabel='dN/dnumpart,jet',
                        filename = f'numparticlesperjetsplitp{jetR}.pdf', output_dir=output_dir)
        print('I saved esplit num')
        #Splt eta phi
            #Split hadrons
        x_list = [results_high[f'jet__{jetR}__hadronjet_eta'][:, 0].flatten(), results_middle[f'jet__{jetR}__hadronjet_eta'][:, 0].flatten(), results_low[f'jet__{jetR}__hadronjet_eta'][:, 0].flatten()]
        bins = np.linspace(-5, 7, bbins)
        plot_histogram_1d(x_list=x_list, label_list=label_splith,
                        bins=bins, logy=True,
                        xlabel='jet_eta,jet', ylabel='dN/djet_eta,jet',
                        filename = f'jet_etasplith{jetR}.pdf', output_dir=output_dir)
        #Split partons
        x_list = [results_high[f'jet__{jetR}__partonjet_eta'][:, 0].flatten(), results_middle[f'jet__{jetR}__partonjet_eta'][:, 0].flatten(), results_low[f'jet__{jetR}__partonjet_eta'][:, 0].flatten()]
        bins = np.linspace(-5, 7, bbins)
        plot_histogram_1d(x_list=x_list, label_list=label_splitp,
                        bins=bins, logy=True,
                        xlabel='jet_eta,jet', ylabel='dN/djet_eta,jet',
                        filename = f'jet_etasplitp{jetR}.pdf', output_dir=output_dir)
                #Split hadrons
        x_list = [results_high[f'jet__{jetR}__hadronjet_phi'][:, 0].flatten(), results_middle[f'jet__{jetR}__hadronjet_phi'][:, 0].flatten(), results_low[f'jet__{jetR}__hadronjet_phi'][:, 0].flatten()]
        bins = np.linspace(-5, 7, bbins)
        plot_histogram_1d(x_list=x_list, label_list=label_splith,
                        bins=bins, logy=True,
                        xlabel='jet_phi,jet', ylabel='dN/djet_phi,jet',
                        filename = f'jet_phisplith{jetR}.pdf', output_dir=output_dir)
        #Split partons
        x_list = [results_high[f'jet__{jetR}__partonjet_phi'][:, 0].flatten(), results_middle[f'jet__{jetR}__partonjet_phi'][:, 0].flatten(), results_low[f'jet__{jetR}__partonjet_phi'][:, 0].flatten()]
        bins = np.linspace(-5, 7, bbins)
        plot_histogram_1d(x_list=x_list, label_list=label_splitp,
                        bins=bins, logy=True,
                        xlabel='jet_eta,jet', ylabel='dN/djet_eta,jet',
                        filename = f'jet_phisplitp{jetR}.pdf', output_dir=output_dir)
        print('I saved phieta')
        #Split m
          # Plot m split had
        x_list = [results_high[f'jet__{jetR}__hadronjet_m'][:, 0].flatten(), results_middle[f'jet__{jetR}__hadronjet_m'][:, 0].flatten(), results_low[f'jet__{jetR}__hadronjet_m'][:, 0].flatten()]
        bins = np.linspace(-5, 15, bbins)
        plot_histogram_1d(x_list=x_list, label_list=label_splith,
                        bins=bins, logy=True,
                        xlabel='jet_m,jet', ylabel='dN/djet_m,jet',
                        filename = f'jet_msplith{jetR}.pdf', output_dir=output_dir)
    #  # Plot m split parton
        x_list = [results_high[f'jet__{jetR}__partonjet_m'][:, 0].flatten(), results_middle[f'jet__{jetR}__partonjet_m'][:, 0].flatten(), results_low[f'jet__{jetR}__partonjet_m'][:, 0].flatten()]
        bins = np.linspace(-5, 15, bbins)
        plot_histogram_1d(x_list=x_list, label_list=label_splitp,
                        bins=bins, logy=True,
                        xlabel='jet_m,jet', ylabel='dN/djet_m,jet',
                        filename = f'jet_msplitp{jetR}.pdf', output_dir=output_dir)
        print('I saved all m')
    
    
    ##

    if Plot_z:
    # Plot pt of jets
        num_events = results['jet_dR'].shape[0]
        z_list = []
        for j in range(num_events):
            jet_pt = results[f'jet__{jetR}__partonjet_pt'][:, 0].flatten()[j]
            for p in range(results[f'jet__{jetR}__partonnumparticlesperjet'][j][0]):
                p_pt = results[f'jet__{jetR}__partonfour_vector'][j][p][0]
                z_list.append(p_pt/jet_pt)
        x_list = [z_list]
        #num_events = results['jet_dR'].shape[0]

        #jet_pt = results[f'jet__{jetR}__partonjet_pt'][:, 0].flatten()  # Array of jet pt values for all events

        #p_pt = results[f'jet__{jetR}__partonfour_vector'][:, :, 0]  # Array of p_pt values for all events and particles

        #z_list = (p_pt / jet_pt[:, np.newaxis]).flatten()  # Compute z values using broadcasting

        #x_list = [z_list]

# Create the histogram
        hist, bins, _ = plt.hist(x_list, bins=30, density=False, alpha=0.5)

# Define the scaling factor
        scaling_factor = 1/num_events

# Rescale the y-axis values
        rescaled_hist = hist * scaling_factor

# Plot the rescaled histogram
        plt.bar(bins[:-1], rescaled_hist, width=np.diff(bins), alpha=0.5)

# Set plot labels
        plt.xlabel('z')
        plt.ylabel('1/N_events * dN/dz')
        plt.savefig(os.path.join(output_dir, 'z.pdf'))
        print('I saved z')
    if Plot_2d:
        # Generate random data
        num_events = results['jet_dR'].shape[0]
        x = results[f'jet__{jetR}__partonjet_m'][:, 0].flatten()
        y = np.array([results[f'jet__{jetR}__partonnumparticlesperjet'][j][0] for j in range(num_events)])

# Create the 2D histogram
        plt.hist2d(x, y, bins=20, cmap='Blues')

# Add colorbar and labels
        plt.colorbar()
        plt.xlabel('Jet m')
        plt.ylabel('Jet constituents')
        plt.title(' ')
       # plt.xlim([0, 20])
       # plt.ylim([0, 10])
# Display the plot
        plt.savefig(os.path.join(output_dir, '2d.pdf'))

                # Split
        quark_index = [i for i in range(results['jet_dR'].shape[0]) if results['leadingparticle_isquark'][i] ]
        gluon_index = [i for i in range(results['jet_dR'].shape[0]) if results['leadingparticle_isgluon'][i] ]
        num_events = results['jet_dR'].shape[0]
        x = np.array([results[f'jet__{jetR}__partonjet_m'][:, 0].flatten()[i] for i in quark_index])
        y = np.array([results[f'jet__{jetR}__partonnumparticlesperjet'][j][0] for j in quark_index])

# Create the 2D histogram
        plt.hist2d(x, y, bins=30, cmap='Reds')

# Add colorbar and labels
        plt.colorbar()
        plt.xlabel('Jet m')
        plt.ylabel('Jet constituents')
        plt.title('Quarks')
        #plt.xlim([-0.5, 15])
        #plt.ylim([-0.5, 15])
# Display the plot
        plt.savefig(os.path.join(output_dir, '2dq.pdf'))
        #Gluons
        x = np.array(results[f'jet__{jetR}__partonjet_m'][:, 0].flatten()[i] for i in gluon_index)
        y = np.array(results[f'jet__{jetR}__partonnumparticlesperjet'][j][0] for j in gluon_index)

# Create the 2D histogram
        plt.hist2d(x, y, bins=30, cmap='Greens')

# Add colorbar and labels
        plt.colorbar()
        plt.xlabel('Jet m')
        plt.ylabel('Jet constituents')
        plt.title('Gluons')
        #plt.xlim([-0.5, 15])
        #plt.ylim([-0.5, 15])
# Display the plot
        plt.savefig(os.path.join(output_dir, '2dg.pdf'))
        
        print('I saved 2d')
    if Plot_dR:
        # Plot dR of jets
        x_list = [results['jet_dR']]
    #print(type(x_list[0]))
        bins = np.linspace(-1, 10, 200)
        plot_histogram_1d(x_list=x_list, label_list=['all'],
                        bins=bins, logy=True,
                        xlabel='dR_jet', ylabel='',
                        filename = f'jet_dR.pdf', output_dir=output_dir)

        print('I saved all dr')
    if Plot_num:

       # Plot number of particles per jets
        x_list = [results[f'jet__{jetR}__hadronnumparticlesperjet'][:, 0].flatten(), results[f'jet__{jetR}__partonnumparticlesperjet'][:, 0].flatten()]
        bins = np.linspace(-5, 30, 35)
        plot_histogram_1d(x_list=x_list, label_list=['hadrons', 'partons'],
                        bins=bins, logy=True,
                        xlabel='numberofparticles,jet', ylabel='dN/dnumpart,jet',
                        filename = f'numparticlesperjet{jetR}.pdf', output_dir=output_dir)
    
    if Plot_phieta:
     # Plot eta
        x_list = [results[f'jet__{jetR}__hadronjet_eta'][:, 0].flatten(), results[f'jet__{jetR}__partonjet_eta'][:, 0].flatten()]
        bins = np.linspace(-5, 7, bbins)
        plot_histogram_1d(x_list=x_list, label_list=['hadrons', 'partons'],
                        bins=bins, logy=True,
                        xlabel='jet_eta,jet', ylabel='dN/djet_eta,jet',
                        filename = f'jet_eta{jetR}.pdf', output_dir=output_dir)

    
     # Plot phi
        x_list = [results[f'jet__{jetR}__hadronjet_phi'][:, 0].flatten(), results[f'jet__{jetR}__partonjet_phi'][:, 0].flatten()]
        bins = np.linspace(-1, 7, bbins)
        plot_histogram_1d(x_list=x_list, label_list=['hadrons', 'partons'],
                        bins=bins, logy=True,
                        xlabel='jet_eta,jet', ylabel='dN/djet_phi,jet',
                        filename = f'jet_phi{jetR}.pdf', output_dir=output_dir)

    
    if Plot_m:
       # Plot m
        x_list = [results[f'jet__{jetR}__hadronjet_m'][:, 0].flatten(), results[f'jet__{jetR}__partonjet_m'][:, 0].flatten()]
        bins = np.linspace(-5, 15, bbins)
        plot_histogram_1d(x_list=x_list, label_list=['hadrons', 'partons'],
                        bins=bins, logy=True,
                        xlabel='jet_m,jet', ylabel='dN/djet_m,jet',
                        filename = f'jet_m{jetR}.pdf', output_dir=output_dir)
     
    #
    # --------
    # Plot multiplicity of particles in jet
    #x = results[f'jet__{jetR}__four_vector'][:,:,0].flatten()
    
    #x = x[x>0.0]
    #bins = np.linspace(0, 200, 200)
    #plot_histogram_1d(x_list=[x], label_list=['hadrons'],
     #                       bins=bins, logy=True,
      #                      xlabel='multiplicity', ylabel='N jets',
       #                     filename = f'multiplicity_R{jetR}.pdf', output_dir=output_dir)


    if Plot_jets:
        num = 0
        for i in range(5):
         
    #Plot Jets 
    # x_parton, x_hadron: 2D arrays representing four-vectors of a single jet, of shape (n_particles, 4) and the 4-vector is e.g. [pt,eta,phi,m]
            x_parton = results[f'jet__{jetR}__partonfour_vector'][i]
            x_hadron = results[f'jet__{jetR}__hadronfour_vector'][i]
           # center_and_scale(x_parton,jetR)
            #center_and_scale(x_hadron,jetR)
        
            colors = ['blue', 'red']
            for i,ev in enumerate([x_hadron, x_parton]):
                
                plt.xlim(-jetR, jetR)
                plt.ylim(-jetR, jetR)
                plt.xlabel('Pseudorapidity')
                plt.ylabel('Azimuthal Angle')
                plt.xticks(np.linspace(-jetR, jetR, 5))
                plt.yticks(np.linspace(-jetR, jetR, 5))
                combined_data = np.concatenate((x_hadron, x_parton), axis=0)
                combined_colors = np.concatenate((np.full(len(x_hadron), colors[0]), np.full(len(x_parton), colors[1])), axis=0)
           # combined_labels = np.concatenate((np.full(len(x_hadron), labels[0]), np.full(len(x_parton), labels[1])), axis=0)
           
                pts = combined_data[:, 0]
                etas = combined_data[:, 1]
                phis = combined_data[:, 2]
            
                plt.scatter(etas, phis, marker='o', s=200 * pts, color=combined_colors, alpha=0.5, lw=0, zorder=10)
           # plt.legend(loc=(0.1, 1.0), frameon=False, ncol=2, handletextpad=0)
                if i % 2 ==0: 
                    plt.scatter([], [], marker='o', s=50, color='blue', label='hadrons')
                    plt.scatter([], [], marker='o', s=50, color='red', label='partons')
                    plt.legend(loc=(0.1, 1.0), frameon=False, ncol=2, handletextpad=0)
                    plt.savefig(os.path.join(output_dir, f'jet__{num/2}__jet.pdf'))
                num = num +1
                plt.close()
                print('I am done plotting')

    
    #--------
    # ...
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
    print('I saved jet plots')
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
                    density=True,#Normalize histogram
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
