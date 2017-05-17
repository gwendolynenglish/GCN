'''
Created on 10 May 2017

@author: gwendolynenglish
'''

import spikingparameters
import defspikingGCN
import spikeoverlay
import spikinganimation
import csv 


(GCN_results, GCN_spikemon) = defspikingGCN.spikingGCN(spikingparameters.network_parameters_highercurrent)

spikeoverlay.spikeoverlay_image(GCN_results, 12304, spikingparameters.network_parameters_highercurrent['trajectory'], spikingparameters.network_parameters_highercurrent['runtime'])
spikeoverlay.spikeoverlay_image(GCN_results, 12334, spikingparameters.network_parameters_highercurrent['trajectory'], spikingparameters.network_parameters_highercurrent['runtime'])
spikeoverlay.spikeoverlay_image(GCN_results, 8190, spikingparameters.network_parameters_highercurrent['trajectory'], spikingparameters.network_parameters_highercurrent['runtime'])
spikeoverlay.spikeoverlay_image(GCN_results, 8176, spikingparameters.network_parameters_highercurrent['trajectory'], spikingparameters.network_parameters_highercurrent['runtime'])
spikeoverlay.spikeoverlay_image(GCN_results, 8120, spikingparameters.network_parameters_highercurrent['trajectory'], spikingparameters.network_parameters_highercurrent['runtime'])
spikeoverlay.spikeoverlay_image(GCN_results, 876, spikingparameters.network_parameters_highercurrent['trajectory'], spikingparameters.network_parameters_highercurrent['runtime'])

#spikinganimation.spikes_ani(GCN_spikemon, spikingparameters.network_parameters_highercurrent['rows'], spikingparameters.network_parameters_highercurrent['runtime'], spikingparameters.network_parameters_highercurrent['sim_dt'])

#csv_out_weights = open('spikingYdir.csv', 'w')
#mywriter = csv.writer(csv_out_weights)
#mywriter.writerows(xdir)
#csv_out_weights.close() 