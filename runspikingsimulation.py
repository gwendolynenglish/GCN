'''
Created on 10 May 2017

@author: gwendolynenglish
'''

import spikingparameters
import defspikingGCN
import spikeoverlay
import spikinganimation


GCN_results = defspikingGCN.spikingGCN(spikingparameters.network_parameters_default)

spikeoverlay.spikeoverlay_image(GCN_results, 12304, spikingparameters.network_parameters_default['trajectory'], spikingparameters.network_parameters_default['runtime'])
spikeoverlay.spikeoverlay_image(GCN_results, 12334, spikingparameters.network_parameters_default['trajectory'], spikingparameters.network_parameters_default['runtime'])
spikeoverlay.spikeoverlay_image(GCN_results, 8190, spikingparameters.network_parameters_default['trajectory'], spikingparameters.network_parameters_default['runtime'])
spikeoverlay.spikeoverlay_image(GCN_results, 8176, spikingparameters.network_parameters_default['trajectory'], spikingparameters.network_parameters_default['runtime'])
spikeoverlay.spikeoverlay_image(GCN_results, 8120, spikingparameters.network_parameters_default['trajectory'], spikingparameters.network_parameters_default['runtime'])
spikeoverlay.spikeoverlay_image(GCN_results, 876, spikingparameters.network_parameters_default['trajectory'], spikingparameters.network_parameters_default['runtime'])

spikinganimation.spikes_ani(GCN_results, spikingparameters.network_parameters_default['rows'], spikingparameters.network_parameters_default['runtime'], spikingparameters.network_parameters_default['sim_dt'])