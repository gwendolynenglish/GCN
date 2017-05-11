'''
Created on 8 May 2017

@author: gwendolynenglish
'''

#from brian2 import * 

import rateparameters
import defrateGCN
import synapticactivationanimation

GCN_results = defrateGCN.rateGCN(rateparameters.network_parameters_default)

synapticactivationanimation.synact_ani(GCN_results, rateparameters.network_parameters_default['rows'])
