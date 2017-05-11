'''
Created on 8 May 2017

@author: gwendolynenglish
'''

network_parameters_default = {
    #Network Parameters
    "runtime"               :                180, #ms
    "sim_dt"                :                0.5, #ms
    "initialization"        :                0, #1 for an initialized network, 0 for an uninitialized network
    "periodicity"           :                0, #1 for a periodic network, 0 for an aperiodic network 
    "rows"                  :                128,
    "cols"                  :                128,
    
    #Neuron Parameters 
    "membraneTC"            :                10,  #ms}
    
    #Input Parameters
    "alphaDrive"            :                0.10315,        #for rate-based models

    #Wrapper Parameters
    "ao"                    :                4,
    "R"                     :                64,
    "deltar"                :                64,

    #Synaptic Parameters  
    "a"                     :                1.03,
    "lambdanet"             :                13 * 13,
    "beta"                  :                3 / 13 * 13,
    "gamma"                 :                1.1 * 3 / 13 * 13,
    "l"                     :                2 
    }

