'''
Created on 8 May 2017

@author: gwendolynenglish
'''

network_parameters_default = {
    #Network Parameters
    "runtime"               :                180,  #ms
    "sim_dt"                :                1, #ms
    "trajectory"            :                'RandomWalk',  #RandomWalk or HaftingFull 
    "initialization"        :                'init', #init initialized network, uninit for an uninitialized network
    "periodicity"           :                'periodic', #1 for a periodic network, 0 for an aperiodic network 
    "connectivity"          :                "local", #local or Gaussian 
    "connectivityradius"    :                32, 
    "rows"                  :                128,
    "cols"                  :                128,
    
    #Neuron Parameters 
    "v_rest"                :               -65,  #mV
    "v_reset"               :               -67, # mV
    "v_threshold"           :               -63,  #mV
    "tau"                   :                10, #ms membrane time constant
    "Rm"                    :                10, #ohms membrane resistance    
    "refractory"            :                10,  #ms}
    
    #Input Parameters
    "I_ext"                 :                2, # mA        #for spiking models 
    "vel_drive"             :                .2, #mA         #for spiking models 

    #Wrapper Parameters
    "ao"                    :                4,
    "R"                     :                64,
    "deltar"                :                64,

    #Synaptic Parameters  
    "a"                     :                1.03,
    "lambdanet"             :                13 * 13,
    "beta"                  :                3 / 13 * 13,
    "gamma"                 :                1.1 * 3 / 13 * 13,
    "l"                     :                2,
    "on_pre"                :                -.2 #mV
    }

