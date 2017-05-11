'''
Created on 8 May 2017

@author: gwendolynenglish
'''

from brian2 import * 
import numpy as np
import csv 
import pandas as pd 
import math 

start_scope()
set_device('cpp_standalone')

def rateGCN(parameters):
    
    runtime = parameters['runtime'] * ms
    defaultclock.dt = parameters['sim_dt'] * ms 
    initialization = parameters['initialization']
    periodicity = parameters['periodicity']
    rows = parameters['rows']
    cols = parameters['cols']
    membraneTC = parameters['membraneTC'] * ms
    alphaDrive = parameters['alphaDrive']
    ao = parameters['ao']
    R = parameters['R']
    deltar = parameters['deltar']
    a = parameters['a']
    lambdanet = parameters['lambdanet']
    beta = parameters['beta']
    gamma = parameters['gamma']
    l = parameters['l']

    #Define neuronal dynamics equations
    eqs = '''
    ds/dt = -s/membraneTC + clip((stot + B + I),0,(stot + B + I))/membraneTC : 1      
    stot : 1      
    B = wrapper * (1 + alphaDrive * (xdir * xmotionarray(t) + ydir * ymotionarray(t))) : 1
    x : 1    
    y : 1
    xdir : 1
    ydir : 1
    wrapper : 1 
    I : 1
    '''
    GridCells = NeuronGroup(rows*cols, model = eqs, method = 'linear')
    
    if initialization == 0: 
        GridCells.run_regularly('''I = ((rand()**.3)*.2)*.6''', dt = 50 * ms)    
        xmotionarray = TimedArray([0], dt = 1000*ms)    
        ymotionarray = TimedArray([0], dt = 1000*ms)  
  
    if initialization == 1: 
        I = 0   
        xmotionarray = TimedArray([0.20, 0.16, 0.59, 1.0, .23, -.29, -.90, .10, 1.1, -.19], dt = 100*ms) 
        ymotionarray = TimedArray([0, 0.12, 0.81, .05, -.42, -.81, .19, .90, .43, -.26], dt = 100*ms)
    
    #Functions that define x & y locations of neurons within the grid field.      
    def xcoord(rows):
        xarray = np.hstack((np.arange(-(rows/2),0, 1), np.arange(1,(rows/2)+1,1)))
        x = np.tile(xarray,rows)
        return x

    def ycoord(rows):
        yarray = np.hstack((np.arange(-(rows/2),0, 1), np.arange(1,(rows/2)+1,1)))
        y = []
        for i in yarray:
            y = np.hstack((y, np.tile(i,rows)))
        return y
    
    #Functions that define x & y components of preferred input directionality 
    #of neurons within the grid field
    def pref_dir_x(rows):
        xdir_list = []
        for i in range(rows): 
            if np.mod(i, 2) == 0:
                for j in range(int(rows/2)):                
                    xdir_list.append(-1)
                    xdir_list.append(0)
            else:
                for j in range(int(rows/2)):
                    xdir_list.append(0)
                    xdir_list.append(1)
        return(xdir_list)
        
    def pref_dir_y(rows):
        ydir_list = []
        for i in range(rows): 
            if np.mod(i, 2) == 0:
                for j in range(int(rows/2)):                
                    ydir_list.append(0)
                    ydir_list.append(-1)
            else:
                for j in range(int(rows/2)):
                    ydir_list.append(1)
                    ydir_list.append(0)
        return(ydir_list)
    
    #Function that creates wrapper function over 2D grid cell network. 
    def wrapper(rows, R, deltar, ao, x, y):
        wrap = []
        for i in range(rows*rows):
            wrap.append(exp(-ao*(((math.sqrt(x[i]**2+y[i]**2)-R+deltar)/deltar)**2)))
        return wrap
    
    #Create values of locations & preferred input directionality and 
    #assign them as to neurons with the GridCell NeuronGroup  
    locations = {'x': xcoord(rows),
                 'y': ycoord(rows)}
    
    #Call above functions and set states of motion orientation preferences.          
    preferred_direction = {'xdir' : pref_dir_x(rows),
                           'ydir' : pref_dir_y(rows)}
    
    GridCells.set_states(locations)
    GridCells.set_states(preferred_direction)
    
    #If an aperiodic network is selected, apply wrapper function to suppress activity of neurons at
    #the network boundaries 
    if periodicity == 0: 
        wrapper_values = {'wrapper' : wrapper(rows, R, deltar, ao, xcoord(rows), ycoord(rows))}
        GridCells.set_states(wrapper_values)
        
    #If an initialized network is selected, apply pre-determined values for the synaptic activations of each neuron     
    if initialization == 1:     
        init_values = pd.read_csv('InitializationValuesHighPass.csv', header=None, skiprows =[0], dtype = float64)
        init_values = np.asarray(init_values)
        init_values = np.insert(init_values, 0, 0.007891481)
        initialization = {'s' : init_values}
        GridCells.set_states(initialization)
        
    #==============================================================================
    # RECURRENT SYNAPSES
    # Create all to all recurrent activity within the Grid Cell field and generate 
    # synaptic weights according to neuronal distance 
    #==============================================================================
    if periodicity == 1: 
        torusweights = pd.read_csv('TorusWeights.csv', header=None, skiprows =[], dtype = float64)
        torusweights = np.asmatrix(torusweights)
        torusweights = torusweights.flatten()
        
    #Generate synapses between grid cells.
    #Use summed variable to pass the synaptic activation between neurons at every time step. 
    RecSyn = Synapses(GridCells, GridCells, '''w : 1
                            stot_post = w * s_pre : 1 (summed)''')
    
    #Connect all grid cells to one another.                       
    RecSyn.connect()
    print("Loading synaptic weights.")
    
    #Apply weights configured to either a periodic or aperiodic network
    #For the twisted-torus periodic configuration, load predetermined values
    if periodicity == 1: RecSyn.w = torusweights
    if periodicity == 0: RecSyn.w = 'a * exp(-gamma* ((sqrt((x_post - x_pre - l * xdir_pre)**2 + (y_post - y_pre - l * ydir_pre)**2)))**2) - exp(-beta * ((sqrt((x_post - x_pre - l * xdir_pre)**2 + (y_post - y_pre - l * ydir_pre)**2)))**2)'
    
    #==============================================================================
    # STATE MONITORS & SIMULATION 
    #==============================================================================
    M = StateMonitor(GridCells, ('s'), record = True)
    run(runtime, report = 'text')
    all_s = M.s 
    
    return((all_s))