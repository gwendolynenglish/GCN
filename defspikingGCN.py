'''
Created on 8 May 2017

@author: gwendolynenglish
'''

from brian2 import * 
import numpy as np
import csv 
import pandas as pd 
import math 
import matplotlib
import matplotlib.pyplot as plt
from brian2.equations import refractory

start_scope()
set_device('cpp_standalone')

def spikingGCN(parameters):
    
    #==============================================================================
    # Import Values from Parameter Dictionary 
    #==============================================================================
    runtime = parameters['runtime'] * ms
    sim_dt = parameters['sim_dt'] * ms
    initialization = parameters['initialization']
    trajectory = parameters['trajectory']
    print(trajectory)
    periodicity = parameters['periodicity']
    connectivity = parameters['connectivity']
    connectivityradius = parameters['connectivityradius']
    rows = parameters['rows']
    cols = parameters['cols']
    v_rest = parameters['v_rest'] * mV
    v_reset = parameters['v_reset'] * mV
    v_threshold = parameters['v_threshold'] * mV
    tau = parameters['tau'] * ms 
    Rm = parameters['Rm'] * ohm
    refractory = parameters['refractory'] * ms 
    I_ext = parameters['I_ext'] * mA
    vel_drive = parameters['vel_drive'] * mA
    ao = parameters['ao']
    R = parameters['R']
    deltar = parameters['deltar']
    a = parameters['a']
    lambdanet = parameters['lambdanet']
    beta = parameters['beta']
    gamma = parameters['gamma']
    l = parameters['l']
    on_pre = parameters['on_pre'] * mV
    index_runtime = parameters['runtime']
    
    #==============================================================================
    # Load relevant external files 
    #==============================================================================
    
    #Load Trajectory/Preconfigured Weights/Initialization Values 
    if trajectory == 'RandomWalk' : traj = pd.read_csv('RandomWalkVelocities.csv', header = None, skiprows =  [], dtype = float64)
    traj = np.asarray(traj)
    traj = traj[:,0:index_runtime]
    x_traj = traj[0,:]
    y_traj = traj[1,:]
    xmotionarray = TimedArray(x_traj, dt = sim_dt) #change back
    ymotionarray = TimedArray(y_traj, dt = sim_dt)   #change back 
    print("Shape of x-traj is", np.shape(x_traj))
    print("Shape of y-traj is", np.shape(y_traj))
         
    if periodicity == 'periodic' and connectivity == 'Gaussian' : 
        torusweights = pd.read_csv('TorusWeights.csv', header=None, skiprows =[], dtype = float64)
        torusweights = np.asmatrix(torusweights)
        torusweights = torusweights.flatten()
        
    if periodicity == 'periodic' and connectivity == 'local':
        torusweightslocal = pd.read_csv('TorusWeightsLocal.csv', header=None, skiprows =[], dtype = int)
        torusweightslocal = np.asmatrix(torusweightslocal)
        torusweightslocal = torusweightslocal.flatten()
        
    #Possibly change to auto adjust the init values according to the v-threshold and v-reset?   
    if initialization == 'init': 
        init_values = pd.read_csv("AperiodicSpikeInit_67-63.csv", header=None, skiprows =[0], dtype = float64)
        init_values = np.asarray(init_values)
        init_values = init_values.flatten()
    
    #==============================================================================
    # Define neuronal equations & Create grid cell neuron group 
    #==============================================================================    
    
    #Set Grid Cell network neuronal dynamics
    eqs = '''
    dv/dt = (v_rest - v + Rm * (I_ext + I_vel)) / tau : volt (unless refractory) 
    I_vel = vel_drive * (xdir * xmotionarray(t) + ydir * ymotionarray(t)) : amp 
    x : 1
    y : 1
    xdir : 1
    ydir: 1
    orientation : 1
    wrapper : 1 
    '''
    
    #Define Grid Cells Neuron Group 
    GridCells = NeuronGroup(rows*cols, model = eqs, threshold = 'v > v_threshold', reset = 'v = v_reset', refractory = refractory,  method = 'euler')

    #==============================================================================
    # Define functions that create parameters for each neuron in the the grid cell group
    #==============================================================================
    #Functions that define x & y locations of neurons within the grid field.      
    def xcoord(rows):
        #xarray = np.hstack(np.arange(1,rows + 1, 1))
        xarray = np.hstack((np.arange(-(rows/2),0, 1), np.arange(0, (rows/2),1)))
        #xarray = np.hstack((np.arange(-(rows/2),0, 1), np.arange(1,(rows/2)+1,1)))
        x = np.tile(xarray,rows)
        return x

    def ycoord(rows):
        #yarray = np.hstack(np.arange(1, rows + 1, 1))
        yarray = np.hstack((np.arange(-(rows/2),0, 1), np.arange(0,(rows/2),1)))
        #yarray = np.hstack((np.arange(-(rows/2),0, 1), np.arange(1,(rows/2)+1,1)))
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
    
    #Function that determines aperiodic network wrapper values 
    def wrapper(rows, R, deltar, ao, x, y):
        wrap = []
        for i in range(rows*rows):
            wrap.append(exp(-ao*(((math.sqrt(x[i]**2+y[i]**2)-R+deltar)/deltar)**2)))
        return wrap
    
    #Create values of locations & preferred input directionality and 
    #assigns them as initial values to neurons with the GridCell NeuronGroup  
    locations = {'x': xcoord(rows),
                 'y': ycoord(rows)}
    
    print(xcoord(rows))
    #Create locations and orientation preferences.          
    preferred_direction = {'xdir' : pref_dir_x(rows),
                           'ydir' : pref_dir_y(rows)}

    #Create values from wrapper function                       
    wrapper_values = {'wrapper' : wrapper(rows, R, deltar, ao, xcoord(rows), ycoord(rows))}

    #Set initialized membrane potential values 
    if initialization == 'init': initialization = {'v' : init_values}
    GridCells.set_states(locations)
    GridCells.set_states(preferred_direction)
    if initialization == 'init' : GridCells.set_states(initialization)
    if initialization == 'uninit' : GridCells.v = '-(rand() * 4 + 63) * mV'
    if periodicity == 'aperiodic': GridCells.set_states(wrapper_values)
    
    #==============================================================================
    # Create recurrent synapses 
    #==============================================================================
    RecSyn = Synapses(GridCells, GridCells, 'w : volt', on_pre = 'v_post +=w')
    if connectivity == 'Gaussian' and periodicity == 'periodic': 
        RecSyn.connect()
        RecSyn.w = on_pre * torusweights
    if connectivity == 'Gaussian' and periodicity == 'aperiodic':
        RecSyn.connect()
        RecSyn.w = on_pre * 'a * exp(-gamma* ((sqrt((x_post - x_pre - l * xdir_pre)**2 + (y_post - y_pre - l * ydir_pre)**2)))**2) - exp(-beta * ((sqrt((x_post - x_pre - l * xdir_pre)**2 + (y_post - y_pre - l * ydir_pre)**2)))**2)'
    if connectivity == 'local' and periodicity == 'periodic':
        RecSyn.connect(condition = 'sqrt((x_pre - x_post - l * xdir_pre)**2 + (y_pre - y_post - l * ydir_pre)**2) <= connectivityradius') 
        RecSyn.connect(condition = 'sqrt((x_pre - x_post + 128 - l * xdir_pre)**2 + (y_pre - y_post - l * ydir_pre)**2) <= connectivityradius')
        RecSyn.connect(condition = 'sqrt((x_pre - x_post - 128 - l * xdir_pre)**2 + (y_pre - y_post - l * ydir_pre)**2) <= connectivityradius')
        RecSyn.connect(condition = 'sqrt((x_pre - x_post - l * xdir_pre)**2 + (y_pre - y_post - 128 - l * ydir_pre)**2) <= connectivityradius')
        RecSyn.connect(condition = 'sqrt((x_pre - x_post - l * xdir_pre)**2 + (y_pre - y_post + 128 - l * ydir_pre)**2) <= connectivityradius')
        RecSyn.connect(condition = 'sqrt((x_pre - x_post + 128 - l * xdir_pre)**2 + (y_pre - y_post + 128 - l * ydir_pre)**2) <= connectivityradius')
        RecSyn.connect(condition = 'sqrt((x_pre - x_post + 128 - l * xdir_pre)**2 + (y_pre - y_post - 128 - l * ydir_pre)**2) <= connectivityradius')
        RecSyn.connect(condition = 'sqrt((x_pre - x_post - 128 - l * xdir_pre)**2 + (y_pre - y_post + 128 - l * ydir_pre)**2) <= connectivityradius')
        RecSyn.connect(condition = 'sqrt((x_pre - x_post - 128 - l * xdir_pre)**2 + (y_pre - y_post - 128 - l * ydir_pre)**2) <= connectivityradius')
        
        #RecSyn.connect(condition = 'sqrt((abs(x_post) - abs(x_pre) - l * xdir_pre)**2 + (abs(y_post) - abs(y_pre) - l * ydir_pre)**2) <= connectivityradius' )
        #RecSyn.connect(condition = 'sqrt((abs(x_post) - abs(x_pre) - l * xdir_post)**2 + (abs(y_post) - abs(y_pre) - l * ydir_post)**2) <= connectivityradius' )
        RecSyn.w = on_pre
    if connectivity == 'local' and periodicity == 'aperiodic':
        RecSyn.connect(condition = 'sqrt((x_post - x_pre - l * xdir_pre)**2 + (y_post - y_pre - l * ydir_pre)**2) <= connectivityradius' )
        RecSyn.w = on_pre
    #RecSyn.w['i==j'] = 0 * volt   #Ensure no impact from self-connections 
    if connectivity == 'local': RecSyn.delay = 'rand() * 5 * ms' #Introduce random synaptic delays between 1-5 ms
    #RecSyn.delay = '(sqrt((abs(x_post) - abs(x_pre) - l * xdir_pre)**2 + (abs(y_post) - abs(y_pre) - l * ydir_pre)**2) * 5/8) * ms'
    #above line makes synaptic delay dependent on physical distance 
    #==============================================================================
    # Create monitors and run simulation 
    #==============================================================================
    
    SpikesMon = SpikeMonitor(GridCells)
    
    #I_vel = StateMonitor(GridCells, ('I_vel'), record= True)    
    
    defaultclock.dt = sim_dt
    run(runtime, report = 'text')
  
    #plt.plot(I_vel.t/ms, I_vel.I_vel[8176]/mA, label='8176')
    #plt.savefig('velocitycurrent8176.png')
    #plt.plot(I_vel.t/ms, I_vel.I_vel[12304]/mA, label='12304')
    #plt.savefig('velocitycurrent12304.png')
    
    SpikeTrainsDict = SpikesMon.spike_trains()
    
   

    return((SpikeTrainsDict, SpikesMon))
