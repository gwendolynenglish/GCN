'''
Created on 10 May 2017

@author: gwendolynenglish
'''


import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import csv 
import pandas as pd 
import prettyplotlib as ppl
from prettyplotlib import plt
from prettyplotlib import brewer2mpl

def spikeoverlay_image(dictionary, neuron, trajectory, runtime):
    plt.clf()     
    if trajectory == 'RandomWalk' : positions = pd.read_csv('RandomWalkPositions.csv', header = None, skiprows =  [], dtype = np.float64)
    positions = np.asarray(positions)
    positions = positions[:,0:runtime] 
    print("in fig shape of positions is", np.shape(positions))
    xs = positions[0,:]
    ys = positions[1,:]

    gc_spikes = dictionary[neuron]

    print(np.shape(gc_spikes))
    print(gc_spikes)

    gc_spikes = gc_spikes*1000  #may be incorrect?
    markers_on = gc_spikes.astype(np.int)  #incorrect
    
    marker_style = dict(marker='.', markeredgecolor='r', markerfacecolor = 'r', markersize=10)
    plt.plot(xs, ys, '-k', markevery=markers_on, **marker_style)

    plt.axis('equal')
    plt.axis([-0.9, 0.9, -0.9, 0.9])
    plt.title("Agent Trajectory with Grid Cell Responses")
    plt.savefig('TrajectorywithSpikes-eclipsefirsttry' + str(neuron) + '.png')
    plt.clf()