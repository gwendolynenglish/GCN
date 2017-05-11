'''
Created on 11 May 2017

@author: gwendolynenglish
'''
from brian2 import * 
import numpy as np
import matplotlib.animation as animation 
import prettyplotlib as ppl
from prettyplotlib import plt
from prettyplotlib import brewer2mpl

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate= -1)

def spikes_ani(monitor, rows, runtime, dt):
    spiketimes = monitor.t/ms
    spikeindex = monitor.i
    
    fig_anim = plt.figure()
    spikes = []
    plt.title("Spikes over Trajectory Time")
    plt.xlabel("Planar Representation of Spatially Organized Grid Cells")
    plt.xlim(1, rows)
    plt.ylim(1, rows)
    plt.axis('equal')
    plt.axis([0, rows, 0, rows])
    yellow_red = brewer2mpl.get_map('Reds', 'sequential', 7, reverse = True).mpl_colormap
    sliceset = []    
    
    for add in np.arange(runtime/dt):
        slice = np.zeros(rows*rows)           #create holder array at each time step for plotting spikes
        times = np.where(spiketimes == add)  #determine if spikes occurred during given time step
        index = np.take(spikeindex, times)   #select the neuron index of the those neurons that spiked during timestep
        for neuron in index:                  #set values of neurons that spiked  
            slice[neuron] = 20        
        sliceplot = slice.reshape((rows, rows))
        sliceset.append(sliceplot)
        if (add >=10): sliceset.pop(0)
        frame = sliceset[0]
        for i in range((len(sliceset)-1)):
            frame = np.add(frame, sliceset[i+1])
        spikes.append((plt.pcolormesh(np.arange(1, rows + 1, 1),np.arange(1, rows + 1, 1),frame, cmap=yellow_red, shading='gouraud'),))

    spikes_ani = animation.ArtistAnimation(fig_anim, spikes, interval = 2000, repeat_delay = 3000, blit = False)
    spikes_ani.save('spiking_animationConradt.mp4', writer = writer)    