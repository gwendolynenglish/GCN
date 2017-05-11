'''
Created on 8 May 2017

@author: gwendolynenglish
'''

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import numpy as np
 

# Set up formatting for the movie files

Writer = animation.writers['ffmpeg']
writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate= -1)


def synact_ani(matrix, rows):
    fig_anim = plt.figure()
    heat = [] 
    plt.title("Synaptic Activation over Time")
    plt.xlabel("Planar representation of spatially organized grid cells")
    plt.xlim(1,rows)
    plt.ylim(1,rows)

    for add in np.arange(np.shape(matrix)[1]):
        slices = matrix[:,add]
        sliceplot = slices.reshape((rows,rows))
        heat.append((plt.pcolormesh(np.arange(1, rows + 1, 1),np.arange(1, rows + 1, 1),sliceplot, shading='gouraud'),))
        
    heat_ani = animation.ArtistAnimation(fig_anim, heat, interval = 50, repeat_delay = 3000, blit = True)
    heat_ani.save('synaptic_animation_eclipse.mp4', writer = writer)
    
