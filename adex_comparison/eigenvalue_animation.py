import tables
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as animation

h5file = tables.open_file('eigenvalues_sorted_animation.h5', mode='r')
root = h5file.root

eigenvals = h5file.root.eigs



framecount = len(eigenvals[0,0,:])
framecounter = range(0, framecount)

def animate(nframe):
    sigma = np.linspace(0.5, 5, 46)
    # create input parameter grid (here only the mu and sigma arrays)
    mu = np.linspace(-1.5, 5., 461)
    plt.cla()
    plt.plot(mu, eigenvals[0,:,nframe], color = 'b', label = '1st pair of regular modes')
    plt.plot(mu, eigenvals[1,:,nframe], color = 'b')
    # plt.plot(eigenvals[2,:,nframe])
    # plt.plot(eigenvals[3,:,nframe])
    if np.abs(eigenvals[4,0,nframe].real) > 0:
        plt.plot(mu, eigenvals[4,:,nframe], color = 'r', label = '1st diffusive mode')
        plt.plot(mu, eigenvals[5,:,nframe], color = 'orange', label = '2nd diffusive mode')
    plt.ylim(-.6, 0.02)
    plt.xlim(-1.5, 2.5)
    plt.xlabel('mu')
    plt.title('Sigma = %f'%(sigma[nframe]))
    plt.legend(loc = 'upper right')

fig = plt.figure(figsize=(5,4))
anim = animation.FuncAnimation(fig, animate, framecounter)
anim.save('anim.gif', writer='imagemagick', fps=10);