#-------------------------------------------------------------------------------
#   Plot of the streamfunction of the base profile for test
#
#
#-------------------------------------------------------------------------------

#MODULES
from scipy import *
from scipy import linalg
import cPickle as pickle
import sys
import matplotlib 
matplotlib.use('tkAgg')
import matplotlib.pyplot
import matplotlib.animation

#SETTINGS----------------------------------------

N = 2              # Number of Fourier modes
M = 20               # Number of Chebychevs (>4)
Re = 5771.0           # The Reynold's number
kx  = 1.4

numYs = 50

#------------------------------------------------

def Cheb_to_real_transform(vec, y_points) :
    """ calculate the Fourier chebychev transform for the 2D coherent state
    finder"""
    rVec = zeros(numYs, dtype='complex')
    for yIndx in range(numYs):
        y = y_points[yIndx]
        for m in range(M):
            term = vec[m] * cos(m*arccos(y))
            rVec[yIndx] += term
    del y,m

    return rVec

def init():
    line.set_data([], [])
    return line,

def animate(i):
    y = y_points
    psi = real(PSIplots[i])
    line.set_data(psi, y)
    return line,


y_points = zeros(numYs, dtype='d')
for yIndx in range(numYs):
    y_points[yIndx] = 1.0 - (2.0*yIndx)/(numYs-1.0)
del yIndx
vecLen = (2*N+1)*M
# Set the oneOverC function: 1/2 for m=0, 1 elsewhere:
oneOverC = ones(M)
oneOverC[0] = 1. / 2.
# Set up the CFunc function: 2 for m=0, 1 elsewhere:
CFunc = ones(M)
CFunc[0] = 2.

PSIplots = pickle.load(open('plots.dat', 'r'))
numSteps = len(PSIplots)

fig = matplotlib.pyplot.figure()
ax = matplotlib.pyplot.axes(xlim=(-1., 2.), ylim=(-1, 1))
line, =ax.plot([], [], lw=1)

anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init, frames=numSteps,
                              interval=1, blit=True)

matplotlib.pyplot.show()
