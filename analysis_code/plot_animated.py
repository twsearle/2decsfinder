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
import ConfigParser

#SETTINGS----------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
kx = config.getfloat('General', 'kx')

dt = config.getfloat('Time Iteration', 'dt')
totTime = config.getfloat('Time Iteration', 'totTime')

numYs = config.getint('Plotting', 'numYs')
fMode = config.getint('Plotting', 'Fourier Mode')
fp.close()

numTimeSteps = int(totTime / dt)
assert totTime % dt, "non-integer number of time steps!"

assert fMode <= N, "must select a Fourier mode smaller than N"

kwargs = {'N': N, 'M': M, 'Re': Re, 'kx': kx,'time': numTimeSteps*dt }
baseFileName  = "-N{N}-M{M}-Re{Re}-kx{kx}-t{time}.pickle".format(**kwargs)
inFileName = "series-PSI{0}".format(baseFileName)

#------------------------------------------------

def cheb_to_real_transform(vec, y_points) :
    """ calculate the Chebyshev transform for the 2D coherent state
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
    psi = real(PsiPlots[i])
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

# Read in the profile series
PsiSeries = []
inFp = open(inFileName, 'r')
while True:
    # Try to read in another pickled vector
    try:
        PsiSeries.append(pickle.load(inFp))
    # Throw exception and exit loop if fail
    except(EOFError):
        break
inFp.close()

# Pick out only the mode that you want
PsiPlots = []
for i in range(len(PsiSeries)):
    PsiPlots.append(PsiSeries[i][(N+fMode)*M: (N+1+fMode)*M])
del i

numFrames = len(PsiPlots)

# Transform to real space
for i in range(numFrames):
    PsiPlots[i] = cheb_to_real_transform(PsiPlots[i], y_points)
del i

fig = matplotlib.pyplot.figure()
ax = matplotlib.pyplot.axes(xlim=(-1., 2.), ylim=(-1, 1))
line, =ax.plot([], [], lw=1)

anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init,
                                          frames=numFrames,
                              interval=1, blit=True)

matplotlib.pyplot.show()
