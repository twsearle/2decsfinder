#------------------------------------------------------------------------------
#   colour map plotter for 2D coherent state finder
#   animated.
#
#   Last modified: Thu  5 Dec 13:58:11 2013
#
#------------------------------------------------------------------------------

#MODULES
from scipy import *
from scipy import linalg
import cPickle as pickle
import ConfigParser
import matplotlib 
matplotlib.use('tkAgg')
import matplotlib.pyplot
import matplotlib.animation

#SETTINGS----------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
kx = config.getfloat('General', 'kx')

totTime = config.getfloat('Time Iteration', 'totTime')
dt = config.getfloat('Time Iteration', 'dt')

numYs = config.getint('Plotting', 'numYs')
numXs = config.getint('Plotting', 'numXs')
fp.close()

numTimeSteps = int(totTime / dt)
assert totTime % dt, "non-integer number of time steps!"

kwargs = {'N': N, 'M': M, 'Re': Re, 'kx': kx, 'time': numTimeSteps*dt }
baseFileName  = "-N{N}-M{M}-Re{Re}-kx{kx}-t{time}.pickle".format(**kwargs)
inFileName = "RS-series-PSI{0}".format(baseFileName)
outFileName = "movie{0}.mp4".format(baseFileName[:-7])

#------------------------------------------------

# FUNCTIONS

def animate(i):
    psi = real(Psi2DList[i])
    im.set_array(psi)
    return im,

# MAIN

x_points = zeros(numXs,dtype='d') 
for xIndx in range(numXs):
    #               2.lambda     * fractional position
    x_points[xIndx] = (4.*pi/kx) * ((1.*xIndx)/numXs)
del xIndx

y_points = zeros(numYs,dtype='d')
for yIndx in range(numYs):
    y_points[yIndx] = (2.0*yIndx)/(numYs-1.0) - 1.0 
del yIndx

# Read in the transformed data

Psi2DList = []
inFp = open(inFileName, 'r')
while True:
    # Try to read in another pickled vector
    try:
        Psi2DList.append(pickle.load(inFp))
    # Throw exception and exit loop if fail
    except(EOFError):
        break
inFp.close()

numFrames = len(Psi2DList)

# Plot 

# make meshes
#grid_x, grid_y = meshgrid(x_points, y_points)

fig = matplotlib.pyplot.figure()
im = matplotlib.pyplot.imshow(real(Psi2DList[0]), origin='lower', extent=[0,22,-1,1], aspect=4)
matplotlib.pyplot.colorbar(orientation='horizontal')

anim = matplotlib.animation.FuncAnimation(fig, animate, frames=numFrames,
                              interval=1, blit=True)
# matplotlib.pyplot.show()
anim.save(outFileName, fps=30)
