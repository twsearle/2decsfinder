#------------------------------------------------------------------------------
#   colour map plotter for 2D coherent state finder
#   animated.
#
#   Last modified: Tue  3 Jun 14:25:53 2014
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
from matplotlib import pylab as plt

#SETTINGS----------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
Wi = config.getfloat('General', 'Wi')
beta = config.getfloat('General', 'beta')
kx = config.getfloat('General', 'kx')

totTime = config.getfloat('Time Iteration', 'totTime')
dt = config.getfloat('Time Iteration', 'dt')
intendedFrames = config.getfloat('Time Iteration', 'numFrames')

numYs = config.getint('Plotting', 'numYs')
numXs = config.getint('Plotting', 'numXs')
fp.close()

numTimeSteps = int(totTime / dt)
assert totTime % dt, "non-integer number of time steps!"

kwargs = {'N': N, 'M': M, 'Re': Re,'Wi': Wi, 'beta': beta, 'kx': kx,'time':
          totTime, 'dt':dt }
baseFileName  = "-N{N}-M{M}-kx{kx}-Re{Re}-b{beta}-Wi{Wi}-dt{dt}.pickle".format(**kwargs)
inFileName = "RS-series-pf{0}".format(baseFileName)
outFileName = "movie{0}.mp4".format(baseFileName[:-7])

#------------------------------------------------

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

max_psi = 0
min_psi = 0 
max_xx = 0
min_xx = 0 
max_yy = 0
min_yy = 0 
max_xy = 0
min_xy = 0 

Psi2DList = []
Cxx2DList = []
Cyy2DList = []
Cxy2DList = []

inFp = open(inFileName, 'r')
while True:
    # Try to read in another pickled vector
    try:
        
        tmppsi, tmpxx, tmpyy, tmpxy = pickle.load(inFp)
        Psi2DList.append(tmppsi)
        Cxx2DList.append(tmpxx)
        Cyy2DList.append(tmpyy)
        Cxy2DList.append(tmpxy)

        if amax(tmppsi) > max_psi:
            max_psi  = amax(tmppsi)

        if amin(tmppsi) < min_psi :
            min_psi  = amin(tmppsi)

        if amax(tmpxx) > max_xx :
            max_xx  = amax(tmpxx)

        if amin(tmpxx) < min_xx :
            min_xx  = amin(tmpxx)

        if amax(tmpxy) > max_xy :
            max_xy  = amax(tmpxy)

        if amin(tmpxy) < min_xy :
            min_xy  = amin(tmpxy)

        if amax(tmpyy) > max_yy :
            max_yy  = amax(tmpyy)

        if amin(tmpyy) < min_yy :
            min_yy  = amin(tmpyy)

    # Throw exception and exit loop if fail
    except(EOFError):
        break
inFp.close()


# Plot 

# make meshes
#grid_x, grid_y = meshgrid(x_points, y_points)
interFrameT = (numTimeSteps/intendedFrames)*dt
fig = plt.figure(figsize=(10.0,6.0))

numFrames=len(Psi2DList)

for i in range(numFrames):

    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(real(Psi2DList[0]), origin='lower', extent=[0,22,-1,1],
                   aspect=4, vmin=min_psi , vmax=max_psi )
    ax1.set_title('psi, time = {0}'.format(interFrameT*i))

    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(real(Cxx2DList[0]), origin='lower', extent=[0,22,-1,1],
                   aspect=4, vmin=min_xx , vmax=max_yy )
    ax2.set_title('Cxx, time = {0}'.format(interFrameT*i))

    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(real(Cyy2DList[0]), origin='lower', extent=[0,22,-1,1],
                   aspect=4, vmin=min_yy , vmax=max_yy )
    ax3.set_title('Cyy, time = {0}'.format(interFrameT*i))

    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(real(Cxy2DList[0]), origin='lower', extent=[0,22,-1,1],
                   aspect=4, vmin=min_xy , vmax=max_xy )
    ax4.set_title('Cxy, time = {0}'.format(interFrameT*i))

    plt.colorbar(orientation='horizontal')
    fig.tight_layout()

    fig.savefig('./gifs/all/cmaps/{0:04d}.png'.format(i))
    fig.clf()
