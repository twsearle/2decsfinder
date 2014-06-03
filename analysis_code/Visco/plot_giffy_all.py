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
from matplotlib import pylab as plt
import ConfigParser

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

dt = config.getfloat('Time Iteration', 'dt')
totTime = config.getfloat('Time Iteration', 'totTime')
intendedFrames = config.getfloat('Time Iteration', 'numFrames')

numYs = config.getint('Plotting', 'numYs')
fMode = config.getint('Plotting', 'Fourier Mode')
fp.close()

numTimeSteps = int(totTime / dt)
assert totTime % dt, "non-integer number of time steps!"

assert fMode <= N, "must select a Fourier mode smaller than N"

kwargs = {'N': N, 'M': M, 'Re': Re,'Wi': Wi, 'beta': beta, 'kx': kx,'time':
          totTime, 'dt':dt }
baseFileName  = "-N{N}-M{M}-kx{kx}-Re{Re}-b{beta}-Wi{Wi}-dt{dt}.pickle".format(**kwargs)
inFileName = "series-pf{0}".format(baseFileName)
print inFileName

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

def animate_xx(i):
    y = y_points
    cxx = real(xxPlots[i])
    line.set_data(y, cxx)
    return line,

def animate_yy(i):
    y = y_points
    cxx = real(xxPlots[i])
    line.set_data(y, cyy)
    return line,

def animate_xy(i):
    y = y_points
    cxx = real(xxPlots[i])
    line.set_data(y, cxy)
    return line,

### MAIN ---------------------------------------------------------------------

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
stressesSeries = []
inFp = open(inFileName, 'r')
while True:
    # Try to read in another pickled vector
    try:
        stressesSeries.append(pickle.load(inFp))
    # Throw exception and exit loop if fail
    except(EOFError):
        break
inFp.close()

# Pick out only the mode that you want for each stress component
psiPlots = []
xxPlots = []
yyPlots = []
xyPlots = []
for i in range(len(stressesSeries)):
    tmppsi, tmpxx, tmpyy, tmpxy = stressesSeries[i]

    psiPlots.append(tmppsi[(N+fMode)*M: (N+1+fMode)*M])
    xxPlots.append(tmpxx[(N+fMode)*M: (N+1+fMode)*M])
    yyPlots.append(tmpyy[(N+fMode)*M: (N+1+fMode)*M])
    xyPlots.append(tmpxy[(N+fMode)*M: (N+1+fMode)*M])

del i

numFrames = len(xxPlots)

# Transform to real space
max_psi_y = 0
min_psi_y = 0 
max_xx_y = 0
min_xx_y = 0 
max_yy_y = 0
min_yy_y = 0 
max_xy_y = 0
min_xy_y = 0 
for i in range(numFrames):
    psiPlots[i] = cheb_to_real_transform(psiPlots[i], y_points)
    xxPlots[i] = cheb_to_real_transform(xxPlots[i], y_points)
    yyPlots[i] = cheb_to_real_transform(yyPlots[i], y_points)
    xyPlots[i] = cheb_to_real_transform(xyPlots[i], y_points)

    if max(psiPlots[i]) > max_psi_y:
        max_psi_y = max(psiPlots[i])

    if min(psiPlots[i]) < min_psi_y:
        min_psi_y = min(psiPlots[i])

    if max(xxPlots[i]) > max_xx_y:
        max_xx_y = max(xxPlots[i])

    if min(xxPlots[i]) < min_xx_y:
        min_xx_y = min(xxPlots[i])

    if max(xyPlots[i]) > max_xy_y:
        max_xy_y = max(xyPlots[i])

    if min(xyPlots[i]) < min_xy_y:
        min_xy_y = min(xyPlots[i])

    if max(yyPlots[i]) > max_yy_y:
        max_yy_y = max(yyPlots[i])

    if min(yyPlots[i]) < min_yy_y:
        min_yy_y = min(yyPlots[i])
del i

interFrameT = (numTimeSteps/intendedFrames)*dt

fig = plt.figure(figsize=(8.0,5.5))


#max_psi_y = 0.1
#min_psi_y = -0.1
#max_xx_y = 0.1
#min_xx_y = -0.1 
#max_yy_y = 0.1
#min_yy_y = -0.1 
#max_xy_y = 0.1
#min_xy_y = -0.1 

for i in range(numFrames):

    ax1 = fig.add_subplot(2,2,1)
    ax1.set_ylim(1.1*min_psi_y, 1.1*max_psi_y)
    line, =ax1.plot(y_points, psiPlots[i], lw=1)
    ax1.set_title('psi Time '+str(interFrameT*i))

    ax2 = fig.add_subplot(2,2,2)
    ax2.set_ylim(1.1*min_xx_y, 1.1*max_xx_y)
    line, = ax2.plot(y_points, xxPlots[i], lw=1)
    ax2.set_title('Cxx Time '+str(interFrameT*i))

    ax3 = fig.add_subplot(2,2,3)
    ax3.set_ylim(1.1*min_xy_y, 1.1*max_xy_y)
    line, =ax3.plot(y_points, xyPlots[i], lw=1)
    ax3.set_title('Cxy Time '+str(interFrameT*i))

    ax4 = fig.add_subplot(2,2,4)
    ax4.set_ylim(1.1*min_yy_y, 1.1*max_yy_y)
    line, =ax4.plot(y_points, yyPlots[i], lw=1)
    ax4.set_title('Cyy Time '+str(interFrameT*i))

    fig.tight_layout()

    fig.savefig('./gifs/all/n{0}/{1:04d}.png'.format(fMode, i))

    fig.clf()

