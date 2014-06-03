# ------------------------------------------------------------------------------
#   Chebyshev/Fourier to Real space transformation code   
#
#   Last modified: Tue  3 Jun 14:04:57 2014
#
# ------------------------------------------------------------------------------

# MODULES
from scipy import *
from scipy import linalg
import cPickle as pickle
import ConfigParser
import RStransform
import numpy as np

# SETTINGS----------------------------------------------------------------------

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

numXs = config.getint('Plotting', 'numXs')
numYs = config.getint('Plotting', 'numYs')
fp.close()


numTimeSteps = int(totTime / dt)
assert totTime % dt, "non-integer number of time steps!"

kwargs = {'N': N, 'M': M, 'Re': Re,'Wi': Wi, 'beta': beta, 'kx': kx,'time':
          totTime, 'dt':dt }
baseFileName  = "-N{N}-M{M}-kx{kx}-Re{Re}-b{beta}-Wi{Wi}-dt{dt}.pickle".format(**kwargs)
inFileName = "series-pf{0}".format(baseFileName)
outFileName = "RS-series-pf{0}".format(baseFileName)

# This dict is necessary for passing to cython version of the transform
# function
# consts = {'N': N, 'M': M, 'numXs': numXs, 'numYs': numYs, 'kx': kx}
# ------------------------------------------------------------------------------

# FUNCTIONS

def Fourier_cheb_transform(vec, x_points, y_points) :
    """ calculate the Fourier chebychev transform for the 2D coherent state
    finder"""
    rVec = zeros((numYs, numXs),dtype='complex')
    for xIndx in range(numXs):
        for yIndx in range(numYs):
            for n in range(2*N+1):
                for m in range(M):
                    x = x_points[xIndx]
                    y = y_points[yIndx]
                    term = vec[n*M + m] * exp(1.j*(n-N)*kx*x) * cos(m*arccos(y))  
                    rVec[yIndx, xIndx] += term
    del x,y,n,m

    return real(rVec)

def faster_FC_transform(vecin) :
    vecout = np.zeros(numXs*numYs, dtype='D')

    RStransform.rstransform(PsiSeries[i], vecout, N, M, numXs, numYs, kx)

    # Reshape back to (xindx, yindx) then transpose to (yindx, xindx) (because)
    vecout = reshape(vecout, (numXs, numYs)).T

    return vecout
    
# MAIN

# Read in the profile series
PsiSeries = []
xxSeries = []
yySeries = []
xySeries = []

inFp = open(inFileName, 'r')
while True:
    # Try to read in another pickled vector
    try:
        tmppsi, tmpxx, tmpyy, tmpxy = pickle.load(inFp)
        PsiSeries.append(tmppsi)
        xxSeries.append(tmpxx)
        yySeries.append(tmpyy)
        xySeries.append(tmpxy)

    # Throw exception and exit loop if fail
    except(EOFError):
        break
inFp.close()

# Set the real space domain

x_points = zeros(numXs,dtype='d') 
for xIndx in range(numXs):
    #               2.lambda     * fractional position
    x_points[xIndx] = (4.*pi/kx) * ((1.*xIndx)/(numXs-1))
del xIndx

y_points = zeros(numYs,dtype='d')
for yIndx in range(numYs):
    y_points[yIndx] = (2.0*yIndx)/(numYs-1.0) - 1.0 
del yIndx

# Transform to real space

outFp = open(outFileName, 'w')
for i in range(len(PsiSeries)):
    
    #tmparr = Fourier_cheb_transform(PsiSeries[i], x_points, y_points)
    tmppsi = faster_FC_transform(PsiSeries[i])
    tmpxx  = faster_FC_transform(xxSeries[i])
    tmpyy = faster_FC_transform(yySeries[i])
    tmpxy = faster_FC_transform(xySeries[i])


    #print allclose(tmparr, tmparr2, atol=1e-4)
    pickle.dump((real(tmppsi), real(tmpxx), real(tmpyy), real(tmpxy)), outFp)
del i
outFp.close()



