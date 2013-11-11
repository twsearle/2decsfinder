
#------------------------------------------------------------------------------
#   colour map plotter for 2D coherent state finder
#
#   Last modified: Mon 11 Nov 18:36:12 2013
#
#------------------------------------------------------------------------------

#MODULES
from scipy import *
from scipy import linalg
import cPickle as pickle
from matplotlib import pyplot as plt
from matplotlib import rc

#SETTINGS----------------------------------------

N = 5              # Number of Fourier modes
M = 30               # Number of Chebychevs (>4)
Re = 5771.0           # The Reynold's number
kx  = 1.0

numYs = 100

#------------------------------------------------

# FUNCTIONS

def Fourier_cheb_transform(vec, x_points, y_points) :
    """ calculate the Fourier chebychev transform for the 2D coherent state
    finder"""
    rVec = zeros((numXs, numYs),dtype='complex')
    for xIndx in range(numXs):
        for yIndx in range(numYs):
            for n in range(2*N+1):
                for m in range(M):
                    x = x_points[xIndx]
                    y = y_points[yIndx]
                    term = vec[(n-N)*M + m] * exp(1.j*n*kx*x) * cos(m*arccos(y))
                    rVec[xIndx,yIndx] += term
    del x,y,n,m

    return real(rVec)

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

    return real(rVec)

def mk_single_diffy():
    """Makes a matrix to differentiate a single vector of Chebyshev's, 
    for use in constructing large differentiation matrix for whole system"""
    # make matrix:
    mat = zeros((M, M), dtype='d')
    for m in range(M):
        for p in range(m+1, M, 2):
            mat[m,p] = 2*p*oneOverC[m]

    return mat

# MAIN

#make plots prettier:
inches_per_Lx = 1.4
inches_per_Ly = 2.2
fig_width =  8
fig_height = 4*inches_per_Ly      
fig_size =  [fig_width,fig_height]
rc('figure', figsize=fig_size)

# Read in
inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=N, M=M, kx=kx, Re=Re)
(Psi, Nu) = pickle.load(open(inFileName, 'r'))

# Set the oneOverC function: 1/2 for m=0, 1 elsewhere:
oneOverC = ones(M)
oneOverC[0] = 1. / 2.
#set up the CFunc function: 2 for m=0, 1 elsewhere:
CFunc = ones(M)
CFunc[0] = 2.
singMDY = mk_single_diffy()
plt.figure()

y_points = zeros(numYs, dtype='d')
for yIndx in range(numYs):
    y_points[yIndx] = 1.0 - (2.0*yIndx)/(numYs-1)
del yIndx


for n in range(N):
    psi = Psi[(N-n)*M : (N-n+1)*M]
    psidy = dot(singMDY, psi)
    psir = Cheb_to_real_transform(psi,y_points)
    psidyr = Cheb_to_real_transform(psidy,y_points)
    
    plt.plot(y_points, psir, 'ro')
    plt.plot(y_points, psidyr, 'bo')
    plt.show()
    print "psi +- 1: ", (psir[0], psir[numYs-1])
    print "dypsi +- 1: ", (psidyr[0], psidyr[numYs-1])
del n




