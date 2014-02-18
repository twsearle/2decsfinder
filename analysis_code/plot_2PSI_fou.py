#-------------------------------------------------------------------------------
#   Plot of the streamfunction of the base profile for test
#
#
#-------------------------------------------------------------------------------
"""
Plots only the final state of the time iterated system.
"""


#MODULES
from scipy import *
from scipy import linalg
import cPickle as pickle
import sys
from matplotlib import pyplot as plt
from matplotlib import rc
import ConfigParser
import argparse

#SETTINGS---------------------------------------------------

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
fp.close()

numTimeSteps = int(totTime / dt)
assert totTime % dt, "non-integer number of time steps!"

kwargs = {'N': N, 'M': M, 'Re': Re, 'kx': kx,'time': numTimeSteps*dt }
baseFileName  = "-N{N}-M{M}-Re{Re}-kx{kx}-t{time}.pickle".format(**kwargs)

inFileName = "psi{0}".format(baseFileName)

# SETUP ARGUMENT PARSER

parser = argparse.ArgumentParser(description='plot Fourier modes from data file')
parser.add_argument('-f1','--filename1', default=inFileName,
                     help='input filename 1')
parser.add_argument('-f2','--filename2', default=inFileName,
                     help='input filename 2')
args = parser.parse_args()
inFileName1 = args.filename1
inFileName2 = args.filename2

print inFileName1
print inFileName2

#-----------------------------------------------------------

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

def mk_single_diffy():
    """Makes a matrix to differentiate a single vector of Chebyshev's, 
    for use in constructing large differentiation matrix for whole system"""
    # make matrix:
    mat = zeros((M, M), dtype='d')
    for m in range(M):
        for p in range(m+1, M, 2):
            mat[m,p] = 2*p*oneOverC[m]

    return mat

def mk_diff_y():
    """Make the matrix to differentiate a velocity vector wrt y."""
    D = mk_single_diffy()
    MDY = zeros( (vecLen,  vecLen) )
     
    for cheb in range(0,vecLen,M):
        MDY[cheb:cheb+M, cheb:cheb+M] = D
    del cheb
    return MDY

def mk_diff_x():
    """Make matrix to do fourier differentiation wrt x."""
    MDX = zeros( (vecLen, vecLen), dtype='complex')

    n = -N
    for i in range(0, vecLen, M):
        MDX[i:i+M, i:i+M] = eye(M, M, dtype='complex')*n*kx*1.j
        n += 1
    del n, i
    return MDX

#
#   MAIN
#

vecLen = (2*N+1)*M
# Set the oneOverC function: 1/2 for m=0, 1 elsewhere:
oneOverC = ones(M)
oneOverC[0] = 1. / 2.
# Set up the CFunc function: 2 for m=0, 1 elsewhere:
CFunc = ones(M)
CFunc[0] = 2.
Psi1 = zeros((2*N+1)*M, dtype='complex')
Psi1 = pickle.load(open(inFileName1, 'r'))

Psi2 = zeros((2*N+1)*M, dtype='complex')
Psi2 = pickle.load(open(inFileName2, 'r'))

y_points = zeros(numYs, dtype='d')
for yIndx in range(numYs):
    y_points[yIndx] = 1.0 - (2.0*yIndx)/(numYs-1.0)
del yIndx


MDY = mk_diff_y()
MDX = mk_diff_x()

Psidy1 = dot(MDY, Psi1)
Psidx1 = dot(MDX, Psi1)

Psidy2 = dot(MDY, Psi2)
Psidx2 = dot(MDX, Psi2)

#make plots prettier:
inches_per_Lx = 1.4
inches_per_Ly = 2.2
fig_width = 10 
fig_height = 2*2*inches_per_Ly      
fig_size =  [fig_width,fig_height]
rc('figure', figsize=fig_size)

for n in range(N,2*N+1):
    PSIr1 = Cheb_to_real_transform(Psi1[n*M: (n+1)*M], y_points)
    PSIdy1 = Cheb_to_real_transform(Psidy1[n*M: (n+1)*M], y_points)
    PSIdx1 = Cheb_to_real_transform(Psidx1[n*M: (n+1)*M], y_points)

    PSIr2 = Cheb_to_real_transform(Psi2[n*M: (n+1)*M], y_points)
    PSIdy2 = Cheb_to_real_transform(Psidy2[n*M: (n+1)*M], y_points)
    PSIdx2 = Cheb_to_real_transform(Psidx2[n*M: (n+1)*M], y_points)

    plt.figure()
    #plt.plot(y_points, f, 'ro')
    plt.subplot(311)
    titleString = 'psi n = {mode} mode'.format(mode=n-N)
    plt.plot(y_points, real(PSIr1), 'b-')
    plt.plot(y_points, imag(PSIr1), 'r-')
    plt.plot(y_points, real(PSIr2), 'b.-')
    plt.plot(y_points, imag(PSIr2), 'r.-')
    plt.title(titleString)

    plt.subplot(312)
    titleString = 'dy psi n = {mode} mode'.format(mode=n-N)
    plt.plot(y_points, real(PSIdy1), 'b-')
    plt.plot(y_points, imag(PSIdy1), 'r-')
    plt.plot(y_points, real(PSIdy2), 'b.-')
    plt.plot(y_points, imag(PSIdy2), 'r.-')
    plt.title(titleString)

    plt.subplot(313)
    titleString = 'dx psi n = {mode} mode'.format(mode=n-N)
    plt.plot(y_points, real(PSIdx1), 'b-')
    plt.plot(y_points, imag(PSIdx1), 'r-')
    plt.plot(y_points, real(PSIdx2), 'b.-')
    plt.plot(y_points, imag(PSIdx2), 'r.-')
    plt.title(titleString)
    plt.savefig(r'psi{n}.pdf'.format(n=n-N))
    plt.show()

#savetxt('test.dat', vstack((real(PSIr1), imag(PSIr1))).T)
