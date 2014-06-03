#------------------------------------------------------------------------------
#   colour map plotter for 2D coherent state finder
#
#   Last modified: Tue  3 Dec 17:53:05 2013
#
#------------------------------------------------------------------------------
#TODO check that the axes are the right way up?

#MODULES
import sys
from scipy import *
from scipy import linalg
import cPickle as pickle
import ConfigParser
from matplotlib import pyplot as plt
from matplotlib import rc

#SETTINGS----------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
kx = config.getfloat('General', 'kx')
numYs = config.getint('Plotting', 'numYs')
numXs = config.getint('Plotting', 'numXs')

fp.close()

numXs = 50

inFileName = sys.argv[1]

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
                    term = vec[n*M + m] * exp(1.j*(n-N)*kx*x) * cos(m*arccos(y))
                    rVec[yIndx, xIndx] += term
    del x,y,n,m

    return real(rVec)

# MAIN

# Read in
#inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=N, M=M, kx=kx, Re=Re)
#(Psi, Nu) = pickle.load(open(inFileName, 'r'))

#amp = 0.05
#Psi = zeros((2*N+1)*M, dtype='complex')
#Psi[(N-1)*M:N*M] = amp*pickle.load(open(inFileName, 'r'))
#Psi[(N+1)*M:(N+2)*M] = amp*conjugate(Psi[(N-1)*M:N*M])

(Psi, Nu) = pickle.load(open(inFileName, 'r'))

#PSI0 = zeros((2*N+1)*M, dtype='complex')
#PSI0[N*M]   = -2.0/3.0
#PSI0[N*M+1] = -3.0/4.0
#PSI0[N*M+2] = 0.0
#PSI0[N*M+3] = 1.0/12.0

Psi[N*M]   = 2.0/3.0
Psi[N*M+1] = 3.0/4.0
Psi[N*M+2] = 0.0
Psi[N*M+3] = -1.0/12.0

#Psi = Psi - PSI0 

x_points = zeros(numXs,dtype='d') 
for xIndx in range(numXs):
    #               2.lambda     * fractional position
    x_points[xIndx] = (4.*pi/kx) * ((1.*xIndx)/numXs)
del xIndx

y_points = zeros(numYs,dtype='d')
for yIndx in range(numYs):
    y_points[yIndx] = (2.0*yIndx)/(numYs-1.0) - 1.0 
del yIndx

# The phase factor correction
PSI0RS = 0. + 0.j
for m in range(M):
    PSI0RS += Psi[(N-1)*M + m]*cos(m*arccos(0.5))
del m

phaseFactor = 1. - 1.j*(imag(PSI0RS)/real(PSI0RS))

print "The Phase factor: ", phaseFactor

Psi[(N-1)*M:N*M] = phaseFactor*Psi[(N-1)*M:N*M] 
Psi[N*M:(N+1)*M] = conjugate(phaseFactor)*Psi[N*M:(N+1)*M] 

# Perform transformation
Psi2D = Fourier_cheb_transform(Psi, x_points, y_points)

# make meshes
grid_x, grid_y = meshgrid(x_points, y_points)

#make plots prettier:
inches_per_Lx = 1.4
inches_per_Ly = 2.2
fig_width = 10 
fig_height = 2*inches_per_Ly      
fig_size =  [fig_width,fig_height]
rc('figure', figsize=fig_size)

plt.figure()
plt.imshow(Psi2D, origin='lower', extent=[0,22,-1,1], aspect=4)
plt.colorbar(orientation='horizontal')
titleString = 'psi for k = {k}, Re = {Re}'.format(k=kx, Re=Re)
plt.title(titleString)
plt.savefig(r'psi.pdf')
