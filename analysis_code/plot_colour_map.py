#------------------------------------------------------------------------------
#   colour map plotter for 2D coherent state finder
#
#   Last modified: Mon 11 Nov 18:39:16 2013
#
#------------------------------------------------------------------------------
#TODO check that the axes are the right way up?

#MODULES
from scipy import *
from scipy import linalg
import cPickle as pickle
from matplotlib import pyplot as plt
from matplotlib import rc

#SETTINGS----------------------------------------

N = 3              # Number of Fourier modes
M = 30               # Number of Chebychevs (>4)
Re = 5771.0           # The Reynold's number
kx  = 1.0

numXs = 50
numYs = 50

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

# MAIN

# Read in
inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=N, M=M, kx=kx, Re=Re)
(Psi, Nu) = pickle.load(open(inFileName, 'r'))
PSI0 = zeros((2*N+1)*M, dtype='complex')
PSI0[N*M]   = -2.0/3.0
PSI0[N*M+1] = -3.0/4.0
PSI0[N*M+2] = 0.0
PSI0[N*M+3] = 1.0/12.0

Psi = Psi - PSI0 

print Nu 

x_points = zeros(numXs,dtype='d') 
for xIndx in range(numXs):
    #               2.lambda     * fractional position
    x_points[xIndx] = (4.*pi/kx) * ((1.*xIndx)/numXs)
del xIndx

y_points = zeros(numYs,dtype='d')
for yIndx in range(numYs):
    y_points[yIndx] = (2.0*yIndx)/(numYs-1.0) - 1.0 
del yIndx

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
