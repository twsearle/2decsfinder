#------------------------------------------------------------------------------
#   colour map plotter for 2D coherent state finder
#
#   Last modified: Thu  6 Nov 16:47:59 2014
#
#------------------------------------------------------------------------------
#TODO check that the axes are the right way up?

#MODULES
import sys
from scipy import *
from scipy import linalg
from scipy import fftpack
import numpy as np
import cPickle as pickle
import ConfigParser
from matplotlib import pyplot as plt
from matplotlib import rc
import brewer2mpl 

import RStransform

#SETTINGS----------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
beta = config.getfloat('General', 'beta')
Wi   = config.getfloat('General', 'Wi')
kx = config.getfloat('General', 'kx')
numYs = config.getint('Plotting', 'numYs')
numXs = config.getint('Plotting', 'numXs')

fp.close()

print "Settings:"
print """------------------------------------
N \t= {N}
M \t= {M}              
Wi \t= {Wi}        
Re \t= {Re}         
beta \t= {beta}
kx \t= {kx}
------------------------------------
        """.format(N=N, M=M, kx=kx, Re=Re, beta=beta, Wi=Wi)


consts = {'N':N, 'M':M, 'kx':kx, 'Re':Re, 'b':beta, 'Wi':Wi}
inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}-b{b}-Wi{Wi}.pickle".format(**consts)

#------------------------------------------------

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

    RStransform.rstransform(vecin, vecout, N, M, numXs, numYs, kx)

    # Reshape back to (xindx, yindx) then transpose to (yindx, xindx) (because)
    vecout = reshape(vecout, (numXs, numYs)).T

    return vecout

def fast_fourier_cheb_transform(vec, x_points, y_points) :
    """ 
    calculate the Fourier chebychev transform for the 2D coherent state
    finder
    WARNING: Don't think this thing actually works right yet
    """
    rVec = zeros((numYs, numXs),dtype='complex')
    # reshape the vector into a matrix
    system2DVec = vec.view().reshape((2*N+1, M)).T

    # First perform fast ifft
    FOnlyMat = zeros((numYs,2*N+1), dtype='complex')
    for n in range(2*N+1):
        for m in range(M):
            for yindx in range(numYs):
                y = y_points[yindx]
                FOnlyMat[yindx, n] += system2DVec[m, n] * cos(m*arccos(y))
    del yindx, n, m

    for yindx in range(numYs):
        reformattedArr = concatenate((FOnlyMat[yindx, N:], FOnlyMat[yindx,
                                                                         :N]))
        reformattedArr = fftpack.ifftshift(FOnlyMat[yindx, :])
        rVec[yindx, :] = fftpack.ifft(reformattedArr, n=numXs) 
        print yindx
    del yindx

    return real(rVec)

# MAIN

# Read in
#inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=N, M=M, kx=kx, Re=Re)
#(Psi, Nu) = pickle.load(open(inFileName, 'r'))

#amp = 0.05
#Psi = zeros((2*N+1)*M, dtype='complex')
#Psi[(N-1)*M:N*M] = amp*pickle.load(open(inFileName, 'r'))
#Psi[(N+1)*M:(N+2)*M] = amp*conjugate(Psi[(N-1)*M:N*M])

(Psi, Cxx, Cyy, Cxy, Nu) = pickle.load(open(inFileName, 'r'))

#PSI0 = zeros((2*N+1)*M, dtype='complex')
#PSI0[N*M]   = -2.0/3.0
#PSI0[N*M+1] = -3.0/4.0
#PSI0[N*M+2] = 0.0
#PSI0[N*M+3] = 1.0/12.0

#Psi = Psi - PSI0 

#Psi[N*M]   = 2.0/3.0
#Psi[N*M+1] = 3.0/4.0
#Psi[N*M+2] = 0.0
#Psi[N*M+3] = -1.0/12.0

#Psi[:N*M]   = 0.0
#Psi[(N+1)*M:]   = 0.0

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
#PSI0RS = 0. + 0.j
#for m in range(M):
#    PSI0RS += Psi[(N-1)*M + m]*cos(m*arccos(0.5))
#del m
#
#phaseFactor = 1. - 1.j*(imag(PSI0RS)/real(PSI0RS))
#
#print "The Phase factor: ", phaseFactor
#
#Psi[(N-1)*M:N*M] = phaseFactor*Psi[(N-1)*M:N*M] 
#Psi[N*M:(N+1)*M] = conjugate(phaseFactor)*Psi[N*M:(N+1)*M] 

# Perform transformation
Psi2D = real(faster_FC_transform(Psi))
Cxx2D = real(faster_FC_transform(Cxx))
Cyy2D = real(faster_FC_transform(Cyy))
Cxy2D = real(faster_FC_transform(Cxy))
#test = Fourier_cheb_transform(Psi, x_points, y_points)
#print allclose(Psi2D, test)
#exit(1)

# make meshes
grid_x, grid_y = meshgrid(x_points, y_points)

#make plots prettier:
inches_per_Lx = 1.4
inches_per_Ly = 2.2
fig_width = 10 
fig_height = 8*inches_per_Ly      
#fig_size =  [fig_width,fig_height]
#rc('figure', figsize=fig_size)

fig = plt.figure(figsize=(10.0,6.0))

bmap = brewer2mpl.get_map('Spectral', 'Diverging', 11)


#min_xx = -1
#max_xx =50
extent_ = [0,4.*pi/kx,-1,1]

ax1 = fig.add_subplot(2,2,1)
im1 = ax1.imshow(real(Psi2D), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap)
plt.colorbar(im1, orientation='horizontal')
ax1.set_title('psi')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2 = fig.add_subplot(2,2,2)
im2 = ax2.imshow(real(Cxx2D), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap )
plt.colorbar(im2, orientation='horizontal')
ax2.set_title('Cxx')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

ax3 = fig.add_subplot(2,2,3)
im3 = ax3.imshow(real(Cyy2D), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap )
plt.colorbar(im3, orientation='horizontal')
ax3.set_title('Cyy')
ax3.set_xlabel('x')
ax3.set_ylabel('y')

ax4 = fig.add_subplot(2,2,4)
im4 = ax4.imshow(real(Cxy2D), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap )
plt.colorbar(im4, orientation='horizontal')
ax4.set_title('Cxy')
ax4.set_xlabel('x')
ax4.set_ylabel('y')

fig.tight_layout()

outFileName = 'cmap_' + inFileName[:-7] + '.pdf'
plt.savefig(outFileName)
