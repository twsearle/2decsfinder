
#------------------------------------------------------------------------------
#   colour map plotter for 2D coherent state finder
#
#   Last modified: Tue 12 Nov 12:00:19 2013
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
numXs = 100

#------------------------------------------------

# FUNCTIONS

def Fourier_cheb_to_real_transform(vec, x_points, y_points) :
    """ calculate the Fourier chebychev transform for the 2D coherent state
    finder"""
    rVec = zeros((numXs, numYs),dtype='complex')
    for xIndx in range(numXs):
        for yIndx in range(numYs):
            x = x_points[xIndx]
            y = y_points[yIndx]
            for n in range(2*N+1):
                for m in range(M):
                    term = vec[(n-N)*M + m] * exp(1.j*n*kx*x) * cos(m*arccos(y))
                    rVec[xIndx,yIndx] += term
    del x,y,n,m

    return real(rVec)

def Cheb_to_real_transform(vec, y_points) :
    """ calculate the chebyshev transform of a 1D vector """
    # Be warned, not sure this should be real. It will still be a Fourier mode
    # after this transform so I may need to examine the imaginary parts?
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
vecLen = (2*N+1)*M
singMDY = mk_single_diffy()
MDX = mk_diff_x()
plt.figure()

y_points = zeros(numYs, dtype='d')
for yIndx in range(numYs):
    y_points[yIndx] = 1.0 - (2.0*yIndx)/(numYs-1)
del yIndx

x_points = zeros(numXs,dtype='d') 
for xIndx in range(numXs):
    #               2.lambda     * fractional position
    x_points[xIndx] = (4.*pi/kx) * ((1.*xIndx)/numXs)
del xIndx

print "x derivative"
Psi2D   = Fourier_cheb_to_real_transform(Psi, x_points, y_points)
DxPsi2D = Fourier_cheb_to_real_transform(dot(MDX,Psi), x_points, y_points)
savetxt('topxcond.dat', DxPsi2D[:, 0])
savetxt('botxcond.dat', DxPsi2D[:, numXs-1])
Psi2Dtop = Psi2D[:,0]
Psi2Dbot = Psi2D[:,numXs-1]
DxPsi2Dtop = DxPsi2D[:,0]
DxPsi2Dbot = DxPsi2D[:,numXs-1]
print "DxPsi top max", amax(DxPsi2Dtop)
print "DxPsi bot max", amax(DxPsi2Dbot)
plt.plot(x_points, Psi2Dtop, 'ro-')
plt.plot(x_points, Psi2Dbot, 'bo-')
plt.plot(x_points, DxPsi2Dtop, 'yx-')
plt.plot(x_points, DxPsi2Dbot, 'gx-')
plt.show()

print "y derivative"

print "\t\t+1\t\t-1"

for n in range(N):
    psi = Psi[(N-n)*M : (N-n+1)*M]
    psidy = dot(singMDY, psi)
    conjPsi = Psi[(N+n)*M : (N+n+1)*M]
    conjDyPsi = dot(singMDY, conjPsi)

    psir = Cheb_to_real_transform(psi, y_points)
    psidyr = Cheb_to_real_transform(psidy, y_points)

    conjPsiR = Cheb_to_real_transform(conjPsi, y_points)
    conjDyPsiR = Cheb_to_real_transform(conjDyPsi, y_points)

    plt.title('Real parts')
    plt.plot(y_points, real(psir), 'ro')
    plt.plot(y_points, real(psidyr), 'bo')
    plt.plot(y_points, real(conjPsiR), 'yx')
    plt.plot(y_points, real(conjDyPsiR), 'gx')
    plt.show()
    plt.title('Imaginary parts')
    plt.plot(y_points, imag(psir), 'ro')
    plt.plot(y_points, imag(psidyr), 'bo')
    plt.plot(y_points, imag(conjPsiR), 'yx')
    plt.plot(y_points, imag(conjDyPsiR), 'gx')
    plt.show()
    print "n = ", n
    print "psi +- 1: \t{a:g}\t{b:g}".format(a=psir[0], b=psir[numYs-1])
    print "dypsi +- 1: \t{a:g}\t{b:g} ".format(a=psidyr[0], b=psidyr[numYs-1])
del n




