
from scipy import *
from scipy import linalg
import matplotlib.pyplot as plt
import cPickle as pickle
import ConfigParser

import TobySpectralMethods as tsm

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')

config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
Wi = config.getfloat('General', 'Wi')
beta = config.getfloat('General', 'beta')
kx = config.getfloat('General', 'kx')

fp.close()

n = 1

kx = n*kx

print 'Linear stability of Poisieulle flow at kx = ', kx

tsm.initTSM(N_=3, M_=M, kx_=kx)

MDY = tsm.mk_single_diffy()
II = eye(M,M)
LPL = (dot(MDY, MDY) - kx**2*II)

# This is Poiseuille flow 
U = zeros(M, dtype='complex')
U[0] += 0.5
U[1] += 0
U[2] += -0.5

dyU = dot(MDY, U)

Txx = zeros(M, dtype='complex')
Txx = 2.*Wi*dot(tsm.cheb_prod_mat(dyU), dyU)

Txy = zeros(M, dtype='complex')
Txy[1] += -2.0

if not allclose(Txy, dyU):
    print 'error in laminar profile'
    exit(1)

Tyy = zeros(M, dtype='complex')

BTOP = ones(M)
BBOT = ones(M)
BBOT[1:M:2] = -1

## Write Jacobian

JAC = zeros((6*M, 6*M), dtype='complex')

## U eq
# U
JAC[:M, :M] =  (beta/Re)*LPL - 1.j*kx*tsm.cheb_prod_mat(U) 
# V
JAC[:M, M:2*M] =  - tsm.cheb_prod_mat(dot(MDY,U))
# P 
JAC[:M, 2*M:3*M] =  -(1./Re)*1.j*kx*II
# Txx
JAC[:M, 3*M:4*M] =  1.j*kx*((1.0-beta)/Re)*II
# Tyy
JAC[:M, 4*M:5*M] =  0
# Txy
JAC[:M, 5*M:6*M] =  ((1.0-beta)/Re)*MDY

## V eq
# U
JAC[M:2*M, :M] =  0
# V
JAC[M:2*M, M:2*M] =  (beta/Re)*LPL - 1.j*kx*tsm.cheb_prod_mat(U)
# P 
JAC[M:2*M, 2*M:3*M] =  - (1./Re)*MDY
# Txx
JAC[M:2*M, 3*M:4*M] = 0
# Tyy
JAC[M:2*M, 4*M:5*M] = ((1.0-beta)/Re)*MDY 
# Txy
JAC[M:2*M, 5*M:6*M] =  1.j*kx*((1.0-beta)/Re)*II

## P eqn
# U
JAC[2*M:3*M, :M] =  1.j*kx*II
# V
JAC[2*M:3*M, M:2*M] =  MDY 
# P 2*M:3*M
JAC[2*M:3*M, 2*M:3*M] = 0
# Txx
JAC[2*M:3*M, 3*M:4*M] = 0
# Tyy
JAC[2*M:3*M, 4*M:5*M] = 0
# Txy
JAC[2*M:3*M, 5*M:6*M] = 0

## Txx eqn
# U
JAC[3*M:4*M, :M]      = (2.0/Wi)*1.j*kx*II + 2.j*kx*tsm.cheb_prod_mat(Txx) \
                                           + 2.0*dot(tsm.cheb_prod_mat(Txy), MDY)
# V
JAC[3*M:4*M, M:2*M]   =  - tsm.cheb_prod_mat(dot(MDY,Txx))
# P 
JAC[3*M:4*M, 2*M:3*M] = 0
# Txx
JAC[3*M:4*M, 3*M:4*M] = -(1.0/Wi)*II - 1.j*kx*tsm.cheb_prod_mat(U)
# Tyy
JAC[3*M:4*M, 4*M:5*M] = 0
# Txy
JAC[3*M:4*M, 5*M:6*M] = 2.0*tsm.cheb_prod_mat(dyU)

## Tyy eqn
# U
JAC[4*M:5*M, :M]      = 0
# V
JAC[4*M:5*M, M:2*M]   = (2.0/Wi)*MDY + 2.0j*kx*tsm.cheb_prod_mat(Txy) 
# P 
JAC[4*M:5*M, 2*M:3*M] = 0
# Txx
JAC[4*M:5*M, 3*M:4*M] = 0
# Tyy
JAC[4*M:5*M, 4*M:5*M] = - (1.0/Wi)*II - 1.j*kx*tsm.cheb_prod_mat(U) 
# Txy
JAC[4*M:5*M, 5*M:6*M] = 0

## Txy eqn
# U
JAC[5*M:6*M, :M]      = (1.0/Wi)*MDY
# V
JAC[5*M:6*M, M:2*M]   = (1.0/Wi)*1.j*kx*II + 1.j*kx*tsm.cheb_prod_mat(Txx) \
                                        - tsm.cheb_prod_mat(dot(MDY,Txy))
# P 
JAC[5*M:6*M, 2*M:3*M] = 0
# Txx
JAC[5*M:6*M, 3*M:4*M] = 0
# Tyy
JAC[5*M:6*M, 4*M:5*M] = tsm.cheb_prod_mat(dyU)
# Txy
JAC[5*M:6*M, 5*M:6*M] = - (1.0/Wi)*II - 1.j*kx*tsm.cheb_prod_mat(U)

### BCs on the Jacobian

# U
JAC[M-2, :] = concatenate((BTOP, zeros(5*M)))
JAC[M-1, :] = concatenate((BBOT, zeros(5*M)))

# V
JAC[2*M-2, :] = concatenate((zeros(M), BTOP, zeros(4*M)))
JAC[2*M-1, :] = concatenate((zeros(M), BBOT, zeros(4*M)))


B = zeros((6*M, 6*M), dtype='complex')
B[:2*M, :2*M] = eye(2*M,2*M)
B[3*M:, 3*M:] = eye(3*M,3*M)

### BCs on RHS matrix
B[M-2, :] = 0
B[M-1, :] = 0
B[2*M-2, :] = 0
B[2*M-1, :] = 0

eigenvalues = linalg.eigvals(JAC, B)
eigenvalues = eigenvalues[~isnan(eigenvalues*conj(eigenvalues))]
eigenvalues = eigenvalues[~isinf(eigenvalues*conj(eigenvalues))]

eigarr = zeros((len(eigenvalues), 2))
eigarr[:,0] = real(eigenvalues)
eigarr[:,1] = imag(eigenvalues)
savetxt("eigs.txt", eigarr)

#actualDecayRate = amax(eigarr[:, 0])

actualDecayRate = -1.0

for i in range(len(eigarr[:,0])):
    if eigarr[i,0] > 1.0:
        continue
    if eigarr[i,0] > actualDecayRate:
        actualDecayRate = eigarr[i, 0]
        imaginaryPart = eigarr[i, 1]


print 'decay rate from linear stability analysis = ', actualDecayRate

# Newtonian analytic eigenvalues

m = arange(5)
newtEig = array([-(1.0/Re)*(0.5*pi*m)**2, zeros(5)]).T

def ve_func(m):
    fac = (0.5*pi*m)**2
    des = (Re + beta*fac*Wi)**2 - 4.*Re*Wi*fac
    return ( - Re - beta*fac*Wi + sqrt( des ) ) / (2.*Re*Wi) 

viscoEig = array([ve_func(m), zeros(5)]).T

plt.figure(figsize=(4,3))
plt.plot(eigarr[:,0], eigarr[:,1], 'ro', markersize=5.0)
#plt.annotate('%g %g i' % (actualDecayRate, imaginaryPart), 
#                     (actualDecayRate, imaginaryPart),
#                     xytext=(actualDecayRate, 1.1*imaginaryPart))
plt.plot(newtEig[:,0], newtEig[:,1], 'b+', markersize=5.0)
plt.plot(viscoEig[:,0], viscoEig[:,1], 'gx', markersize=5.0)
plt.axhline(color='gray', linewidth=0.5, linestyle='--')
plt.axvline(color='gray', linewidth=0.5, linestyle='--')
plt.xlim([-.01, 0])
plt.show(block=True)

print """
==============================================================================
==============================================================================
==============================================================================
==============================================================================
==============================================================================

"""
