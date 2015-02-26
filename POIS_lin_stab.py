
## Linear stability of Poisieulle flow

"""
Pretty sure this code doesn't work at the moment :(
"""

from scipy import *
from scipy import linalg
import matplotlib.pyplot as plt

import TobySpectralMethods as tsm


M = 40

# This is Poiseuille flow 
U = zeros(M, dtype='complex')
U[0]   += 0.5
U[1] += 0
U[2] += -0.5

II = eye(M,M)
BTOP = ones(M)
BBOT = ones(M)
BBOT[1:M:2] = -1

for Re in r_[2000:30000:2000]:
    evals = []
    for kx in r_[0.0:1.0:0.01]:
        tsm.initTSM(N_=3, M_=M, kx_=kx)
        MDY = tsm.mk_single_diffy()
        LPL = (dot(MDY, MDY) - kx**2*II)

        ## Write Jacobian

        JAC = zeros((3*M, 3*M), dtype='complex')

        ## U eq
        # U
        JAC[:M, :M] =  (1./Re) * LPL - 1.j*kx*tsm.cheb_prod_mat(U) 
        # V
        JAC[:M, M:2*M] =  0
        # P 
        JAC[:M, 2*M:3*M] =  - 1.j*kx*II

        ## V eq
        # U
        JAC[M:2*M, :M] =  0
        # V
        JAC[M:2*M, M:2*M] =  (1./Re) * LPL - 1.j*kx*tsm.cheb_prod_mat(U)
        # P 
        JAC[M:2*M, 2*M:3*M] =  - MDY

        ## P eqn
        # U
        JAC[2*M:3*M, :M] =  1.j*kx*II
        # V
        JAC[2*M:3*M, M:2*M] =  MDY 
        # P 2*M:3*M
        JAC[2*M:3*M, 2*M:3*M] = 0

        ### BCs on the Jacobian

        # U
        JAC[M-2, :] = concatenate((BTOP, zeros(2*M)))
        JAC[M-1, :] = concatenate((BBOT, zeros(2*M)))

        # V
        JAC[2*M-2, :] = concatenate((zeros(M), BTOP, zeros(M)))
        JAC[2*M-1, :] = concatenate((zeros(M), BBOT, zeros(M)))


        B = zeros((3*M, 3*M), dtype='complex')
        B[:2*M, :2*M] = eye(2*M,2*M)

        ### BCs on RHS matrix
        B[M-2, :] = 0
        B[M-1, :] = 0
        B[2*M-2, :] = 0
        B[2*M-1, :] = 0

        eigenvalues = linalg.eig(JAC, B, left=False, right=False)
        eigenvalues = eigenvalues[~isnan(eigenvalues*conj(eigenvalues))]
        eigenvalues = eigenvalues[~isinf(eigenvalues*conj(eigenvalues))]

        eigarr = zeros((len(eigenvalues), 2))
        eigarr[:,0] = real(eigenvalues)
        eigarr[:,1] = imag(eigenvalues)
        #savetxt("eigs.txt", eigarr)

        
        actualDecayRate = amax(eigarr[:, 0])

        while actualDecayRate > 100:
            eigarr[argmax(eigarr[:,0]),0] = -10.0
            actualDecayRate = amax(eigarr[:, 0])

        evals.append([kx, actualDecayRate])

    savetxt("RE{0}.dat".format(Re), evals)

