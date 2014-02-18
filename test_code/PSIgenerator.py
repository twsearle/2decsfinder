
# MODULES
import sys
sys.path.append('~/Documents/Project_3/REPO')
from scipy import *
from scipy import linalg
import cPickle as pickle
import ConfigParser
import TobySpectralMethods as tsm

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
kx = config.getfloat('General', 'kx')
dt = config.getfloat('Time Iteration', 'dt')
totTime = config.getfloat('Time Iteration', 'totTime')
numFrames = config.getint('Time Iteration', 'numFrames')
fp.close()

amp = 0.025

tsm.initTSM(N_=N, M_=M, kx_=kx)

def mk_PSI_ECS_guess():

    PSI = zeros(vecLen, dtype='complex')

    PSI[N*M]   += 2.0/3.0
    PSI[N*M+1] += 3.0/4.0
    PSI[N*M+2] += 0.0
    PSI[N*M+3] += -1.0/12.0

    # Perturb 3 of 4 of first Chebyshevs of the 1st Fourier mode
    PSI[(N-1)*M] = -random.normal(loc=amp, scale=0.001) 
    PSI[(N-1)*M+2] = random.normal(loc=amp, scale=0.001) 
    PSI[(N-1)*M+4] = -0.1*random.normal(loc=amp, scale=0.001) 
    PSI[(N-1)*M+6] = -0.05*random.normal(loc=amp, scale=0.001) 

    PSI[(N+1)*M:(N+2)*M] = conjugate(PSI[(N-1)*M:N*M])

    # reduce the base flow KE by a roughly corresponding amount (8pc), with this
    # energy in the perturbation (hopefully). ( 0.96 is about root(0.92) )
    bfReduc = 0.8
    PSI[N*M:(N+1)*M] = bfReduc*PSI[N*M:(N+1)*M]

    # Check to make sure energy is large enough to get an ECS
    U = dot(MDY, PSI)
    V = - dot(MDX, PSI)
    MMU = tsm.prod_mat(U)
    MMV = tsm.prod_mat(V)
    Usq = dot(MMU, U) + dot(MMV, V)
    Usq1 = Usq[(N-1)*M:N*M] + Usq[(N+1)*M:(N+2)*M]
    Usq2 = Usq[(N-2)*M:(N-1)*M] + Usq[(N+2)*M:(N+3)*M]
    KE0 = 0.5*dot(INTY, Usq[N*M:(N+1)*M])
    KE1 = 0.5*dot(INTY, Usq1)
    KE2 = 0.5*dot(INTY, Usq2)
    print 'Kinetic energy of 0th mode is: ', KE0
    print 'Kinetic energy of 1st mode is: ', KE1
    print 'TOTAL: ', KE0+KE1+KE2

    print 'norm of 0th mode is: ', linalg.norm(PSI[N*M:(N+1)*M], 2)
    print 'norm of 1st mode is: ', linalg.norm(PSI[(N-1)*M:N*M] +
                                               PSI[(N+1)*M:(N+2)*M], 2)

    return PSI


# MAIN
vecLen = (2*N+1)*M
# Useful operators 

MDY = tsm.mk_diff_y()
MDYY = dot(MDY,MDY)
MDYYY = dot(MDY,MDYY)
MDX = tsm.mk_diff_x()
MDXX = dot(MDX, MDX)
MDXY = dot(MDX, MDY)
LAPLAC = dot(MDX,MDX) + dot(MDY,MDY)
BIHARM = dot(LAPLAC, LAPLAC)
MDXLAPLAC = dot(MDX, LAPLAC)
MDYLAPLAC = dot(MDY, LAPLAC)

# single mode Operators
SMDY = tsm.mk_single_diffy()
SMDYY = dot(SMDY, SMDY)
SMDYYY = dot(SMDY, SMDYY)

INTY = tsm.mk_cheb_int()

# Identity
SII = eye(M, M, dtype='complex')


PSI = mk_PSI_ECS_guess()

pickle.dump(PSI, open('psi.init', 'w'))
