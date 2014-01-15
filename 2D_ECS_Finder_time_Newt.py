#-----------------------------------------------------------------------------
#   2D Newtonian Poiseuille flow time iteration
#
#   Last modified:
#
#-----------------------------------------------------------------------------

""" Program to find Exact coherent states from a given flow profile using a time
iteration of the flow. When the flow settles to a steady state, we will know
that we have a exact solution to the Navier-Stokes equations.
"""

# MODULES
from scipy import *
from scipy import linalg
import cPickle as pickle
import ConfigParser
import TobySpectralMethods as tsm

# SETTINGS---------------------------------------------------------------------

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

numTimeSteps = int(totTime / dt)
assert totTime % dt, "non-integer number of time steps!"

amp = 0.015

kwargs = {'N': N, 'M': M, 'Re': Re, 'kx': kx,'time': totTime}
baseFileName  = "-N{N}-M{M}-Re{Re}-kx{kx}-t{time}.pickle".format(**kwargs)
outFileName  = "psi{0}".format(baseFileName)
outFileNameTrace = "trace{0}.dat".format(baseFileName[:-7])
outFileNameTime = "series-PSI{0}".format(baseFileName)
inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(**kwargs)

tsm.initTSM(N_=N, M_=M, kx_=kx)

# -----------------------------------------------------------------------------

# FUNCTIONS

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

print"=====================================\n"
print "Settings:"
print """------------------------------------
N \t\t= {N}
M \t\t= {M}              
Re \t\t= {Re}         
kx \t\t= {kx}
dt\t\t= {dt}
totTime\t\t= {t}
NumTimeSteps\t= {NT}
------------------------------------
        """.format(N=N, M=M, kx=kx, Re=Re, dt=dt, NT=numTimeSteps, t=totTime)

# SET UP

vecLen = (2*N+1)*M

oneOverRe = 1. / Re
assert oneOverRe != infty, "Can't set Reynold's to zero!"

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

# Boundary arrays
BTOP = ones(M)
BBOT = ones(M)
BBOT[1:M:2] = -1

DERIVTOP = zeros((M), dtype='complex')
DERIVBOT = zeros((M), dtype='complex')
for j in range(M):
    DERIVTOP[j] = dot(BTOP, SMDY[:,j]) 
    DERIVBOT[j] = dot(BBOT, SMDY[:,j])
del j

#### The initial stream-function
PSI = zeros(vecLen, dtype='complex')

PSI[N*M]   += 2.0/3.0
PSI[N*M+1] += 3.0/4.0
PSI[N*M+2] += 0.0
PSI[N*M+3] += -1.0/12.0

# Perturb 3 of 4 of first Chebyshevs of the 1st Fourier mode
PSI[(N-1)*M] = random.normal(loc=amp, scale=0.001) 
PSI[(N-1)*M+1] = random.normal(loc=amp, scale=0.001) 
PSI[(N-1)*M+3] = random.normal(loc=amp, scale=0.001) 

PSI[(N+1)*M:(N+2)*M] = conjugate(PSI[(N-1)*M:N*M])

# reduce the base flow KE by a roughly corresponding amount (8pc), with this
# energy in the perturbation (hopefully). ( 0.96 is about root(0.92) )
PSI[N*M:(N+1)*M] = 0.9*PSI[N*M:(N+1)*M]

# Check to make sure energy is large enough to get an ECS
U = dot(MDY, PSI)
V = - dot(MDX, PSI)
MMU = tsm.prod_mat(U)
MMV = tsm.prod_mat(V)
Usq = dot(MMU, U) + dot(MMV, V)
Usq1 = Usq[(N-1)*M:N*M] + Usq[(N+1)*M:(N+2)*M]
KE0 = 0.5*dot(INTY, Usq[N*M:(N+1)*M])
KE1 = 0.5*dot(INTY, Usq1)
print 'Kinetic energy of 0th mode is: ', KE0
print 'Kinetic energy of 1st mode is: ', KE1

print 'norm of 0th mode is: ', linalg.norm(PSI[N*M:(N+1)*M], 2)
print 'norm of 1st mode is: ', linalg.norm(PSI[(N-1)*M:N*M] +
                                           PSI[(N+1)*M:(N+2)*M], 2)
exit(1)

# Read in stream function from file
#(PSI, Nu) = pickle.load(open(inFileName,'r'))


# Form the operators
PsiOpInvList = []
for i in range(N):
    n = i-N

    PSIOP = zeros((2*M, 2*M), dtype='complex')
    SLAPLAC = -n*n*kx*kx*SII + SMDYY

    PSIOP[0:M, 0:M] = 0
    PSIOP[0:M, M:2*M] = SII - 0.5*oneOverRe*dt*SLAPLAC

    PSIOP[M:2*M, 0:M] = SLAPLAC
    PSIOP[M:2*M, M:2*M] = -SII

    # Apply BCs
    # dypsi(+-1) = 0
    PSIOP[M-2, :] = concatenate((DERIVTOP, zeros(M, dtype='complex')))
    PSIOP[M-1, :] = concatenate((DERIVBOT, zeros(M, dtype='complex')))
    
    # dxpsi(+-1) = 0
    PSIOP[2*M-2, :] = concatenate((BTOP, zeros(M, dtype='complex')))
    PSIOP[2*M-1, :] = concatenate((BBOT, zeros(M, dtype='complex')))

    # store the inverse of the relevent part of the matrix
    PSIOP = linalg.inv(PSIOP)
    PSIOP = PSIOP[0:M, 0:M]

    PsiOpInvList.append(PSIOP)

del PSIOP

# zeroth mode
Psi0thOp = zeros((M,M), dtype='complex')
Psi0thOp = SMDY - 0.5*dt*oneOverRe*SMDYYY + 0j

# Apply BCs

# dypsi0(+-1) = 0
Psi0thOp[M-3, :] = DERIVTOP
Psi0thOp[M-2, :] = DERIVBOT
# psi0(-1) =  0
Psi0thOp[M-1, :] = BBOT

# compute lu factorisation, PSIOPLU blocks have l and u together, without
# diagonal elements of l. PSIOPPIV elements are pivot vectors for the columns.


lu0, piv0 = linalg.lu_factor(Psi0thOp)

# # ITERATE THE FLOW PROFILE

RHSVec = zeros(vecLen, dtype='complex')

# form a list of times
timesList = r_[dt:dt*numTimeSteps:dt]

# open the files
traceOutFp = open(outFileNameTrace, 'w')
psiSeriesFp = open(outFileNameTime, 'w')
pickle.dump(PSI, psiSeriesFp)

print """
Beginning Time Iteration:
=====================================

"""
for tindx, currTime in enumerate(timesList):
    
    # Make the vector for the RHS of the equations
    U =   dot(MDY, PSI)
    V = - dot(MDX, PSI)
    MMU = tsm.prod_mat(U)
    MMV = tsm.prod_mat(V)
    
    RHSVec = dt*0.5*oneOverRe*dot(BIHARM, PSI) \
            + dot(LAPLAC, PSI) \
            - dt*dot(MMU, dot(MDXLAPLAC, PSI)) \
            - dt*dot(MMV, dot(MDYLAPLAC, PSI)) 

    # Zeroth mode
    RHSVec[N*M:(N+1)*M] = 0
    RHSVec[N*M:(N+1)*M] = dt*0.5*oneOverRe*dot(MDYYY, PSI)[N*M:(N+1)*M] \
            + dot(MDY, PSI)[N*M:(N+1)*M] \
            - dt*dot(dot(MMV, MDYY), PSI)[N*M:(N+1)*M]
    RHSVec[N*M] += dt*2*oneOverRe

    # Apply BC's
    
    for n in range (N+1): 
        # dyPsi(+-1) = 0  
        # Only impose the BC which is actually present in the inverse operator
        # we are dealing with. Remember that half the Boundary Conditions were
        # imposed on phi, which was accounted for implicitly when we ignored it.
        RHSVec[n*M + M-2] = 0
        RHSVec[n*M + M-1] = 0
    del n

    # dyPsi0(+-1) = 0
    RHSVec[N*M + M-3] = 0
    RHSVec[N*M + M-2] = 0

    # Psi0(-1) = 0
    RHSVec[N*M + M-1] = 0

    # Take the time step

    PSI[N*M:(N+1)*M] = linalg.lu_solve((lu0, piv0), RHSVec[N*M:(N+1)*M])

    for n in range(N):
        PSI[n*M:(n+1)*M] = dot(PsiOpInvList[n], RHSVec[n*M:(n+1)*M])
    del n  
 
    for n in range(N+1,2*N+1):
        PSI[n*M:(n+1)*M] = conj(PSI[(2*N-n)*M:(2*N-n+1)*M])
 

    # KE0 is from the previous timestep rather than the current one.

    Usq = dot(MMU, U) + dot(MMV, V)
    KE0 = 0.5*dot(INTY, Usq[N*M:(N+1)*M])
    Usq1 = Usq[(N-1)*M:N*M] + Usq[(N+1)*M:(N+2)*M]
    KE1 = 0.5*dot(INTY, Usq1)

    dataout = {'t':currTime-dt, 'KE0':KE0, 'KE1':KE1}
    
    if not tindx % (numTimeSteps/numFrames):
        pickle.dump(PSI, psiSeriesFp)
        print "{t:15.8g} {KE0:15.8g} {KE1:15.8g}".format(**dataout)

    traceOutFp.write("{t:15.8g} {KE0:15.8g} {KE1:15.8g}\n".format(**dataout))

traceOutFp.close()
psiSeriesFp.close()
pickle.dump(PSI, open(outFileName, 'w'))
