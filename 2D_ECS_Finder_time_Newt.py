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
from matrix_checker import matrix_checker
import cPickle as pickle

# SETTINGS---------------------------------------------------------------------

N = 1              # Number of Fourier modes
M = 20               # Number of Chebychevs (>4)
Re = 2000.0           # The Reynold's number
kx  = 1.01
dt = 0.01
amp = 0.1
numTimeSteps = 1000

outFileName = "Psi_iterated.pickle"

# -----------------------------------------------------------------------------

# FUNCTIONS
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

def cheb_prod_mat(velA):
    """Function to return a matrix for left-multiplying two Chebychev vectors"""

    D = zeros((M, M), dtype='complex')

    for n in range(M):
        for m in range(-M+1,M):     # Bottom of range is inclusive
            itr = abs(n-m)
            if (itr < M):
                D[n, abs(m)] += 0.5*oneOverC[n]*CFunc[itr]*CFunc[abs(m)]*velA[itr]
    del m, n, itr
    return D

def prod_mat(velA):
    """Function to return a matrix ready for the left dot product with another
    velocity vector"""
    MM = zeros((vecLen, vecLen), dtype='complex')

    #First make the middle row
    midMat = zeros((M, vecLen), dtype='complex')
    for n in range(2*N+1):       # Fourier Matrix is 2*N+1 cheb matricies
        yprodmat = cheb_prod_mat(velA[n*M:(n+1)*M])
        endind = 2*N+1-n
        midMat[:, (endind-1)*M:endind*M] = yprodmat
    del n

    #copy matrix into MM, according to the matrix for spectral space
    # top part first
    for i in range(0, N):
        MM[i*M:(i+1)*M, :] = column_stack((midMat[:, (N-i)*M:], zeros((M, (N-i)*M))) )
    del i
    # middle
    MM[N*M:(N+1)*M, :] = midMat
    # bottom 
    for i in range(0, N):
        MM[(i+N+1)*M:(i+2+N)*M, :] = column_stack((zeros((M, (i+1)*M)), midMat[:, :(2*N-i)*M] ))
    del i

    return MM

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
NumTimeSteps\t= {NT}
------------------------------------
        """.format(N=N, M=M, kx=kx, Re=Re, dt=dt, NT=numTimeSteps)

# SET UP

vecLen = (2*N+1)*M
# Set the oneOverC function: 1/2 for m=0, 1 elsewhere:
oneOverC = ones(M)
oneOverC[0] = 1. / 2.
# Set up the CFunc function: 2 for m=0, 1 elsewhere:
CFunc = ones(M)
CFunc[0] = 2.

oneOverRe = 1. / Re
assert oneOverRe != infty, "Can't set Reynold's to zero!"

# The initial stream-function
PSI = zeros(vecLen, dtype='D')
# Perturb first 3 Chebyshevs
PSI[(N-1)*M:(N-1)*M + 3] = amp*(random.random(3) + 1.j*random.random(3))
PSI[(N+1)*M:(N+2)*M] = conjugate(PSI[(N-1)*M:N*M])

# Which way round do we want this (cambridge or 'Drazin'?)
PSI[N*M]   += 2.0/3.0
PSI[N*M+1] += 3.0/4.0
PSI[N*M+2] += 0.0
PSI[N*M+3] += -1.0/12.0

# Useful operators 

MDY = mk_diff_y()
MDYY = dot(MDY,MDY)
MDYYY = dot(MDY,MDYY)
MDX = mk_diff_x()
MDXX = dot(MDX, MDX)
MDXY = dot(MDX, MDY)
LAPLAC = dot(MDX,MDX) + dot(MDY,MDY)
BIHARM = dot(LAPLAC, LAPLAC)
MDXLAPLAC = dot(MDX, LAPLAC)
MDYLAPLAC = dot(MDY, LAPLAC)

# single mode Operators
SMDY = mk_single_diffy()
SMDYY = dot(SMDY, SMDY)
SMDYYY = dot(SMDY, SMDYY)

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
    PSIOP[M-2, :] = concatenate((DERIVTOP, zeros(M, dtype='D')))
    PSIOP[M-1, :] = concatenate((DERIVBOT, zeros(M, dtype='D')))
    
    # dxpsi(+-1) = 0
    PSIOP[2*M-2, :] = concatenate((BTOP, zeros(M, dtype='D')))
    PSIOP[2*M-1, :] = concatenate((BBOT, zeros(M, dtype='D')))

    # store the inverse of the relevent part of the matrix
    PSIOP = linalg.inv(PSIOP)
    PSIOP = PSIOP[0:M, 0:M]

    PsiOpInvList.append(PSIOP)

del PSIOP

# zeroth mode
Psi0thOp = zeros((M,M), dtype='D')
Psi0thOp = SMDY - 0.5*dt*oneOverRe*SMDYYY

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

print """
Beginning Time Iteration:
=====================================

"""
for tindx, currTime in enumerate(timesList):
    
    # Make the vector for the RHS of the equations
    MMU =   prod_mat(dot(MDY, PSI))
    MMV = - prod_mat(dot(MDX, PSI))
    
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
 

    L2Norm = linalg.norm(PSI, 2)

    print "{0:15.8g} \t {1:15.8g}".format(currTime, L2Norm)

pickle.dump(PSI, open(outFileName, 'w'))

