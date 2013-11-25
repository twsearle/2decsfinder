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
import matplotlib.pyplot as plt
import cPickle as pickle

# SETTINGS---------------------------------------------------------------------

N = 5              # Number of Fourier modes
M = 40               # Number of Chebychevs (>4)
Re = 100000.0           # The Reynold's number
kx  = 1.1
dt = 0.00001

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

oneOverRe = 1./Re
assert oneOverRe != infty, "Can't set Reynold's to zero!"

# The initial stream-function
# PSI = random.random(vecLen)/1000000.0
PSI = zeros(vecLen, dtype='complex')

PSI[N*M]   = -2.0/3.0
PSI[N*M+1] = -3.0/4.0
PSI[N*M+2] = 0.0
PSI[N*M+3] = 1.0/12.0

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

#Identity
II = eye(vecLen, vecLen, dtype='complex')

# Boundary arrays
BTOP = ones(M)
BBOT = ones(M)
BBOT[1:M:2] = -1

singleDY = mk_single_diffy()
DERIVTOP = zeros((M), dtype='complex')
DERIVBOT = zeros((M), dtype='complex')
for j in range(M):
    DERIVTOP[j] = dot(BTOP, singleDY[:,j]) 
    DERIVBOT[j] = dot(BBOT, singleDY[:,j])
del j


# ITERATE THE FLOW PROFILE

PSIOP = zeros(((2*N+1)*M, (2*N+1)*M), dtype='complex')
RHSVec = zeros(vecLen, dtype='complex')

# form a list of times
timesList = r_[dt:dt*numTimeSteps:dt]

print """
Beginning Time Iteration:
=====================================

"""

for tindx, currTime in enumerate(timesList):
    
    # Form the operators
    PSIOP = LAPLAC - 0.5*oneOverRe*dt*BIHARM

    # zeroth mode
    PSIOP[N*M:(N+1)*M, :] = 0
    PSIOP[N*M:(N+1)*M, :] = MDY[N*M:(N+1)*M] - 0.5*dt*oneOverRe*MDYYY[N*M:(N+1)*M]

    # Apply BCs

    for n in range(2*N+1):
        if n == N: continue     # Don't apply bcs to psi0 mode here
        # dxPsi(+-1) = 0
        PSIOP[n*M + M-2, 0 : vecLen] = \
            concatenate( (zeros(n*M), (n-N)*kx*1.j*BTOP, zeros((2*N-n)*M)) )
        PSIOP[n*M + M-1, 0 : vecLen] = \
            concatenate( (zeros(n*M), (n-N)*kx*1.j*BBOT, zeros((2*N-n)*M)) )
        # dypsi(+-1) = 0 
        PSIOP[n*M + M-4, 0:vecLen] = \
            concatenate( (zeros(n*M), DERIVTOP, zeros((2*N-n)*M)) )
        PSIOP[n*M + M-3, 0:vecLen] = \
            concatenate( (zeros(n*M), DERIVBOT, zeros((2*N-n)*M)) )
    del n

    # dypsi0(+-1) = 0
    PSIOP[N*M + M-3, 0:vecLen] = \
        concatenate( (zeros(N*M), DERIVTOP, zeros(N*M)) )
    PSIOP[N*M + M-2, 0:vecLen] = \
        concatenate( (zeros(N*M), DERIVBOT, zeros(N*M)) )
    # psi0(-1) =  0
    PSIOP[N*M + M-1, 0:vecLen] = \
        concatenate( (zeros(N*M), BBOT, zeros(N*M)) )

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
            - dt*dot(dot(MMV, MDY), PSI)[N*M:(N+1)*M] \
            + dt*2*oneOverRe

    # Apply BC's
    
    # dxPsi(+-1) = 0   
    for k in range (2*N+1): 
        if k == N: continue # skip the 0th component 
        RHSVec[k*M + M-2] = dot((k-N)*1.j*kx*BTOP, PSI[k*M:(k+1)*M])
        RHSVec[k*M + M-1] = dot((k-N)*1.j*kx*BBOT, PSI[k*M:(k+1)*M])
    del k

    # dyPsi(+-1) = 0 
    for k in range (2*N+1):
        if k == N: continue # skip the 0th component 
        RHSVec[k*M + M-4] = dot(DERIVTOP, PSI[k*M:(k+1)*M])
        RHSVec[k*M + M-3] = dot(DERIVBOT, PSI[k*M:(k+1)*M])
    del k

    # dyPsi0(+-1) = 0
    RHSVec[N*M + M-3] = dot(DERIVTOP, (PSI[N*M:(N+1)*M]))
    RHSVec[N*M + M-2] = dot(DERIVBOT, (PSI[N*M:(N+1)*M]))

    # Psi0(-1) = 0
    RHSVec[N*M + M-1] = dot(BBOT, (PSI[N*M:(N+1)*M]))

    # Take the time step
    PSI = linalg.solve(PSIOP, RHSVec)
    L2Norm = linalg.norm(PSI, 2)

    print currTime, "\t", L2Norm
    pickle.dump(PSI, open(outFileName, 'w'))
