#-----------------------------------------------------------------------------
#   2D Newtonian Poiseuille flow time iteration
#
#   Last modified: Fri 24 Jan 18:18:15 2014
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

# SETTINGS---------------------------------------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
Wi = config.getfloat('General', 'Wi')
beta = config.getfloat('General', 'beta')
kx = config.getfloat('General', 'kx')
dt = config.getfloat('Time Iteration', 'dt')
amp = config.getfloat('Time Iteration', 'amp')
totTime = config.getfloat('Time Iteration', 'totTime')
numFrames = config.getint('Time Iteration', 'numFrames')
fp.close()

numTimeSteps = int(totTime / dt)
assert totTime % dt, "non-integer number of time steps!"
assert Wi != 0.0, "cannot have Wi = 0!"

NOld = 5
MOld = 40
kwargs = {'N': N, 'M': M, 'Re': Re,'Wi': Wi, 'beta': beta, 'kx': kx,'time': totTime}
baseFileName  = "-N{N}-M{M}-Re{Re}-Wi{Wi}-b{beta}-kx{kx}-t{time}.pickle".format(**kwargs)
outFileName  = "pf{0}".format(baseFileName)
outFileNameTrace = "trace{0}.dat".format(baseFileName[:-7])
outFileNameTime = "series-pf{0}".format(baseFileName)
inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=NOld, M=MOld, 
                                                        kx=kx, Re=Re)


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

def increase_resolution(vec):
    """increase resolution from Nold, Mold to N, M and return the higher res
    vector"""
    highMres = zeros((2*NOld+1)*M, dtype ='complex')
    for n in range(2*NOld+1):
        highMres[n*M:n*M + MOld] = vec[n*MOld:(n+1)*MOld]
    del n
    fullres = zeros(vecLen, dtype='complex')
    fullres[(N-NOld)*M:(N-NOld)*M + M*(2*NOld+1)] = highMres[0:M*(2*NOld+1)]
    return fullres

def mk_cheb_int():
    integrator = zeros(M, dtype='d')
    for m in range(0,M,2):
        integrator[m] = (1 + cos(m*pi)) / (1-m*m)
    del m
    return integrator

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

formKW = {'N':N, 'M':M, 'kx':kx, 'Re':Re, 'b':beta, 'Wi':Wi, 'dt':dt, 'NT':numTimeSteps, 't':totTime}
print"=====================================\n"
print "Settings:"
print """------------------------------------
N \t\t= {N}
M \t\t= {M}              
Re \t\t= {Re}
beta \t\t={b}
Wi \t\t= {Wi}
kx \t\t= {kx}
dt\t\t= {dt}
totTime\t\t= {t}
NumTimeSteps\t= {NT}
------------------------------------
        """.format(**formKW)

# SET UP

vecLen = (2*N+1)*M
# Set the oneOverC function: 1/2 for m=0, 1 elsewhere:
oneOverC = ones(M)
oneOverC[0] = 1. / 2.
# Set up the CFunc function: 2 for m=0, 1 elsewhere:
CFunc = ones(M)
CFunc[0] = 2.

assert Re != 0, "Setting Reynold's number to zero is dangerous!"
oneOverWi = 1. / Wi 


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

II = eye((2*N+1)*M, (2*N+1)*M, dtype='complex')
constOneVec = zeros((2*N+1)*M, dtype='complex')
constOneVec[N*M] = 1.

# single mode Operators
SMDY = mk_single_diffy()
SMDYY = dot(SMDY, SMDY)
SMDYYY = dot(SMDY, SMDYY)

INTY = mk_cheb_int()

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

# The initial stream-function

PSI = zeros(vecLen, dtype='complex')
# Perturb first 3 Chebyshevs
#PSI[(N-1)*M:(N-1)*M + 3] = amp*(random.random(3) + 1.j*random.random(3))
#PSI[(N+1)*M:(N+2)*M] = conjugate(PSI[(N-1)*M:N*M])

#PSI[N*M]   += 2.0/3.0
#PSI[N*M+1] += 3.0/4.0
#PSI[N*M+2] += 0.0
#PSI[N*M+3] += -1.0/12.0

PSIOld, Nu = pickle.load(open(inFileName, 'r'))

PSI = increase_resolution(PSIOld)

# Initial Stress is Newtonian

Cxx = zeros(vecLen, dtype='complex')
Cxx = Wi*2*dot(MDX, dot(MDY, PSI))
Cxx[N*M] += 1.0
Cyy = zeros(vecLen, dtype='complex')
Cyy = - Wi*2*dot(MDY, dot(MDX, PSI))
Cyy[N*M] += 1.0
Cxy = zeros(vecLen, dtype='complex')
Cxy = Wi*dot(MDYY, PSI) - Wi*dot(MDX, dot(MDX, PSI))

# Form the operators
PsiOpInvList = []
for i in range(N):
    n = i-N

    PSIOP = zeros((2*M, 2*M), dtype='complex')
    SLAPLAC = -n*n*kx*kx*SII + SMDYY

    PSIOP[0:M, 0:M] = 0
    PSIOP[0:M, M:2*M] = Re*SII - 0.5*beta*dt*SLAPLAC

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
Psi0thOp = Re*SMDY - 0.5*dt*beta*SMDYYY + 0j

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
    PSIOld = copy(PSI)
    CxxOld = copy(Cxx)
    CyyOld = copy(Cyy)
    CxyOld = copy(Cxy)
    
    # Make the vector for the RHS of the equations
    U = dot(MDY, PSIOld)
    V = - dot(MDX, PSIOld)    
    MMU = prod_mat(U)
    MMV = prod_mat(V)

    # PSI
    RHSVec = Re*dot(LAPLAC, PSIOld) \
            + dt*0.5*beta*dot(BIHARM, PSIOld) \
            - dt*Re*dot(MMU, dot(MDXLAPLAC, PSIOld)) \
            - dt*Re*dot(MMV, dot(MDYLAPLAC, PSIOld))\
            + (1.-beta)*oneOverWi*(dot(MDXX, Cxy) \
                    + dot(MDXY,(Cyy - Cxx)) \
                    - dot(MDYY, Cxy) )

    # Zeroth mode
    RHSVec[N*M:(N+1)*M] = 0
    RHSVec[N*M:(N+1)*M] = + Re*dot(MDY, PSIOld)[N*M:(N+1)*M] \
              + dt*0.5*beta*dot(MDYYY, PSIOld)[N*M:(N+1)*M] \
              - dt*Re*dot(dot(MMV, MDYY), PSIOld)[N*M:(N+1)*M] \
              + dt*(1.-beta)*oneOverWi*dot(MDY, Cxy)[N*M:(N+1)*M]
    RHSVec[N*M] += dt*2
    
    # This vector was exploding, was trying to test for the cause. 
    #print 'RHSvec0 ', linalg.norm(RHSVec[N*M:(N+1)*M])
    #print 'term1 ', linalg.norm(+ Re*dot(MDY, PSIOld)[N*M:(N+1)*M], 2)
    #print 'term2 ', linalg.norm(+ dt*0.5*beta*dot(MDYYY, PSIOld)[N*M:(N+1)*M], 2)
    #print 'term3 ', linalg.norm(- dt*Re*dot(dot(MMV, MDYY), PSIOld)[N*M:(N+1)*M], 2)
    #print 'term4 ', linalg.norm(+ dt*(1.-beta)*oneOverWi*dot(MDY, Cxy)[N*M:(N+1)*M], 2)

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

    # PSI
    PSI[N*M:(N+1)*M] = linalg.lu_solve((lu0, piv0), RHSVec[N*M:(N+1)*M])

    for n in range(N):
        PSI[n*M:(n+1)*M] = dot(PsiOpInvList[n], RHSVec[n*M:(n+1)*M])
    del n  
 
    for n in range(N+1, 2*N+1):
        PSI[n*M:(n+1)*M] = conj(PSI[(2*N-n)*M:(2*N-n+1)*M])
 
    # Use Backward (implicit?) Euler method to step the stresses

    # Calculate polymeric stress components
    TxxOld = oneOverWi*CxxOld
    TxxOld[N*M] += -1
    TyyOld = oneOverWi*CyyOld
    TyyOld[N*M] += -1
    TxyOld = oneOverWi*CxyOld

    MMCXX = prod_mat(CxxOld)
    MMCYY = prod_mat(CyyOld)
    MMCXY = prod_mat(CxyOld)
    VGRAD = dot(MMU, MDX) + dot(MMV, MDY)

    # CXX
    #Cxx = CxxOld + 2*dt*dot(MMCXX, dot(MDX, U)) + 2*dt*dot(MMCXY, dot(MDX, V))\
    #     - dt*dot(VGRAD, CxxOld) - dt*TxxOld 

    # Cxx via backwards Euler
    AAA = 2*dot(MMCXY, dot(MDX, V)) + oneOverWi*constOneVec
    MMM = VGRAD + prod_mat(2*dot(MDX, U)) - oneOverWi*II
    CxxOp = II - dt*MMM
    Cxx = linalg.solve(CxxOp, CxxOld + AAA*dt)

    # CYY
    #Cyy = CyyOld + 2*dt*dot(MMCXY, dot(MDY, U)) + 2*dt*dot(MMCYY, dot(MDY, V))\
    #     - dt*dot(VGRAD, CyyOld) - dt*TyyOld

    # Cyy via backwards Euler
    AAA = 2*dot(MMCXY, dot(MDY, U)) + oneOverWi*constOneVec
    MMM = VGRAD + 2*prod_mat(dot(MDY, V)) - oneOverWi*II
    CyyOp = II - dt*MMM
    Cyy = linalg.solve(CyyOp, CyyOld + AAA*dt)


    # CXY
    #Cxy = CxyOld + dt*dot(MMCXX, dot(MDY, U)) + dt*dot(MMCYY, dot(MDX, V))\
    #     - dt*dot(VGRAD, CxyOld) - dt*TxyOld

    # Cxy via backwards Euler
    AAA = dot(MMCXX, dot(MDY, U)) + dot(MMCYY, dot(MDX, V))
    MMM = VGRAD - oneOverWi*II
    CxyOp = II - dt*MMM
    Cxy = linalg.solve(CxyOp, CxyOld + AAA*dt)

    #print linalg.norm(CxxOld - Cxx)#, linalg.norm(CxxOld -CxxBack)
    #print linalg.norm(CyyOld - Cyy)#, linalg.norm(CyyOld -CyyBack)
    #print linalg.norm(CxyOld - Cxy)#, linalg.norm(CxyOld -CxyBack)
    #print linalg.norm(PSIOld - PSI)
    #exit(1)
    

    # KE outputted is always that of the previous step
    Usq = dot(MMU,U) + dot(MMV,V)
    KE0 = 0.5*real(dot(INTY, Usq[N*M:(N+1)*M]))
    Usq1 = Usq[(N-1)*M:N*M]*exp(1.j*kx) + Usq[(N+1)*M:(N+2)*M]*exp(-1.j*kx)
    KE1  = 0.5*real(dot(INTY, Usq1))

    if not tindx % (numTimeSteps/numFrames):
        pickle.dump(PSI, psiSeriesFp)
        print "{0:15.8g} {1:15.8g} {2:15.8g}".format(currTime-dt, KE0, KE1)

    traceOutFp.write("{0:15.8g} {1:15.8g} {2:15.8g}\n".format(currTime-dt, KE0, KE1))

traceOutFp.close()
psiSeriesFp.close()
pickle.dump(PSI, open(outFileName, 'w'))
