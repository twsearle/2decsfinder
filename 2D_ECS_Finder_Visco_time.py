#-----------------------------------------------------------------------------
#   2D Plane Poiseuille flow time iteration
#
#   Last modified: Mon  9 Mar 12:21:57 2015
#
#-----------------------------------------------------------------------------

"""

Simulation of plane Poiseuille flow. In the past I have used a numerical scheme
which uses the streamfunction at the wrong time in the Runge-Kutta method. Here
I attempt to correct this by calculating the streamfunction twice, Once at t
+ dt/2 and once at t + dt.

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
Wi = config.getfloat('General', 'Wi')
beta = config.getfloat('General', 'beta')
kx = config.getfloat('General', 'kx')

dt = config.getfloat('Time Iteration', 'dt')
amp = 1e-8
totTime = config.getfloat('Time Iteration', 'totTime')
numFrames = config.getint('Time Iteration', 'numFrames')
fp.close()

numTimeSteps = int(totTime / dt)
assert totTime % dt, "non-integer number of time steps!"
assert Wi != 0.0, "cannot have Wi = 0!"

NOld = N 
MOld = M
kwargs = {'N': N, 'M': M, 'Re': Re,'Wi': Wi, 'beta': beta, 'kx': kx,'time':
          totTime, 'dt':dt }
baseFileName  = "-N{N}-M{M}-kx{kx}-Re{Re}-b{beta}-Wi{Wi}-dt{dt}.pickle".format(**kwargs)
outFileName  = "pf{0}".format(baseFileName)
outFileNameTrace = "trace{0}.dat".format(baseFileName[:-7])
outFileNameTime = "series-pf{0}".format(baseFileName)
#inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=NOld, M=MOld, 
#                                                        kx=kx, Re=Re)
inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}-b{beta}-Wi{Wi}.pickle".format(**kwargs)


tsm.initTSM(N_=N, M_=M, kx_=kx)

# -----------------------------------------------------------------------------

# FUNCTIONS

def increase_resolution(vec, NOld, MOld, N_, M_):
    """increase resolution from Nold, Mold to N, M and return the higher res
    vector"""
    highMres = zeros((2*NOld+1)*M_, dtype ='complex')
    for n in range(2*NOld+1):
        highMres[n*M_:n*M_ + MOld] = vec[n*MOld:(n+1)*MOld]
    del n
    fullres = zeros((2*N_+1)*M_, dtype='complex')
    fullres[(N_-NOld)*M_:(N_-NOld)*M_ + M_*(2*NOld+1)] = highMres[0:M_*(2*NOld+1)]
    return fullres

def decrease_resolution(vec, NOld, MOld, N_, M_):
    """ 
    decrease both the N and M resolutions
    """

    lowMvec = zeros((2*NOld+1)*M_, dtype='complex')
    for n in range(2*NOld+1):
        lowMvec[n*M_:(n+1)*M_] = vec[n*MOld:n*MOld + M_]
    del n

    lowNMvec = zeros((2*N_+1)*M_, dtype='D')
    lowNMvec = lowMvec[(NOld-N_)*M_:(NOld-N_)*M_ + (2*N_+1)*M_]

    return lowNMvec

def mk_cheb_int():
    integrator = zeros(M, dtype='d')
    for m in range(0,M,2):
        integrator[m] = (1 + cos(m*pi)) / (1-m*m)
    del m
    return integrator

def iplct_euler_step(stressVec, dt):

    CxxOld = stressVec[0:vecLen]
    CyyOld = stressVec[vecLen:2*vecLen]
    CxyOld = stressVec[2*vecLen:3*vecLen]

    MMCXX = tsm.c_prod_mat(CxxOld)
    MMCYY = tsm.c_prod_mat(CyyOld)
    MMCXY = tsm.c_prod_mat(CxyOld)

    SVNew = zeros(3*vecLen, dtype='D')

    # Cxx via backwards Euler
    AAA = 2*dot(MMCXY, dot(MDY, U)) + oneOverWi*constOneVec
    MMM = VGRAD + tsm.c_prod_mat(2*dot(MDX, U)) - oneOverWi*II
    CxxOp = II - dt*MMM
    SVNew[:vecLen] = linalg.solve(CxxOp, CxxOld + AAA*dt)

    # Cyy via backwards Euler
    AAA = 2*dot(MMCXY, dot(MDX, U)) + oneOverWi*constOneVec
    MMM = VGRAD + 2*tsm.c_prod_mat(dot(MDY, V)) - oneOverWi*II
    CyyOp = II - dt*MMM
    SVNew[vecLen:2*vecLen] = linalg.solve(CyyOp, CyyOld + AAA*dt)

    # Cxy via backwards Euler
    AAA = dot(MMCXX, dot(MDX, V)) + dot(MMCYY, dot(MDY, U))
    MMM = VGRAD - oneOverWi*II
    CxyOp = II - dt*MMM
    SVNew[2*vecLen:3*vecLen] = linalg.solve(CxyOp, CxyOld + AAA*dt)

    return SVNew

def solve_stress_eqns(PSI, stressVec):
    """
    Returns the time derivative of the stresses given the current state of the
    flow.
    """

    U = dot(MDY, PSI)
    V = - dot(MDX, PSI)    
    MMU = tsm.c_prod_mat(U)
    MMV = tsm.c_prod_mat(V)

    # set up gradient operator
    VGRAD = dot(MMU, MDX) + dot(MMV, MDY)
    CxxOld = stressVec[0:vecLen]
    CyyOld = stressVec[vecLen:2*vecLen]
    CxyOld = stressVec[2*vecLen:3*vecLen]

    MMCXX = tsm.c_prod_mat(CxxOld)
    MMCYY = tsm.c_prod_mat(CyyOld)
    MMCXY = tsm.c_prod_mat(CxyOld)

    # Calculate polymeric stress components
    TxxOld = oneOverWi*CxxOld
    TxxOld[N*M] += -oneOverWi
    TyyOld = oneOverWi*CyyOld
    TyyOld[N*M] += -oneOverWi
    TxyOld = oneOverWi*CxyOld
    
    SVNew = zeros(3*vecLen, dtype='D')

    #dCxxdt 
    SVNew[:vecLen] = 2*dot(MMCXX, dot(MDX, U)) + 2*dot(MMCXY, dot(MDY, U))\
                     - dot(VGRAD, CxxOld) - TxxOld 
    #dCyydt
    SVNew[vecLen:2*vecLen] = + 2*dot(MMCXY, dot(MDX, V)) \
                             + 2*dot(MMCYY, dot(MDY, V)) \
                             - dot(VGRAD, CyyOld) - TyyOld
    #dCxydt
    SVNew[2*vecLen:3*vecLen] = dot(MMCXX, dot(MDX, V)) \
                              + dot(MMCYY, dot(MDY, U))\
                              - dot(VGRAD, CxyOld) - TxyOld
    
    return SVNew


def form_operators(_dt):
    """
    Make operators according to the timestep, _dt.
    """

    PsiOpInvList = []
    for i in range(N):
        n = i-N

        PSIOP = zeros((2*M, 2*M), dtype='complex')
        SLAPLAC = -n*n*kx*kx*SII + SMDYY

        PSIOP[0:M, 0:M] = 0
        PSIOP[0:M, M:2*M] = Re*SII - 0.5*beta*_dt*SLAPLAC

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
    Psi0thOp = Re*SMDY - 0.5*_dt*beta*SMDYYY + 0j

    # Apply BCs

    # dypsi0(+-1) = 0
    Psi0thOp[M-3, :] = DERIVTOP
    Psi0thOp[M-2, :] = DERIVBOT
    # psi0(-1) =  0
    Psi0thOp[M-1, :] = BBOT

    # compute lu factorisation, PSIOPLU blocks have l and u together, without
    # diagonal elements of l. PSIOPPIV elements are pivot vectors for the columns.

    Psi0thOp = linalg.lu_factor(Psi0thOp)

    
    return PsiOpInvList, Psi0thOp

def step_PSI(PSIOld, PsiOpInvList, Psi0thOp, time, _dt):

    """
    Step the streamfunction forward _dt in time via the fancy semi-implicit
    method.
    """

    # Make the vector for the RHS of the equations
    U = dot(MDY, PSIOld)
    V = - dot(MDX, PSIOld)    
    MMU = tsm.c_prod_mat(U)
    MMV = tsm.c_prod_mat(V)

    # PSI
    RHSVec = Re*dot(LAPLAC, PSIOld) \
            + _dt*0.5*beta*dot(BIHARM, PSIOld) \
            - _dt*Re*dot(MMU, dot(MDXLAPLAC, PSIOld)) \
            - _dt*Re*dot(MMV, dot(MDYLAPLAC, PSIOld))\
            + _dt*(1.-beta)*oneOverWi*(dot(MDXX, Cxy) \
                    + dot(MDXY,(Cyy - Cxx)) \
                    - dot(MDYY, Cxy) )

    # Zeroth mode
    RHSVec[N*M:(N+1)*M] = 0
    RHSVec[N*M:(N+1)*M] = + Re*dot(MDY, PSIOld)[N*M:(N+1)*M] \
              + _dt*0.5*beta*dot(MDYYY, PSIOld)[N*M:(N+1)*M] \
              - _dt*Re*dot(dot(MMV, MDYY), PSIOld)[N*M:(N+1)*M] \
              + _dt*(1.-beta)*oneOverWi*dot(MDY, Cxy)[N*M:(N+1)*M]
    # TODO: work out the implications of having dependence on the current time
    # rather than just the time step.
    RHSVec[N*M] += 2*_dt
    
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
    lu0, piv0 = Psi0thOp
    PSI[N*M:(N+1)*M] = linalg.lu_solve((lu0, piv0), RHSVec[N*M:(N+1)*M])

    for n in range(N):
        PSI[n*M:(n+1)*M] = dot(PsiOpInvList[n], RHSVec[n*M:(n+1)*M])
    del n  
 
    for n in range(N+1, 2*N+1):
        PSI[n*M:(n+1)*M] = conj(PSI[(2*N-n)*M:(2*N-n+1)*M])

    return PSI

    # calculations.

    U = dot(MDY, PSI)
    V = - dot(MDX, PSI)    
    MMU = tsm.c_prod_mat(U)
    MMV = tsm.c_prod_mat(V)

    # set up gradient operator
    VGRAD = dot(MMU, MDX) + dot(MMV, MDY)

    # Calculate polymeric stress components
    TxxOld = oneOverWi*CxxOld
    TxxOld[N*M] += -oneOverWi
    TyyOld = oneOverWi*CyyOld
    TyyOld[N*M] += -oneOverWi
    TxyOld = oneOverWi*CxyOld
    
    SVNew = zeros(3*vecLen, dtype='D')

    #dCxxdt 
    # - kx
    SVNew[:(N+1)*M] = 2*dot(MMCXX, dot(MDX, U))[:(N+1)*M]\
                + 2*dot(MMCXY, dot(MDY, U))[:(N+1)*M]\
                - dot(VGRAD, CxxOld)[:(N+1)*M] - TxxOld[:(N+1)*M]

    # + kx
    SVNew[(N+1)*M:vecLen] = conj(SVNew[:N*M])

    #dCyydt
    # - kx
    SVNew[vecLen:vecLen + (N+1)*M] = + 2*dot(MMCXY, dot(MDX, V))[:(N+1)*M] \
                                 + 2*dot(MMCYY, dot(MDY, V))[:(N+1)*M] \
                                 - dot(VGRAD, CyyOld)[:(N+1)*M] - TyyOld[:(N+1)*M]

    # + kx
    SVNew[vecLen + (N+1)*M : 2*vecLen] = conj(SVNew[vecLen : vecLen + N*M])

    #dCxydt
    # - kx
    SVNew[2*vecLen:2*vecLen + (N+1)*M] = dot(MMCXX, dot(MDX, V))[:(N+1)*M] \
                                     + dot(MMCYY, dot(MDY, U))[:(N+1)*M]\
                                     - dot(VGRAD, CxyOld)[:(N+1)*M] - TxyOld[:(N+1)*M]

    # + kx
    SVNew[2*vecLen + (N+1)*M : 3*vecLen] = conj(SVNew[2*vecLen : 2*vecLen + N*M])
    
    return SVNew

def rk4_step(x_old, dt):
    """ 
    Numerical recipies Runge-Kutta fourth order, where the derivative has no
    time dependence. 
    """

    k1 = dt*solve_stress_eqns(PSIOld, x_old) 
    k2 = dt*solve_stress_eqns(PSIhalf, x_old+k1/2.)
    k3 = dt*solve_stress_eqns(PSIhalf, x_old+k2/2.)
    k4 = dt*solve_stress_eqns(PSI, x_old+k3)

    x_new = x_old + k1/6. + k2/3. + k3/3. + k4/6.

    return x_new 

def rk5_step(x_old, dt):
    """ 
    Numerical recipies Runge-Kutta Cash-Karp.
    """

    b21 = 1./5.
    b31 = 3./40.
    b32 = 9./40.
    b41 = 3./10.
    b42 = -9./10.
    b43 = 6./5.
    b51 = -11./54.
    b52 = 5./2.
    b53 = -70./27.
    b54 = 35./27.
    b61 = 1631./55296.
    b62 = 175./512.
    b63 = 575./13824.
    b64 = 44275./110592.
    b65 = 253./4096.

    k1 = dt*solve_eqns(x_old) 
    k2 = dt*solve_eqns(x_old + b21*k1)
    k3 = dt*solve_eqns(x_old + b31*k1 + b32*k2)
    k4 = dt*solve_eqns(x_old + b41*k1 + b42*k2 + b43*k3)
    k5 = dt*solve_eqns(x_old + b51*k1 + b52*k2 + b53*k3 + b54*k4)
    k6 = dt*solve_eqns(x_old + b61*k2 + b62*k2 + b63*k3 + b64*k4)

    return x_old + (37./378.)*k1 + (250./621.)*k3 + (124./594.)*k4 \
            + (512./1771.)*k6

def explct_euler_step(x_old, dt): 
    return x_old + dt*solve_eqns(x_old)

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
invRe = 1. / Re
oneOverWi = 1. / Wi 


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

II = eye((2*N+1)*M, (2*N+1)*M, dtype='complex')
constOneVec = zeros((2*N+1)*M, dtype='complex')
constOneVec[N*M] = 1.

# single mode Operators
SMDY = tsm.mk_single_diffy()
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

# Perturb first 3 Chebyshevs for linear stability
#PSI[(N-1)*M:(N-1)*M + 3] = 1e-3#*(random.random(3) + 1.j*random.random(3))
#PSI[(N+1)*M:(N+2)*M] = conjugate(PSI[(N-1)*M:N*M])

#PSI[N*M]   += 2.0/3.0
#PSI[N*M+1] += 3.0/4.0
#PSI[N*M+2] += 0.0
#PSI[N*M+3] += -1.0/12.0

PSI,Cxx,Cyy,Cxy,Nu = pickle.load(open(inFileName, 'r'))

# Initial Stress is Newtonian laminar base profile stress.

#Cxx = zeros(vecLen, dtype='complex')
#Cxx = 2*(Wi**2)*dot(MDYY, PSI)
#Cxx[N*M] += 1.0
#Cyy = zeros(vecLen, dtype='complex')
#Cyy[N*M] += 1.0
#Cxy = zeros(vecLen, dtype='complex')
#Cxy = Wi*(dot(MDYY, PSI))

stressOld = zeros(3*vecLen, dtype='D')
stressOld[:vecLen] = Cxx
stressOld[vecLen:2*vecLen] = Cyy
stressOld[2*vecLen:3*vecLen] = Cxy

# FORM THE STREAMFUNCTION OPERATORS

# Form the full-step operators
PsiOpInvListFull, Psi0thOpFull = form_operators(dt) 

# Form the half-step operators
PsiOpInvListHalf, Psi0thOpHalf = form_operators(dt/2.) 

## INITIALISE BEFORE THE LOOP

RHSVec = zeros(vecLen, dtype='complex')
# open the files
traceOutFp = open(outFileNameTrace, 'w')
psiSeriesFp = open(outFileNameTime, 'w')
pickle.dump((PSI, Cxx, Cyy, Cxy), psiSeriesFp)

# form a list of times
timesList = r_[dt:dt*numTimeSteps:dt]

## ITERATE THE FLOW PROFILE

print """
Beginning Time Iteration:
=====================================

"""
for tindx, currTime in enumerate(timesList):

    PSIOld = copy(PSI)
    CxxOld = copy(Cxx)
    CyyOld = copy(Cyy)
    CxyOld = copy(Cxy)
    
    # Step PSI to t + dt/2
    PSIhalf = step_PSI(PSIOld, PsiOpInvListHalf, Psi0thOpHalf, currTime, dt/2.)

    # Step PSI to t + dt
    PSI = step_PSI(PSIOld, PsiOpInvListFull, Psi0thOpFull, currTime, dt)

    # Step the Stresses
    stressOld[:vecLen] = CxxOld
    stressOld[vecLen:2*vecLen] = CyyOld
    stressOld[2*vecLen:3*vecLen] = CxyOld

    stressNew = rk4_step(stressOld,  dt)

    Cxx = stressNew[:vecLen] 
    Cyy = stressNew[vecLen:2*vecLen]
    Cxy = stressNew[2*vecLen:3*vecLen]

    # KE outputted is always that of the previous step

    U = dot(MDY, PSIOld)
    V = - dot(MDX, PSIOld)    
    MMU = tsm.c_prod_mat(U)
    MMV = tsm.c_prod_mat(V)

    Usq = dot(MMU,U) + dot(MMV,V)
    KE0 = (15./8.)*0.5*real(dot(INTY, Usq[N*M:(N+1)*M]))
    Usq1 = Usq[(N-1)*M:N*M]*exp(1.j*kx) + Usq[(N+1)*M:(N+2)*M]*exp(-1.j*kx)
    KE1  = (15./8.)*0.5*real(dot(INTY, Usq1))

    if not tindx % (numTimeSteps/numFrames):
        pickle.dump((PSI, Cxx, Cyy, Cxy), psiSeriesFp)
        print "{0:15.8g} {1:15.8g} {2:15.8g}".format(currTime-dt, KE0, KE1)

    traceOutFp.write(
        "{0:15.8g} {1:15.8g} {2:15.8g} \n".format(
            currTime-dt, KE0, KE1, ))

traceOutFp.close()
psiSeriesFp.close()
pickle.dump(PSI, open(outFileName, 'w'))
