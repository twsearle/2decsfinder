#-----------------------------------------------------------------------------
#   2D ECS finder
#
#   Last modified: Wed 18 Dec 16:58:29 2013
#
#-----------------------------------------------------------------------------

""" Program to find Exact coherent states from a given flow profile using
Newton-Raphson and the Oldroyd-B model."""

#MODULES
from scipy import *
from scipy import linalg
import ConfigParser
import cPickle as pickle
import TobySpectralMethods as tsm
import sys

#SETTINGS----------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
amp = config.getfloat('Newton-Raphson', 'amp')
relax = config.getfloat('Newton-Raphson', 'relax')

kx = float(sys.argv[1])

fp.close()

NRdelta = 1e-06     # Newton-Rhaphson tolerance
y_star  = 0.5

ReOld = Re 
kxOld = kx
baseFileName = "-N{N}-M{M}-Re{Re}-kx{kx}".format(N=N, M=M, kx=kx, Re=Re)
inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=N, M=M, kx=kxOld,
                                                       Re=ReOld)
outTraceFileName = "KE-trace-N{N}-M{M}-kx{kx}.dat".format(N=N, M=M, kx=kx)

ReList = flipud(r_[0:Re+0.5:0.5])

tsm.initTSM(N, M, kx)
#------------------------------------------------

#FUNCTIONS

def solve_eq(xVec):
    """calculates the residuals of equations and the jacobian that ought to
    generate them for minimisation via Newton-Rhaphson"""
    
    PSI = xVec[0:vecLen] 
    Nu  = xVec[vecLen]


    U         = + dot(MDY, PSI)
    V         = - dot(MDX, PSI)
    LAPLACPSI = dot(LAPLAC, PSI)

    # Useful Operators
    MMU    = tsm.prod_mat(U)
    MMV    = tsm.prod_mat(V)
    VGRAD  = dot(MMU,MDX) + dot(MMV,MDY)
    MMDXU  = tsm.prod_mat(dot(MDX, U))
    MMDXV  = tsm.prod_mat(dot(MDX, V))
    MMDYU  = tsm.prod_mat(dot(MDY, U))
    MMDYV  = tsm.prod_mat(dot(MDY, V))

    MMDXPSI   = tsm.prod_mat(dot(MDX, LAPLACPSI))

    # a vector with only constant component
    constVec = zeros(((2*N+1)*M), dtype='complex')
    constVec[N*M] = 1.0

    #######calculate the Residuals########

    residualsVec = zeros((vecLen + 1), dtype='complex')

    #####psi
    residualsVec[0:vecLen] = + Re*Nu*dot(dot(MDX,LAPLAC),PSI) \
                             - Re*dot(MMU, dot(MDX, LAPLACPSI)) \
                             - Re*dot(MMV, dot(MDY, LAPLACPSI))  \
                             + dot(BIHARM, PSI)

    #####Nu
    #residualsVec[vecLen] = dot(SPEEDCONDITION[0:(2*N+1)*M] , PSI)
    residualsVec[vecLen] = imag(dot(SPEEDCONDITION, PSI[(N+1)*M:(N+2)*M]))


    #####psi0
    residualsVec[N*M:(N+1)*M] = - Re*dot(dot(MMV,MDY), U)[N*M:(N+1)*M] \
                                + dot(MDYYY, PSI)[N*M:(N+1)*M] \
    # set the pressure gradient (pressure driven flow)
    residualsVec[N*M] += 2.0


    ##### Apply boundary conditions to residuals vector

    # dxPsi = 0   
    for k in range (2*N+1): 
        if k == N: continue # skip the 0th component 
        residualsVec[k*M + M-2] = dot((k-N)*1.j*kx*BTOP, PSI[k*M:(k+1)*M])
        residualsVec[k*M + M-1] = dot((k-N)*1.j*kx*BBOT, PSI[k*M:(k+1)*M])
    del k

    # dyPsi(+-1) = 0 
    for k in range (2*N+1):
        if k == N: continue # skip the 0th component 
        residualsVec[k*M + M-4] = dot(DERIVTOP, PSI[k*M:(k+1)*M])
        residualsVec[k*M + M-3] = dot(DERIVBOT, PSI[k*M:(k+1)*M])
    del k

    # dyPsi0(+-1) = 0
    residualsVec[N*M + M-3] = dot(DERIVTOP, (PSI[N*M:(N+1)*M]))
    residualsVec[N*M + M-2] = dot(DERIVBOT, (PSI[N*M:(N+1)*M]))

    # Psi0(-1) = 0
    residualsVec[N*M + M-1] = dot(BBOT, (PSI[N*M:(N+1)*M]))

    #################SET THE JACOBIAN MATRIX####################

    jacobian = zeros((vecLen+1,  vecLen+1), dtype='complex')

    ###### psi
    ##psi
    jacobian[0:vecLen, 0:vecLen] = + Nu*Re*dot(MDX, LAPLAC) \
                                   - Re*dot(dot(MMU, MDX), LAPLAC) \
                                   - Re*dot(dot(MMV, MDY), LAPLAC) \
                                   + Re*dot(tsm.prod_mat(dot(MDY, LAPLACPSI)), MDX) \
                                   - Re*dot(tsm.prod_mat(dot(MDX, LAPLACPSI)), MDY) \
                                   + BIHARM 
    ##Nu - vector not a product matrix
    jacobian[0:vecLen, vecLen] = Re*dot(MDX, LAPLACPSI)

    ##### Nu 
    jacobian[vecLen, (N+1)*M:(N+2)*M] = -1.j*0.5*SPEEDCONDITION
    jacobian[vecLen, (N-1)*M:N*M] = 1.j*0.5*SPEEDCONDITION

    ###### U0 equation
    #set row to zero
    jacobian[N*M:(N+1)*M, :] = 0
    ##u0
    jacobian[N*M:(N+1)*M, 0:vecLen] = \
                            + Re*dot(tsm.prod_mat(dot(MDX, PSI)), MDYY)[N*M:(N+1)*M, :]\
                            + Re*dot(tsm.prod_mat(dot(MDYY, PSI)), MDX)[N*M:(N+1)*M, :]\
                            + MDYYY[N*M:(N+1)*M, :]
    ##nu
    jacobian[N*M:(N+1)*M, vecLen] = 0

    #######apply BC's to jacobian

    # Apply BC to zeroth mode
    # dypsi0 = const
    jacobian[N*M + M-3, 0:vecLen + 1] = \
        concatenate( (zeros(N*M), DERIVTOP, zeros(N*M+1)) )
    jacobian[N*M + M-2, 0:vecLen + 1] = \
        concatenate( (zeros(N*M), DERIVBOT, zeros(N*M+1)) )
    # psi(-1) = const 
    jacobian[N*M + M-1, 0:vecLen + 1] = \
        concatenate( (zeros(N*M), BBOT, zeros(N*M+1)) )

    for n in range(2*N+1):
        if n == N: continue     # Don't apply bcs to psi0 mode here
        # dxpsi = 0
        jacobian[n*M + M-2, 0 : vecLen + 1] = \
            concatenate( (zeros(n*M), (n-N)*kx*1.j*BTOP, zeros((2*N-n)*M+1)) )
        jacobian[n*M + M-1, 0 : vecLen + 1] = \
            concatenate( (zeros(n*M), (n-N)*kx*1.j*BBOT, zeros((2*N-n)*M+1)) )
        # -dypsi = const
        jacobian[n*M + M-4, 0:vecLen + 1] = \
            concatenate( (zeros(n*M), DERIVTOP, zeros((2*N-n)*M+1)) )
        jacobian[n*M + M-3, 0:vecLen + 1] = \
            concatenate( (zeros(n*M), DERIVBOT, zeros((2*N-n)*M+1)) )
    del n

    return(jacobian, residualsVec)

def symmetrise(vec):
    """symmetrise the vector thingy to make unbroke"""

    tmp = zeros(vecLen, dtype='complex')
    for n in range(N):
        tmp[n*M:(n+1)*M] = 0.5*conj(vec[vecLen-(n+1)*M:vecLen-n*M])\
                         + 0.5*vec[n*M:(n+1)*M]
        tmp[vecLen-(n+1)*M:vecLen-n*M] = conj(tmp[n*M:(n+1)*M])
    del n
    tmp[N*M:(N+1)*M] = real(vec[N*M:(N+1)*M])

    return tmp

def mk_cheb_int():
    integrator = zeros(M, dtype='d')
    for m in range(0,M,2):
        integrator[m] = (1 + cos(m*pi)) / (1-m*m)
    del m
    return integrator

#MAIN


# setup the initial conditions 

vecLen = M*(2*N+1)

print "=====================================\n"
print "Settings:"
print """------------------------------------
N \t= {N}
M \t= {M}              
Re \t= {Re}         
kx \t= {kx}
amp\t= {amp}
------------------------------------
        """.format(N=N, M=M, kx=kx, Re=Re, amp=amp )

print "The length of a vector is: ", vecLen
NuIndx = M - 5           #Choose a high Fourier mode

almostZero = zeros(M, dtype='D') + 1e-14

PSI = zeros(vecLen, dtype='complex')

# Calculate the phase factor

#PSI[N*M]   += 2.0/3.0
#PSI[N*M+1] += 3.0/4.0
#PSI[N*M+2] += 0.0
#PSI[N*M+3] += -1.0/12.0


# PSI = pickle.load(open('Alex_psi.pickle','r'))

# Read in profile from previous step

PSI, Nu = pickle.load(open(inFileName, 'r'))
print inFileName
print "Nu = ", Nu

# PSI = pickle.load(open('psi'+baseFileName+'-t1000.0.pickle', 'r'))

# apply the phase factor
#PSI0RS = 0. + 0.j
#for m in range(M):
#    PSI0RS += PSI[(N-1)*M + m]*cos(m*arccos(y_star))
#del m

#phaseFactor = 1. - 1.j*(imag(PSI0RS)/real(PSI0RS))

#print "The Phase factor: ", phaseFactor

#PSI[(N-1)*M:N*M] = phaseFactor*PSI[(N-1)*M:N*M] 
#PSI[N*M:(N+1)*M] = conjugate(phaseFactor)*PSI[N*M:(N+1)*M] 

xVec = zeros((vecLen + 1), dtype='complex')
xVec[0:vecLen] = PSI
xVec[vecLen] = Nu 

# Useful operators 

MDY = tsm.mk_diff_y()
MDYY = dot(MDY,MDY)
MDYYY = dot(MDY,MDYY)
MDX = tsm.mk_diff_x()
MDXX = dot(MDX, MDX)
MDXY = dot(MDX, MDY)
LAPLAC = dot(MDX,MDX) + dot(MDY,MDY)
BIHARM = dot(LAPLAC, LAPLAC)

INTY = mk_cheb_int()

#Identity
II = eye(vecLen, vecLen, dtype='complex')

# Boundary arrays
BTOP = ones(M)
BBOT = ones(M)
BBOT[1:M:2] = -1

singleDY = tsm.mk_single_diffy()
DERIVTOP = zeros((M), dtype='complex')
DERIVBOT = zeros((M), dtype='complex')
for j in range(M):
    DERIVTOP[j] = dot(BTOP, singleDY[:,j]) 
    DERIVBOT[j] = dot(BBOT, singleDY[:,j])
del j

# Set only the imaginary part of a Fourier component to zero to constrain
#  nu. I will choose y = 0.5. Use the 1st mode for this condition
SPEEDCONDITION = zeros(M, dtype = 'complex')
for m in range(M):
    SPEEDCONDITION[m] = cos(m*arccos(y_star)) 

outKEfp = open(outTraceFileName, 'w')
for Re in ReList:
    outFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=N, M=M, kx=kx, Re=Re)

    print "Begin Newton-Rhaphson at Re = ", Re
    print "------------------------------------"
    print "L2norm:"
    L2norm = 1.0
    counter = 0

    while (L2norm > NRdelta):
        counter +=1
        # Iterate until you find a solution
        (J_x0, f_x0) = solve_eq(xVec)
        dx = linalg.solve(J_x0, -f_x0)
        xVec = xVec + relax*dx
        L2norm = linalg.norm(f_x0,2)
        print "\t {L2norm}".format(L2norm=L2norm)
        PSIans = xVec[0:vecLen] 
        Nuans  = xVec[vecLen]
        
        if (L2norm > 1e20) :
            # If convergence fails once, end the program
            print "Lost stability"
            outKEfp.close()
            exit(1)

        if (counter > 30) :
            print "Not going to converge"
            outKEfp.close()
            exit(1)

        # Force symmetrisation
        xVec[:vecLen] = symmetrise(xVec[:vecLen])

    print "------------------------------------\n"
    PSIans = xVec[0:vecLen] 
    Nuans  = xVec[vecLen]
    print "Nu = ", Nu
    print "=====================================\n"
    # print PSIans[(N-1)*M: N*M]

    # Make sure solution is not trivial before outputing
    if any(greater(PSIans[(N-1)*M: N*M], almostZero)):
        print 'Solution Found!'
        pickle.dump((PSIans,Nuans), open(outFileName, 'w'))
        PSI = PSIans
        Nu = Nuans

        realityArr = []
        for n in range(N):
            cond = allclose(PSI[n*M:(n+1)*M],conj(PSI[(2*N-n)*M:(2*N+1-n)*M]))
            realityArr.append(cond)
        del n
        realityTest = all(realityArr)
        
        U = dot(MDY, PSI)
        V = -dot(MDX, PSI)
        MMU = tsm.prod_mat(U)
        MMV = tsm.prod_mat(V)
        U0sq = (dot(MMU,U) + dot(MMV,V))[N*M:(N+1)*M]
        assert allclose(almostZero, imag(U0sq)), "Imaginary velocities!"
        KE0 = 0.5*real(dot(INTY, U0sq))
        print 'KE0 = ', KE0
        outKEfp.write("{0}\t{1}\t{2}\t{3}\n".format(Re, kx, KE0, realityTest))
        outKEfp.flush()

outKEfp.close()
