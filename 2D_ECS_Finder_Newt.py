#-----------------------------------------------------------------------------
#   2D ECS finder
#
#   Last modified: Fri  6 Dec 14:42:18 2013
#
#-----------------------------------------------------------------------------

""" Program to find Exact coherent states from a given flow profile using
Newton-Raphson and the Oldroyd-B model."""

#MODULES
from scipy import *
from scipy import linalg
import ConfigParser
import matplotlib.pyplot as plt
import cPickle as pickle
import TobySpectralMethods as tsm

#SETTINGS----------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
kx = config.getfloat('General', 'kx')
amp = config.getfloat('Newton-Raphson', 'amp')
relax = config.getfloat('Newton-Raphson', 'relax')

fp.close()

NRdelta = 1e-06     # Newton-Rhaphson tolerance

ReOld = Re
kxOld = kx+0.001
outFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=N, M=M, kx=kx, Re=Re)
inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=N, M=M, kx=kxOld,
                                                        Re=ReOld)
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
    residualsVec[0:vecLen] = - Re*Nu*dot(dot(MDX,LAPLAC),PSI) \
                             + Re*dot(MMU, dot(MDX, LAPLACPSI)) \
                             + Re*dot(MMV, dot(MDY, LAPLACPSI))  \
                             - dot(BIHARM, PSI)

    #####Nu
    residualsVec[vecLen] = dot(SPEEDCONDITION[0:(2*N+1)*M] , PSI)

    #####psi0
    residualsVec[N*M:(N+1)*M] = - Re*dot(VGRAD, U)[N*M:(N+1)*M] \
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
    jacobian[0:vecLen, 0:vecLen] = - Nu*Re*dot(MDX, LAPLAC) \
                                   + Re*dot(dot(MMU, MDX), LAPLAC) \
                                   + Re*dot(dot(MMV, MDY), LAPLAC) \
                                   - Re*dot(tsm.prod_mat(dot(MDY, LAPLACPSI)), MDX) \
                                   + Re*dot(tsm.prod_mat(dot(MDX, LAPLACPSI)), MDY) \
                                   - BIHARM 
    ##Nu - vector not a product matrix
    jacobian[0:vecLen, vecLen] = Re*dot(MDX, LAPLACPSI)

    ##### Nu 
    jacobian[vecLen, :] = SPEEDCONDITION

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

# Normal people's stream-function notation

PSI[N*M]   += 2.0/3.0
PSI[N*M+1] += 3.0/4.0
PSI[N*M+2] += 0.0
PSI[N*M+3] += -1.0/12.0

PSI[(N-1)*M:N*M] = amp*pickle.load(open(inFileName,'r'))
PSI[(N+1)*M:(N+2)*M] = conjugate(PSI[(N-1)*M:N*M])

PSI0RS = 0. + 0.j
for m in range(M):
    PSI0RS += PSI[(N-1)*M + m]*cos(m*arccos(0.5))
del m

phaseFactor = 1. - 1.j*(imag(PSI0RS)/real(PSI0RS))

print "The Phase factor: ", phaseFactor

PSI[(N-1)*M:N*M] = phaseFactor*PSI[(N-1)*M:N*M] 
PSI[N*M:(N+1)*M] = conjugate(phaseFactor)*PSI[N*M:(N+1)*M] 

PSI = pickle.load(open('Alex_psi.pickle','r'))

Nu = 0.35811001226 + 3.61762360029j*1.e-14


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
SPEEDCONDITION = zeros(vecLen+1, dtype = 'complex')
for m in range(M):
    SPEEDCONDITION[(N-1)*M + m] = cos(m*arccos(0.5)) 
    SPEEDCONDITION[(N+1)*M + m] = -cos(m*arccos(0.5))

print "Begin Newton-Rhaphson"
print "------------------------------------"
print "L2norm:"
while True:
    (J_x0, f_x0) = solve_eq(xVec)
    dx = linalg.solve(J_x0, -f_x0)
    xVec = xVec + relax*dx
    L2norm = linalg.norm(f_x0,2)
    print "\t {L2norm}".format(L2norm=L2norm)
    PSIans = xVec[0:vecLen] 
    Nuans  = xVec[vecLen]
    pickle.dump((PSIans,Nuans), open(outFileName, 'w'))
    if (L2norm < NRdelta): 
        print 
        print "------------------------------------\n"
        PSIans = xVec[0:vecLen] 
        Nuans  = xVec[vecLen]
        print "Nu = ", Nu
        print "=====================================\n"
        print PSIans[(N-1)*M: N*M]

        if any(greater(PSIans[(N-1)*M: N*M], almostZero)):
            print 'Solution Found!'
            pickle.dump((PSIans,Nuans), open(outFileName, 'w'))
        break
    
    if (L2norm > 1e20): break

