#-----------------------------------------------------------------------------
#   2D ECS finder
#
#   Last modified: Wed 17 Jun 12:31:19 2015
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
from numpy.fft import fftshift
import h5py

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

NRdelta = 1e-7   # Newton-Rhaphson tolerance
y_star  = 0.5

ReOld = Re #+ 10
kxOld = kx #- 0.01
NOld = N#-1
MOld = M
baseFileName = "-N{N}-M{M}-Re{Re}-kx{kx}".format(N=N, M=M, kx=kx, Re=Re)
outFileName = "psi-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=N, M=M, kx=kx, Re=Re)
inFileName= "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=NOld, M=MOld, kx=kxOld,
                                                       Re=ReOld)
tsm.initTSM(N, M, kx)
#------------------------------------------------

#FUNCTIONS

def solve_eq(xVec):
    """calculates the residuals of equations and the jacobian that ought to
    generate them for minimisation via Newton-Rhaphson"""
    
    PSI = xVec[0:vecLen] 


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

    residualsVec = zeros((vecLen), dtype='complex')

    #####psi
    residualsVec[0:vecLen] = - Re*dot(MMU, dot(MDX, LAPLACPSI)) \
                             - Re*dot(MMV, dot(MDY, LAPLACPSI))  \
                             + dot(BIHARM, PSI)

    #####psi0
    residualsVec[N*M:(N+1)*M] = - Re*dot(dot(MMV,MDY), U)[N*M:(N+1)*M] \
                                + dot(MDYYY, PSI)[N*M:(N+1)*M] \
    # set the pressure gradient (pressure driven flow)
    residualsVec[N*M:(N+1)*M] += forcingVec


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

    # dyPsi0(+-1) = +-1
    residualsVec[N*M + M-3] = dot(DERIVTOP, (PSI[N*M:(N+1)*M])) -1.0
    residualsVec[N*M + M-2] = dot(DERIVBOT, (PSI[N*M:(N+1)*M])) +1.0

    # Psi0(-1) = 0
    residualsVec[N*M + M-1] = dot(BBOT, (PSI[N*M:(N+1)*M]))

    #################SET THE JACOBIAN MATRIX####################

    jacobian = zeros((vecLen,  vecLen), dtype='complex')

    ###### psi
    ##psi
    jacobian[0:vecLen, 0:vecLen] = - Re*dot(dot(MMU, MDX), LAPLAC) \
                                   - Re*dot(dot(MMV, MDY), LAPLAC) \
                                   + Re*dot(tsm.prod_mat(dot(MDY, LAPLACPSI)), MDX) \
                                   - Re*dot(tsm.prod_mat(dot(MDX, LAPLACPSI)), MDY) \
                                   + BIHARM 

    ###### U0 equation
    #set row to zero
    jacobian[N*M:(N+1)*M, :] = 0
    ##u0
    jacobian[N*M:(N+1)*M, 0:vecLen] = \
                            + Re*dot(tsm.prod_mat(dot(MDX, PSI)), MDYY)[N*M:(N+1)*M, :]\
                            + Re*dot(tsm.prod_mat(dot(MDYY, PSI)), MDX)[N*M:(N+1)*M, :]\
                            + MDYYY[N*M:(N+1)*M, :]

    #######apply BC's to jacobian

    # Apply BC to zeroth mode
    # dypsi0 = const
    jacobian[N*M + M-3, 0:vecLen] = \
        concatenate( (zeros(N*M), DERIVTOP, zeros(N*M)) )
    jacobian[N*M + M-2, 0:vecLen] = \
        concatenate( (zeros(N*M), DERIVBOT, zeros(N*M)) )
    # psi(-1) = const 
    jacobian[N*M + M-1, 0:vecLen] = \
        concatenate( (zeros(N*M), BBOT, zeros(N*M)) )

    for n in range(2*N+1):
        if n == N: continue     # Don't apply bcs to psi0 mode here
        # dxpsi = 0
        jacobian[n*M + M-2, 0 : vecLen] = \
            concatenate( (zeros(n*M), (n-N)*kx*1.j*BTOP, zeros((2*N-n)*M)) )
        jacobian[n*M + M-1, 0 : vecLen] = \
            concatenate( (zeros(n*M), (n-N)*kx*1.j*BBOT, zeros((2*N-n)*M)) )
        # -dypsi = const
        jacobian[n*M + M-4, 0:vecLen] = \
            concatenate( (zeros(n*M), DERIVTOP, zeros((2*N-n)*M)) )
        jacobian[n*M + M-3, 0:vecLen] = \
            concatenate( (zeros(n*M), DERIVBOT, zeros((2*N-n)*M)) )
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

def decrease_resolution(vec):
    """ 
    decrease both the N and M resolutions
    """

    lowMvec = zeros((2*NOld+1)*M, dtype='complex')
    for n in range(2*NOld+1):
        lowMvec[n*M:(n+1)*M] = vec[n*MOld:n*MOld + M]
    del n

    lowNMvec = zeros((2*N+1)*M, dtype='D')
    lowNMvec = lowMvec[(NOld-N)*M:(NOld-N)*M + (2*N+1)*M]

    return lowNMvec

def decide_resolution(vec):
    """
    Choose to increase or decrease resolution depending on values of N,M
    NOld,MOld.
    """
    if N >= NOld and M >= MOld:
        ovec = increase_resolution(vec)

    elif N <= NOld and M <= MOld:
        ovec = decrease_resolution(vec)

    return ovec

def stupid_transform(GLreal):
    """
    apply the Chebyshev transform the stupid way.
    """

    out = zeros(M)

    for i in range(M):
        out[i] += (1./(M-1.))*GLreal[0]
        for j in range(1,M-1):
            out[i] += (2./(M-1.))*GLreal[j]*cos(pi*i*j/(M-1))
        out[i] += (1./(M-1.))*GLreal[M-1]*cos(pi*i)
    del i,j

    out[0] = out[0]/2.
    out[M-1] = out[M-1]/2.

    return out



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
#f = open('field_Re=3000.0_kx=1.313_N=2_M_40.txt','r')
#_ = pickle.load(f)
#_ = pickle.load(f)
#xVec = pickle.load(f)
#PSI = xVec[:(2*N+1)*M]

# read in from ti output
f = h5py.File("output/final.h5","r")
PSIOld = array(f["psi"])
PSIOld = PSIOld.reshape((2*NOld+1, MOld)).T
PSIOld = fftshift(PSIOld, axes=1)
PSIOld = PSIOld.T.flatten()

PSI = PSIOld
#PSI = increase_resolution(PSIOld)
#PSI = decrease_resolution(PSIOld)

#PSI = pickle.load(open('psi.init', 'r'))

print inFileName

# Set up shear layer forcing

y_points = cos(pi*arange(M)/(M-1))
delta = 0.1

forcing = zeros((M), dtype='d')

for i in range(M):
    y =y_points[i]
    forcing[i] = ( 2.0/tanh(1.0/delta)) * (1.0/cosh(y/delta)**2) * tanh(y/delta)
    forcing[i] *= 1.0/(delta**2) 
del y, i

forcingVec = stupid_transform(forcing)

# Try to move to the other branch
#push = 3.02e-2
#PSI[N*M] -= push
#PSI[N*M+1] -= push
#PSI[(N-1)*M] += push
#PSI[(N-1)*M+1] += push
#PSI[(N+1)*M:(N+2)*M] = conj(PSI[(N-1)*M:N*M])

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

xVec = zeros((vecLen), dtype='complex')
xVec[0:vecLen] = PSI

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

INTY = tsm.mk_cheb_int()

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

print "Begin Newton-Rhaphson"
print "------------------------------------"
print "L2norm:"
while True:
    (J_x0, f_x0) = solve_eq(xVec)
    print shape(J_x0)
    print shape(f_x0)
    dx = linalg.solve(J_x0, -f_x0)
    xVec = xVec + relax*dx
    L2norm = linalg.norm(f_x0,2)
    print "\t {L2norm}".format(L2norm=L2norm)
    if (L2norm < NRdelta): 
        PSIans = xVec[0:vecLen] 
        print 
        print "------------------------------------\n"
        PSIans = xVec[0:vecLen] 
        print "=====================================\n"
        print PSIans[(N-1)*M: N*M]

        print 'Solution Found!'
        U = dot(MDY, PSIans)
        V = -dot(MDX, PSIans)
        MMU = tsm.prod_mat(U)
        MMV = tsm.prod_mat(V)
        U0sq = (dot(MMU,U) + dot(MMV,V))[N*M:(N+1)*M]
        if not allclose(almostZero, imag(U0sq)):
            print "Caution! Imaginary velocities with norm ", linalg.norm(imag(U0sq))
            print "Symmetrising anyway"
            PSIans = symmetrise(PSIans)

        pickle.dump(PSIans, open(outFileName, 'w'))

        KE0 = 0.5*real(dot(INTY, U0sq))
        print 'KE0 = ', KE0
        break

    xVec[:(2*N+1)*M] = symmetrise(xVec[:(2*N+1)*M])
    
    if (L2norm > 1e20): break

