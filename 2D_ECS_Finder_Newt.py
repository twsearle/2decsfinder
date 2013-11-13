#-----------------------------------------------------------------------------
#   2D ECS finder
#
#   Last modified: Wed 13 Nov 13:30:02 2013
#
#-----------------------------------------------------------------------------

""" Program to find Exact coherent states from a given flow profile using
Newton-Rhaphson and the Oldroyd-B model."""

#MODULES
from scipy import *
from scipy import linalg
import matplotlib.pyplot as plt
import cPickle as pickle

#SETTINGS----------------------------------------

N = 5              # Number of Fourier modes
M = 30               # Number of Chebychevs (>4)
Re = 5771.0           # The Reynold's number
kx  = 1.0

NRdelta = 1e-06     # Newton-Rhaphson tolerance

#------------------------------------------------

#FUNCTIONS

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

def solve_eq(xVec):
    """calculates the residuals of equations and the jacobian that ought to
    generate them for minimisation via Newton-Rhaphson"""
    
    PSI = xVec[0:vecLen] 
    Nu  = xVec[vecLen]


    U         = + dot(MDY, PSI)
    V         = - dot(MDX, PSI)
    LAPLACPSI = dot(LAPLAC, PSI)

    # Useful Operators
    MMU    = prod_mat(U)
    MMV    = prod_mat(V)
    VGRAD  = dot(MMU,MDX) + dot(MMV,MDY)
    MMDXU  = prod_mat(dot(MDX, U))
    MMDXV  = prod_mat(dot(MDX, V))
    MMDYU  = prod_mat(dot(MDY, U))
    MMDYV  = prod_mat(dot(MDY, V))

    MMDXPSI   = prod_mat(dot(MDX, LAPLACPSI))

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
                                   - Re*dot(prod_mat(dot(MDY, LAPLACPSI)), MDX) \
                                   + Re*dot(prod_mat(dot(MDX, LAPLACPSI)), MDY) \
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
                            + Re*dot(prod_mat(dot(MDX, PSI)), MDYY)[N*M:(N+1)*M, :]\
                            + Re*dot(prod_mat(dot(MDYY, PSI)), MDX)[N*M:(N+1)*M, :]\
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

outFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=N, M=M,
                                                                     kx=kx, Re=Re)

# setup the initial conditions 

vecLen = M*(2*N+1)  
print"=====================================\n"
print "Settings:"
print """------------------------------------
N \t= {N}
M \t= {M}              
Re \t= {Re}         
kx \t= {kx}
------------------------------------
        """.format(N=N, M=M, kx=kx, Re=Re )

print "The length of a vector is: ", vecLen
NuIndx = M - 5           #Choose a high Fourier mode
# Set the oneOverC function: 1/2 for m=0, 1 elsewhere:
oneOverC = ones(M)
oneOverC[0] = 1. / 2.
#set up the CFunc function: 2 for m=0, 1 elsewhere:
CFunc = ones(M)
CFunc[0] = 2.

PSI = random.random(vecLen)/1000000.0

#PSI = zeros(vecLen, dtype='complex')
PSI[N*M]   = -2.0/3.0
PSI[N*M+1] = -3.0/4.0
PSI[N*M+2] = 0.0
PSI[N*M+3] = 1.0/12.0

Nu  = 0.01

xVec = zeros((vecLen + 1), dtype='complex')
xVec[0:vecLen]          = PSI
xVec[vecLen]          = Nu 

# Useful operators 

MDY = mk_diff_y()
MDYY = dot(MDY,MDY)
MDYYY = dot(MDY,MDYY)
MDX = mk_diff_x()
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

singleDY = mk_single_diffy()
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
    xVec = xVec + dx
    L2norm = linalg.norm(f_x0,2)
    print "\t {L2norm}".format(L2norm=L2norm)
    if (L2norm < NRdelta): break

print "------------------------------------\n"
PSI = xVec[0:vecLen] 
Nu  = xVec[vecLen]
print "Nu = ", Nu
print"=====================================\n"

pickle.dump((PSI,Nu), open(outFileName, 'w'))
