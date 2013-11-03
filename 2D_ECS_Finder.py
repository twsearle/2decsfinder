#-----------------------------------------------------------------------------
#   2D ECS finder
#
#   Last modified: Sun  3 Nov 21:06:59 2013
#
#-----------------------------------------------------------------------------

""" Program to find Exact coherent states from a given flow profile using
Newton-Rhaphson and the Oldroyd-B model."""

#MODULES
from scipy import *
from scipy import linalg
import cPickle as pickle

#SETTINGS----------------------------------------

N = 10              # Number of Fourier modes
M = 20              # Number of Chebychevs (>6)
Wi = 0.01           # The Weissenberg number
Re = 10.0           # The Reynold's number
beta = 0.1
kx  = 0.06

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
    generate them for comparison via Newton-Rhaphson"""
    
    PSI = xVec[0:vecLen] 
    Cxx = xVec[1*vecLen:2*vecLen] 
    Cyy = xVec[2*vecLen:3*vecLen] 
    Cxy = xVec[3*vecLen:4*vecLen]
    Nu  = xVec[4*vecLen]


    Txx = oneOverWi * (Cxx - 1.)
    Tyy = oneOverWi * (Cyy - 1.)
    Txy = oneOverWi * Cxy
    print size(Txy)
    print size(MDYY)

    # Useful Operators
    U      = -dot(MDY,PSI)
    V      =  dot(MDX,PSI)
    VGRAD  = dot(U,MDX) + dot(V,MDY)
    MMU    = prod_mat(U)
    MMV    = prod_mat(V)
    MMDXU  = prod_mat(dot(MDX,U))
    MMDXV  = prod_mat(dot(MDX,V))
    MMDYU  = prod_mat(dot(MDY,U))
    MMDYV  = prod_mat(dot(MDY,V))

    LAPLACPSI = dot(LAPLAC, PSI)

    #######calculate the Residuals########

    residualsVec = zeros((4*vecLen + 1), dtype='complex')

    #####psi
    residualsVec[0:vecLen] = Re*Nu*dot(LAPLAC,PSI) \
                            - Re*dot(MMU, dot(MDX, LAPLACPSI)) \
                            - Re*dot(MMV, dot(MDY,LAPLACPSI))  \
                            + beta*dot(BIHARM,PSI) \
                            - (1.-beta)*(dot(MDXX, Txx) + dot(MDXY, (Txy - Txy)) \
                                         - dot(MDYY, Txy))

    #####xx
    residualsVec[vecLen:2*vecLen] = Nu*dot(MDX,Cxx) - dot(VGRAD, Cxx) \
                                    + 2.*dot(MMDXU, Cxx) \
                                    + 2.*dot(MMDXV, Cxy) - Txx

    #####yy
    residualsVec[2*vecLen:3*vecLen] = Nu*dot(MDX,Cyy) - dot(VGRAD, Cyy) \
                                      + 2.*dot(MMDYU,Cxy) \
                                      + 2.*dot(MMDYV, Cyy) - Tyy

    #####xy
    residualsVec[3*vecLen:4*vecLen] = Nu*dot(MDY,Cxy) - dot(VGRAD, Cxy) \
                                      + dot(MMDYU, Cxx) \
                                      + dot(MMDXV, Cyy) - Txy

    #####Nu
    residualsVec[vecLen] = Cxy[NuIndx] - Cxy[2*N + NuIndx]

    ##### Apply boundary conditions to residuals vector
    Cheb0 = zeros(M, dtype='complex')
    Cheb0[0] = 1

    # dxPsi TODO
    # Not sure if any of this is going to work - at the moment the dxpsi
    # boundary condition is going into the zeroth mode of psi. Is this ok? Do I
    # not need to restrict all the modes due to this derivative? 

    residualsVec[N*M + M-2] = dot(BTOP, (PSI[N*M:(N+1)*M] - Cheb0))
    residualsVec[N*M + M-1] = dot(BBOT, (PSI[N*M:(N+1)*M] + Cheb0))

    for k in range (N):
        residualsVec[k*M + M-2] = dot(BTOP, PSI[k*M:k*M + M])
        residualsVec[k*M + M-1] = dot(BBOT, PSI[k*M:k*M + M])
    del k
    for k in range(N+1, 2*N+1):
        residualsVec[k*M + M-2] = dot(BTOP, PSI[k*M:k*M + M])
        residualsVec[k*M + M-1] = dot(BBOT, PSI[k*M:k*M + M])
    del k

    #dyPsi BC TODO
    # to set the derivative, must restrict every Fourier mode?

    residualsVec[N*M + M-4] = dot(DERIVTOP, (PSI[N*M:(N+1)*M] - Cheb0))
    residualsVec[N*M + M-3] = dot(DERIVBOT, (PSI[N*M:(N+1)*M] + Cheb0))
    for k in range (N):
        residualsVec[k*M + M-4] = dot(DERIVTOP, PSI[k*M:k*M + M])
        residualsVec[k*M + M-3] = dot(DERIVBOT, PSI[k*M:k*M + M])
    del k
    for k in range(N+1, 2*N+1):
        residualsVec[k*M + M-4] = dot(DERIVTOP, PSI[k*M:k*M + M])
        residualsVec[k*M + M-3] = dot(DERIVBOT, PSI[k*M:k*M + M])
    del k

    #################SET THE JACOBIAN MATRIX####################

    jacobian = zeros((4*vecLen+1,  4*vecLen+1), dtype='complex')

    ###### psi
    ##psi
    jacobian[0:vecLen, 0:vecLen]                      = Re*dot(MDX, LAPLAC) \
                                  - Re*dot(MMU, LAPLAC) \
                                  - Re*dot(MMV, LAPLAC) \
                                  - Re*dot(prod_mat(dot(MDY, LAPLACPSI)), MDX) \
                                  + Re*dot(prod_mat(dot(MDX, LAPLACPSI)), MDY) \
                                  + beta*BIHARM 
    ##cxx
    jacobian[0:vecLen, vecLen:2*vecLen] = + (1.-beta)*oneOverWi*MDXY
    ##cyy
    jacobian[0:vecLen, 2*vecLen:3*vecLen] = - (1.-beta)*oneOverWi*MDXY
    ##cxy
    jacobian[0:vecLen, 2*vecLen:3*vecLen] = + (1.-beta)*oneOverWi*(MDYY - MDXX)

    ###### Cxx
    ##psi                                   - dv.grad cxx
    jacobian[vecLen:2*vecLen, 0:vecLen] = + dot(prod_mat(dot(MDX, Cxx)), MDY) \
                                          - dot(prod_mat(dot(MDY, Cxx)), MDX) \
                                          - 2.*dot(prod_mat(Cxx), MDXY) \
                                          + 2.*dot(prod_mat(Cxy), MDXX) \
    ##cxx
    jacobian[vecLen:2*vecLen, vecLen:2*vecLen] = Nu*MDX - VGRAD + 2.*MMDXU \
                                                 - oneOverWi*II 
    ##cyy
    jacobian[vecLen:2*vecLen, 2*vecLen:3*vecLen] = 0
    ##cxy
    jacobian[vecLen:2*vecLen, 2*vecLen:3*vecLen] = 2.*MMDXV

    ###### Cyy
    ##psi
    jacobian[2*vecLen:3*vecLen, 0:vecLen]  = + dot(prod_mat(dot(MDX, Cyy)), MDY) \
                                             - dot(prod_mat(dot(MDY, Cyy)), MDX) \
                                             + 2.*dot(prod_mat(Cyy), MDXY) \
                                             - 2.*dot(prod_mat(Cxy), MDYY) \
    ##cxx
    jacobian[2*vecLen:3*vecLen, vecLen:2*vecLen] = 0
    ##cyy
    jacobian[2*vecLen:3*vecLen, 2*vecLen:3*vecLen] = Nu*MDX - VGRAD \
                                                    + 2.*MMDYV - oneOverWi*II
    ##cxy
    jacobian[2*vecLen:3*vecLen, 2*vecLen:3*vecLen] = 2.*MMDYU

    ###### Cxy
    ##psi
    jacobian[3*vecLen:4*vecLen, 0:vecLen]   = + dot(prod_mat(dot(MDX, Cyy)), MDY) \
                                              - dot(prod_mat(dot(MDY, Cyy)), MDX) \
                                              - dot(prod_mat(Cxx), MDYY) \
                                              + dot(prod_mat(Cyy), MDXY) \
    ##cxx
    jacobian[3*vecLen:4*vecLen, vecLen:2*vecLen] = MMDYU
    ##cyy
    jacobian[3*vecLen:4*vecLen, 2*vecLen:3*vecLen] = MMDXV
    ##cxy
    jacobian[3*vecLen:4*vecLen, 2*vecLen:3*vecLen] = Nu*MDX - VGRAD \
                                                     - oneOverWi*II
    ###### Nu
    #how to set only the imaginary part of this in the jacobian?
    jacobian[4*vecLen] = SPEEDCONDITION

    #######apply BC's to jacobian
    # TODO: Check this is the right way to implement the BC's
    # This doesn't seem like it will set the dirivative of x to be zero?
    for n in range(2*N+1):
        jacobian[n*M + M-2, 0 : 4*vecLen + 1] = \
            concatenate( (zeros(n*M), BTOP, zeros((2*N-n)*M+3*vecLen+1)) )
        jacobian[n*M + M-1, 0 : 4*vecLen + 1] = \
            concatenate( (zeros(n*M), BBOT, zeros((2*N-n)*M+3*vecLen+1)) )

        jacobian[n*M + M-4, 0:4*vecLen + 1] = \
            concatenate( (zeros(n*M), DERIVTOP, zeros((2*N-n)*M+3*vecLen+1)) )
        jacobian[n*M + M-3, 0:4*vecLen + 1] = \
            concatenate( (zeros(n*M), DERIVBOT, zeros((2*N-n)*M+3*vecLen+1)) )
    del n


    return(jacobian, residualsVec)

#MAIN

outFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}-b{b}-Wi{Wi}.pickle".format(N=N, M=M,
                                                                     kx=kx, Re=Re, 
                                                                     b=beta, Wi=Wi)

# setup the initial conditions 

vecLen = M*(2*N+1)  
print "The length of a vector is: ", vecLen
NuIndx = int(floor(M - 5)) #Choose a high Fourier mode
oneOverWi = 1. / Wi
# Set the oneOverC function: 1/2 for m=0, 1 elsewhere:
oneOverC = ones(M)
oneOverC[0] = 1. / 2.
#set up the CFunc function: 2 for m=0, 1 elsewhere:
CFunc = ones(M)
CFunc[0] = 2.

PSI = zeros(vecLen, dtype='complex')
Cxx = zeros(vecLen, dtype='complex')
Cyy = zeros(vecLen, dtype='complex')
Cxy = zeros(vecLen, dtype='complex')
Nu  = 0

xVec = zeros((4*vecLen + 1), dtype='complex')
xVec[0:vecLen]          = PSI
xVec[vecLen:2*vecLen]   = Cxx
xVec[2*vecLen:3*vecLen] = Cyy
xVec[3*vecLen:4*vecLen] = Cxy

# Useful operators 

MDY = mk_diff_y()
MDYY = dot(MDY,MDY)
MDX = mk_diff_x()
MDXX = dot(MDX,MDX)
MDXY = dot(MDX,MDY)
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

#TODO work out how to set only the imaginary part
SPEEDCONDITION = zeros(4*vecLen+1, dtype = 'complex')
SPEEDCONDITION[3*vecLen + NuIndx] = 1.
SPEEDCONDITION[3*vecLen + 2*N + NuIndx] = -1.


while True:
    (J_x0, f_x0) = solve_eq(xVec)
    dx = linalg.solve(J_x0, -f_x0)
    xVec = xVec + dx
    L2norm = linalg.norm(f_x0,2)
    print "L2 norm = {L2norm}".format(L2norm=L2norm)
    if (L2norm < NRdelta): break

PSI = xVec[0:vecLen] 
Cxx = xVec[1*vecLen:2*vecLen] 
Cyy = xVec[2*vecLen:3*vecLen] 
Cxy = xVec[3*vecLen:4*vecLen]
Nu  = xVec[4*vecLen]

save_pickle((PSI,Cxx,Cyy,Cxy,Nu), outFileName)
