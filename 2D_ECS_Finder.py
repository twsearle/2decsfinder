#-----------------------------------------------------------------------------
#   2D ECS finder
#
#   Last modified: Sun  3 Nov 13:01:19 2013
#
#-----------------------------------------------------------------------------

""" Program to find Exact coherent states from a given flow profile using
Newton-Rhaphson and the Oldroyd-B model."""

#MODULES
from scipy import *

#SETTINGS----------------------------------------

N = 10              # Number of Fourier modes
M = 20              # Number of Chebychevs

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
        MDZ[i:i+M, i:i+M] = eye(M, M, dtype='complex')*n*kx*1.j
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
    Cxx = xVec[1*vecLen:vecLen] 
    Cyy = xVec[2*vecLen:vecLen] 
    Cxy = xVec[3*vecLen:vecLen]

    # Useful Operators
    U = -dot(MDY,PSI)
    V =  dot(MDX,PSI)
    VGRAD = dot(U,MDX) + dot(V,MDY)
    LAPLAC = dot(MDX,MDX) + dot(MDY,MDY)
    MMDXU  = prod_mat(dot(MDX,U))
    MMDXV  = prod_mat(dot(MDX,V))
    MMDYU  = prod_mat(dot(MDY,U))
    MMDYV  = prod_mat(dot(MDY,V))

    return(jacobianMat, residualsVec)

#MAIN

# setup the initial conditions 

vecLen = 4*M*(2*N+1)
PSI = zeros(vecLen, dtype='complex')
Cxx = zeros(vecLen, dtype='complex')
Cyy = zeros(vecLen, dtype='complex')
Cxy = zeros(vecLen, dtype='complex')

xVec = concatenate((PSI, Cxx, Cyy, Cxy))

# Useful operators 

MDY = mk_diff_y()
MDYY = dot(MDY,MDY)
MDX = mk_diff_x()
MDXX = dot(MDX,MDX)

# Boundary arrays
BTOP = ones(M)
BBOT = ones(M)
BBOT[1:M:2] = -1

oneOverWeiss = 1. / Weiss

while True:
    (J_x0, f_x0) = solve_eq(xVec)
    dx = linalg.solve(J_x0, -f_x0)
    xVec = xVec + dx
    L2norm = linalg.norm(f_x0,2)
    print "L2 norm = {L2norm}".format(L2norm=L2norm)
    if (L2norm < NRdelta): break

PSI = xVec[0:vecLen] 
Cxx = xVec[1*vecLen:vecLen] 
Cyy = xVec[2*vecLen:vecLen] 
Cxy = xVec[3*vecLen:vecLen] 

save_pickle((PSI,Cxx,Cyy,Cxy), outFileName)
