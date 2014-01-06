from scipy import *

N = None
M = None 
kx = None

def initTSM(N_, M_, kx_):
    global oneOverC
    global CFunc
    global N
    global M
    global kx
    N = N_
    M = M_
    kx = kx_
    
    # Set the oneOverC function: 1/2 for m=0, 1 elsewhere:
    oneOverC = ones(M)
    oneOverC[0] = 1. / 2.
    #set up the CFunc function: 2 for m=0, 1 elsewhere:
    CFunc = ones(M)
    CFunc[0] = 2.

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
    MDY = zeros( ((2*N+1)*M,  (2*N+1)*M) )
     
    for cheb in range(0,(2*N+1)*M,M):
        MDY[cheb:cheb+M, cheb:cheb+M] = D
    del cheb
    return MDY

def mk_diff_x():
    """Make matrix to do fourier differentiation wrt x."""
    MDX = zeros( ((2*N+1)*M, (2*N+1)*M), dtype='complex')

    n = -N
    for i in range(0, (2*N+1)*M, M):
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
    MM = zeros(((2*N+1)*M, (2*N+1)*M), dtype='complex')

    #First make the middle row
    midMat = zeros((M, (2*N+1)*M), dtype='complex')
    for n in range(2*N+1):       # Fourier Matrix is 2*N+1 cheb matricies
        yprodmat = cheb_prod_mat(velA[n*M:(n+1)*M])
        endind = 2*N+1-n
        midMat[:, (endind-1)*M:endind*M] = yprodmat
    del n

    #copy matrix into MM, according to the matrix for spectral space
    # top part first
    for i in range(0, N):
        MM[i*M:(i+1)*M, :(N+1+i)*M] = midMat[:, (N-i)*M:]
    del i
    # middle
    MM[N*M:(N+1)*M, :] = midMat
    #  bottom - beware! This thing is pretty horribly written. i = 0 is actually
    #  row index N+1
    for i in range(0, N):
        MM[(i+N+1)*M:(i+2+N)*M, (i+1)*M:] = midMat[:, :(2*N-i)*M]
    del i

    return MM

def mk_cheb_int():
    integrator = zeros(M, dtype='d')
    for m in range(0,M,2):
        integrator[m] = 2. / (1.-m*m)
    del m
    
    return integrator
