#-----------------------------------------------------------------------------
#   2D ECS finder
#
#   Last modified: Sun  2 Mar 00:58:42 2014
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

N = 3              # Number of Fourier modes
M = 30             # Number of Chebychevs (>4)
Wi = 1.e-5           # The Weissenberg number
Re = 3000.0           # The Reynold's number
beta = 0.1
kx  = 1.31
y_star = 0.5

NRdelta = 1e-06     # Newton-Rhaphson tolerance

consts = {'N':N, 'M':M, 'kx':kx, 'Re':Re, 'b':beta, 'Wi':Wi}
NOld = N #-2 
MOld = M #- 10
kxOld = kx
ReOld = Re
bOld = beta 
WiOld = Wi
oldConsts = {'N':NOld, 'M':MOld, 'kx':kxOld, 'Re':ReOld, 'b':bOld, 'Wi':WiOld}
inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}-b{b}-Wi{Wi}.pickle".format(**oldConsts)
outFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}-b{b}-Wi{Wi}.pickle".format(**consts)
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
    Cxx = xVec[1*vecLen:2*vecLen] 
    Cyy = xVec[2*vecLen:3*vecLen] 
    Cxy = xVec[3*vecLen:4*vecLen]
    Nu  = xVec[4*vecLen]


    # Useful Vectors
    Txx = oneOverWi * Cxx 
    Txx[N*M] -= 1.0
    Tyy = oneOverWi * Cyy 
    Tyy[N*M] -= 1.0
    Txy = oneOverWi * Cxy

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
    MMDXCXX   = prod_mat(dot(MDX, Cxx))
    MMDXCYY   = prod_mat(dot(MDX, Cyy))
    MMDXCXY   = prod_mat(dot(MDX, Cxy))

    #######calculate the Residuals########

    residualsVec = zeros((4*vecLen + 1), dtype='complex')

    #####psi
    residualsVec[0:vecLen] = + Re*Nu*dot(dot(MDX,LAPLAC),PSI) \
                             - Re*dot(MMU, dot(MDX, LAPLACPSI)) \
                             - Re*dot(MMV, dot(MDY, LAPLACPSI)) \
                             + beta*dot(BIHARM, PSI) \
                             - (1.-beta)*(dot(MDXX, Txy) + dot(MDXY, (Tyy - Txx)) \
                                          - dot(MDYY, Txy))

    #####xx
    residualsVec[vecLen:2*vecLen] = Nu*dot(MDX,Cxx) - dot(VGRAD, Cxx) \
                                    + 2.*dot(MMDXU, Cxx) \
                                    + 2.*dot(MMDYU, Cxy) - Txx

    #####yy
    residualsVec[2*vecLen:3*vecLen] = Nu*dot(MDX,Cyy) - dot(VGRAD, Cyy) \
                                      + 2.*dot(MMDXV, Cxy) \
                                      + 2.*dot(MMDYV, Cyy) - Tyy

    #####xy
    residualsVec[3*vecLen:4*vecLen] = Nu*dot(MDX,Cxy) - dot(VGRAD, Cxy) \
                                      + dot(MMDXV, Cxx) + dot(MMDYU, Cyy)\
                                      - Txy

    #####Nu
    residualsVec[4*vecLen] = imag(dot(SPEEDCONDITION, PSI[(N+1)*M:(N+2)*M]))

    #####psi0
    residualsVec[N*M:(N+1)*M] = - Re*dot(VGRAD, U)[N*M:(N+1)*M] \
                                + beta*dot(MDYYY, PSI)[N*M:(N+1)*M] \
                                + (1.-beta)*dot(MDY,Txy)[N*M:(N+1)*M]
    # set the pressure gradient (pressure driven flow)
    residualsVec[N*M] += 2.0


    ##### Apply boundary conditions to residuals vector

    # dxPsi = 0   
    for k in range (2*N+1): 
        if k == N: continue # skip the 0th component 
        residualsVec[k*M + M-2] = dot((k-N)*kx*BTOP, PSI[k*M:(k+1)*M])
        residualsVec[k*M + M-1] = dot((k-N)*kx*BBOT, PSI[k*M:(k+1)*M])
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

    jacobian = zeros((4*vecLen+1,  4*vecLen+1), dtype='complex')

    ###### psi
    ##psi
    jacobian[0:vecLen, 0:vecLen] = + Nu*Re*dot(MDX, LAPLAC) \
                                   - Re*dot(dot(MMU, MDX), LAPLAC) \
                                   - Re*dot(dot(MMV, MDY), LAPLAC) \
                                   + Re*dot(prod_mat(dot(MDY, LAPLACPSI)), MDX) \
                                   - Re*dot(prod_mat(dot(MDX, LAPLACPSI)), MDY) \
                                   + beta*BIHARM 
    ##cxx
    jacobian[0:vecLen, vecLen:2*vecLen] = + (1.-beta)*oneOverWi*MDXY
    ##cyy
    jacobian[0:vecLen, 2*vecLen:3*vecLen] = - (1.-beta)*oneOverWi*MDXY
    ##cxy
    jacobian[0:vecLen, 3*vecLen:4*vecLen] = - (1.-beta)*oneOverWi*(MDXX - MDYY)
    ##Nu - vector not a product matrix
    jacobian[0:vecLen, 4*vecLen] = Re*dot(MDX, LAPLACPSI)

    ###### Cxx
    ##psi                                   - dv.grad cxx
    jacobian[vecLen:2*vecLen, 0:vecLen] = - dot(prod_mat(dot(MDX, Cxx)), MDY) \
                                          + dot(prod_mat(dot(MDY, Cxx)), MDX) \
                                          + 2.*dot(prod_mat(Cxx), MDXY) \
                                          + 2.*dot(prod_mat(Cxy), MDXY) \
    ##cxx
    jacobian[vecLen:2*vecLen, vecLen:2*vecLen] = Nu*MDX - VGRAD + 2.*MMDXU \
                                                 - oneOverWi*II 
    ##cyy
    jacobian[vecLen:2*vecLen, 2*vecLen:3*vecLen] = 0
    ##cxy
    jacobian[vecLen:2*vecLen, 3*vecLen:4*vecLen] = 2.*MMDYU
    ##Nu - vector not a product matrix
    jacobian[vecLen:2*vecLen, 4*vecLen] = dot(MDX, Cxx) 

    ###### Cyy
    ##psi
    jacobian[2*vecLen:3*vecLen, 0:vecLen]  = - dot(prod_mat(dot(MDX, Cyy)), MDY) \
                                             + dot(prod_mat(dot(MDY, Cyy)), MDX) \
                                             - 2.*dot(prod_mat(Cyy), MDXY) \
                                             - 2.*dot(prod_mat(Cxy), MDXX) \
    ##cxx
    jacobian[2*vecLen:3*vecLen, vecLen:2*vecLen] = 0
    ##cyy
    jacobian[2*vecLen:3*vecLen, 2*vecLen:3*vecLen] = Nu*MDX - VGRAD \
                                                    + 2.*MMDYV - oneOverWi*II
    ##cxy
    jacobian[2*vecLen:3*vecLen, 3*vecLen:4*vecLen] = 2.*MMDXV
    ##Nu - vector not a product matrix
    jacobian[2*vecLen:3*vecLen, 4*vecLen] = dot(MDX, Cyy)

    ###### Cxy
    ##psi
    jacobian[3*vecLen:4*vecLen, 0:vecLen]   = - dot(prod_mat(dot(MDX, Cxy)), MDY) \
                                              + dot(prod_mat(dot(MDY, Cxy)), MDX) \
                                              + dot(prod_mat(Cyy), MDYY) \
                                              - dot(prod_mat(Cxx), MDXX) \
    ##cxx
    jacobian[3*vecLen:4*vecLen, vecLen:2*vecLen] =  MMDXV
    ##cyy
    jacobian[3*vecLen:4*vecLen, 2*vecLen:3*vecLen] = MMDYU
    ##cxy
    jacobian[3*vecLen:4*vecLen, 3*vecLen:4*vecLen] = Nu*MDX - VGRAD \
                                                     - oneOverWi*II
    ##Nu - vector not a product matrix
    jacobian[3*vecLen:4*vecLen, 4*vecLen] = dot(MDX, Cxy)

    ###### Nu 
    jacobian[4*vecLen, (N+1)*M:(N+2)*M] = -1.j*0.5*SPEEDCONDITION
    jacobian[4*vecLen, (N-1)*M:N*M] = 1.j*0.5*SPEEDCONDITION

    ###### psi0 equation
    #set row to zero
    jacobian[N*M:(N+1)*M, :] = 0
    ##u0
    jacobian[N*M:(N+1)*M, 0:vecLen] = \
                            + Re*dot(prod_mat(dot(MDX, PSI)), MDYY)[N*M:(N+1)*M, :]\
                            + Re*dot(prod_mat(dot(MDYY, PSI)), MDX)[N*M:(N+1)*M, :]\
                            + MDYYY[N*M:(N+1)*M, :]
    ##cxx
    jacobian[N*M:(N+1)*M, vecLen:2*vecLen] = 0
    ##cyy
    jacobian[N*M:(N+1)*M, 2*vecLen:3*vecLen] = 0
    ##cxy
    jacobian[N*M:(N+1)*M, 3*vecLen:4*vecLen] = \
                                            + (1-beta)*oneOverWi*MDY[N*M:(N+1)*M, :]
    ##nu
    jacobian[N*M:(N+1)*M, 4*vecLen] = 0


    #######apply BC's to jacobian

    # Apply BC to zeroth mode
    # dypsi0 = const
    jacobian[N*M + M-3, 0:4*vecLen + 1] = \
        concatenate( (zeros(N*M), DERIVTOP, zeros(N*M+3*vecLen+1)) )
    jacobian[N*M + M-2, 0:4*vecLen + 1] = \
        concatenate( (zeros(N*M), DERIVBOT, zeros(N*M+3*vecLen+1)) )
    # psi(-1) = const 
    jacobian[N*M + M-1, 0:4*vecLen + 1] = \
        concatenate( (zeros(N*M), BBOT, zeros(N*M+3*vecLen+1)) )

    for n in range(2*N+1):
        if n == N: continue     # Don't apply bcs to psi0 mode here
        # dxpsi = 0
        jacobian[n*M + M-2, 0 : 4*vecLen + 1] = \
            concatenate( (zeros(n*M), (n-N)*kx*BTOP, zeros((2*N-n)*M+3*vecLen+1)) )
        jacobian[n*M + M-1, 0 : 4*vecLen + 1] = \
            concatenate( (zeros(n*M), (n-N)*kx*BBOT, zeros((2*N-n)*M+3*vecLen+1)) )
        # -dypsi = const
        jacobian[n*M + M-4, 0:4*vecLen + 1] = \
            concatenate( (zeros(n*M), DERIVTOP, zeros((2*N-n)*M+3*vecLen+1)) )
        jacobian[n*M + M-3, 0:4*vecLen + 1] = \
            concatenate( (zeros(n*M), DERIVBOT, zeros((2*N-n)*M+3*vecLen+1)) )
    del n

    return(jacobian, residualsVec)


def symmetrise(vec):
    """symmetrise each vector to manually impose real space condition""" 

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

def mk_cheb_int():
    integrator = zeros(M, dtype='d')
    for m in range(0,M,2):
        integrator[m] = (1 + cos(m*pi)) / (1-m*m)
    del m
    return integrator
    
def newtonian_profile(PSI):
"""
given a Newtonian ECS profile, find the stresses.
"""

    U = dot(MDX, PSI)
    V = - dot(MDY, PSI)
    VGRAD = dot(U,MDX) + dot(V,MDY)

    BPFEQNS = zeros((3*vecLen, 3*vecLen), dtype='D')
    # Cxx eqn
    # Cxx
    BPFEQNS[0:vecLen, 0:vecLen] = - VGRAD \
                                + 2*prod_mat(dot(MDX,U)) - oneOverWi*II
    # Cyy
    BPFEQNS[0:vecLen, vecLen:2*vecLen] = 0
    # Cxy
    BPFEQNS[0:vecLen, 2*vecLen:3*vecLen] = 2*prod_mat(dot(MDY, U))
    # Cyy eqn
    # Cxx
    BPFEQNS[vecLen:2*vecLen, 0:vecLen] = 0
    # Cyy
    BPFEQNS[vecLen:2*vecLen, vecLen:2*vecLen] = - VGRAD - oneOverWi*II\
                                              + 2.*prod_mat(dot(MDY, V))
    # Cxy
    BPFEQNS[vecLen:2*vecLen, 2*vecLen:3*vecLen] = 2.*prod_mat(dot(MDX, V))
    #Cxy eqn
    # Cxx
    BPFEQNS[2*vecLen:3*vecLen, 0:vecLen] = prod_mat(dot(MDX, V))
    # Cyy 
    BPFEQNS[2*vecLen:3*vecLen, vecLen:2*vecLen] = prod_mat(dot(MDY, U))
    # Cxy
    BPFEQNS[2*vecLen:3*vecLen, 2*vecLen:3*vecLen] = -VGRAD - oneOverWi*II 

    RHS = zeros(3*vecLen, dtype='D')
    RHS[0] = -oneOverWi
    RHS[vecLen] = -oneOverWi
    RHS[2*vecLen:3*vecLen] = 0

    soln = linalg.solve(BPFEQNS, RHS)

    Cxx = soln[0:vecLen]
    Cyy = soln[vecLen:2*vecLen]
    Cxy = soln[2*vecLen:3*vecLen]

    return Cxx, Cyy, Cxy

#MAIN


# setup the initial conditions 

vecLen = M*(2*N+1)  
print "Settings:"
print """------------------------------------
N \t= {N}
M \t= {M}              
Wi \t= {Wi}        
Re \t= {Re}         
beta \t= {beta}
kx \t= {kx}
------------------------------------
        """.format(N=N, M=M, kx=kx, Re=Re, beta=beta, Wi=Wi)

print "The length of a vector is: ", vecLen
print "The size of the jacobian Matrix is: {x} by {y}".format(x=(4*vecLen+1), y=
                                                             (4*vecLen+1))
NuIndx = M - 5           #Choose a high Fourier mode
oneOverWi = 1. / Wi
# Set the oneOverC function: 1/2 for m=0, 1 elsewhere:
oneOverC = ones(M)
oneOverC[0] = 1. / 2.
#set up the CFunc function: 2 for m=0, 1 elsewhere:
CFunc = ones(M)
CFunc[0] = 2.

almostZero = zeros(M, dtype='D') + 1e-14


# Useful operators 

MDY = mk_diff_y()
MDYY = dot(MDY,MDY)
MDYYY = dot(MDY,MDYY)
MDX = mk_diff_x()
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

singleDY = mk_single_diffy()
DERIVTOP = zeros((M), dtype='complex')
DERIVBOT = zeros((M), dtype='complex')
for j in range(M):
    DERIVTOP[j] = dot(BTOP, singleDY[:,j]) 
    DERIVBOT[j] = dot(BBOT, singleDY[:,j])
del j

#PSI = random.random(vecLen)/10.0

#PSI = zeros(vecLen, dtype='complex')
#PSI[N*M+1] =  3.0/8.0
#PSI[N*M+2] = -1.0
#PSI[N*M+3] = -1.0/8.0

# Forgotten which of these is right.
#PSI[N*M]   += 2.0/3.0
#PSI[N*M+1] += 3.0/4.0
#PSI[N*M+2] += 0.0
#PSI[N*M+3] += -1.0/12.0
#Nu  = 0.35

inFp = open('pf-N3-M30-kx1.31-Re3000.0.pickle', 'r')
PSI, Nu = pickle.load(inFp)

Cxx, Cyy, Cxy = newtonian_profile(PSI)

#Cyy = zeros(vecLen, dtype='complex')
#Cyy[N*M] += 1.0
#Cxy = zeros(vecLen, dtype='complex')
#Cxy = Wi*dot(MDYY, PSI)
#Cxx = zeros(vecLen, dtype='complex')
#Cxx = 2*Wi*Wi*Cxy*Cxy
#Cxx[N*M] += 1.0


#PSI, Cxx, Cyy, Cxy, Nu = pickle.load(open(inFileName, 'r'))

#PSI = increase_resolution(PSIOld)
#Cxx = increase_resolution(CxxOld)
#Cyy = increase_resolution(CyyOld)
#Cxy = increase_resolution(CxyOld)

#PSI[(N-NOld)*M:(N-NOld)*M + M*(2*NOld+1)] = PSItmp[0:M*(2*NOld+1)]
#Cxx[(N-NOld)*M:(N-NOld)*M + M*(2*NOld+1)] = Cxxtmp[0:M*(2*NOld+1)]
#Cyy[(N-NOld)*M:(N-NOld)*M + M*(2*NOld+1)] = Cyytmp[0:M*(2*NOld+1)]
#Cxy[(N-NOld)*M:(N-NOld)*M + M*(2*NOld+1)] = Cxytmp[0:M*(2*NOld+1)]

xVec = zeros((4*vecLen + 1), dtype='complex')
xVec[0:vecLen]          = PSI
xVec[vecLen:2*vecLen]   = Cxx
xVec[2*vecLen:3*vecLen] = Cyy
xVec[3*vecLen:4*vecLen] = Cxy
xVec[4*vecLen]          = Nu 

# Set only the imaginary part at a point to zero to constrain nu. I will choose
# y = 0.5
SPEEDCONDITION = zeros(M, dtype = 'complex')
for m in range(M):
    SPEEDCONDITION[m] = cos(m*arccos(y_star)) 

print "Begin Newton-Rhaphson"
print "------------------------------------"
print "L2norm:"
while True:
    (J_x0, f_x0) = solve_eq(xVec)
    dx = linalg.solve(J_x0, -f_x0)
    xVec = xVec + dx
    L2norm = linalg.norm(f_x0,2)
    print "\t {L2norm}".format(L2norm=L2norm)
    if (L2norm < NRdelta): 
        break
    xVec[0:vecLen] = symmetrise(xVec[0:vecLen])
    xVec[vecLen:2*vecLen] = symmetrise(xVec[vecLen:2*vecLen])
    xVec[2*vecLen:3*vecLen] = symmetrise(xVec[2*vecLen:3*vecLen])
    xVec[3*vecLen:4*vecLen] = symmetrise(xVec[3*vecLen:4*vecLen])

PSI = xVec[0:vecLen] 
Cxx = xVec[1*vecLen:2*vecLen] 
Cyy = xVec[2*vecLen:3*vecLen] 
Cxy = xVec[3*vecLen:4*vecLen]
Nu  = xVec[4*vecLen]
print "------------------------------------\n"
print " Nu = ", Nu
U = dot(MDY, PSI)
V = -dot(MDX, PSI)
MMU = prod_mat(U)
MMV = prod_mat(V)
U0sq = (dot(MMU,U) + dot(MMV,V))[N*M:(N+1)*M]
assert allclose(almostZero, imag(U0sq)), "Imaginary velocities!"
KE0 = 0.5*real(dot(INTY, U0sq))
print 'KE0 = ', KE0
print "norm of 1st psi mode = ", linalg.norm(PSI[(N+1)*M:(N+2)*M], 2)

if KE0 > 1e-4:
    pickle.dump((PSI,Cxx,Cyy,Cxy,Nu), open(outFileName, 'w'))
