#-----------------------------------------------------------------------------
#   2D ECS finder
#
#   Last modified: Mon  9 Jun 15:41:47 2014
#
#-----------------------------------------------------------------------------

""" Program to find Exact coherent states from a given flow profile using
Newton-Rhaphson and the Oldroyd-B model."""

#MODULES
from scipy import *
from scipy import linalg
from scipy import optimize
import matplotlib.pyplot as plt
import cPickle as pickle
import ConfigParser
import argparse

import TobySpectralMethods as tsm
import matrix_checker as mmc

#SETTINGS----------------------------------------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
Wi = config.getfloat('General', 'Wi')
beta = config.getfloat('General', 'beta')
kx = config.getfloat('General', 'kx')
dS = config.getfloat('Newton-Raphson', 'dS')
fp.close()

y_star = 0.5

NRdelta = 1e-06     # Newton-Rhaphson tolerance

argparser = argparse.ArgumentParser()

argparser.add_argument("-N", type=int, default=N, 
                help='Override Number of Fourier modes given in the config file')
argparser.add_argument("-M", type=int, default=M, 
                help='Override Number of Chebyshev modes in the config file')
argparser.add_argument("-Re", type=float, default=Re, 
                help="Override Reynold's number in the config file") 
argparser.add_argument("-b", type=float, default=beta, 
                help='Override beta of the config file')
argparser.add_argument("-Wi", type=float, default=Wi, 
                help='Override Weissenberg number of the config file')
argparser.add_argument("-kx", type=float, default=kx, 
                help='Override wavenumber of the config file')

args = argparser.parse_args()
N = args.N 
M = args.M
Re = args.Re
beta = args.b
Wi = args.Wi
kx = args.kx

consts = {'N':N, 'M':M, 'kx':kx, 'Re':Re, 'b':beta, 'Wi':Wi}

NOld = N#3 
MOld = M#30
kxOld = kx
ReOld = Re
bOld = beta#+0.1 
WiOld = Wi
oldConsts = {'N':NOld, 'M':MOld, 'kx':kxOld, 'Re':ReOld, 'b':bOld, 'Wi':WiOld}
inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}-b{b}-Wi{Wi}.pickle".format(**oldConsts)
traceOutFileName = "cont-trace{0}.dat".format(inFileName[2:-7])

tsm.initTSM(N_=N, M_=M, kx_=kx)

#------------------------------------------------------------------------------

#FUNCTIONS

def solve_eq(xVec):
    """calculates the residuals of equations and the jacobian that ought to
    generate them for minimisation via Newton-Rhaphson"""
    
    PSI = xVec[0:vecLen] 
    Cxx = xVec[1*vecLen:2*vecLen] 
    Cyy = xVec[2*vecLen:3*vecLen] 
    Cxy = xVec[3*vecLen:4*vecLen]
    Nu  = xVec[4*vecLen]
    Re_ = real(xVec[4*vecLen+1])


    # Useful Vectors
    Txx = oneOverWi * Cxx 
    Txx[N*M] -= oneOverWi
    Tyy = oneOverWi * Cyy 
    Tyy[N*M] -= oneOverWi
    Txy = oneOverWi * Cxy

    U         = + dot(MDY, PSI)
    V         = - dot(MDX, PSI)
    LAPLACPSI = dot(LAPLAC, PSI)

    # Useful Operators
    MMU    = tsm.c_prod_mat(U)
    MMV    = tsm.c_prod_mat(V)
    VGRAD  = dot(MMU,MDX) + dot(MMV,MDY)
    MMDXU  = tsm.c_prod_mat(dot(MDX, U))
    MMDXV  = tsm.c_prod_mat(dot(MDX, V))
    MMDYU  = tsm.c_prod_mat(dot(MDY, U))
    MMDYV  = tsm.c_prod_mat(dot(MDY, V))

    MMDXPSI   = tsm.c_prod_mat(dot(MDX, LAPLACPSI))
    MMDXCXX   = tsm.c_prod_mat(dot(MDX, Cxx))
    MMDXCYY   = tsm.c_prod_mat(dot(MDX, Cyy))
    MMDXCXY   = tsm.c_prod_mat(dot(MDX, Cxy))

    #######calculate the Residuals########

    residualsVec = zeros((4*vecLen + 2), dtype='complex')

    #####psi
    residualsVec[0:vecLen] = + Re_*Nu*dot(dot(MDX,LAPLAC),PSI) \
                             - Re_*dot(MMU, dot(MDX, LAPLACPSI)) \
                             - Re_*dot(MMV, dot(MDY, LAPLACPSI)) \
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

    #####Re
    residualsVec[4*vecLen + 1] = (Re0-Re_)**2 \
                                 + sum((xVec0-xVec[:4*vecLen+1])**2)\
                                 - dS**2
    
    #####psi0
    residualsVec[N*M:(N+1)*M] = - Re_*dot(VGRAD, U)[N*M:(N+1)*M] \
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

    return(residualsVec)

def find_jacobian(x):

    PSI = xVec[0:vecLen] 
    Cxx = xVec[1*vecLen:2*vecLen] 
    Cyy = xVec[2*vecLen:3*vecLen] 
    Cxy = xVec[3*vecLen:4*vecLen]
    Nu  = xVec[4*vecLen]
    Re_  = real(xVec[4*vecLen+1])


    # Useful Vectors
    Txx = oneOverWi * Cxx 
    Txx[N*M] -= oneOverWi
    Tyy = oneOverWi * Cyy 
    Tyy[N*M] -= oneOverWi
    Txy = oneOverWi * Cxy

    U         = + dot(MDY, PSI)
    V         = - dot(MDX, PSI)
    LAPLACPSI = dot(LAPLAC, PSI)

    # Useful Operators
    MMU    = tsm.c_prod_mat(U)
    MMV    = tsm.c_prod_mat(V)
    VGRAD  = dot(MMU,MDX) + dot(MMV,MDY)
    MMDXU  = tsm.c_prod_mat(dot(MDX, U))
    MMDXV  = tsm.c_prod_mat(dot(MDX, V))
    MMDYU  = tsm.c_prod_mat(dot(MDY, U))
    MMDYV  = tsm.c_prod_mat(dot(MDY, V))

    MMDXPSI   = tsm.c_prod_mat(dot(MDX, LAPLACPSI))
    MMDXCXX   = tsm.c_prod_mat(dot(MDX, Cxx))
    MMDXCYY   = tsm.c_prod_mat(dot(MDX, Cyy))
    MMDXCXY   = tsm.c_prod_mat(dot(MDX, Cxy))


    #################SET THE JACOBIAN MATRIX####################

    jacobian = zeros((4*vecLen+2,  4*vecLen+2), dtype='complex')

    ###### psi
    ##psi
    jacobian[0:vecLen, 0:vecLen] = + Nu*Re_*dot(MDX, LAPLAC) \
                                   - Re_*dot(dot(MMU, MDX), LAPLAC) \
                                   - Re_*dot(dot(MMV, MDY), LAPLAC) \
                                   + Re_*dot(tsm.c_prod_mat(dot(MDY, LAPLACPSI)), MDX) \
                                   - Re_*dot(tsm.c_prod_mat(dot(MDX, LAPLACPSI)), MDY) \
                                   + beta*BIHARM 
    ##cxx
    jacobian[0:vecLen, vecLen:2*vecLen] = + (1.-beta)*oneOverWi*MDXY
    ##cyy
    jacobian[0:vecLen, 2*vecLen:3*vecLen] = - (1.-beta)*oneOverWi*MDXY
    ##cxy
    jacobian[0:vecLen, 3*vecLen:4*vecLen] = - (1.-beta)*oneOverWi*(MDXX - MDYY)
    ##Nu - vector not a product matrix
    jacobian[0:vecLen, 4*vecLen] = Re_*dot(MDX, LAPLACPSI)
    ##Re_ - for the continuation in Reynold's number
    jacobian[0:vecLen, 4*vecLen+1] = + Nu*dot(MDX, LAPLACPSI) \
                                   - dot(dot(MMU, MDX), LAPLACPSI) \
                                   - dot(dot(MMV, MDY), LAPLACPSI) \
                                   + dot(tsm.c_prod_mat(dot(MDY, LAPLACPSI)),
                                            dot(MDX, PSI)) \
                                   - dot(tsm.c_prod_mat(dot(MDX, LAPLACPSI)),
                                            dot(MDY, PSI)) 

    ###### Cxx
    ##psi                                   - dv.grad cxx
    jacobian[vecLen:2*vecLen, 0:vecLen] = - dot(tsm.c_prod_mat(dot(MDX, Cxx)), MDY) \
                                          + dot(tsm.c_prod_mat(dot(MDY, Cxx)), MDX) \
                                          + 2.*dot(tsm.c_prod_mat(Cxx), MDXY) \
                                          + 2.*dot(tsm.c_prod_mat(Cxy), MDXY) \
    ##cxx
    jacobian[vecLen:2*vecLen, vecLen:2*vecLen] = Nu*MDX - VGRAD + 2.*MMDXU \
                                                 - oneOverWi*II 
    ##cyy
    jacobian[vecLen:2*vecLen, 2*vecLen:3*vecLen] = 0
    ##cxy
    jacobian[vecLen:2*vecLen, 3*vecLen:4*vecLen] = 2.*MMDYU
    ##Nu - vector not a product matrix
    jacobian[vecLen:2*vecLen, 4*vecLen] = dot(MDX, Cxx) 
    ##Re - for the continuation in Reynold's number
    jacobian[vecLen:2*vecLen, 4*vecLen+1] = 0

    ###### Cyy
    ##psi
    jacobian[2*vecLen:3*vecLen, 0:vecLen]  = - dot(tsm.c_prod_mat(dot(MDX, Cyy)), MDY) \
                                             + dot(tsm.c_prod_mat(dot(MDY, Cyy)), MDX) \
                                             - 2.*dot(tsm.c_prod_mat(Cyy), MDXY) \
                                             - 2.*dot(tsm.c_prod_mat(Cxy), MDXX) \
    ##cxx
    jacobian[2*vecLen:3*vecLen, vecLen:2*vecLen] = 0
    ##cyy
    jacobian[2*vecLen:3*vecLen, 2*vecLen:3*vecLen] = Nu*MDX - VGRAD \
                                                    + 2.*MMDYV - oneOverWi*II
    ##cxy
    jacobian[2*vecLen:3*vecLen, 3*vecLen:4*vecLen] = 2.*MMDXV
    ##Nu - vector not a product matrix
    jacobian[2*vecLen:3*vecLen, 4*vecLen] = dot(MDX, Cyy)
    ##Re - for the continuation in Reynold's number
    jacobian[2*vecLen:3*vecLen, 4*vecLen+1] = 0

    ###### Cxy
    ##psi
    jacobian[3*vecLen:4*vecLen, 0:vecLen]   = - dot(tsm.c_prod_mat(dot(MDX, Cxy)), MDY) \
                                              + dot(tsm.c_prod_mat(dot(MDY, Cxy)), MDX) \
                                              + dot(tsm.c_prod_mat(Cyy), MDYY) \
                                              - dot(tsm.c_prod_mat(Cxx), MDXX) \
    ##cxx
    jacobian[3*vecLen:4*vecLen, vecLen:2*vecLen] =  MMDXV
    ##cyy
    jacobian[3*vecLen:4*vecLen, 2*vecLen:3*vecLen] = MMDYU
    ##cxy
    jacobian[3*vecLen:4*vecLen, 3*vecLen:4*vecLen] = Nu*MDX - VGRAD \
                                                     - oneOverWi*II
    ##Nu - vector not a product matrix
    jacobian[3*vecLen:4*vecLen, 4*vecLen] = dot(MDX, Cxy)
    ##Re - for the continuation in Reynold's number
    jacobian[3*vecLen:4*vecLen, 4*vecLen+1] = 0

    ###### Nu 
    jacobian[4*vecLen, (N+1)*M:(N+2)*M] = -1.j*0.5*SPEEDCONDITION
    jacobian[4*vecLen, (N-1)*M:N*M] = 1.j*0.5*SPEEDCONDITION

    ###### Re equation
    jacobian[4*vecLen+1, 0:4*vecLen+1] = -2*(xVec0-xVec[0:4*vecLen+1])
    jacobian[4*vecLen+1, 4*vecLen+1] = -2*(Re0-Re_) 

    ###### psi0 equation
    #set row to zero
    jacobian[N*M:(N+1)*M, :] = 0
    ##u0
    jacobian[N*M:(N+1)*M, 0:vecLen] = \
                            + Re_*dot(tsm.c_prod_mat(dot(MDX, PSI)), MDYY)[N*M:(N+1)*M, :]\
                            + Re_*dot(tsm.c_prod_mat(dot(MDYY, PSI)), MDX)[N*M:(N+1)*M, :]\
                            + beta*MDYYY[N*M:(N+1)*M, :]
    ##cxx
    jacobian[N*M:(N+1)*M, vecLen:2*vecLen] = 0
    ##cyy
    jacobian[N*M:(N+1)*M, 2*vecLen:3*vecLen] = 0
    ##cxy
    jacobian[N*M:(N+1)*M, 3*vecLen:4*vecLen] = \
                                            + (1-beta)*oneOverWi*MDY[N*M:(N+1)*M, :]
    ##nu
    jacobian[N*M:(N+1)*M, 4*vecLen] = 0

    ##Re
    jacobian[N*M:(N+1)*M, 4*vecLen+1] = + dot(tsm.c_prod_mat(dot(MDX, PSI)), 
                                            dot(MDYY, PSI))[N*M:(N+1)*M]\
                                      + dot(tsm.c_prod_mat(dot(MDYY, PSI)),
                                            dot(MDX, PSI))[N*M:(N+1)*M]
    

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

    return jacobian


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
    
def line_search(solve_eq, find_jac, x, alpha=1e-4, NRtol=1e-6):
    """
    My implementation of the line search global quasi Newton Raphson method for
    when you have a function to calculate the analytic jacobian. If you don't
    have this just use the scipy Newton Krylov solver (has a finite difference
    approximation to the jacobian built in).  This method only takes a Newton
    step if this decreases the residual squared.  This function is known as the
    master function f in this code. 

    I have added the symmeterise function to make sure I don't screw with the
    complex conjugation of the fourier modes. This is dirty, but it works!

    Parameters:
        alpha: sets the size of average rate of decrease of f should be at least
               this fraction of the initial rate of decrease of f. Never greater
               than 1. Ideally as low as possible. Numerical recipies thinks
               1e-4 is fine.
        NRtol: The stopping criteria on the L2 norm of the residual.
    
    WARNING: Could clash with global variables if you have been silly enough to
    choose rubbish names.
    """

    print "\n\tBegin Newton line search method"
    print "\t------------------------------------"
    finCond = False
    while not finCond:
        # Calculate the newton step, dx
        F_x0 = solve_eq(x)
        j_x0 = find_jac(x)
        dx = linalg.solve(j_x0, -F_x0)

        # Define the master function
        f_x0 = real(0.5*dot(conj(F_x0), F_x0))

        slope_x0dx = real(-2*f_x0) #-dot(conj(F_x0), F_x0)

        # Decide whether to take the Newton Step by Armijo line search method
        # First initialise variables so that first iteration happens 
        lam = 1 
        lamPrev = 1  
        f_xn = f_x0 + alpha*lam*slope_x0dx + 1
        f_lam2Prev = 0 # Doesn't matter, will be set before it is used
        f_lamPrev = 0 
        counter = 0

        # Now choose a lambda and see if it is good.
        while f_xn >= f_x0 + alpha*lam*slope_x0dx:

            if counter == 1:
                # set lambda by a quadratic model for the residual master function f
                lam = - slope_x0dx / 2*(f_xn - f_x0 - slope_x0dx)
                #print "square model lambda =", lam
                
                # impose upper and lower bounds on lambda 
                if lam > 0.5:
                    lam = 0.5
                if lam < 0.1:
                    lam = 0.1

            elif counter > 1:
                # set lambda by a cubic model for the residual master function f
                abmat = zeros((2,2))
                abmat[0,0] = 1/(lamPrev*lamPrev)
                abmat[0,1] = -1/(lam2Prev*lam2Prev)
                abmat[1,0] = -lam2Prev/(lamPrev*lamPrev)
                abmat[1,1] = lamPrev/(lam2Prev*lam2Prev)

                f3vec = zeros(2)
                f3vec[0] = f_lamPrev - f_x0 - slope_x0dx*lamPrev
                f3vec[1] = f_lam2Prev - f_x0 - slope_x0dx*lam2Prev

                abvec = (1./(lamPrev-lam2Prev)) * dot(abmat, f3vec)
                aaa = abvec[0]
                bbb = abvec[1]
                lam = (- bbb + sqrt(bbb**2 - 3*aaa*slope_x0dx)) / 3*aaa

                # impose upper and lower bounds on lambda 
                if lam > 0.5*lamPrev:
                    lam = 0.5*lamPrev
                if lam < 0.1*lamPrev:
                    lam = 0.1*lamPrev

                #print "cubic model lambda", lam

                if lam < 1e-6:
                    print " loop counter of last step = ", counter-1
                    print "step too small, take full Newton step and hope for the best."
                    lam = 1
                    break

            # calculate the residual and master function so we can see if the
            # step was a good one.
            F_xn = solve_eq(x + lam*dx)
            f_xn = real(0.5*dot(conj(F_xn), F_xn))
            #print """   |F_xn| = """, linalg.norm(F_xn) 

            # update old values for cubic method
            lam2Prev = lamPrev
            lamPrev = lam
            f_lam2Prev = f_lamPrev
            f_lamPrev = f_xn

            counter += 1
        

        # change x to the value at the step we just took
        x = x + lam*dx

        # Extra symmerterisation step
        x[0:vecLen] = symmetrise(x[0:vecLen])
        x[vecLen:2*vecLen] = symmetrise(x[vecLen:2*vecLen])
        x[2*vecLen:3*vecLen] = symmetrise(x[2*vecLen:3*vecLen])
        x[3*vecLen:4*vecLen] = symmetrise(x[3*vecLen:4*vecLen])
        
        # Print norm and check if we can exit yet.
        L2 = linalg.norm(F_xn)
        print """|F_xn| = {0:10.5g}, |dx| = {1:10.5g}, lambda = {2}""".format(
            L2, linalg.norm(dx), lam)

        # Quit if L2 norm is getting huge
        if L2 > 1e50:
            print "Error: Shooting off to infinity!"
            exit(1)

        if L2 < NRtol:
            print "Solution found!"
            finCond = True

    return x

def old_newton (xVec):
    print "\n\tBegin Newton-Rhaphson"
    print "\t------------------------------------"
    print "\t |F_x0|: \t|dx|:" 
    while True:

        f_x0 = solve_eq(xVec)
        J_x0 = find_jacobian(xVec)
        dx = linalg.solve(J_x0, -f_x0)

        L2norm = linalg.norm(f_x0,2)
        print "\t {L2norm} \t{dx}".format(L2norm=L2norm, dx=linalg.norm(dx))

        if (L2norm < NRdelta): 
            break

        xVec = xVec + dx

        xVec[0:vecLen] = symmetrise(xVec[0:vecLen])
        xVec[vecLen:2*vecLen] = symmetrise(xVec[vecLen:2*vecLen])
        xVec[2*vecLen:3*vecLen] = symmetrise(xVec[2*vecLen:3*vecLen])
        xVec[3*vecLen:4*vecLen] = symmetrise(xVec[3*vecLen:4*vecLen])

    return xVec

#MAIN


# setup the initial conditions 

vecLen = M*(2*N+1)  
print """
\tSettings:
\t------------------------------------
\tN \t= {N}
\tM \t= {M}              
\tWi \t= {Wi}        
\tRe \t= {Re}         
\tbeta \t= {beta}
\tkx \t= {kx}
\tdS \t= {dS}
\t------------------------------------
        """.format(N=N, M=M, kx=kx, Re=Re, beta=beta, Wi=Wi, dS=dS)

print "The length of a vector is: ", vecLen
print "The size of the jacobian Matrix is: {x} by {y}".format(x=(4*vecLen+1), y=
                                                             (4*vecLen+1))
oneOverWi = 1. / Wi
# Set the oneOverC function: 1/2 for m=0, 1 elsewhere:
oneOverC = ones(M)
oneOverC[0] = 1. / 2.
#set up the CFunc function: 2 for m=0, 1 elsewhere:
CFunc = ones(M)
CFunc[0] = 2.

almostZero = zeros(M, dtype='D') + 1e-14


# Useful operators 

MDY = tsm.mk_diff_y()
MDYY = dot(MDY,MDY)
MDYYY = dot(MDY,MDYY)
MDX = tsm.mk_diff_x()
MDXX = dot(MDX, MDX)
MDXY = dot(MDX, MDY)
LAPLAC = dot(MDX,MDX) + dot(MDY,MDY)
BIHARM = dot(LAPLAC, LAPLAC)

INTY = tsm.mk_cheb_int()

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

# Set only the imaginary part at a point to zero to constrain nu. I will choose
# y = 0.5
SPEEDCONDITION = zeros(M, dtype = 'complex')
for m in range(M):
    SPEEDCONDITION[m] = cos(m*arccos(y_star)) 

PSI, Cxx, Cyy, Cxy, Nu = pickle.load(open(inFileName, 'r'))

traceOutFp = open(traceOutFileName, 'w')

for cCounter in range(1000000):

    Re0 = copy(Re)
    Re = Re - 0.001

    xVec = zeros((4*vecLen + 2), dtype='complex')
    xVec[0:vecLen]          = PSI
    xVec[vecLen:2*vecLen]   = Cxx
    xVec[2*vecLen:3*vecLen] = Cyy
    xVec[3*vecLen:4*vecLen] = Cxy
    xVec[4*vecLen]          = Nu 
    xVec[4*vecLen + 1]      = Re

    xVec0 = xVec[:4*vecLen+1]

    #mmc.matrix_checker(find_jacobian(xVec), vecLen, False)
    #xVec = line_search(solve_eq, find_jacobian, xVec)
    #xVec = old_newton(xVec)

    inv_jac = eye(4*vecLen+2, 4*vecLen +2) #linalg.inv(find_jacobian(xVec))
    xVec = optimize.newton_krylov(solve_eq, xVec, inner_M=inv_jac, verbose=True)


    PSI = xVec[0:vecLen] 
    Cxx = xVec[1*vecLen:2*vecLen] 
    Cyy = xVec[2*vecLen:3*vecLen] 
    Cxy = xVec[3*vecLen:4*vecLen]
    Nu  = xVec[4*vecLen]
    Re  = xVec[4*vecLen + 1]

    print "\t------------------------------------\n"
    print "\tNu = ", Nu

    U = dot(MDY, PSI)
    V = -dot(MDX, PSI)
    MMU = tsm.c_prod_mat(U)
    MMV = tsm.c_prod_mat(V)
    U0sq = (dot(MMU,U) + dot(MMV,V))[N*M:(N+1)*M]
    assert allclose(almostZero, imag(U0sq)), "Imaginary velocities!"
    KE0 = (15.0/8.0)*0.5*real(dot(INTY, U0sq))

    print '\tKE0 = {0}, Re = {1}'.format(KE0, Re)
    print "\tnorm of 1st psi mode = ", linalg.norm(PSI[(N+1)*M:(N+2)*M], 2)

    if KE0 > 1e-4 and KE0 < 1.0:
        traceOutFp.write("{0:10.5f} {1:10.5f} \n".format( Re, kx ))
        traceOutFp.flush()
       
        if not cCounter % 10:
            consts = {'N':N, 'M':M, 'kx':kx, 'Re':Re, 'b':beta, 'Wi':Wi}
            outFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re:6.2f}-b{b}-Wi{Wi}.pickle".format(**consts)
            pickle.dump((PSI,Cxx,Cyy,Cxy,Nu), open(outFileName, 'w'))
    else:
        print 'KE0 is way too small or it is probably laminar'
        break

traceOutFp.close()
