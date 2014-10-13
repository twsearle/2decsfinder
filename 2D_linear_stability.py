#----------------------------------------------------------------------------#
#   Fully spectral linear stability analysis of a 2D exact solution
#
#   Last modified: Tue  7 Oct 12:45:15 2014
#
#
#----------------------------------------------------------------------------#

"""
Code to perform linear stability analysis for a 2D streamwise-wall-normal
channel flow. Input ought to be in the streamfunction form. Output will be an
eigenvalue spectrum or an eigenfunction profile depending on the commandline
arguments provided.
"""

# MODULES
import sys
import time
from scipy import *
from scipy import linalg
import cPickle as pickle
import ConfigParser
import argparse

import TobySpectralMethods as tsm

# SETTINGS -------------------------------------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)

cfgN = config.getint('General', 'N')
cfgM = config.getint('General', 'M')
cfgkx = config.getfloat('General', 'kx')
cfgRe = config.getfloat('General', 'Re')
cfgbeta = config.getfloat('General','beta')
cfgWi = config.getfloat('General','Wi')

cfgkz = config.getfloat('Linear Stability', 'kz')

fp.close()

argparser = argparse.ArgumentParser()

argparser.add_argument("-N", type=int, default=cfgN, 
                help='Override Number of Fourier modes given in the config file')
argparser.add_argument("-M", type=int, default=cfgM, 
                help='Override Number of Chebyshev modes in the config file')
argparser.add_argument("-Re", type=float, default=cfgRe, 
                help="Override Reynold's number in the config file") 
argparser.add_argument("-b", type=float, default=cfgbeta, 
                help='Override beta of the config file')
argparser.add_argument("-Wi", type=float, default=cfgWi, 
                help='Override Weissenberg number of the config file')
argparser.add_argument("-kx", type=float, default=cfgkx,
                help='Override kx from the config file')
argparser.add_argument("-kz", type=float, default=cfgkz,
                help='Override kz from the config file')
argparser.add_argument("-evecs", 
                help = 'output eigenvectors instead of eigenvalues',
                       action="store_true")
argparser.add_argument("-NumEvecs", type=int, default=5, 
help="If eigenvectors are to be outputted, change the number of leading eigenvectors")

args = argparser.parse_args()
N = args.N 
M = args.M
Re = args.Re
beta = args.b
Wi = args.Wi
kx = args.kx
kz = args.kz


filename = '-N{N}-M{M}-kx{kx}-Re{Re}-b{beta}-Wi{Wi}.pickle'.format(\
            N=N, M=M, kx=kx, Re=Re, beta=beta, Wi=Wi)

tsm.initTSM(N_=N, M_=M, kx_=kx)

# -----------------------------------------------------------------------------

#FUNCTIONS

def mk_bigM():
    """ makes the matrix to appear on the left hand side of the generalised
    eigenvalue problem"""
    bigM = zeros((10*vLen, 10*vLen), dtype=complex)

    ################Navier Stokes x direction:###############
    #*u
    bigM[0:vLen, 0:vLen] = + Re*Nu*MDX - Re*GRAD - Re*MMDXU \
                           + beta*LAPLACIAN
    #*v
    bigM[0:vLen, vLen:2*vLen] = -Re*MMDYU
    #*w
    #*p
    bigM[0:vLen, 3*vLen:4*vLen] = - MDX
    #cxx
    bigM[0:vLen, 4*vLen:5*vLen] = (1-beta)*oneOverWi*MDX
    #cyy
    #czz
    #cxy
    bigM[0:vLen, 7*vLen:8*vLen] = (1-beta)*oneOverWi*MDY

    #cxz
    bigM[0:vLen, 8*vLen:9*vLen] = (1-beta)*oneOverWi*1.j*kz*eye(vLen,vLen)
    #cyz
    bigM[0:vLen, 9*vLen:10*vLen] = 0 

    ################Navier Stokes y direction:###############
    #*u
    bigM[vLen:2*vLen, 0:vLen] = - Re*MMDXV
    #*v
    bigM[vLen:2*vLen, vLen:2*vLen] = Re*Nu*MDX - Re*GRAD - Re*MMDYV \
                                     + beta*LAPLACIAN
    #*w
    #*p
    bigM[vLen:2*vLen, 3*vLen:4*vLen] = - MDY
    #cxx
    #cyy
    bigM[vLen:2*vLen, 5*vLen:6*vLen] = (1-beta)*oneOverWi*MDY
    #czz
    #cxy
    bigM[vLen:2*vLen, 7*vLen:8*vLen] = (1-beta)*oneOverWi*MDX
    #cxz
    #cyz
    bigM[vLen:2*vLen, 9*vLen:10*vLen] = (1-beta)*oneOverWi*1.j*kz*eye(vLen,vLen)

    ################Navier Stokes z direction:###############

    #*u
    bigM[2*vLen:3*vLen, 0:vLen] = - Re*MMDXW
    #*v
    bigM[2*vLen:3*vLen, vLen:2*vLen] = - Re*MMDYW
    #*w
    bigM[2*vLen:3*vLen, 2*vLen:3*vLen] = Re*Nu*MDX - Re*GRAD + beta*LAPLACIAN
    #*p
    bigM[2*vLen:3*vLen, 3*vLen:4*vLen] = - 1.j*kz*eye(vLen,vLen)
    #cxx
    bigM[2*vLen:3*vLen, 4*vLen:5*vLen] = 0
    #cyy
    #czz
    bigM[2*vLen:3*vLen, 6*vLen:7*vLen] = (1-beta)*oneOverWi*1.j*kz*eye(vLen,vLen)
    #cxy
    bigM[2*vLen:3*vLen, 7*vLen:8*vLen] = 0
    #cxz
    bigM[2*vLen:3*vLen, 8*vLen:9*vLen] = (1-beta)*oneOverWi*MDX
    #cyz
    bigM[2*vLen:3*vLen, 9*vLen:10*vLen] = (1-beta)*oneOverWi*MDY

    ################Incompressability equation:###############
    #*u
    bigM[3*vLen:4*vLen, 0:vLen] = MDX 
    #*v
    bigM[3*vLen:4*vLen, vLen:2*vLen] = MDY
    #*w
    bigM[3*vLen:4*vLen, 2*vLen:3*vLen] = 1.j*kz*eye(vLen,vLen)
    #*p
    #cxx
    #cyy
    #czz
    #cxy
    #cxz
    #cyz

    ################cxx equation:####################

    #*u
    bigM[4*vLen:5*vLen, 0*vLen:vLen] = - tsm.c_prod_mat(dot(MDX,Cxx)) + 2.j*kz*MMCXZ \
                                       +  2*dot(MMCXY,MDY) + 2*dot(MMCXX,MDX)
    #*v
    bigM[4*vLen:5*vLen, vLen:2*vLen] = -tsm.c_prod_mat(dot(MDY,Cxx))
    #*w
    #*p
    #cxx
    bigM[4*vLen:5*vLen, 4*vLen:5*vLen] = Nu*MDX - oneOverWi*eye(vLen,vLen) - GRAD \
                                       + 2*MMDXU 
    #cyy
    #czz
    bigM[4*vLen:5*vLen, 6*vLen:7*vLen] = 0
    #cxy
    bigM[4*vLen:5*vLen, 7*vLen:8*vLen] = 2*MMDYU
    #cxz
    #cyz
    bigM[4*vLen:5*vLen, 9*vLen:10*vLen] = 0

    ################cyy equation:####################
    #*u
    bigM[5*vLen:6*vLen, 0:vLen] =  -tsm.c_prod_mat(dot(MDX,Cyy))
    #*v
    bigM[5*vLen:6*vLen, vLen:2*vLen] = +2j*kz*MMCYZ +2*dot(MMCYY,MDY) \
                                     + 2*dot(MMCXY,MDX) \
                                     - tsm.c_prod_mat(dot(MDY,Cyy))
    #*w
    #*p
    #cxx
    #cyy
    bigM[5*vLen:6*vLen, 5*vLen:6*vLen] = Nu*MDX - oneOverWi*eye(vLen,vLen) - GRAD \
                                       + 2*MMDYV
    #czz
    #cxy
    bigM[5*vLen:6*vLen, 7*vLen:8*vLen] = 2*MMDXV
    #cxz
    #cyz

    ################czz equation:####################

    #*u
    bigM[6*vLen:7*vLen, 0:vLen] = -tsm.c_prod_mat(dot(MDX,Czz))
    #*v
    bigM[6*vLen:7*vLen, vLen:2*vLen] = -tsm.c_prod_mat(dot(MDY,Czz)) 
    #*w
    bigM[6*vLen:7*vLen, 2*vLen:3*vLen] = 2.j*kz*MMCZZ + 2*dot(MMCYZ,MDY) \
                                        + 2*dot(MMCXZ,MDX) 
    #*p
    #cxx
    #cyy
    #czz
    bigM[6*vLen:7*vLen, 6*vLen:7*vLen] = Nu*MDX - oneOverWi*eye(vLen,vLen) - GRAD
    #cxy
    #cxz
    bigM[6*vLen:7*vLen, 8*vLen:9*vLen] = 2*MMDXW
    #cyz
    bigM[6*vLen:7*vLen, 9*vLen:10*vLen] = 2*MMDYW
    
    ################cxy equation:####################

    #*u
    bigM[7*vLen:8*vLen, 0:vLen] = - tsm.c_prod_mat(dot(MDX,Cxy)) + 1.j*kz*MMCYZ \
                                         + dot(MMCYY,MDY) 
    #*v
    bigM[7*vLen:8*vLen, vLen:2*vLen] = - tsm.c_prod_mat(dot(MDY,Cxy)) + 1.j*kz*MMCXZ \
                                       + dot(MMCXX,MDX)
    #*w
    bigM[7*vLen:8*vLen, 2*vLen:3*vLen] = -1.j*kz*MMCXY
    #*p
    #cxx
    bigM[7*vLen:8*vLen, 4*vLen:5*vLen] =  MMDXV
    #cyy
    bigM[7*vLen:8*vLen, 5*vLen:6*vLen] =  MMDYU
    #czz
    #cxy
    bigM[7*vLen:8*vLen, 7*vLen:8*vLen] = Nu*MDX - oneOverWi*eye(vLen,vLen) - GRAD
    #cxz
    #cyz

    ################cxz equation:####################
    #*u
    bigM[8*vLen:9*vLen, 0:vLen] = - tsm.c_prod_mat(dot(MDX,Cxz)) + 1.j*kz*MMCZZ\
                                         + dot(MMCYZ,MDY) 
    #*v
    bigM[8*vLen:9*vLen, vLen:2*vLen] = - tsm.c_prod_mat(dot(MDY,Cxz)) \
                                      - dot(MMCXZ,MDY)
    #*w
    bigM[8*vLen:9*vLen, 2*vLen:3*vLen] = + dot(MMCXY,MDY) + dot(MMCXX,MDX)
    #*p
    #cxx
    bigM[8*vLen:9*vLen, 4*vLen:5*vLen] = MMDXW
    #cyy
    #czz
    #cxy
    bigM[8*vLen:9*vLen, 7*vLen:8*vLen] = MMDYW
    #cxz
    bigM[8*vLen:9*vLen, 8*vLen:9*vLen] = Nu*MDX - oneOverWi*eye(vLen,vLen) - GRAD\
                                         + MMDXU
    #cyz
    bigM[8*vLen:9*vLen, 9*vLen:10*vLen] =  MMDYU

    ###############cyz equation:####################

    #*u
    bigM[9*vLen:10*vLen, 0:vLen] = - tsm.c_prod_mat(dot(MDX,Cyz)) \
                                         - dot(MMCYZ,MDX)
    #*v
    bigM[9*vLen:10*vLen, vLen:2*vLen] = -tsm.c_prod_mat(dot(MDY,Cyz)) + 1.j*kz*MMCZZ \
                                        + dot(MMCXZ,MDX)
    #*w
    bigM[9*vLen:10*vLen, 2*vLen:3*vLen] =  + dot(MMCYY,MDY) + dot(MMCXY,MDX)
    #*p
    #cxx
    #cyy
    bigM[9*vLen:10*vLen, 5*vLen:6*vLen] =  MMDYW
    #czz
    #cxy
    bigM[9*vLen:10*vLen, 7*vLen:8*vLen] =  MMDXW
    #cxz
    bigM[9*vLen:10*vLen, 8*vLen:9*vLen] =  MMDXV
    #cyz
    bigM[9*vLen:10*vLen, 9*vLen:10*vLen] = Nu*MDX - oneOverWi*eye(vLen,vLen) - GRAD\
                                         + MMDYV
    

    #Apply Boundary Conditions for u, v, w:
    for i in range(3*(2*N+1)):
        bigM[M*(i+1)-2,:] = hstack((zeros(M*i), BTOP, zeros(10*vLen-M*(i+1))))
        bigM[M*(i+1)-1,:] = hstack((zeros(M*i), BBOT, zeros(10*vLen-M*(i+1))))
    del i

    return bigM

#
# MAIN
#
#Start the clock:
startTime = time.time()

print """
----------------------------------------
N     = {0}
M     = {1}
Re    = {2}
beta  = {3}
Wi    = {4}
kx    = {5}
kz    = {6}
----------------------------------------
""". format(N, M, Re, beta, Wi, kx, kz)

# Unpickle the 2D exact solution

fpickle = open('pf'+filename, 'r')
(PSI, Cxx, Cyy, Cxy, Nu) = pickle.load(fpickle)
fpickle.close()

# Setup variables:
vecLen = M*(2*N+1)
vLen = vecLen
print vLen, type(vLen)
oneOverWi = 1./Wi

#Boundary arrays:
BTOP = ones(M)
BBOT = ones(M)
BBOT[1:M:2] = -1

# Set up U,V and the other stress components 
MDY = tsm.mk_diff_y()
MDX = tsm.mk_diff_x()

U = dot(MDY, PSI)
V = -dot(MDX, PSI)

W = zeros(vecLen, dtype='D')
Czz = zeros(vecLen, dtype='D')
Czz[N*M] = 1
Cxz = zeros(vecLen, dtype='D')
Cyz = zeros(vecLen, dtype='D')

# make some useful matrices
MMW = tsm.c_prod_mat(W)
MMV = tsm.c_prod_mat(V)
MMU = tsm.c_prod_mat(U)
MMCXX = tsm.c_prod_mat(Cxx)
MMCYY = tsm.c_prod_mat(Cyy)
MMCZZ = tsm.c_prod_mat(Czz)
MMCXY = tsm.c_prod_mat(Cxy)
MMCXZ = tsm.c_prod_mat(Cxz)
MMCYZ = tsm.c_prod_mat(Cyz)

MMDYW = tsm.c_prod_mat(dot(MDY, W))
MMDXW = tsm.c_prod_mat(dot(MDX, W))
MMDYV = tsm.c_prod_mat(dot(MDY, V))
MMDXV = tsm.c_prod_mat(dot(MDX, V))
MMDYU = tsm.c_prod_mat(dot(MDY, U))
MMDXU = tsm.c_prod_mat(dot(MDX, U))

LAPLACIAN = -(kz**2)*eye(vLen,vLen) + dot(MDY,MDY) + dot(MDX,MDX)
GRAD      = 1.j*kz*MMW + dot(MMV,MDY) + dot(MMU,MDX)

# Make the matrix for the generalised eigenvalue problem

equations_matrix= mk_bigM()

# Make the scaling matrix for RHS of equation
RHS = eye(10*vLen,10*vLen) 
RHS[:3*vLen, :3*vLen] = Re*eye(3*vLen,3*vLen)
# Zero all elements corresponding to p equation
RHS[3*vLen:4*vLen, :] = zeros((vLen,10*vLen))

# Apply boundary conditions to RHS
for i in range(3*(2*N+1)):
    RHS[M*(i+1)-1, M*(i+1)-1] = 0
    RHS[M*(i+1)-2, M*(i+1)-2] = 0
del i

if args.evecs:
    print 'finding eigenvectors and eigenvalues'
    # Use library function to solve for eigenvalues/vectors
    print 'in linalg.eig time=', (time.time() - startTime)
    eigenvals, evecs = linalg.eig(equations_matrix, RHS, overwrite_a=True)

    # Save output

    #make large_evs, as large as eigenvals, but only contains real part of large, 
    #physical eigenvalues. Rest are zeros. Index of large_evs same as that of 
    #eigenvalues 
    large_evs = zeros(len(eigenvals))
    for i in xrange(10*(2*N+1)*M):
        if (real(eigenvals[i]) > 0) and (real(eigenvals[i]) < 50):
            large_evs[i] = real(eigenvals[i])
    del i
    
    # sort the eigenvalues in large_evs so that the last NumEvecs are the
    # largest

    large_evs = sort(large_evs)

    eigarray = vstack((real(eigenvals), imag(eigenvals))).T
    #remove nans and infs from eigenvalues
    #eigarray = eigarray[~isnan(eigarray).any(1), :]
    #eigarray = eigarray[~isinf(eigarray).any(1), :]

    savetxt('ev-kz{kz}{fn}.dat'.format(kz=kz, fn=filename[:-7],
                                                 ), eigarray)

    for evec_index in range(args.NumEvecs): 

        len_levs = len(large_evs)
        lead_index = large_evs[len_levs - 1 - evec_index]

        print 'chosen eig: {e}'.format(e=lead_index)
        du =   evecs[           :(2*N+1)*M,   lead_index]
        dv =   evecs[  (2*N+1)*M:2*(2*N+1)*M, lead_index]
        dw =   evecs[2*(2*N+1)*M:3*(2*N+1)*M, lead_index]
        dp =   evecs[3*(2*N+1)*M:4*(2*N+1)*M, lead_index]
        dcxx = evecs[4*(2*N+1)*M:5*(2*N+1)*M, lead_index]
        dcyy = evecs[5*(2*N+1)*M:6*(2*N+1)*M, lead_index]
        dczz = evecs[6*(2*N+1)*M:7*(2*N+1)*M, lead_index]
        dcxy = evecs[7*(2*N+1)*M:8*(2*N+1)*M, lead_index]
        dcxz = evecs[8*(2*N+1)*M:9*(2*N+1)*M, lead_index]
        dcyz = evecs[9*(2*N+1)*M:10*(2*N+1)*M, lead_index]

        Nux = Nu
        Nuz = eigarray[lead_index, 1] / kz

        pickle.dump((Nux,Nuz,du,dv,dw,dp,dcxx,dcyy,dczz,dcxy,dcxz,dcyz), 
                    open('full-evecs-{n}-kz{kz}{fn}'.format(n=evec_index, kz=kz,fn=filename), 'w'))



else:
    # eigenvalues only
    print 'finding eigenvalues only'
    # Use library function to solve for eigenvalues/vectors
    print 'in linalg.eig time=', (time.time() - startTime)
    eigenvals = linalg.eigvals(equations_matrix, RHS, overwrite_a=True)

    # Save output

    eigarray = vstack((real(eigenvals), imag(eigenvals))).T
    #remove nans and infs from eigenvalues
    eigarray = eigarray[~isnan(eigarray).any(1), :]
    eigarray = eigarray[~isinf(eigarray).any(1), :]

    savetxt('ev-kz'+str(kz)+filename[:-7]+'.dat', eigarray)

    #stop the clock
    print 'done in', (time.time()-startTime)

######################TESTS####################################################

###############################################################################
