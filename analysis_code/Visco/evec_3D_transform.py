
from scipy import *
from scipy import fftpack
import numpy as np
from numpy import linalg
import cPickle as pickle
import ConfigParser
import argparse
import sys
import scipy.weave as weave
from scipy.weave.converters import blitz
import TobySpectralMethods as tsm

# SETTINGS --------------------------------------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
cfgN = config.getint('General', 'N')
cfgM = config.getint('General', 'M')
cfgRe = config.getfloat('General', 'Re')
cfgbeta = config.getfloat('General','beta')
cfgWi = config.getfloat('General','Wi')
cfgkx = config.getfloat('General', 'kx')
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
                help='Override kx of the config file')
argparser.add_argument("-kz", type=float, default=cfgkz, 
                help='Override kz of the config file')
argparser.add_argument("-evecNum", type=int, default=0, 
                help='the number of the leading eigenvector')

argparser.add_argument("-dim2", 
                help='specify just yz eigenvector transform', action='store_true')

args = argparser.parse_args()
N = args.N 
M = args.M
Re = args.Re
beta = args.b
Wi = args.Wi
kx = args.kx
kz = args.kz
evecNum = args.evecNum

tsm.initTSM(N_=N, M_=M, kx_=kx)

baseFileName = '-kz{kz}-N{N}-M{M}-kx{kx}-Re{Re}-b{b}-Wi{Wi}.pickle'.format(
                    kz=kz, kx=kx, N=N, M=M, Re=Re, b=beta, Wi=Wi)

inFilename = 'full-evecs-{0}{1}'.format(evecNum, baseFileName)
outFilename = 'real-evec-{0}{1}'.format(evecNum, baseFileName)

bothFilename = 'real-both-{0}{1}.txt'.format(evecNum, baseFileName[:-7])


element_number = r_[0:M]
YPOINTS = cos(pi*element_number/(M-1))
zDataPts = 50
zDataPts_2D = 50 # 2D output uses double the number of points per period 
yDataPts = 50
xDataPts = 50

# -----------------------------------------------------------------------------


#FUNCTIONS

def to_3Dreal(velA):
    """convert velocity to a 2D real space velocity array"""
    print 'converting velocity to real space...'
    Vel2D = zeros((yDataPts, zDataPts), dtype='complex')
    for zind in xrange(zDataPts):
        for yind in xrange(yDataPts):
            for n in xrange(2*N+1):
                for m in xrange(M):
                    Vel2D[yind, zind] += velA[n*M+m]*cheb(yind, m)*fou(zind, n)
    del m, n, yind, zind

    #double the size in the z dimension to make it easier to see.
    realVel = zeros((xDataPts,yDataPts, 2*zDataPts))
    realVel[0,:,:] = hstack((Vel2D,Vel2D))
    del Vel2D

    print 'filling out x dimension...'
    for xindx in xrange(1,xDataPts):
        realVel[xindx,:,:] = realVel[0,:,:]*exp(1.j*kx*x_point(xindx)) +  \
                            conjugate(realVel[0,:,:])*exp(-1.j*kx*x_point(xindx))
    del xindx
    realVel[0,:,:] = realVel[0,:,:] + conjugate(realVel[0,:,:])

    return realVel

def c_to_3Dreal(vecIn):

    """
    use scipy weave to do the fourier transform. Blitz arrays ought to be
    accurate and fast
    """

    weaveCode = r"""
    #include <cmath>
    #include <complex>
    int n, m, xind, yind;
    double cheb, tmp, tmp2;
    double PI;
    PI = 3.14159265358979323846264338327950288419716939937510582097494459230781640628;
    for (xind=0; xind<xDataPts; xind++) {
        for (yind=0; yind<yDataPts; yind++) {
            for (n=0; n<2*N+1; n++) {
                for (m=0; m<M; m++) {
                tmp = acos(-1. + (2./((double)yDataPts-1.))*(double)yind);
                cheb = cos((double)m*tmp);
                tmp2=2.*PI*((double)n-(double)N)*(((double)xind/((double)xDataPts-1.))-0.5);
                fou.imag() = sin(tmp2);
                fou.real() = cos(tmp2);
                vecOut(yind, xind) += vecIn(n*M+m)*cheb*fou;
                }
            }
        }
    }
    """
    vecOut = zeros((yDataPts, xDataPts), dtype='D', order='C')
    EI = 1.j
    fou = 1. + 1.j
    weave.inline(weaveCode,['vecOut', 'vecIn', 'N', 'M', 'xDataPts',
                            'yDataPts', 'fou'],type_converters = blitz, compiler='gcc',
                 headers=["<cmath>", "<complex>" ] )

    #double the size in the z dimension to make it easier to see.
    realVel = zeros((yDataPts, zDataPts, xDataPts))
    realVel[:,0,:] = vecOut
    del vecOut

    print 'filling out z dimension...'
    for zindx in xrange(1,zDataPts):
        realVel[:,zindx,:] = realVel[:,0,:]*exp(1.j*kz*z_point(zindx)) +  \
                conjugate(realVel[:,0,:])*exp(-1.j*kz*z_point(zindx))
    del zindx
    realVel[:,0,:] = realVel[:,0,:] + conjugate(realVel[:,0,:])

    return realVel

def fft_to_3Dreal(vecIn):
    """
    THIS FUNCTION IS BROKEN 
    Use fast fourier transforms to transform to real space
    Full 3 dimensional transform from spectral to real space. 
    
    First use the 2D Fast fourier transform, then use my Chebyshev transform on
    the result in the y direction.

    I have attempted to minimize the number of 3D arrays created. 

    Note: dealiasing removes a third of the effective degrees of freedom. The
    true resolution is then much lower than that assumed by N,L,M this ought to
    be fixed in future versions as it will be a huge waste of computation.
    """
    print 'THIS FUNCTION IS BROKEN: transforms to GL grid. Needs interpolation.'
    sys.exit(1)

    assert yDataPts > M#, print 'Not enough y data pts'
    assert xDataPts > 2*N+1#, print 'Not enough y data pts'
    assert zDataPts > 3#, print 'Not enough z data pts'


    # MOVE ONTO A 3D FFT ARRAY OF THE CORRECT SIZE
    in3D = zeros((yDataPts, zDataPts, xDataPts), dtype='complex')
    vecIn2D = np.fft.fftshift(vecIn.reshape((2*N+1, M)).T, axes=1) 

    for y_ in range(M):

        # zeroth x mode
        in3D[y_, 1, 0] = vecIn2D[y_, 0]
        in3D[y_, -1, 0] = conj(vecIn2D[y_, 0])

        for x_ in range(1, N+1): 

            # - x modes 
            # first z mode
            in3D[y_, 1, x_] = vecIn2D[y_, x_]

            # -first z mode
            in3D[y_, -1, x_] = conj(vecIn2D[y_, x_])

            # +ve x modes
            # first z mode
            in3D[y_, 1, xDataPts - x_] = vecIn2D[y_, 2*N+1 - x_]

            # -first z mode
            in3D[y_, -1, xDataPts - x_] = conj(vecIn2D[y_, 2*N+1 - x_])

    # PERFORM THE REQUIRED FFT

    # Perform the FFT across the x and z directions   

    _realtmp = zeros((2*yDataPts-2, zDataPts, xDataPts), dtype='double')
    out3D = zeros((2*yDataPts-2, zDataPts, xDataPts), dtype='complex')

    out3D[:yDataPts, :, :] = fftpack.ifft2(in3D)

    normImag = linalg.norm(imag(out3D))

    _realtmp = real(out3D)
    
    # Perform the Chebyshev transformation across the y direction

    # The second half contains the vector on the Gauss-Labatto points excluding
    # the first and last elements and in reverse order
    # do this before filling out the first half! 

    _realtmp[yDataPts:, :, :] = _realtmp[yDataPts-2:0:-1, :, :]

    # The first half contains the vector on the Gauss-Labatto points * c_k
    _realtmp[0, :, :] = 2*_realtmp[0, :, :]
    _realtmp[yDataPts-1, :, :] = 2*_realtmp[yDataPts-1, :, :]

    # Perform the chebyshev transformation
    out3D = 0.5*fftpack.rfft(_realtmp, axis=0)

    normImag = linalg.norm(imag(out3D[0:yDataPts, :, :]))

    out3D = real(out3D)
    
    return out3D[0:yDataPts, :, :]




def c_to_2Dreal(vecIn):
    """
    use scipy weave to do the fourier transform. Blitz arrays ought to be
    accurate and fast
    """

    weaveCode = r"""
    #include <cmath>
    #include <complex>
    int n, m, zind, yind;
    double cheb, tmp, tmp2;
    double PI;
    PI = 3.14159265358979323846264338327950288419716939937510582097494459230781640628;
    for (zind=0; zind<zDataPts; zind++) {
        for (yind=0; yind<yDataPts; yind++) {
            for (n=0; n<2*N+1; n++) {
                for (m=0; m<M; m++) {
                tmp = acos(-1. + (2./((double)yDataPts-1.))*(double)yind);
                cheb = cos((double)m*tmp);
                tmp2=2.*PI*((double)n-(double)N)*(((double)zind/((double)zDataPts-1.))-0.5);
                fou.imag() = sin(tmp2);
                fou.real() = cos(tmp2);
                vecOut(yind, zind) += vecIn(n*M+m)*cheb*fou;
                }
            }
        }
    }
    """
    vecOut = zeros((yDataPts, zDataPts), dtype='D', order='C')
    EI = 1.j
    fou = 1. + 1.j
    weave.inline(weaveCode,['vecOut', 'vecIn', 'N', 'M', 'zDataPts',
                            'yDataPts', 'fou'],type_converters = blitz, compiler='gcc',
                 headers=["<cmath>", "<complex>" ] )

    return real(vecOut)

def cheb(yIndex, chebIndex):
    """Take a yIndex(cies) in the array, change it into a y value in the system,
    then calculate the Chebyshev polynomial."""
    return cos(chebIndex*arccos(-1. + (2./(yDataPts-1.))*yIndex))

def y_point(yIndex):
    return -1. + (2./(yDataPts-1.))*yIndex

def z_point(zIndex):
    zLength = 2*pi/kz 
    return zLength*(zIndex/(zDataPts-1.))

def x_point(xIndex):
    if kx==0:
        xLength = 10
    else:
        xLength = 2.*pi/kx     
    return xIndex/(xDataPts-1.)*xLength

def fou(zIndex, fouIndex):
    return exp(1j*(2*pi)*(fouIndex-N)*(((1.*zIndex)/(zDataPts-1.))-0.5))

def save_field(mat, name):
    """takes a matrix of real values in the y and z planes and saves them 
    in a file such that they are readable by gnuplot."""

    settingsline = '#Reynolds: '+str(Re)+' beta: '+str(beta)+' Weissenburg: '\
                +str(Wi)+ ' Amp: '+str(Amp)

    #Open file, write data, close file
    f = open(name+'.dat', 'w')
    delim = ', '
    f.write(settingsline)
    f.write('\n')
    for n in range(len(mat[0,:])):
        for m in range(len(mat[:,0])):
            f.write(str(z_point(n)))
            f.write(delim)
            f.write(str(y_point(m)))
            f.write(delim)
            f.write(str(mat[m,n]))
            f.write('\n')
        f.write('\n')
    f.close()

def save_csv_for_paraview((xVec, yVec, zVec), filename):

    Nst = 50

    xLength = 2.*pi/kx
    zLength = 2.*pi/kz
    yLength = 2.

    xspace = xLength/(Nst-1.) 
    yspace = yLength/(Nst-1.)
    zspace = zLength/(Nst-1.)

    xPoints = r_[0 : xLength + xspace : xspace] 
    yPoints = r_[-yLength/2. : yLength/2. + yspace : yspace]
    zPoints = r_[-zLength/2. : zLength/2. + zspace : zspace]

    fp = open(filename, 'w')
    fp.write('x,    y,   z,   u,   v,   w\n')
    for xIndx in range(xDataPts):
        for yIndx in range(yDataPts):
            for zIndx in range(zDataPts):
                fp.write('{0:10.5f}, '.format(xPoints[xIndx]))
                fp.write('{0:10.5f}, '.format(yPoints[yIndx]))
                fp.write('{0:10.5f}, '.format(zPoints[zIndx]))
                fp.write('{0:10.5f}, '.format(xVec[yIndx, zIndx, xIndx]))
                fp.write('{0:10.5f}, '.format(yVec[yIndx, zIndx, xIndx]))
                fp.write('{0:10.5f}'.format(zVec[yIndx, zIndx, xIndx]))
                fp.write('\n')
        fp.flush()
    del xIndx, yIndx, zIndx

    fp.close()



#MAIN


(Nux,Nuz,du,dv,dw,_,dcxx,dcyy,dczz,dcxy,dcxz,dcyz) = pickle.load(open(inFilename, 'r'))

print 'test', linalg.norm(du)

print 'finished reading in files'

# Vorticity
oneOverC = ones(M)
oneOverC[0] = 1. / 2.
#set up the CFunc function: 2 for m=0, 1 elsewhere:
CFunc = ones(M)
CFunc[0] = 2.
zLength = 2.*pi/kz
vecLen = (2*N+1)*M
MDY = tsm.mk_diff_y()
MDX = tsm.mk_diff_x()

print 'vorticity'
dOmegaX = dot(MDY,dw) - 1.j*kz*dv
dOmegaY = 1.j*kz*du   - dot(MDX,dw)
dOmegaZ = dot(MDX,dv) - dot(MDY,du)

dOmegaX = c_to_3Dreal(dOmegaX)
dOmegaY = c_to_3Dreal(dOmegaY)
dOmegaZ = c_to_3Dreal(dOmegaZ)

print 'du'
du = c_to_3Dreal(du)
print 'dv'
dv = c_to_3Dreal(dv)
print 'dw'
dw = c_to_3Dreal(dw)
print 'dcxx'
dcxx = c_to_3Dreal(dcxx)
print 'dcyy'
dcyy = c_to_3Dreal(dcyy)
print 'dczz'
dczz = c_to_3Dreal(dczz)
print 'dcxy'
dcxy = c_to_3Dreal(dcxy)
print 'dcxz'
dcxz = c_to_3Dreal(dcxz)
print 'dcyz'
dcyz = c_to_3Dreal(dcyz)

# RESCALING, so that velocities are between + - dumax 

duMax = 1.0

if amax(abs(du)) != 0:
    rescaleFactor = duMax / amax(abs(du))
else:
    print 'maximum du is zero'
    rescaleFactor = 1

du = rescaleFactor*du
dv = rescaleFactor*dv
dw = rescaleFactor*dw

dcxx = rescaleFactor*dcxx
dcyy = rescaleFactor*dcyy
dczz = rescaleFactor*dczz
dcxy = rescaleFactor*dcxy
dcxz = rescaleFactor*dcxz
dcyz = rescaleFactor*dcyz

save_csv_for_paraview((du,dv,dw), outFilename[:-7]+'.txt')

outVortFilename = 'vorticity-evec{0}.txt'.format(baseFileName[:-7])
save_csv_for_paraview((dOmegaX, dOmegaY, dOmegaZ), outVortFilename)

pickle.dump((du,dv,dw,dcxx,dcyy,dczz,dcxy,dcxz,dcyz), open(outFilename, 'w'))

basePfName = 'real-pf-N{N}-M{M}-kx{kx}-Re{Re}-b{b}-Wi{Wi}.pickle'.format(
            N=N, M=M, kx=kx, Re=Re, b=beta, Wi=Wi)


(Nu,U,V,W,Cxx,Cyy,Czz,Cxy,Cxz,Cyz) = pickle.load(open(basePfName, 'r'))

tmpU = copy(U)
tmpV = copy(V)
tmpW = copy(W)

U = zeros((yDataPts, zDataPts, xDataPts))
V = zeros((yDataPts, zDataPts, xDataPts))
W = zeros((yDataPts, zDataPts, xDataPts))
for i_ in range(zDataPts):
    U[:,i_,:] = tmpU
    V[:,i_,:] = tmpV
    W[:,i_,:] = tmpW

save_csv_for_paraview((U+2*du, V+2*dv, W+2*dw), bothFilename)

pickle.dump((dOmegaX, dOmegaY, dOmegaZ),
            open('vorticity-evec{0}'.format(baseFileName), 'w'))
