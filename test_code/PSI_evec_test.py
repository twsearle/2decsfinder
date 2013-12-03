#-------------------------------------------------------------------------------
#   Plot of the streamfunction of the base profile for test
#
#
#-------------------------------------------------------------------------------

#MODULES
from scipy import *
from scipy import linalg
import cPickle as pickle
from matplotlib import pyplot as plt
from matplotlib import rc

#SETTINGS----------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
kx = config.getfloat('General', 'kx')
numYs = config.getint('Plotting', 'numYs')

#------------------------------------------------

def Cheb_to_real_transform(vec, y_points) :
    """ calculate the Fourier chebychev transform for the 2D coherent state
    finder"""
    rVec = zeros(numYs, dtype='complex')
    for yIndx in range(numYs):
        y = y_points[yIndx]
        for m in range(M):
            term = vec[m] * cos(m*arccos(y))
            rVec[yIndx] += term
    del y,m

    return rVec

#
#   MAIN
#

#PSI = random.random(M)/100000.0
PSI = zeros(M,dtype='d')

Psi = zeros((2*N+1)*M, dtype='complex')
Psi[(N-1)*M:N*M] = pickle.load(open('psi.init', 'r'))

PSI[0]   = -2.0/3.0
PSI[1]   = -3.0/4.0
PSI[2]   = 0.0
PSI[3]   = 1.0/12.0

#Psi[N*M]   = -2.0/3.0
#Psi[N*M+1] = -3.0/4.0
#Psi[N*M+2] = 0.0
#Psi[N*M+3] = 1.0/12.0

y_points = zeros(numYs, dtype='d')
for yIndx in range(numYs):
    y_points[yIndx] = 1.0 - (2.0*yIndx)/(numYs-1.0)
del yIndx

PSIr1 = Cheb_to_real_transform(Psi[(N-1)*M: N*M], y_points)

# Set the phase according to our phase condition of Im(psi(0.5)) = 0.
# To fix this: psi1(y) = psi1_in(y).(1-ib/a) where psi(0.5) = a + ib
PSI0RS = 0. + 0.j
for m in range(M):
    PSI0RS += Psi[(N-1)*M + m]*cos(m*arccos(0.5))
del m

PhaseFactor = 1. - 1.j*(imag(PSI0RS)/real(PSI0RS))

print PhaseFactor

Psi[(N-1)*M:N*M] = PhaseFactor*Psi[(N-1)*M:N*M] 
Psi[N*M:(N+1)*M] = conjugate(PhaseFactor)*Psi[N*M:(N+1)*M] 

PSIrPhase = Cheb_to_real_transform(Psi[(N-1)*M: N*M], y_points)

#f = 1./3. * y_points**3 -y_points -2./3.

#make plots prettier:
inches_per_Lx = 1.4
inches_per_Ly = 2.2
fig_width = 10 
fig_height = 2*inches_per_Ly      
fig_size =  [fig_width,fig_height]
rc('figure', figsize=fig_size)


plt.figure()
#plt.plot(y_points, f, 'ro')
plt.plot(y_points, real(PSIr1), 'bx-')
plt.plot(y_points, imag(PSIr1), 'rx-')
plt.plot(y_points, real(PSIrPhase), 'go-')
plt.plot(y_points, imag(PSIrPhase), 'ko-')
titleString = 'psi for k = {k}, Re = {Re}'.format(k=kx, Re=Re)
plt.title(titleString)
plt.show()
#plt.savefig(r'psi.pdf')

savetxt('test.dat', vstack((real(PSIr1), imag(PSIr1))).T)
