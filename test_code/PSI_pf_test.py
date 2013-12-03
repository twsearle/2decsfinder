#-------------------------------------------------------------------------------
#   Plot of the streamfunction of the base profile for test
#
#
#-------------------------------------------------------------------------------

#MODULES
from scipy import *
from scipy import linalg
import cPickle as pickle
import sys
from matplotlib import pyplot as plt
from matplotlib import rc

#SETTINGS----------------------------------------

N = 1              # Number of Fourier modes
M = 20               # Number of Chebychevs (>4)
Re = 5770.0           # The Reynold's number
kx  = 1.18

numYs = 50

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
#PSI = zeros(M,dtype='d')

Psi = zeros((2*N+1)*M, dtype='complex')
#Psi[(N-1)*M:N*M] = pickle.load(open('psi.init', 'r'))

(Psi, Nu) = pickle.load(open(sys.argv[1], 'r'))
print Psi

#pickle.dump(Psi[(N-1)*M:N*M],open('psi.init', 'w'))

#PSI[0]   = -2.0/3.0
#PSI[1]   = -3.0/4.0
#PSI[2]   = 0.0
#PSI[3]   = 1.0/12.0

#Psi[N*M]   = -2.0/3.0
#Psi[N*M+1] = -3.0/4.0
#Psi[N*M+2] = 0.0
#Psi[N*M+3] = 1.0/12.0

y_points = zeros(numYs, dtype='d')
for yIndx in range(numYs):
    y_points[yIndx] = 1.0 - (2.0*yIndx)/(numYs-1.0)
del yIndx

PSIr0 = Cheb_to_real_transform(Psi[N*M: (N+1)*M], y_points)

PSIr1 = Cheb_to_real_transform(Psi[(N-1)*M: N*M], y_points)

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
titleString = 'psi n=0 mode'
plt.title(titleString)
plt.plot(y_points, real(PSIr0), 'bo-')
plt.plot(y_points, imag(PSIr0), 'ro-')
plt.show()

plt.plot(y_points, real(PSIr1), 'bo-')
plt.plot(y_points, imag(PSIr1), 'ro-')
titleString = 'psi n=1 mode'
plt.title(titleString)
plt.show()
#plt.savefig(r'psi.pdf')

savetxt('test.dat', vstack((real(PSIr1), imag(PSIr1))).T)
