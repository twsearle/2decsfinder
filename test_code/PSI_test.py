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

N = 3              # Number of Fourier modes
M = 30               # Number of Chebychevs (>4)
Re = 5771.0           # The Reynold's number
kx  = 1.0

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

    return real(rVec)

def real_to_Cheb_transform(vec, y_points) :
    # DOESN'T WORK
    cVec = zeros(M, dtype='d')
    y = y_points
    for k in range(M):
        for j in range(numYs):
            cVec[k] += 2./numYs * vec[j] * cos(pi*k*j/numYs)


    return cVec

#
#   MAIN
#

#PSI = random.random(M)/100000.0
PSI = zeros(M,dtype='d')

PSI[0]   = -2.0/3.0
PSI[1]   = -3.0/4.0
PSI[2]   = 0.0
PSI[3]   = 1.0/12.0


y_points = zeros(numYs, dtype='d')
for yIndx in range(numYs):
    y_points[yIndx] = 1.0 - (2.0*yIndx)/(numYs-1.0)
del yIndx

PSIr = Cheb_to_real_transform(PSI, y_points)

f = 1./3. * y_points**3 -y_points -2./3.

#make plots prettier:
inches_per_Lx = 1.4
inches_per_Ly = 2.2
fig_width = 10 
fig_height = 2*inches_per_Ly      
fig_size =  [fig_width,fig_height]
rc('figure', figsize=fig_size)


plt.figure()
plt.plot(y_points, f, 'ro')
plt.plot(y_points, PSIr, 'bo')
titleString = 'psi for k = {k}, Re = {Re}'.format(k=kx, Re=Re)
plt.title(titleString)
plt.show()
#plt.savefig(r'psi.pdf')
