#------------------------------------------------------------------------------
#   colour map plotter for 2D coherent state finder
#   animated.
#
#   Last modified: Tue  3 Dec 14:12:36 2013
#
#------------------------------------------------------------------------------

#MODULES
from scipy import *
from scipy import linalg
import cPickle as pickle
import ConfigParser
import matplotlib 
matplotlib.use('tkAgg')
import matplotlib.pyplot
import matplotlib.animation

#SETTINGS----------------------------------------

N = 1
M = 30
Re = 5770
kx = 1.302
numYs = 50
numXs = 50

#------------------------------------------------

# FUNCTIONS

def Fourier_cheb_transform(vec, x_points, y_points) :
    """ calculate the Fourier chebychev transform for the 2D coherent state
    finder"""
    rVec = zeros((numXs, numYs),dtype='complex')
    for xIndx in range(numXs):
        for yIndx in range(numYs):
            for n in range(2*N+1):
                for m in range(M):
                    x = x_points[xIndx]
                    y = y_points[yIndx]
                    term = vec[n*M + m] * exp(1.j*(n-N)*kx*x) * cos(m*arccos(y))
                    rVec[yIndx, xIndx] += term
    del x,y,n,m

    return real(rVec)

def animate(i):
    psi = real(Psi2DList[i])
    im.set_array(psi)
    return im,

# MAIN

x_points = zeros(numXs,dtype='d') 
for xIndx in range(numXs):
    #               2.lambda     * fractional position
    x_points[xIndx] = (4.*pi/kx) * ((1.*xIndx)/numXs)
del xIndx

y_points = zeros(numYs,dtype='d')
for yIndx in range(numYs):
    y_points[yIndx] = (2.0*yIndx)/(numYs-1.0) - 1.0 
del yIndx

# Read in
#PsiList = pickle.load(open('psi_time.pickle', 'r'))

#print PsiList[0][(N+1)*M:(N+2)*M]
#numFrames = len(PsiList)

# Perform transformation
#Psi2DList = []
#for i in range(numFrames):
#    print i
#    Psi2DList.append(Fourier_cheb_transform(PsiList[i], x_points, y_points))

#pickle.dump(Psi2DList, open('transformed_time.pickle', 'w'))
Psi2DList = pickle.load(open('transformed_time.pickle', 'r'))
numFrames = len(Psi2DList)
#print all(equal(Psi2DList[0], Psi2DList[numFrames-1]))

# make meshes
#grid_x, grid_y = meshgrid(x_points, y_points)

fig = matplotlib.pyplot.figure()
im = matplotlib.pyplot.imshow(real(Psi2DList[0]), origin='lower', extent=[0,22,-1,1], aspect=4)
matplotlib.pyplot.colorbar(orientation='horizontal')
titleString = 'psi frame {i}'.format(i=0)
matplotlib.pyplot.title(titleString)

anim = matplotlib.animation.FuncAnimation(fig, animate, frames=numFrames,
                              interval=1, blit=True)
matplotlib.pyplot.show()

