# -----------------------------------------------------------------------------
#   plot_3D_surface.py
#   Last modified: Sun  5 Jan 12:24:13 2014
#
#
# -----------------------------------------------------------------------------
"""
Read in Data from KE trace files. Order this data so that points that are
adjacent are connected on the surface. Also make it so that contours can be
drawn from the data set in any direction.
"""


from scipy import *
from matplotlib import pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import glob
import operator
import sys

# PARAMETERS ------------------------------------------------------------------

N = 7
M = 50
ReSpacing = 0.5
# -----------------------------------------------------------------------------

# FUNCTIONS

def gen_list_pts_from_file(N, M, rootName):
    fileList = glob.glob(rootName + '*.dat')
    numKxPts = len(fileList)
    bothArr = []
    for filename in fileList:
        splitString = filename.split('-')
        if len(splitString) < 5:
            continue
        NString = splitString[2]
        NFile = int(NString[1:])
        MString = splitString[3]
        MFile = int(MString[1:])
        if NFile != N or MFile != M:
            continue
        print filename, NFile, MFile
        kxString = splitString[4]
        kx = float(kxString[2:-4])
        bothArr.append([filename, kx])
    del filename

    bothArr.sort(key=operator.itemgetter(1))
    bothArr = array(bothArr)

    fileList = bothArr[:,0]

    data3Darr = zeros(3, dtype='d')
    for filename in fileList:
        g = genfromtxt(filename)
        #print filename, shape(g)
        try:
            g = g[:, :3]
            data3Darr = vstack((data3Darr, g))
        except:
            g = g[:3]
            data3Darr = vstack((data3Darr, g))
    del filename

    data3Darr = data3Darr[1:,:]
    #savetxt( 'bif-surf-N{N}-M{M}.dat'.format(N=N, M=M), data3Darr)

    return data3Darr

def slice_by_Re(ptsArr, searchReVal):
    # Return the points in the array which correspond ot the Reynold's number
    # search value 

    answerArgs = argwhere(ptsArr[:,0]==searchReVal)
    answer2 = []
    for a in answerArgs:
        answer2.append(a[0])
    answer2 = array(answer2)

    return ptsArr[answer2, :]

def plot_hacky_contour(ReList, BranchesPts):
    numCtrs = 6
    ctrStep = len(ReList) / numCtrs

    ctrList = arange(0,len(ReList), ctrStep)
    for ctrIndx in ctrList:
        Re = ReList[ctrIndx]
        tmpSlice = slice_by_Re(BranchesPts, Re)
        plt.plot(tmpSlice[:,1], tmpSlice[:,2])

    plt.show()

def mk_single_diffy():
    """Makes a matrix to differentiate a single vector of Chebyshev's, 
    for use in constructing large differentiation matrix for whole system"""
    # make matrix:
    mat = zeros((M, M), dtype='d')
    for m in range(M):
        for p in range(m+1, M, 2):
            mat[m,p] = 2*p*oneOverC[m]

    return mat

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

def mk_cheb_int():
    integrator = zeros(M, dtype='d')
    for m in range(0,M,2):
        integrator[m] = (1 + cos(m*pi)) / (1-m*m)
    del m
    return integrator

# MAIN

# Set the oneOverC function: 1/2 for m=0, 1 elsewhere:
oneOverC = ones(M)
oneOverC[0] = 1. / 2.
#set up the CFunc function: 2 for m=0, 1 elsewhere:
CFunc = ones(M)
CFunc[0] = 2.

topBranchPts = gen_list_pts_from_file(N, M, 'topKE')
botBranchPts = gen_list_pts_from_file(N, M, 'KE')

# sort by KE then kx then Re

# mergesort makes the sort stable, do multiple sorts so that list is ordered by
# different columns
# order of columns goes RE, kx, KE
topBranchPts = topBranchPts[flipud(topBranchPts[:,1].argsort(kind='mergesort'))]
topBranchPts = topBranchPts[topBranchPts[:,0].argsort(kind='mergesort')]

# The flip of the argument list should bring about a reverse sort

botBranchPts = botBranchPts[botBranchPts[:,1].argsort(kind='mergesort')]
botBranchPts = botBranchPts[botBranchPts[:,0].argsort(kind='mergesort')]

# Take the top and bottom branches and put them into a larger list, where
# adjacent rows are connected 

# form a list of all Re

minRe = min(topBranchPts[:,0])
maxRe = max(topBranchPts[:,0])
if min(botBranchPts[:,0]) < minRe:
    minRe = min(botBranchPts[:,0])
if max(botBranchPts[:,0]) > maxRe:
    maxRe = max(botBranchPts[:,0])

ReList = arange(minRe, maxRe+ReSpacing, ReSpacing)

# for each Re, add the top and bottom branches to a new array
BranchesPts = zeros(3, dtype='d')
for Re in ReList:
    tmpTop = slice_by_Re(topBranchPts, Re)
    tmpBot = slice_by_Re(botBranchPts, Re)
    BranchesPts = vstack((BranchesPts, tmpTop, tmpBot))

BranchesPts = BranchesPts[1:,:]

# Change units to KEpois - KE0
INTY = mk_cheb_int()
SMDY = mk_single_diffy()

PSI0      = zeros(M, dtype='D') 
PSI0[0]   = 2.0/3.0
PSI0[1] = 3.0/4.0
PSI0[2] = 0.0
PSI0[3] = -1.0/12.0
U0 = dot(SMDY, PSI0)
KEpois = 0.5*dot(INTY, dot(cheb_prod_mat(U0), U0) )

BranchesPts[:,2] = KEpois - BranchesPts[:,2]

# Find approximation of the Critical point:
# BranchesPts array is already ordered, so 
myMinPt = BranchesPts[0,:]
print myMinPt

# Hacked together contour plot:
# plot_hacky_contour(ReList, BranchesPts)

# proper matplotlib contour plot

minkx = min(BranchesPts[:,1])
maxkx = max(BranchesPts[:,1])

minKE = min(BranchesPts[:,2])
maxKE = max(BranchesPts[:,2])

x = BranchesPts[:,1]
y = BranchesPts[:,2]
z = BranchesPts[:,0]

# define grid.
fig = plt.figure()
xi = np.linspace(minkx-0.01,maxkx+0.01, 200)
yi = np.linspace(minKE-0.01,maxKE+0.01, 200)
# grid the data.
zi = griddata(x,y,z,xi,yi,interp='linear')
# contour the gridded data, plotting dots at the nonuniform data points.
CS = plt.contour(xi,yi,zi)
#CS = plt.contourf(xi,yi,zi)
#plt.colorbar() # draw colorbar
# plot data points.
plt.scatter(x,y,marker='o',c='b',s=1,zorder=10)
#plt.plot(myMinPt[1], myMinPt[2], 'k+')

# Move numbers to an array (now that it is gridded) and plot using imshow

print type(xi), type(yi), type(zi)
print shape(xi), shape(yi), shape(zi)
imArr = zeros((len(xi), len(yi)), dtype='d')

for xIndx in range(len(xi)):
    for yIndx in range(len(yi)):
        imArr[xIndx, yIndx] = zi[xIndx,yIndx]

cPlot = plt.imshow(imArr, origin='lower', extent=(min(xi), max(xi), min(yi), max(yi)))
plt.colorbar() # draw colorbar
plt.show()

fig.savefig('bifurcation_contour_map.pdf')

###### Plot the 3D connected surface
# This is going to be possible because we rotated the data so that the gridded z
# axis is single valued everywhere

# NOTE: The data types of xi,yi differ from zi. stupid matplotlib example
XGrid, YGrid = meshgrid(xi, yi)
ZGrid = zi

# Convert from masked grid to standard ndarray
ZGridUnmasked = zeros(ZGrid.shape, dtype='d')
for xIndx in range(len(ZGrid[:,0])):
    for yIndx in range(len(ZGrid[0,:])):
        ZGridUnmasked[xIndx, yIndx] = ZGrid[xIndx, yIndx]

ZGrid = ZGridUnmasked

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(XGrid, YGrid, ZGrid, alpha=0.3, antialiased=False)
fig.savefig('surface.png')

print XGrid.shape, YGrid.shape, ZGrid.shape
print type(XGrid), type(YGrid), type(ZGrid)

# Plot contours on the walls. Doesn't work because XGrid and YGrid are not in
# the right format

#cset = ax.contour(XGrid, YGrid, ZGrid, zdir='z', offset=2939, cmap=cm.coolwarm)
#cset = ax.contour(XGrid, YGrid, ZGrid, zdir='x', offset=min(xi), cmap=cm.coolwarm)
#cset = ax.contour(XGrid, YGrid, ZGrid, zdir='y', offset=max(yi), cmap=cm.coolwarm)

plt.show()

