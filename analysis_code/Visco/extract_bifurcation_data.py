# -----------------------------------------------------------------------------
#   extract_birfurcation_data.py
#   Last modified: Mon  6 Oct 13:36:38 2014
#
#
# -----------------------------------------------------------------------------
"""
Read in data from the continuation trace files and organise it into a list of
points for paraview, and maybe for a colour map if I get time. Adjacent points
should be such that they are adjacent on the surface, this helps programs like
paraview plot a correct connected surface.
"""


from scipy import *
import numpy as np
import glob
import operator
import sys

from matplotlib import pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rc
# This next import is supposed to make sure fonts are good, regardless of os
from matplotlib.font_manager import fontManager, FontProperties

# SETTINGS ---------------------------------------------------------------------

N = 8
M = 50
beta = 0.1
Wi = 1.0

# ------------------------------------------------------------------------------

# FUNCTIONS

def gen_list_pts_from_files(N, M, rootName):

    fileList = glob.glob(rootName + '*.dat')

    numKxPts = len(fileList)
    bothArr = []

    for filename in fileList:
        splitString = filename.split('-')
        if len(splitString) < 7:
            continue
        NString = splitString[2]
        NFile = int(NString[1:])
        MString = splitString[3]
        MFile = int(MString[1:])
        if NFile != N or MFile != M:
            continue

        kxString = splitString[4]
        kx = float(kxString[2:])

        print filename, NFile, MFile, kx

        bothArr.append([filename, kx])

    if bothArr == []:
        print 'warning: no files found in this folder: ', rootName
        return None

    bothArr.sort(key=operator.itemgetter(1))
    bothArr = array(bothArr)

    fileList = bothArr[:,0]

    data3Darr = zeros(3, dtype='d')

    for filename in fileList:
        fileData = genfromtxt(filename)

        # attempt to copy the data into the larger data array. If this is
        # unsuccesful, it could just be that the file only contains one point,
        # so treat that casse as an exception. 
        try:
            fileData = fileData[:, :3]
            data3Darr = vstack((data3Darr, fileData))
        except:
            fileData = fileData[:3]
            data3Darr = vstack((data3Darr, fileData))
    del filename

    data3Darr = data3Darr[1:,:]
    #savetxt( 'bif-surf-N{N}-M{M}.dat'.format(N=N, M=M), data3Darr)

    return data3Darr

def slice_by_Re(ptsArr, searchReVal):
    # Return the points in the array which correspond to the Reynold's number
    # search value 

    answerArgs = argwhere(ptsArr[:,0]==searchReVal)
    answer2 = []
    for a in answerArgs:
        answer2.append(a[0])

    if len(answer2) is 0: 
        print 'Warning, looking at an Re with no points on one of the branches'
        print answer2, type(answer2)
        return None

    answer2 = array(answer2)
    return ptsArr[answer2, :]


# MAIN

botBranchPts = gen_list_pts_from_files(N, M,
                                       'N{N}_M{M}/lower_branch/cont-trace-'.format(N=N, M=M))
topBranchPts = gen_list_pts_from_files(N, M, 'N{N}_M{M}/upper_branch/cont-trace-'.format(N=N, M=M))

## The sort ##

# order of columns goes RE, kx, KE
# sort by kx first, in reverse
topBranchPts = topBranchPts[flipud(topBranchPts[:,1].argsort(kind='mergesort'))]
# Then Re
topBranchPts = topBranchPts[topBranchPts[:,0].argsort(kind='mergesort')]

# For the bottom branch, we need to connect with the kx of the top branch.
# so don't reverse the kx sort
botBranchPts = botBranchPts[botBranchPts[:,1].argsort(kind='mergesort')]
# sort by Re second
botBranchPts = botBranchPts[botBranchPts[:,0].argsort(kind='mergesort')]

## Making one big array ##

# First we need to find out all of the Reynolds numbers that we have, then for
# each Reynold's number, put all points next to each other which are at that Re
# and adjacent in kx.

ReList = unique(concatenate((topBranchPts[:,0], botBranchPts[:,0])))
ReList.sort()

branchesPts = zeros(3, dtype='d')
for Re in ReList:
    tmpTop = slice_by_Re(topBranchPts, Re)
    tmpBot = slice_by_Re(botBranchPts, Re)

    if tmpTop is not None and tmpBot is None:
        branchesPts = vstack((branchesPts, tmpTop))
    if tmpTop is None and tmpBot is not None:
        branchesPts = vstack((branchesPts, tmpBot))
    if tmpTop is None and tmpBot is None:
        pass
    if tmpTop is not None and tmpBot is not None: 
        branchesPts = vstack((branchesPts, tmpTop, tmpBot))

branchesPts = branchesPts[1:,:]

## Output for paraview ##

BranchesPtsPv = copy(branchesPts)
BranchesPtsPv[:,0] = branchesPts[:,0]/1000

savetxt('b{b}_Wi{Wi}_pv_bifurcation.txt'.format(b=beta, Wi=Wi), BranchesPtsPv,
        delimiter=',')
    
## Write a colour map ##

minkx = min(branchesPts[:,1])
maxkx = max(branchesPts[:,1])

minKE = min(branchesPts[:,2])
maxKE = max(branchesPts[:,2])

minRePt = branchesPts[0,:]

x = branchesPts[:,1]
y = branchesPts[:,2]
z = branchesPts[:,0]

# define grid.
fig = plt.figure()
xi = np.linspace(minkx-0.01,maxkx+0.01, 500)
yi = np.linspace(minKE-0.01,maxKE+0.01, 500)

# grid the data.
zi = griddata(x,y,z,xi,yi,interp='linear')
# contour the gridded data, plotting dots at the nonuniform data points.
#CS = plt.contour(xi,yi,zi)

# Move numbers to an array (now that it is gridded) and plot using imshow

print type(xi), type(yi), type(zi)
print shape(xi), shape(yi), shape(zi)
imArr = zeros((len(xi), len(yi)), dtype='d')

for xIndx in range(len(xi)):
    for yIndx in range(len(yi)):
        imArr[xIndx, yIndx] = zi[xIndx,yIndx]

ax1 = fig.add_subplot(111)
ax1.set_ylim([0.4,1])
ax1.set_xlim([1.1,1.6])

ax1.autoscale(False)
cPlot = plt.imshow(imArr, origin='lower', extent=(min(xi), max(xi), min(yi), max(yi)))
# Add minimum Reynold's number
minRePlt, = ax1.plot(minRePt[1], minRePt[2], 'og')

lStr = "Re = {0}, kx = {1}, KE0 = {2:4.3f}".format(minRePt[0],
                                                           minRePt[1], minRePt[2])
ax1.legend([lStr], fontsize='small', loc='best')
ax1
plt.colorbar() # draw colorbar
plt.xlabel('$k_{x}$')
plt.ylabel('$KE0 / KE0_{lam}$')
plt.show()

fig.savefig('b{b}_Wi{Wi}_pv_bifurcation_map.pdf'.format(b=beta, Wi=Wi))

