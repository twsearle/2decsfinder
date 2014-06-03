#------------------------------------------------------------------------------
#   Take energy data from a series of velocity profiles
#   Last modified: Sat  7 Dec 18:28:01 2013
#
#------------------------------------------------------------------------------

from scipy import *
from scipy import linalg
import glob
import operator
import ConfigParser
import cPickle as pickle
import TobySpectralMethods as tsm

#-----------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
kx = config.getfloat('General', 'kx')

fp.close()


#-----------------------------------------

# MAIN


fileList = glob.glob('*.pickle')
print fileList

OutputList = []

for i, filename in enumerate(fileList):

    splitString = filename.split('-')
    typeString = splitString[0]
    if typeString != 'pf':
        continue

    # Find the Fourier mode for this file
    NString = splitString[1]
    if NString[0] != 'N':
        continue
    
    files_N = int(NString[1:])
    if files_N != N:
        continue

    # Find the Chebyshev mode for this file
    MString = splitString[2]
    if MString[0] != 'M':
        continue
    
    files_M = int(MString[1:])
    if files_M != M:
        continue

    # Find the wavenumber, kx for this file
    kxString = splitString[3]
    if kxString[0:2] != 'kx':
        continue
    
    files_kx = float(kxString[2:])

    # Find the Reynold's number for this file
    ReString = splitString[4]
    if ReString[0:2] != 'Re':
        continue
    # These parameters must be consistent with the Spectral Method functions in
    # tsm!
    if files_Re != Re:
        continue
    
    files_Re = float(ReString[2:-7])

    fileConsts={'N': files_N, 'M': files_M, 'kx': files_kx, 'Re': files_Re}

    # Print the constants for the file to stdout
    print 'N: {N}, M: {M}, kx: {kx}, Re: {Re}'.format(**fileConsts)

    tsm.initTSM(N, M, files_kx)
    MDY = tsm.mk_diff_y()
    MDX = tsm.mk_diff_x()
    (Psi, Nu) = pickle.load(open(filename, 'r'))
    U = dot(MDY, Psi)
    V = dot(MDX, Psi)
    MMU = tsm.prod_mat(U)
    MMV = tsm.prod_mat(V)

    # Need to change this so that it finds the kinetic energy (integrate)
    KE = linalg.norm(dot(MMU,U) + dot(MMV,V), 2)

    dataPoint = [files_kx, files_Re, KE]

    OutputList.append(dataPoint)
del i

OutputList.sort(key=operator.itemgetter(1))
Consts={'N': N, 'M': M, 'Re': Re}
outFileName = 'bif-Re-KE-N{N}-M{M}-Re{Re}.dat'.format(**Consts)
outFp = open(outFileName, 'w')

for row in OutputList:
    outFp.write('{kx}\t{Re}\t{KE}\n'.format(
                     kx=row[0], Re=row[1], KE=row[2]))
outFp.close()

