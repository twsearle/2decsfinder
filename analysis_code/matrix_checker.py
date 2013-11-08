#------------------------------------------------------------------------------
#
#   matrix checking module
#   Last modified: Fri  8 Nov 17:26:18 2013
#
#------------------------------------------------------------------------------
""" Checks properties of the matrix used in the problem to diagnose and test
whether it is working. Can be run as a stand alone program if provided with
scipy format matrix text file a length of the equation blocks """


from scipy import *
from scipy import linalg
from numpy import linalg as nplinalg
import sys
import matplotlib.pyplot as plt

def matrix_checker(mat, vecLen, output) :
    """Gives you a bunch of facts about the mat"""

    print "The matrix determinant is ", linalg.det(mat)
    zeroColCond = any(mat, axis=0)
    print "The matrix condition number is ", nplinalg.cond(mat)

    print "If false there is a zero col: ", all(zeroColCond)
    zeroColsIndxs = nonzero(invert(zeroColCond))
    print "The indices of the zero cols: ", zeroColsIndxs
    cIndx = zeroColsIndxs
    #if output==True : savetxt("zeroCols.dat", mat[zeroColsIndxs].T)
    print "sum over the first zero col check, ", sum(mat[:,cIndx[0]])
    print "Col Indx divided by vecLen", double(cIndx)/vecLen

    zeroRowCond = any(mat, axis=1) 
    print "If false there is a zero row: ", all(zeroRowCond)
    zeroRowsIndxs = nonzero(invert(zeroRowCond))
    print "The indices of the zero rows: ", zeroRowsIndxs
    rIndx = zeroRowsIndxs
    #if output==True : savetxt("zeroRows.dat", mat.T[zeroRowsIndxs].T)
    print "sum over the first zero row check, ", \
            sum(mat[rIndx[0],:])
    print "Row Indx divided by vecLen", double(rIndx)/vecLen
    print "Search for NaN's, are there any?", any(isnan(mat))

    print "BLOCK BY BLOCK"
    jdim = len(mat[:,0]) 

    for j in range(int(jdim/vecLen)): 
        for i in range(int(jdim/vecLen)):
            d = linalg.det(mat[j*vecLen:(j+1)*vecLen,
                                    i*vecLen:(i+1)*vecLen])
            norm = linalg.norm(mat[j*vecLen:(j+1)*vecLen,
                                    i*vecLen:(i+1)*vecLen], 2)
            s = "BLOCK: row {r}, column {c} (det, norm) = {d} {n}".format(c=i+1, 
                                                                        r=j+1,
                                                                        d=d,
                                                                         n=norm)
            print s
    del i,j,d

    print "calculate singular values"
    singvals = linalg.svd(mat, compute_uv=False)
    print "are the singular values positive-definite? ", all(nonzero(singvals))
    if output==True: savetxt("singvals.dat", singvals)


    plt.figure()
    plt.imshow(log(real(mat*conjugate(mat))))
    plt.show()
    exit(1)



if __name__=='__main__':
    filein = sys.argv[1]
    vecLen = sys.argv[2]
    print " checking {filein} ".format(filein=filein)

    jac = genfromtxt(filein)
    matrix_checker(jac, vecLen, True)

