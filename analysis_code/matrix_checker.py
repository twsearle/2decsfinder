from scipy import *
from scipy import linalg
import matplotlib.pyplot as plt
import cPickle as pickle

def matrix_checker(jacobian) :
    """Gives you a bunch of facts about the jacobian"""
    print "The jacobian determinant is ", linalg.det(jacobian)
    zeroColCond = any(jacobian, axis=0)
    print "If false there is a zero col: ", all(zeroColCond)
    zeroColsIndxs = nonzero(invert(zeroColCond))
    print "The indices of the zero cols: ", zeroColsIndxs
    cIndx = zeroColsIndxs
    savetxt("zeroCols.dat", jacobian[zeroColsIndxs].T)
    print "sum over the first zero col check, ", sum(jacobian[:,cIndx[0]])
    print "Col Indx divided by vecLen", double(cIndx)/vecLen

    zeroRowCond = any(jacobian, axis=1) 
    print "If false there is a zero row: ", all(zeroRowCond)
    zeroRowsIndxs = nonzero(invert(zeroRowCond))
    print "The indices of the zero rows: ", zeroRowsIndxs
    rIndx = zeroRowsIndxs
    savetxt("zeroRows.dat", jacobian.T[zeroRowsIndxs].T)
    print "sum over the first zero row check, ", \
            sum(jacobian[rIndx[0],:])
    print "Row Indx divided by vecLen", double(rIndx)/vecLen
    plt.figure()
    plt.imshow(log(real(jacobian*conjugate(jacobian))))
    plt.show()
    exit(1)

