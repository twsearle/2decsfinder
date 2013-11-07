from scipy import *
from scipy import linalg
import sys
import matplotlib.pyplot as plt

def matrix_checker(jacobian, vecLen, output) :
    """Gives you a bunch of facts about the jacobian matrix"""
    print "The jacobian determinant is ", linalg.det(jacobian)
    zeroColCond = any(jacobian, axis=0)
    print "If false there is a zero col: ", all(zeroColCond)
    zeroColsIndxs = nonzero(invert(zeroColCond))
    print "The indices of the zero cols: ", zeroColsIndxs
    cIndx = zeroColsIndxs
    if output==True : savetxt("zeroCols.dat", jacobian[zeroColsIndxs].T)
    print "sum over the first zero col check, ", sum(jacobian[:,cIndx[0]])
    print "Col Indx divided by vecLen", double(cIndx)/vecLen

    zeroRowCond = any(jacobian, axis=1) 
    print "If false there is a zero row: ", all(zeroRowCond)
    zeroRowsIndxs = nonzero(invert(zeroRowCond))
    print "The indices of the zero rows: ", zeroRowsIndxs
    rIndx = zeroRowsIndxs
    if output==True : savetxt("zeroRows.dat", jacobian.T[zeroRowsIndxs].T)
    print "sum over the first zero row check, ", \
            sum(jacobian[rIndx[0],:])
    print "Row Indx divided by vecLen", double(rIndx)/vecLen

    print "Search for NaN's, are there any?", any(isnan(jacobian))

    print "BLOCK BY BLOCK"
    jdim = len(jacobian[:,0]) 

    for j in range(int(jdim/vecLen)): 
        for i in range(int(jdim/vecLen)):
            d = linalg.det(jacobian[j*vecLen:(j+1)*vecLen,
                                    i*vecLen:(i+1)*vecLen])
            norm = linalg.norm(jacobian[j*vecLen:(j+1)*vecLen,
                                    i*vecLen:(i+1)*vecLen], 2)
            s = "BLOCK: row {r}, column {c} (det, norm) = {d} {n}".format(c=i+1, 
                                                                        r=j+1,
                                                                        d=d,
                                                                         n=norm)
            print s
    del i,j,d

    plt.figure()
    plt.imshow(log(real(jacobian*conjugate(jacobian))))
    plt.show()
    exit(1)



if __name__=='__main__':
    filein = sys.argv[1]
    print " checking {filein} ".format(filein=filein)

    jac = genfromtxt(filein)
    matrix_checker(jac, True)

