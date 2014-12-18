
# MODULES
from scipy import *
from scipy import linalg
import cPickle as pickle
import ConfigParser
import argparse

import TobySpectralMethods as tsm
import physReWi

# SETTINGS---------------------------------------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
Wi = config.getfloat('General', 'Wi')
beta = config.getfloat('General', 'beta')
kx = config.getfloat('General', 'kx')

numXs = config.getint('Plotting', 'numXs')
numYs = config.getint('Plotting', 'numYs')

fp.close()

# Use argparse to interpret the command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("-Newt", 
                help = 'Examine pure newtonian ECS',
                       action="store_true")
argparser.add_argument("-N", type=int, default=N, 
                help='Override Number of Fourier modes given in the config file')
argparser.add_argument("-M", type=int, default=M, 
                help='Override Number of Chebyshev modes in the config file')
argparser.add_argument("-Re", type=float, default=Re, 
                help="Override Reynold's number in the config file") 
argparser.add_argument("-b", type=float, default=beta, 
                help='Override beta of the config file')
argparser.add_argument("-Wi", type=float, default=Wi, 
                help='Override Weissenberg number of the config file')
argparser.add_argument("-kx", type=float, default=kx, 
                help='Override wavenumber of the config file')
argparser.add_argument("-calcReWiMeth", default='mean_profile')

args = argparser.parse_args()
N = args.N 
M = args.M
Re = args.Re
beta = args.b
Wi = args.Wi
kx = args.kx
method = args.calcReWiMeth

if args.Newt:

    CONSTS = {'N': N, 'M': M, 'Re': Re, 'kx': kx, 'numXs':numXs, 'numYs':numYs}
    inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(**CONSTS)

    print "Settings:"
    print """
    ------------------------------------
    N \t= {N}
    M \t= {M}              
    Re \t= {Re}         
    kx \t= {kx}
    ------------------------------------
        """.format(**CONSTS)

else:

    CONSTS = {'N': N, 'M': M, 'Re': Re, 'Wi':Wi, 'beta':beta, 'kx': kx, 'numXs':numXs, 'numYs':numYs}
    inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}-b{beta}-Wi{Wi}.pickle".format(**CONSTS)

    print "Settings:"
    print """
    ------------------------------------
    N \t= {N}
    M \t= {M}              
    Wi \t= {Wi}        
    Re \t= {Re}         
    beta \t= {beta}
    kx \t= {kx}
    ------------------------------------
        """.format(**CONSTS)


tsm.initTSM(N_=N, M_=M, kx_=kx)

# -----------------------------------------------------------------------------


# Useful operators 

oneOverWi = 1./Wi

MDY = tsm.mk_diff_y()
MDYY = dot(MDY,MDY)
MDYYY = dot(MDY,MDYY)
MDX = tsm.mk_diff_x()
MDXX = dot(MDX, MDX)
MDXY = dot(MDX, MDY)
LAPLAC = dot(MDX,MDX) + dot(MDY,MDY)
BIHARM = dot(LAPLAC, LAPLAC)
MDXLAPLAC = dot(MDX, LAPLAC)
MDYLAPLAC = dot(MDY, LAPLAC)

# single mode Operators
SMDY = tsm.mk_single_diffy()
SMDYY = dot(SMDY, SMDY)
SMDYYY = dot(SMDY, SMDYY)

INTY = tsm.mk_cheb_int()

if args.Newt:
    (PSI, Nu) = pickle.load(open(inFileName,'r'))
    U = dot(MDY, PSI)
    V = - dot(MDX, PSI)

    # Newtonian force term:
    lplU2 = dot( (dot(LAPLAC, U)), conj(dot(LAPLAC, U)) ) 
    lplV2 = dot( (dot(LAPLAC, V)), conj(dot(LAPLAC, V)) ) 
    NewtStress = sqrt(lplU2+lplV2)
    print 'Strength of the Newtonian stress term in Navier stokes: ', NewtStress

else:
    (PSI, Cxx, Cyy, Cxy, Nu) = pickle.load(open(inFileName,'r'))
    U = dot(MDY, PSI)
    V = - dot(MDX, PSI)

    # Newtonian force term:
    lplU2 = dot( (dot(LAPLAC, U)), conj(dot(LAPLAC, U)) ) 
    lplV2 = dot( (dot(LAPLAC, V)), conj(dot(LAPLAC, V)) ) 
    NewtStress = beta*sqrt(lplU2+lplV2)
    print 'Strength of the Newtonian stress term in Navier Stokes: ', NewtStress

    # Viscoelastic force term
    polyU = dot(MDX, oneOverWi*(Cxx - 1)) + dot(MDY, oneOverWi*Cxy)
    polyV = dot(MDY, oneOverWi*(Cyy - 1)) + dot(MDX, oneOverWi*Cxy)
    ViscoStress = (1-beta)*sqrt( dot(polyU, conj(polyU)) + dot(polyV, conj(polyV)) )
    
    print 'Strength of the Viscoelastic stress term in Navier Stokes: ', ViscoStress

    # Calculate the physical Re and Wi 

    ReN, WiN1 = physReWi.calc_Re_Wi((PSI, Cxx, Cyy, Cxy, Nu), CONSTS,
                                   method='mean_profile')
    print 'calculated Re, Wi'
    print ReN, WiN1

    ReN, WiN2 = physReWi.calc_Re_Wi((PSI, Cxx, Cyy, Cxy, Nu), CONSTS,
                                   method='stress')
    print ReN, WiN2

    ReN, WiN3 = physReWi.calc_Re_Wi((PSI, Cxx, Cyy, Cxy, Nu), CONSTS,
                                   method='corrected_stress')
    print ReN, WiN3


PSI0norm = linalg.norm(PSI[N*M:(N+1)*M], 2)
print 'Norm of 0th mode', PSI0norm
# output size of the first mode
PSI1norm = linalg.norm(PSI[(N-1)*M:N*M]+PSI[(N+1)*M:(N+2)*M], 2)
print 'Norm of 1st mode', PSI1norm

MMU = tsm.prod_mat(U)
MMV = tsm.prod_mat(V)
Usq = dot(MMU, U) + dot(MMV, V)
Usq1 = Usq[(N-1)*M:N*M] + Usq[(N+1)*M:(N+2)*M]
KE0 = 0.5*dot(INTY, Usq[N*M:(N+1)*M])
KE0 = (15.0/8.0)*real(KE0)
KE1 = (15.0/8.0)*linalg.norm(0.5*dot(INTY, Usq1))
print 'Kinetic energy of 0th mode is: ', KE0
print 'Kinetic energy of 1st mode is: ', KE1


U0sq = dot(tsm.c_cheb_prod_mat(U[N*M:(N+1)*M]), U[N*M:(N+1)*M])
KE0A = (15.0/16.0)*dot(INTY, U0sq)

URsq = zeros(M, dtype='complex')

for i_ in range(N):
    n_ = N-i_
    URsq += dot(tsm.c_cheb_prod_mat(1.j*n_*kx*U[i_*M:(i_+1)*M]),
                -1.j*n_*kx*U[(2*N-i_)*M:(2*N+1-i_)*M])
del i_

KERA = (15.0/16.0)*dot(INTY, URsq) 

print "Alexander's KE0: ", KE0A
print "Alexander's KErest: ", KERA
fp = open('trace.txt', 'a')

if args.Newt:
    fp.write("{0} {1} {2} {3}\n".format(Re, kx, KE0, real(Nu)))
else:
    fp.write("{0} {1} {2} {3} {4} {5} {6} {7}\n".format(Re, kx, KE0, real(Nu),
                                                        Wi, WiN1, WiN2, WiN3,
                                                        ReN))

fp.close()
