
# MODULES
from scipy import *
from scipy import linalg
import cPickle as pickle
import ConfigParser
import argparse
import TobySpectralMethods as tsm

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

args = argparser.parse_args()
N = args.N 
M = args.M
Re = args.Re
beta = args.b
Wi = args.Wi
kx = args.kx

if args.Newt:

    kwargs = {'N': N, 'M': M, 'Re': Re, 'kx': kx}
    inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(**kwargs)

    print "Settings:"
    print """
    ------------------------------------
    N \t= {N}
    M \t= {M}              
    Re \t= {Re}         
    kx \t= {kx}
    ------------------------------------
        """.format(**kwargs)

else:

    kwargs = {'N': N, 'M': M, 'Re': Re, 'Wi':Wi, 'beta':beta, 'kx': kx}
    inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}-b{beta}-Wi{Wi}.pickle".format(**kwargs)

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
        """.format(**kwargs)


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
fp = open('trace.txt', 'a')
fp.write("{0} {1} {2}\n".format(Re, kx, KE0))
fp.close()
