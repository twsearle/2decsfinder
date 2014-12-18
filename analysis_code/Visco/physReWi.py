#-----------------------------------------------------------------------------#
#
#   Calculate Physical Reynold's and Weissenberg numbers
#   physReWi.py
#
#
#-----------------------------------------------------------------------------#

"""
Calculate the physical Reynolds and Weissenberg numbers of a 2D exact solution.
Multiple methods are used for the Weissenberg number, the maximum velocity
gradient, the stresses, the stresses adjusted for the principle direction.

TODO:
    Check that I am defining the stress correctly. Might need to include Tyy
    somewhere?
"""


from scipy import *
import ConfigParser
import cPickle as pickle

import TobySpectralMethods as tsm
import RStransform


def faster_FC_transform(vecin, N, M, numXs, numYs, kx) :
    vecout = zeros(numXs*numYs, dtype='D')

    RStransform.rstransform(vecin, vecout, N, M, numXs, numYs, kx)

    # Reshape back to (xindx, yindx) then transpose to (yindx, xindx) (because)
    vecout = reshape(vecout, (numXs, numYs)).T

    return vecout

def calc_Re_Wi(flow_profile, CONSTS, method='mean_profile'):
    """
    Take a given flow profile and calculate the Reynold's number from the mean
    flow and the Weissenberg number using one of 3 methods.

    First need to calculate: U, gradVxx, gradVyy, gradVxy, DyU, Txx, Tyy, Txy.

    'mean_profile':
        Via the mean profile, Wi = Wi * max(U'(y))

    'stress':
        Via the stresses.

        Make an array of Wi at all points from the stresses and take the
        maximum.

        Wi = max( Txx(x,y) / Txy(x,y) )

    'corrected_stress':
        Take account of the reorientation of the principle stress direction.

        Make an array of the Wi at all points using an approximation of the
        principle stress direction.

        Wi = max |            2*sqrt(T:T)            |
                 |-----------------------------------|
                 |sqrt(Dv + (Dv)`) : sqrt(Dv + (Dv)`)|

        So the bottom line is a guess of the shear stress and the top line is
        an approximation to the eigendirection of the maximum stress. like a
        really shit krylov method.

    """

    # caculate the stresses and gradients from the flow profile

    (PSI, Cxx, Cyy, Cxy, Nu) = flow_profile

    tsm.initTSM(N_=CONSTS['N'], M_=CONSTS['M'], kx_=CONSTS['kx'])

    N = CONSTS['N']
    M = CONSTS['M']
    Wi = CONSTS['Wi']
    Re = CONSTS['Re']
    kx = CONSTS['kx']
    numXs = CONSTS['numXs']
    numYs = CONSTS['numYs']

    MDX = tsm.mk_diff_x()
    MDY = tsm.mk_diff_y()

    U = dot(MDY, PSI)
    V = dot(MDX, PSI)

    DyU = dot(MDY, U)

    Txx = Cxx / Wi
    Txx[N*M] = Txx[N*M] - 1./Wi

    Tyy = Cyy / Wi
    Tyy[N*M] = Tyy[N*M] - 1./Wi

    Txy = Cxy / Wi

    gradVxx = 2*dot(MDX, U)
    gradVyy = 2*dot(MDY, V)
    gradVxy = dot(MDX, V) + dot(MDY, U)

    # transform to real space in 2D

    Txx = faster_FC_transform(Txx, N, M, numXs, numYs, kx)
    Txy = faster_FC_transform(Txy, N, M, numXs, numYs, kx)
    Tyy = faster_FC_transform(Tyy, N, M, numXs, numYs, kx)

    gradVxx = faster_FC_transform(gradVxx, N, M, numXs, numYs, kx)
    gradVxy = faster_FC_transform(gradVxy, N, M, numXs, numYs, kx)
    gradVyy = faster_FC_transform(gradVyy, N, M, numXs, numYs, kx)

    U = faster_FC_transform(U, N, M, numXs, numYs, kx)
    DyU = faster_FC_transform(DyU, N, M, numXs, numYs, kx)
    
    # calculate the new Reynold's number and Wi

    ReNew = Re * amax(U)

    if method == 'mean_profile':

        WiNew = Wi * amax( DyU )

    elif method == 'stress':

        WiNew = Wi * amax( Txx / Txy )

    elif method == 'corrected_stress':

        TTdyad = Txx*Txx + Tyy*Tyy + 2*Txy*Txy
        gradVdyad = gradVxx**2 + gradVyy**2 + 2*gradVxy**2

        WiNew = 2 * amax(sqrt(TTdyad) / sqrt(gradVdyad) )
    
    return real(ReNew), real(WiNew)

if __name__ == "__main__":

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

    CONSTS = {'N': N, 'M': M, 'Re': Re,'Wi': Wi, 'beta': beta, 'kx': kx, 'numXs':numXs, 'numYs':numYs}

    filename = "pf-N{N}-M{M}-kx{kx}-Re{Re}-b{beta}-Wi{Wi}.pickle".format(**CONSTS)

    flow_profile = pickle.load(open(filename, 'r'))
    print 'Re = {Re}, Wi = {Wi}'.format(**CONSTS) 

    ReA, WiA = calc_Re_Wi(flow_profile, CONSTS, 'mean_profile')
    print 'ReA = {Re}, WiA = {Wi}'.format(Re=ReA, Wi=WiA) 

    ReB, WiB = calc_Re_Wi(flow_profile, CONSTS, 'stress')
    print 'ReB = {Re}, WiB = {Wi}'.format(Re=ReB, Wi=WiB) 

    ReC, WiC = calc_Re_Wi(flow_profile, CONSTS, 'corrected_stress')
    print 'ReC = {Re}, WiC = {Wi}'.format(Re=ReC, Wi=WiC) 

