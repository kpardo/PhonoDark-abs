'''
rate.py
'''

import numpy as np
from pda.constants import *
import pda.selfenergy as se
import pda.new_physics as physics


def generate_q_mesh(q_mag, num_q_theta, num_q_phi):
    '''
    creates x,y,z mesh of q values with given magnitude
    distributed across sphere
    '''
    start = (0.5)/num_q_phi  # start and end away from pole

    # sample thetas using arccos function to create more even grid?
    betas = np.linspace(start, (1-start), num_q_theta)
    thetas = np.arccos(2*betas - 1.)

    phis = np.linspace(start*2*np.pi, 2*np.pi*(1-start), num_q_phi)
    pp, tt = np.meshgrid(phis, thetas, indexing='xy')

    xx = np.array([np.sin(tt)*np.cos(pp)], dtype=np.float64)
    yy = np.array([np.sin(tt)*np.sin(pp)], dtype=np.float64)
    zz = np.array([np.cos(tt)], dtype=np.float64)
    qxyz = q_mag*np.vstack([xx, yy, zz]).T

    return qxyz.reshape((num_q_theta*num_q_phi, 3))


def get_vel_contrib(q_XYZ_list, vEVec):
    '''
    calculates velocity distribution function
    Inputs: q list
    Outputs: list of velocity distribution function evaluated at q's.
    '''
    q_dir = q_XYZ_list / np.linalg.norm(q_XYZ_list, axis=1)[:, np.newaxis]

    # get theta/phi values
    # q_dir has shape N x 3, columns are x,y,z
    theta = np.arccos(q_dir[:, 2])
    phi = np.arctan2(q_dir[:, 1], q_dir[:, 0])

    int_vel_dist_val = np.array([physics.int_vel_dist(
        t, p, vEVec)[0] for t, p in zip(theta, phi)])
    return int_vel_dist_val


def rate(mass_list, q_XYZ_list, mat, coupling=None, pol_mixing=True, width='best', width_val=10**(-3)):
    selfenergy = se.SelfEnergy(nu=mass_list, k=q_XYZ_list, mat=mat,
                               coupling=coupling, pol_mixing=pol_mixing,
                               lam='vi', width=width, width_val=width_val)
    if coupling.se_shape == 'vector':
        sesum = np.einsum('ikjn -> ikj', selfenergy.se)
    else:
        sesum = selfenergy.se
    # Get Absorption Rate, Eqn. 1
    absrate = -1. / mass_list * np.imag(sesum)
    # FIXME: should probably make a separate class for vel?
    # for now, just using dressed up version of Tanner's code.
    jacob = 4 * np.pi / len(q_XYZ_list)
    vel_contrib = get_vel_contrib(q_XYZ_list, np.array([0, 0, VE]))
    # FIXME: need to be more careful with vel int over angles here?
    velint = np.einsum('ikj, i -> kj', absrate, jacob * vel_contrib)
    # finally sum over mat.energies -- do not sum over last 2 energies
    # these last 2 modes are not physical -- zero energy modes.
    totself = np.einsum('kj -> j', velint[:-2])
    prefac = RHO_DM * mass_list**(-1) * 1./mat.m_cell
    # Get total rate, Eqn. 2
    rate = prefac * totself
    return rate
