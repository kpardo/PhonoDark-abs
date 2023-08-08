'''
rate.py
'''

import numpy as np
from pda.constants import *
import pda.selfenergy as se
import pda.new_physics as physics


# def generate_q_mesh(q_mag, num_q_theta, num_q_phi):
def generate_q_mesh(q_mag, n_theta, n_phi):
    '''
    creates x,y,z mesh of q values with given magnitude
    distributed across sphere
    '''
    ##FIXME make sure this is equivalent to what we were doing before and/or update code.
    # start = (0.5)/num_q_phi  # start and end away from pole

    # # sample thetas using arccos function to create more even grid?
    # betas = np.linspace(start, (1-start), num_q_theta)
    # thetas = np.arccos(2*betas - 1.)

    # phis = np.linspace(start*2*np.pi, 2*np.pi*(1-start), num_q_phi)
    # pp, tt = np.meshgrid(phis, thetas, indexing='xy')

    # xx = np.array([np.sin(tt)*np.cos(pp)], dtype=np.float64)
    # yy = np.array([np.sin(tt)*np.sin(pp)], dtype=np.float64)
    # zz = np.array([np.cos(tt)], dtype=np.float64)
    # qxyz = q_mag*np.vstack([xx, yy, zz]).T

    # return qxyz.reshape((num_q_theta*num_q_phi, 3))
    q_list = []

    for i in range(n_theta):
        for j in range(n_phi):

            a = i / max(n_phi - 1.0, 1.0)
            b = j / max(n_theta - 1.0, 1.0)

            phi = 2*np.pi*a
            theta = np.arccos( 2.0*b - 1.0 )

            q_list.append([
                    q_mag*np.sin(theta)*np.cos(phi),
                    q_mag*np.sin(theta)*np.sin(phi),
                    q_mag*np.cos(theta)
                ])

    return np.array(q_list)


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


def rate(mass_list, q_XYZ_list, mat, coupling=None, pol_mixing=True, width='best', width_val=10**(-2)):
    selfenergy = se.SelfEnergy(nu=mass_list, k=q_XYZ_list, mat=mat,
                               coupling=coupling, pol_mixing=pol_mixing,
                               lam='vi', width=width, width_val=width_val)
    # if 'scalar' in coupling.se_shape:
    sesum = selfenergy.se
    # else:
        # sesum = np.einsum('ikjn -> ikj', selfenergy.se)
    # Get Absorption Rate, Eqn. 1
    absrate = -1.*np.imag(sesum)
    # absrate = -1. / mass_list * sesum
    totself = np.sum(absrate/len(q_XYZ_list),axis=0)
    # totself = np.einsum('kj -> j', velint)
    prefac = RHO_DM * mass_list**(-2) * 1./mat.rho_T
    # Get total rate, Eqn. 2 ## FIXME added abs for ElectricDipole...need to check minus signs.
    rate = prefac * totself
    if coupling.se_shape != 'scalar':
        rate *= 1./3.
    return rate
