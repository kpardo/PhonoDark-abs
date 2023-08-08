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


def rate(mass_list, mat, q_XYZ_list=None, coupling=None, pol_mixing=True, width='best', width_val=10**(-2)):
    '''
    calculates velocity distribution function
    Inputs: mass list, q list, material (optional: coupling, width type, width value)
    Outputs: rate as a function of mass
    '''
    if q_XYZ_list == None:
        q_XYZ_list = generate_q_mesh(10**(-2), 5, 5)
    selfenergy = se.SelfEnergy(omega=mass_list, k=q_XYZ_list, mat=mat,
                               coupling=coupling, pol_mixing=pol_mixing,
                               lam='vi', width=width, width_val=width_val)
    sesum = selfenergy.se
    # Get Absorption Rate, Eqn. 2
    absrate = -1.* mass_list**(-1) * np.imag(sesum)
    totself = np.sum(absrate/len(q_XYZ_list),axis=0)
    # Get full rate, Eqn. 5
    prefac = RHO_DM * mass_list**(-1) * 1./mat.rho_T
    rate = prefac * totself
    if coupling.se_shape != 'scalar':
        ## if coupling to a vector, average over polarizations
        ## Eqn. 7
        rate *= 1./3.
    return rate
