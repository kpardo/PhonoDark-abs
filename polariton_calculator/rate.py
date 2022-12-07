'''
rate.py
'''

from dataclasses import dataclass
import numpy as np
from scipy import linalg as sla
import sys
from constants import *
from material import Material
import transfer_matrix as tm
import selfenergy as se
import new_physics as physics


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


def L_func(omega, omega_0, width):
    '''
    lorentzian
    '''
    return (
        4.0*omega[:, np.newaxis]*omega_0*width *
        ((omega[:, np.newaxis]**2 - omega_0**2) **
         2 + (omega[:, np.newaxis]*width)**2)**(-1)
    )


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


def rate(mass_list, q_XYZ_list, mat, width='proportional', pol_mixing=False):
    selfenergy = se.ScalarSE(nu=mat.energies, k=q_XYZ_list, mat=mat,
                             pol_mixing=pol_mixing, lam='vi', uv_op1=r'scalar',
                             uv_op2='scalar')
    if width == 'proportional':
        width_list = 10**(-3)*np.ones((len(mat.energies[0])))
    lorentz = L_func(mass_list, mat.energies[0], width_list)
    prefac = RHO_DM * mass_list**(-2)
    fullself = np.einsum('ij, ljk -> ijkl',
                         lorentz, selfenergy.se)
    # FIXME: should probably make a separate class for vel?
    # for now, just using dressed up version of Tanner's code.
    vel_contrib = get_vel_contrib(q_XYZ_list, np.array([0, 0, VE]))
    # FIXME: need to be more careful with vel int over angles here?
    velint = np.einsum('ijkl, l -> ij', fullself, vel_contrib)
    totself = np.einsum('ij -> i', velint)
    absrate = -1. * np.imag(totself) / mass_list
    rate = prefac * absrate
    return rate


def rate_eff(mass_list, q_XYZ_list, mat, width='proportional', pol_mixing=False):
    selfenergy = se.EffectiveCoup(nu=mat.energies, k=q_XYZ_list, mat=mat,
                                  pol_mixing=pol_mixing, lam='vi')
    print(np.shape(selfenergy.se))
    if width == 'proportional':
        width_list = 10**(-3)*np.ones((len(mat.energies[0])))
    lorentz = L_func(mass_list, mat.energies[0], width_list)
    prefac = RHO_DM * mass_list**(-2)
    # average with B field
    bfield = 10*T_To_eV2**2*1/3  # 10 Tesla averaged over 3 directions.
    fullself = np.einsum('ij, ljk -> ijkl',
                         lorentz, selfenergy.se*bfield)
    # FIXME: should probably make a separate class for vel?
    # for now, just using dressed up version of Tanner's code.
    vel_contrib = get_vel_contrib(q_XYZ_list, np.array([0, 0, VE]))
    # FIXME: need to be more careful with vel int over angles here?
    velint = np.einsum('ijkl, l -> ij', fullself, vel_contrib)
    totself = np.einsum('ij -> i', velint)
    absrate = -1. * np.imag(totself) / mass_list
    rate = prefac * absrate
    return rate
