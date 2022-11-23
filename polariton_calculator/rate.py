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


def generate_q_mesh(q_mag, num_q_theta, num_q_phi):
	'''
	creates x,y,z mesh of q values with given magnitude
	distributed across sphere
	'''
	start = (0.5)/num_q_phi ## start and end away from pole

	## sample thetas using arccos function to create more even grid?
	betas = np.linspace(start, (1-start), num_q_theta)
	thetas = np.arccos(2*betas - 1.)

	phis = np.linspace(start*2*np.pi, 2*np.pi*(1-start), num_q_phi)
	pp, tt = np.meshgrid(phis, thetas, indexing='xy')

	xx = np.array([np.sin(tt)*np.cos(pp)], dtype=np.float64)
	yy = np.array([np.sin(tt)*np.sin(pp)], dtype=np.float64)
	zz = np.array([np.cos(tt)], dtype=np.float64)
	qxyz = q_mag*np.vstack([xx,yy,zz]).T

	return qxyz.reshape((num_q_theta*num_q_phi, 3))


def L_func(omega, omega_0, width):
    '''
    lorentzian
    '''
    return (
    	4.0*omega[:,np.newaxis]*omega_0*width*( ( omega[:, np.newaxis]**2 - omega_0**2 )**2 + (omega[:, np.newaxis]*width)**2 )**(-1)
    	)


def rate(mass_list, q_XYZ_list, mat, width='proportional', pol_mixing=False):
    selfenergy = se.ScalarSE(nu=mat.energies, k=q_XYZ_list, j=np.ones(5), mat=mat,
                         pol_mixing=pol_mixing, lam='vi', uv_op1=r'$g_\chi \phi \bar{\psi} \psi$',
                        uv_op2=r'$g_\chi \phi \bar{\psi} \psi$')
    if width == 'proportional':
        width_list = 10**(-3)*np.ones((len(mat.energies[0])))
    lorentz = L_func(mass_list, mat.energies[0], width_list)

    prefac = RHO_DM * mass_list**(-2)
    sumse = np.sum(selfenergy.se[::len(mat.energies[0]), :, :], axis=0) ## FIXME ?
    fullself = np.dot(lorentz, sumse)
    totself = np.sum(fullself, axis=1)
    absrate = -1. * np.imag(totself) / mass_list
    rate = prefac * absrate
    return rate
