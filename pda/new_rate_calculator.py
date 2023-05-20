# calculates the scattering rate

import os
import numpy as np
import phonopy
import math
import sys
from dataclasses import dataclass
import pandas as pd

from constants import *
import new_physics as physics
import new_diagonalization as diagonalization
import phonopy_funcs
import my_math
from material import Material


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


def calculate_phi_mat(q_XYZ_list, dielectric, T_mat_list, bare_ph_energy_o, xi_vec_list, vEVec):

    x_hat = np.array([1, 0, 0])
    y_hat = np.array([0, 1, 0])
    z_hat = np.array([0, 0, 1])

    num_q = len(q_XYZ_list)

    num_pol_modes = len(T_mat_list[0])//2

    phi_mat = np.zeros((num_pol_modes, 3, 3), dtype=complex)

    jacob = 4*PI/num_q

    for q in range(num_q):

        q_vec = q_XYZ_list[q]

        q_dir = q_vec / np.linalg.norm(q_vec)

        # get theta/phi values
        theta = math.acos(np.dot(z_hat, q_dir))
        phi = math.atan2(np.dot(q_dir, y_hat), np.dot(q_dir, x_hat))

        int_vel_dist_val = physics.int_vel_dist(theta, phi, vEVec)[0]

        xi_vec = xi_vec_list[q]

        dielectric_inv = np.linalg.inv(dielectric)

        T11_conj = np.conj(T_mat_list[q][:num_pol_modes-2, :num_pol_modes])

        T21_conj = np.conj(
            T_mat_list[q][num_pol_modes:2*num_pol_modes - 2, :num_pol_modes])

        # checking with no polariton mixing
        # T11_conj = np.eye(len(T11_conj))
        # T21_conj = np.zeros(np.shape(T21_conj))

        for lam in range(num_pol_modes-2):
            for nu in range(num_pol_modes - 2):
                for nup in range(num_pol_modes - 2):
                    for a in range(3):
                        for b in range(3):

                            phi_mat[lam][a][b] += jacob*int_vel_dist_val *\
                                (T11_conj[nu][lam] + T21_conj[nu][lam]) *\
                                np.conj((T11_conj[nup][lam] + T21_conj[nup][lam])) *\
                                np.sqrt(bare_ph_energy_o[q][nu]*bare_ph_energy_o[q][nup])**(-1) *\
                                np.conj(np.matmul(dielectric_inv, xi_vec[nu])[a]) *\
                                np.matmul(dielectric_inv, xi_vec[nup])[b]

    return phi_mat


@dataclass
class PhiMatrix():
    '''
    computes the velocity and matrix element parts of the rate
    Inputs: ...
    Outputs: PhiMatrix
    '''
    q_XYZ_list: np.ndarray
    dielectric: np.ndarray
    T_mat_list: np.ndarray
    bare_ph_energy_o: np.ndarray
    xi_vec_list: np.ndarray
    vEVec: np.ndarray

    def __post_init__(self):
        self.phi = self.get_phi()

    @staticmethod
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

    def get_phonon_polariton_contrib(self):
        '''
        gets T_{\nu\nu'}(q), which is |U + V|^2
        Inputs: T matrix list.
        Outputs T_{\nu\nu'}(q) (q=axis0, nu=axis1, nup=axis2)
        '''
        num_pol_modes = len(self.T_mat_list[0])//2
        T11_conj = np.conj(
            self.T_mat_list[:, :num_pol_modes-2, :num_pol_modes])
        T21_conj = np.conj(
            self.T_mat_list[:, num_pol_modes:2*num_pol_modes - 2, :num_pol_modes])
        # 25 x 3 x 5 -- qs, xs, pols
        term1 = T11_conj + T21_conj
        term2 = np.conj(T11_conj + T21_conj)
        return term1, term2

    def get_dielectric_contrib(self):
        '''
        gets dielectric tensor
        Inputs: dielectric, xi_vec
        Outputs:
        '''
        dielectric_inv = np.linalg.inv(self.dielectric)
        term1 = np.conj(np.matmul(self.xi_vec_list, dielectric_inv))
        term2 = np.matmul(self.xi_vec_list, dielectric_inv)
        return term1, term2

    def get_phi(self):
        '''
        gets phi matrix
        '''
        int_vel_dist_val = self.get_vel_contrib(self.q_XYZ_list, self.vEVec)
        Ttensor1, Ttensor2 = self.get_phonon_polariton_contrib()
        dielec1, dielec2 = self.get_dielectric_contrib()
        energyterm = (np.einsum('ij, ik -> ijk', self.bare_ph_energy_o,
                      self.bare_ph_energy_o))**(-0.5)
        jacob = 4 * np.pi / len(self.q_XYZ_list)
        all_but_q = np.einsum('ijk, ilk, ija, ilb, ijl -> ikab', Ttensor1, Ttensor2,
                              dielec1, dielec2, energyterm)
        return np.sum(jacob*int_vel_dist_val[:, np.newaxis, np.newaxis, np.newaxis] *
                      all_but_q, axis=0)


def rate1(m, pol_energy_list, width_list, b_field, phi_mat, m_cell):

    rate = 0.0

    for lam in range(len(pol_energy_list[0]) - 2):

        rate += (1/m_cell)*(E_EM**2*RHO_DM)*m**(-2) *\
            physics.L_func(m, pol_energy_list[0][lam], width_list[lam]) *\
            np.dot(b_field, np.matmul(phi_mat[lam], b_field))

    return rate


def rate1_b_average(m, pol_energy_list, width_list, phi_mat, m_cell, widthfunc='lorentzian'):
    '''
    calculates rate averaged over B-field directions
    '''
    prefac = (1/m_cell)*(E_EM**2*RHO_DM)*m**(-2)*(1./3.)*T_To_eV2**2
    if widthfunc == 'lorentzian':
        width = physics.L_func(m, pol_energy_list[0], width_list)
    elif widthfunc == 'gaussian':
        width = physics.gaussian(m, pol_energy_list[0], width_list)

    return np.sum(prefac[:, np.newaxis] * width * np.trace(phi_mat, axis1=1, axis2=2), axis=1)


def gayy_reach(m, exposure, pol_energy_list, width_list, b_field,
               phi_mat, n_cut, m_cell):
    """
    returns the reach in gayy in GeV
    """

    return np.sqrt(n_cut/(rate1(m, pol_energy_list, width_list, b_field, phi_mat, m_cell) *
                          exposure*KG_YR))*inveV_To_invGeV


def gayy_reach_b_average(m, exposure, pol_energy_list, width_list,
                         phi_mat, n_cut, m_cell):
    """
    returns the reach in gayy in GeV
    """

    return np.sqrt(n_cut/(rate1_b_average(m, pol_energy_list, width_list, phi_mat, m_cell) *
                          exposure*KG_YR))*inveV_To_invGeV


def get_rel_mode_contribution(phi_mat):

    phi_mat_trace = np.zeros(len(phi_mat))

    for lam in range(len(phi_mat) - 2):

        phi_mat_trace[lam] = np.real(np.trace(phi_mat[lam]))

    phi_mat_trace_max = max(phi_mat_trace)

    phi_mat_trace = phi_mat_trace / phi_mat_trace_max

    return phi_mat_trace


def get_mode_contribution(phi_mat):

    phi_mat_trace = np.zeros(len(phi_mat))

    for lam in range(len(phi_mat) - 2):

        phi_mat_trace[lam] = np.real(np.trace(phi_mat[lam]))

    return phi_mat_trace


##################################
dir_path = os.path.dirname(os.path.realpath(__file__))
all_material_list = os.listdir(os.path.join(dir_path, 'material_data'))

b_field_mag = 10  # T
exp = 1			 # kg-yr

n_pol_cut = 3

run_dict = {
    'materials': ['GaAs', 'SiO2', 'Al2O3', 'CaWO4'],
    # 'materials': ['GaAs'],
    # 'bfield': [
    # 			T_To_eV2*np.array([b_field_mag, 0, 0]),
    # 			T_To_eV2*np.array([0, b_field_mag, 0]),
    # 			T_To_eV2*np.array([0, 0, b_field_mag]),
    # 			'average'
    # 		],
    'bfield': ['average'],
    # 'descrip': [
    # 			'Bx',
    # 			'By',
    # 			'Bz',
    # 			'Baverage'
    # ]
    'descrip': ['Baverage']
}


def get_results(m):
    MATERIAL = run_dict['materials'][m]

    print('Material number '+str(m + 1)+' / ' +
          str(len(run_dict['materials']))+' : '+MATERIAL)
    print()

    q_XYZ_list = generate_q_mesh(10**(-4), 5, 5)

    mat = Material(name=MATERIAL, q_xyz=q_XYZ_list)
    width_list = 10**(-3)*np.ones((len(mat.energies[0])))

    print('Computing phi matrix for '+MATERIAL+'...')
    print()

    # phi_mat = calculate_phi_mat(q_XYZ_list, mat.dielectric, mat.UVmats,
    #                             mat.bare_ph_energy_o, mat.xi_vec_list,
    #                             np.array([0, 0, VE]))
    phi_mat = PhiMatrix(q_XYZ_list=q_XYZ_list,
                        dielectric=mat.dielectric,
                        T_mat_list=mat.UVmats,
                        bare_ph_energy_o=mat.bare_ph_energy_o,
                        xi_vec_list=mat.xi_vec_list,
                        vEVec=np.array([0, 0, VE])).phi
    ##### transfer matrix will go here. #####

    # mode_contrib = get_rel_mode_contribution(phi_mat)
    #
    # file = open('./data/'+MATERIAL+'_rel_mode_contribution_and_energy.csv', 'w')
    # file.write('relative contribution to rate (dimensionless), Energy (eV) \n')
    #
    # for lam in range(len(pol_energy_list[0]) - 2):
    # 	file.write(str(mode_contrib[lam])+' , '+str(pol_energy_list[0][lam])+'\n')
    #
    # file.close()
    #
    # print('Done!\n')

    for f in range(len(run_dict['bfield'])):

        B_field = run_dict['bfield'][f]
        descrip = run_dict['descrip'][f]

        print(f'Computing rate for B field : {B_field}')
        print(f'Description is {descrip}')
        print()

        fn = f'./data/{MATERIAL}_gayy_Reach_{descrip}_new.csv'

        num_m = int(1.e6)
        m_list = np.logspace(-2, 0, num_m)
        indices = np.searchsorted(m_list, mat.energies[0, :-2])
        m_list = np.insert(m_list, indices, mat.energies[0, :-2])
        # to -2 in pol_energy_list to avoid acoustic modes.
        if not isinstance(B_field, str):
            reach = np.real(gayy_reach(m_list, exp, mat.energies, width_list,
                                       B_field, phi_mat,
                                       n_pol_cut, mat.m_cell))

        else:
            reach = np.real(gayy_reach_b_average(m_list, exp*b_field_mag**2,
                                                 mat.energies,
                                                 width_list, phi_mat,
                                                 n_pol_cut, mat.m_cell))

            reach_res = np.real(gayy_reach_b_average(mat.energies[0, :-2],
                                                     exp*b_field_mag**2,
                                                     mat.energies,
                                                     width_list, phi_mat,
                                                     n_pol_cut, mat.m_cell))

        table = pd.DataFrame({'m (eV)': m_list,
                              'gayy (GeV^(-1))': reach})
        table.to_csv(fn)

        table_res = pd.DataFrame({'m (eV)': mat.energies[0, :-2],
                                  'gayy (GeV^(-1))': reach_res})

        table_res.to_csv(f'./data/{MATERIAL}_gayy_Reach_{descrip}_new_res.csv')

        print('Done!')
        print()


# for m in range(len(run_dict['materials'])):
#     get_results(m)
