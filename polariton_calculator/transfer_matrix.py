'''
transfer_matrix.py
'''

from dataclasses import dataclass
import numpy as np
from scipy import linalg as sla
import sys
from constants import *
from material import Material


@dataclass
class TransferMatrix:
    nu: np.ndarray
    k: np.ndarray
    mat: Material
    pol_mixing: bool = False  # turn off pol mixing for now.
    lam: str = 'vi'  # default is velocity independent
    ground_state: str = 'right'  # default is to calculate matrix from 0 to v,k

    def __post_init__(self):
        # make container for TM
        # get all ingredients for calc.
        # get the transfer matrix
        # save to file.
        self.fn = f'''transfer_{self.mat}_{self.lam}_{self.ground_state}
                    _{self.pol_mixing}.dat'''
        self.tm = self.get_transfer()
        return 0

    def get_mass_energy_term(self):
        return (2. * self.mat.bare_ph_energy_o)**(-0.25)

    def get_transfer(self):
        me = self.get_mass_energy_term()
        if self.pol_mixing:
            UV = self.mat.UVmats
            num_pol_modes = len(UV[0])//2
            U = UV[:, :num_pol_modes-2, :num_pol_modes]
            V = UV[:, num_pol_modes:2*num_pol_modes - 2, :num_pol_modes]
            if self.ground_state == 'right':
                uvcontrib = (np.conj(U) + V)
                dielectricwithxi = np.conj(np.matmul(
                    self.mat.xi_vec_list, self.mat.dielectric))
                dot = np.einsum('ijk, ik -> ijk', dielectricwithxi, self.k)
                tm = np.einsum('ij, ilk, ilj -> ijk',
                               1j*me, dot, uvcontrib)
            elif self.ground_state == 'left':
                uvcontrib = np.conj((np.conj(U) + V))
                dielectricwithxi = np.matmul(
                    self.mat.xi_vec_list, self.mat.dielectric)
                dot = np.einsum('ijk, ik -> ijk', dielectricwithxi, self.k)
                tm = np.einsum('ij, ilk, ilj -> ijk',
                               1j*me, dot, uvcontrib)
        else:
            if self.ground_state == 'right':
                dot = np.dot(self.k, np.conj(self.mat.dielectric))
            elif self.ground_state == 'left':
                dot = np.dot(self.k, self.mat.dielectric)
            tm = np.einsum('ij, ik -> ijk', 1j*me, dot)
        return tm


@dataclass
class TMEff(TransferMatrix):
    '''
    for effective couplings. doesn't dot dielectric with k.
    '''

    def get_transfer(self):
        me = self.get_mass_energy_term(pol_mixing=self.pol_mixing)
        if self.pol_mixing:
            UV = self.mat.UVmats
            num_pol_modes = len(UV[0])//2
            U = UV[:, :num_pol_modes-2, :num_pol_modes]
            V = UV[:, num_pol_modes:2*num_pol_modes - 2, :num_pol_modes]
            # FIXME: Don't understand why I need inv dielectric ?
            # FIXME: Also why conj(U)+conj(V) instead of just conj(U) + V?
            if self.ground_state == 'right':
                uvcontrib = (np.conj(U) + np.conj(V))
                dielectricwithxi = np.conj(np.matmul(
                    self.mat.xi_vec_list, np.linalg.inv(self.mat.dielectric)))
                tm = np.einsum('ijl, ija, ijk -> ijkla',
                               1j*me, dielectricwithxi, uvcontrib)
            elif self.ground_state == 'left':
                uvcontrib = np.conj(np.conj(U) + np.conj(V))
                dielectricwithxi = np.matmul(
                    self.mat.xi_vec_list, np.linalg.inv(self.mat.dielectric))
                tm = np.einsum('ijl, ilb, ilk -> ijklb',
                               1j*me, dielectricwithxi, uvcontrib)
        else:
            if self.ground_state == 'right':
                dielec = np.conj(self.mat.dielectric)
            elif self.ground_state == 'left':
                dielec = self.mat.dielectric
            tm = np.einsum('ij, kl -> ijkl', 1j*me, dielec)
        return tm

    def get_mass_energy_term(self, pol_mixing=False):
        if pol_mixing:
            energy = (np.einsum('ij, ik -> ijk', self.mat.bare_ph_energy_o,
                                self.mat.bare_ph_energy_o))**(-0.25)
        else:
            energy = (2. * self.mat.bare_ph_energy_o)**(-0.25)
        return energy
