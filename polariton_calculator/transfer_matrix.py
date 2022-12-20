'''
transfer_matrix.py
'''

from dataclasses import dataclass
import numpy as np
from constants import *
from material import Material


@dataclass
class TransferMatrix:
    nu: np.ndarray
    k: np.ndarray
    mat: Material
    pol_mixing: bool = True  # set to False at your own risk.
    lam: str = 'vi'  # default is velocity independent
    ground_state: str = 'right'  # default is to calculate matrix from 0 to v,k

    def __post_init__(self):
        # set filename. Right now, this isn't used at all...
        self.fn = f'''transfer_{self.mat}_{self.lam}_{self.ground_state}
                    _{self.pol_mixing}.dat'''
        # get the transfer matrix
        self.tm = self.get_transfer()
        return 0

    def get_mass_energy_term(self, pol_mixing=False):
        if pol_mixing:
            energy = (np.einsum('ij, ik -> ijk', self.mat.bare_ph_energy_o,
                                self.mat.bare_ph_energy_o))**(-0.25)
        else:
            # FIXME: this is probably wrong...
            energy = (2. * self.mat.bare_ph_energy_o)**(-0.25)
        return energy

    def get_transfer(self):
        me = self.get_mass_energy_term(pol_mixing=self.pol_mixing)
        if self.pol_mixing:
            UV = self.mat.UVmats
            num_pol_modes = len(UV[0])//2
            U = UV[:, :num_pol_modes-2, :num_pol_modes]
            V = UV[:, num_pol_modes:2*num_pol_modes - 2, :num_pol_modes]
            if self.ground_state == 'right':
                uvcontrib = (np.conj(U) + np.conj(V))
                dielectricwithxi = np.conj(np.matmul(
                    self.mat.xi_vec_list, np.linalg.inv(self.mat.dielectric)))
                # final tm indices are: q, nu, lambda, nu', a
                tm = np.einsum('ijl, ija, ijk -> ijkla',
                               1j*me, dielectricwithxi, uvcontrib)
            elif self.ground_state == 'left':
                uvcontrib = np.conj(np.conj(U) + np.conj(V))
                dielectricwithxi = np.matmul(
                    self.mat.xi_vec_list, np.linalg.inv(self.mat.dielectric))
                # final tm indices are: q, nu, lambda, nu', b
                tm = np.einsum('ijl, ilb, ilk -> ijklb',
                               1j*me, dielectricwithxi, uvcontrib)
        else:
            # FIXME: this is probably wrong.
            if self.ground_state == 'right':
                dot = np.dot(self.k, np.conj(
                    np.linalg.inv(self.mat.dielectric)))
            elif self.ground_state == 'left':
                dot = np.dot(self.k, np.linalg.inv(self.mat.dielectric))
            tm = np.einsum('ij, ik -> ijk', 1j*me, dot)
        return tm
